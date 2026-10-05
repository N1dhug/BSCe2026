"""Small G4 equation/unit, bounded-fit, archive and causal replay controls."""
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.deterministic import _jsonable
from src.exe.run_model.bm_1.deterministic_response import main
from src.method.bm_1_deterministic_fit import BM1DeterministicInputs
from src.method.bm_1_deterministic_response import BM1ResponseInputs, fit_response
from src.model.bm_1 import bm_1
from src.params.bm_1_response import BM1PiecewiseResponse
from src.params.bm_1_deterministic_response import BM1ResponseModel
from src.tools.input_generator.input_builder import input_builder
from src.tools.bm_1_deterministic_snapshot import export_fit_snapshot
from src.tools.bm_1_deterministic_response import (read_training_archive,fit_training_archive,
    validate_response_snapshot,load_response_snapshot,forecast_response,score_response_forecast)
from src.method.bm_1_deterministic_benchmark import run_benchmark
from src.params.bm_1_deterministic import BM1DeterministicConfig


class ResponseNumericsTests(unittest.TestCase):
    def test_linear_delegation_is_exact_and_hourly_mean_balances_energy(self):
        time = np.linspace(0,24,289); baseline = .4+.1*np.sin(time/3)
        price = .5+.3*np.cos(np.arange(24)/2); p = [2.,2.,.5,.5]
        for contract in ('hourly-point','hourly-mean'):
            original = BM1DeterministicInputs(time,baseline,price,observation_contract=contract)
            adapter = BM1ResponseInputs(time,baseline,price,observation_contract=contract)
            before, after = original.solve(p,keep_fine=True), adapter.solve(p,keep_fine=True)
            for name in ('state','demand','fine_state','fine_demand'): np.testing.assert_array_equal(before[name],after[name])
        g = BM1PiecewiseResponse.from_inner_values([.35],pieces=2)
        adapter.price_response = g
        curved = adapter.solve(p)
        np.testing.assert_allclose(curved['demand'],adapter.b_mean+p[0]*np.diff(curved['state']),rtol=0,atol=1e-15)

    def test_curved_demand_drift_and_price_jumps_match_general_bm_in_hours_and_days(self):
        for fmid in (0.,-.2):
            f = BM1PiecewiseResponse.from_inner_values([fmid],pieces=2)
            for gmid in (-1.,.3,1.):
                g = BM1PiecewiseResponse.from_inner_values([gmid],pieces=2)
                spec = BM1ResponseModel(2.,2.,.6,f,g)
                baseline = np.array([.3,.7,.4,.6,.5]); prices = np.array([0.,.5,1.,.2])
                time = np.linspace(0,4,97); fine_baseline = np.interp(time,np.arange(5),baseline)
                for x0 in (0.,.45,1.):
                    inputs = BM1ResponseInputs(time,fine_baseline,prices,state_response=f,price_response=g,steps_per_hour=24)
                    out = inputs.solve([2.,2.,.6,x0],keep_fine=True)
                    for unit,scale in (('hours',1.),('days',24.)):
                        b = input_builder.energy((time/scale,fine_baseline))
                        u = input_builder.price_intervals((np.arange(5)/scale,prices))
                        params = spec.as_bm1_parameters(time_unit=unit).parameter_template(np.arange(5)/scale); params[0] = u
                        model = bm_1(b,b,(0.,4/scale),97)
                        general = model.deterministic_solver([x0],params,t_eval=time/scale,
                            breakpoints=np.arange(1,4)/scale,rtol=1e-10,atol=1e-12,max_step=1/(24*scale))
                        np.testing.assert_allclose(out['fine_state'],general[1],rtol=0,atol=3e-7)
                        np.testing.assert_allclose(out['fine_demand'],general[-1],rtol=0,atol=3e-7)
                        # Boundary/interior point demand implies exactly the same drift.
                        for x in (0.,.5,1.):
                            probe = BM1ResponseInputs([0.,1.],[.4,.4],[.5],state_response=f,price_response=g)
                            demand = probe.solve([2.,2.,.6,x])['demand'][0]
                            _,bt,_,_,_,dt = model.eval(.5/scale,x,u,f,g,2.,.6)
                            expected = .4+.6*np.tanh((f(x)+g(.5)))*(.6 if f(x)+g(.5)>=0 else .4)
                            self.assertAlmostEqual(demand,expected,places=14)
                            self.assertAlmostEqual(model._drift(.5/scale,x,params),(dt-bt)/(2/scale),places=13)

    def fixture(self):
        time = np.linspace(0,36,433); baseline = .45+.1*np.sin(time/3)
        price = .5+.4*np.sin(np.arange(36)*1.17)
        g = BM1PiecewiseResponse.from_inner_values([.3],pieces=2)
        inputs = BM1ResponseInputs(time,baseline,price,price_response=g)
        actual = inputs.solve([2.5,2.,.6,.5])['demand']
        actual[[5,19]] = np.nan
        return BM1ResponseInputs(time,baseline,price,actual,observation_mask=np.isfinite(actual))

    def test_joint_shared_fit_recovers_synthetic_shape_and_separates_penalty(self):
        inputs = self.fixture()
        result = fit_response(inputs,[4.,2.,.5,.5],regularizations=(0.,.001),starts=(0.,),max_calls=350,calls_per_start=175,seconds=20.)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['observed_samples'],34)
        self.assertLessEqual(result['counts']['residual_calls'],350)
        best = result['candidates'][0]['best_converged']
        np.testing.assert_allclose(best['parameters'],[2.5,2.,.6,.5],rtol=0,atol=3e-4)
        self.assertAlmostEqual(best['g_mid'],.3,places=4)
        self.assertLess(best['mse'],1e-11)
        for candidate in result['candidates']:
            best = candidate['best_converged']
            self.assertAlmostEqual(best['penalty'],candidate['regularization']*best['g_mid']**2)
            self.assertAlmostEqual(best['regularized_objective'],best['mse']+best['penalty'])
        self.assertIn('not-selected',result['selection'])

    def test_optional_x0_and_global_probe_budget_are_explicit(self):
        inputs = self.fixture()
        result = fit_response(inputs,[4.,2.,.5,.5],fixed_x0=None,regularizations=(0.,),starts=(0.,),max_calls=300,calls_per_start=300,seconds=20.)
        self.assertEqual(result['fixed_x0'],None)
        self.assertEqual(result['optimized_coordinates'][-1],'x0')
        np.testing.assert_allclose(result['candidates'][0]['best_converged']['parameters'],[2.5,2.,.6,.5],rtol=0,atol=5e-4)
        short = fit_response(inputs,[4.,2.,.5,.5],max_calls=2,calls_per_start=2,seconds=20.)
        self.assertEqual(short['counts']['residual_calls'],2)
        self.assertEqual(short['status'],'budget-stopped')
        self.assertIsNone(short['candidates'][0]['best_converged'])
        for option in ({'regularizations':(0.,.001,.01)},{'max_calls':3001},{'seconds':601.},{'starts':(1.1,)},{'regularizations':(.01,)}):
            with self.assertRaises(ValueError): fit_response(inputs,[4.,2.,.5,.5],**option)


class ResponseArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range('2022-01-01',periods=360,freq='h'); t = np.arange(len(index))
        cls.raw = pd.DataFrame({'Demand':2+.3*np.sin(t/5)+.01*np.cos(t/2),
            'Price':100+30*np.sin(t/4),'Temp':5+np.cos(t/8)},index=index)
        config = BM1DeterministicConfig(starts=((4.,2.,.6,.5),),fixed_x0=.5,steps_per_hour=12)
        fitted = {'status':'converged','fixed_x0':.5,'best_converged':{'parameters':[4.,2.,.6,.5]}}
        with patch('src.method.bm_1_deterministic_benchmark.fit_deterministic',return_value=fitted):
            cls.reference_result = run_benchmark(cls.raw,index[0],index[192],config=config,forecast_hours=24,wm_steps_per_hour=12,price_bounds=(60.,140.))
        cls.reference_result['fit']['best_converged']['mse'] = cls.reference_result['fit_metrics_normalized']['bm1']['mse']
        cls.reference_result['fit']['observed_samples'] = 192
        cls.reference = _jsonable(export_fit_snapshot(cls.reference_result))

    def archive(self, path):
        (path/'fit_snapshot.json').write_text(json.dumps(self.reference),encoding='utf-8')
        self.reference_result['fit_normalized'].to_csv(path/'fit_predictions_normalized.csv',index_label='Datetime')
        self.reference_result['fit_signals'].to_csv(path/'fit_signals_normalized.csv',index_label='Datetime')

    def candidate_report(self, inputs, p, **kwargs):
        # Declared curved fixture isolates archive/export/replay from LS testing above.
        q = [3.,p[1],.5,p[3]]
        output = inputs.solve(q,price_response=BM1PiecewiseResponse.from_inner_values([.25],pieces=2))
        mask = inputs.observation_mask
        mse = float(np.mean((output['demand'][mask]-inputs.observations[mask])**2))
        best = dict(parameters=q,g_mid=.25,mse=mse,penalty=0.,regularized_objective=mse,converged=True)
        return dict(status='complete',fixed_x0=.5,fixed_k=p[1],counts={},candidates=[dict(regularization=0.,best_converged=best)])

    def fitted(self, root):
        self.archive(root)
        with patch('src.tools.bm_1_deterministic_response.fit_response',side_effect=self.candidate_report):
            report, results = fit_training_archive(root,seconds=20.)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(len(results),1)
        return results[0]

    def test_archive_replay_rejects_changed_inputs_masks_cutoff_and_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.archive(root)
            reference, inputs, hourly, fine, hashes, _ = read_training_archive(root)
            self.assertEqual(reference['origin'],self.reference['origin'])
            self.assertEqual(len(hashes),2)
            original = (root/'fit_predictions_normalized.csv').read_bytes()
            for column in ('U','x','observed','demand_imputed'):
                changed = hourly.copy()
                if column == 'demand_imputed': changed.iloc[1,changed.columns.get_loc(column)] = True
                else: changed.iloc[1,changed.columns.get_loc(column)] += .01
                changed.to_csv(root/'fit_predictions_normalized.csv',index_label='Datetime')
                with self.assertRaises((ValueError,AssertionError)): read_training_archive(root)
                (root/'fit_predictions_normalized.csv').write_bytes(original)
            fine.iloc[:-1].to_csv(root/'fit_signals_normalized.csv',index_label='Datetime')
            with self.assertRaisesRegex(ValueError,'complete hourly/fine'): read_training_archive(root)
            self.archive(root)
            reference['manifest']['warmup_hours'] = 24
            (root/'fit_snapshot.json').write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError,'zero-warmup'): read_training_archive(root)

    def test_snapshot_curved_origin_round_trip_is_causal_and_preserves_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); result = self.fitted(root); snapshot = _jsonable(result['fit_snapshot'])
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            g4 = root/'g4'; g4.mkdir(); (g4/'fit_snapshot.json').write_text(json.dumps(snapshot))
            with patch('src.tools.bm_1_deterministic_response.fit_response',side_effect=AssertionError('refit')), \
                 patch('src.exe.run_model.wm_4.output_model.statistical_output.fit',side_effect=AssertionError('baseline refit')):
                loaded = load_response_snapshot(g4)
                first = forecast_response(loaded,self.raw,forecast_hours=24)
                second = forecast_response(loaded,self.raw.drop(columns='Demand'),forecast_hours=24)
            pd.testing.assert_frame_equal(first['predictions'],second['predictions'])
            self.assertEqual(first['normalized_predictions'].x.iloc[0],snapshot['x_origin'])
            self.assertNotAlmostEqual(snapshot['x_origin'],self.reference['bm']['x_origin'],places=4)
            np.testing.assert_array_equal(first['normalized_predictions'].iloc[0][['F','R']],self.reference['wm']['state_origin'])
            self.assertTrue(first['numerics']['passed'])
            scale = self.reference['manifest']['scales']
            np.testing.assert_allclose(first['predictions'].linear_bm1,first['normalized_predictions'].linear_bm1*scale['demand_scale']+scale['demand_min'])
            y = self.raw.Demand.copy(); y.loc[first['predictions'].index[:3]] = np.nan
            with pd.option_context('mode.copy_on_write',True): scored = score_response_forecast(first,y)
            self.assertEqual(scored['scoring']['scored_samples'],21)
            self.assertEqual(scored['scoring']['models'],['O','bm1','wm4','linear_bm1'])
            self.assertEqual(score_response_forecast(first,None)['metrics'],{})
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir() if p.is_file()})
            for mutation in (lambda s:s.update(x_origin=1.1),lambda s:s['model'].update(time_unit='days'),
                lambda s:s['fit']['best_converged']['parameters'].__setitem__(1,3.),
                lambda s:s['fit']['best_converged'].update(g_mid=.5),lambda s:s['numerics'].update(tolerance=.01),
                lambda s:s['forward_source_sha256'].update({next(iter(s['forward_source_sha256'])):'bad'})):
                corrupt = deepcopy(snapshot); mutation(corrupt)
                with self.assertRaises(ValueError): validate_response_snapshot(corrupt)

    def test_cli_persists_input_only_forecast_before_scoring_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); result = self.fitted(root)
            g4 = root/'g4'; g4.mkdir(); snapshot = _jsonable(result['fit_snapshot'])
            snapshot['reference']['manifest']['data_audit'] = {'tag':'MED'}
            (g4/'fit_snapshot.json').write_text(json.dumps(snapshot))
            out = root/'forecast'
            def failed_scoring(*args):
                self.assertTrue((out/'forecast_predictions.csv').exists())
                self.assertTrue((out/'forecast_status.json').exists())
                raise OSError('Missing observation workbook')
            with patch('src.tools.bm_1_deterministic_data.load_forecast_inputs',return_value=self.raw.drop(columns='Demand')), \
                 patch('src.tools.bm_1_deterministic_data.load_forecast_observations',side_effect=failed_scoring),redirect_stdout(io.StringIO()):
                code = main(['--from-fit',str(g4),'--output',str(out)])
            self.assertEqual(code,2)
            self.assertEqual(json.loads((out/'result.json').read_text())['scoring']['status'],'failed')
            self.assertNotIn('observed',pd.read_csv(out/'forecast_predictions.csv'))
            self.assertTrue((out/'comparison.png').exists())
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(['--from-fit',str(g4),'--seconds','10'])

    def test_cli_fit_exports_candidates_without_loading_forecast_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.archive(root); out = root/'pilot'
            with patch('src.tools.bm_1_deterministic_response.fit_response',side_effect=self.candidate_report), \
                 patch('src.tools.bm_1_deterministic_data.load_forecast_inputs',side_effect=AssertionError('future input')),redirect_stdout(io.StringIO()):
                code = main(['--reference-fit',str(root),'--output',str(out),'--seconds','20'])
            self.assertEqual(code,0)
            variant = out/'variants/lambda_00'
            self.assertEqual(load_response_snapshot(variant)['kind'],'G4-deterministic-fit')
            self.assertTrue((variant/'fit.png').exists())
            self.assertTrue((variant/'response.png').exists())
            self.assertEqual(json.loads((out/'pilot.json').read_text())['candidates'][0]['export_status'],'complete')

    def test_real_archive_optimizer_exports_a_consistent_checked_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.archive(root)
            report, results = fit_training_archive(root,seconds=20.,regularizations=(0.,),
                starts=(0.,),max_calls=250,calls_per_start=250)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(len(results),1)
            result = results[0]
            self.assertAlmostEqual(result['fit']['best_converged']['mse'],result['fit_metrics_normalized']['bm1']['mse'],places=13)
            self.assertLessEqual(report['counts']['residual_calls'],250)
            self.assertEqual(report['counts']['all_forward_solves'],report['counts']['forward_solves']+3)
            self.assertEqual(result['fit_snapshot']['reference'],self.reference)


if __name__ == '__main__': unittest.main()

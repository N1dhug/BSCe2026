"""Physical process limits, measured-only scoring, and explicit invalidity."""
from copy import deepcopy
import io
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests import test_bm1_s3_forecast as s3_fixture
from src.method.bm_1_s3_forecast import predict_s3,save_snapshot
from src.method.bm_1_s3_uncertainty import (
    demand_values,fit_variance_amplitude,_fv_day,process_forecast,score_process_forecast)
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.params.bm_1_response import BM1PiecewiseResponse
from src.model.bm_1 import bm_1
from src.exe.run_model.bm_1.s3 import main


class S3UncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s3_fixture.S3ForecastTests.setUpClass()
        cls.snapshot=s3_fixture.S3ForecastTests.snapshot;cls.raw=s3_fixture.S3ForecastTests.raw

    def test_variance_ls_recovers_amplitude_and_excludes_missing_targets(self):
        fit=fit_variance_amplitude([1.,2.,100.],[3.,6.,np.nan])
        self.assertEqual(fit['q_per_hour'],3.)
        self.assertAlmostEqual(fit['sigma_per_sqrt_day']**2,72.)
        self.assertEqual(fit_variance_amplitude([0.,0.],[1.,2.])['status'],'unidentifiable')
        with self.assertRaises(ValueError):fit_variance_amplitude([1.],[-1.])

    def test_readout_matches_existing_model_and_monotone_support(self):
        x=np.linspace(0,1,101);g=BM1PiecewiseResponse.from_inner_values([.3],pieces=2)
        b=.4;k=2.;xi=.6
        model=bm_1(lambda t:b,lambda t:b,(0.,1.),2)
        native=model.eval(0.,x,lambda t:.3,BM1PiecewiseResponse.linear(pieces=2),g,k,xi)[-1]
        actual=demand_values(x,b,g(.3),k,xi)
        np.testing.assert_allclose(actual,native,rtol=0,atol=5e-16)
        self.assertTrue(np.all(np.diff(actual)<=0))

    def test_zero_noise_is_exact_s3_and_ignores_future_demand_without_refit(self):
        with patch('src.exe.run_model.wm_4.output_model.statistical_output.fit',side_effect=AssertionError('refit')):
            a=process_forecast(self.snapshot,self.raw,sigma_x=0.)
            changed=self.raw.copy();changed.Demand=1e20
            b=process_forecast(self.snapshot,changed,sigma_x=0.)
        pd.testing.assert_frame_equal(a['predictions'],b['predictions'])
        np.testing.assert_array_equal(a['predictions'].prediction,a['predictions'].process_mean)
        self.assertTrue((a['predictions'].process_variance==0).all())
        self.assertEqual(a['manifest']['stochastic_status'],'implemented-conditional-process-scenario')

    def test_real_fv_pure_diffusion_conserves_mass_and_bound_on_variance(self):
        # Xi=0: demand is constant, drift zero. State variance must grow while
        # demand variance remains zero; this catches wrong readout/noise coupling.
        t=np.arange(121)/60
        inp=BM1ResponseInputs(t,np.full_like(t,.4),[.3,.7],steps_per_hour=12)
        out=_fv_day(inp,[4.,2.,0.,.5],BM1PiecewiseResponse.linear(pieces=2),.5,.2,64,1/24,perf_counter()+10)
        self.assertGreater(out['state_variance'][1],0.)
        self.assertLessEqual(out['state_variance'][1],.25)
        np.testing.assert_allclose(out['demand_mean'],.4,atol=1e-12)
        np.testing.assert_allclose(out['demand_variance'],0.,atol=1e-25)
        self.assertLess(out['diagnostics']['maximum_abs_mass_error'],1e-10)
        self.assertFalse(out['diagnostics']['used_clipping'])
        with self.assertRaisesRegex(RuntimeError,'budget stop'):
            _fv_day(inp,[4.,2.,0.,.5],BM1PiecewiseResponse.linear(pieces=2),.5,.2,64,1/24,perf_counter()-1)

    def test_scores_have_correct_units_mask_interval_score_and_no_mutation(self):
        a=process_forecast(self.snapshot,self.raw,sigma_x=0.)
        observed=a['predictions'].process_mean+1.;observed.iloc[0]=np.inf
        b=score_process_forecast(a,observed)
        self.assertEqual(b['scoring']['samples'],167)
        self.assertAlmostEqual(b['scoring']['metrics']['process_mean']['mse'],1.)
        self.assertAlmostEqual(b['scoring']['mean_process_interval_score95'],40.)
        self.assertEqual(b['scoring']['process_band_coverage95'],0.)
        self.assertNotIn('observed',a['predictions'])
        scale=a['manifest']['scales']
        np.testing.assert_allclose(b['normalized_predictions'].observed,
            (b['predictions'].observed-scale['demand_min'])/scale['demand_scale'],equal_nan=True)

    def test_failed_refinement_never_reports_complete(self):
        def fake(inp,*args):
            cells=args[4];n=inp.n_hours
            out={k:np.zeros(n) for k in ('demand_mean','demand_variance','state_mean','state_variance')}
            out['state_mean'][:]=.4 if cells==32 else .6
            out.update(demand_quantiles=np.zeros((n,3)),state_cdf_after_start=np.zeros((n-1,3)),
                diagnostics={},initial_state_discretization_error=0.)
            return out
        with patch('src.method.bm_1_s3_uncertainty._fv_day',side_effect=fake):
            result=process_forecast(self.snapshot,self.raw,sigma_x=.2,cells=32)
        self.assertEqual(result['status'],'numerics-failed')
        self.assertFalse(result['numerics']['passed'])

    def test_nonzero_spring_forecast_preserves_physical_hours_and_raw_variance(self):
        s=deepcopy(self.snapshot);origin=pd.Timestamp('2022-03-21');s['origin']=str(origin)
        s['history']['days_by_weekday']=[[str((origin+pd.Timedelta(days=d)-pd.Timedelta(weeks=w)).date())
            for w in range(1,6)] for d in range(7)]
        index=pd.date_range(origin,origin+pd.Timedelta(days=7),freq='h',inclusive='left')
        inputs=pd.DataFrame({'Price':100.,'Temp':5.},index=index)
        result=process_forecast(s,inputs,sigma_x=.2)
        p=result['predictions'];n=result['normalized_predictions']
        self.assertEqual(len(p),167)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'),p.index)
        self.assertEqual(p.time_utc.nunique(),167)
        self.assertTrue((p.process_state_variance.between(0,.25)).all())
        self.assertTrue((p.process_lower95<=p.process_upper95).all())
        np.testing.assert_allclose(p.process_variance,n.process_variance*s['scales']['demand_scale']**2)

    def test_cli_forecast_survives_missing_observation_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);save_snapshot(root/'s.json',self.snapshot)
            self.raw[['Price','Temp']].to_csv(root/'inputs.csv',index_label='Datetime')
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=main(['point-process','predict','--snapshot',str(root/'s.json'),'--inputs-csv',str(root/'inputs.csv'),
                    '--sigma-x','0','--output',str(root/'out'),'--observations-csv',str(root/'missing.csv')])
            self.assertEqual(code,2);self.assertTrue((root/'out/forecast.json').exists())


class S3CommandTests(unittest.TestCase):
    def test_variants_keep_explicit_observation_semantics(self):
        from src.exe.run_model.bm_1.s3 import parser
        common = ['--snapshot', 'saved.json', '--inputs-csv', 'inputs.csv', '--output', 'out']
        interval = parser().parse_args(['interval', 'predict', *common])
        self.assertIsNone(interval.sigma_x)
        process = parser().parse_args(['point-process', 'predict', *common, '--sigma-x', '.2'])
        self.assertEqual(process.sigma_x, .2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['point-process', 'predict', *common])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['residual', 'diagnose', *common])

    def test_all_prediction_variants_persist_before_target_access(self):
        from contextlib import ExitStack
        from src.exe.run_model.bm_1.s3 import parser, _predict
        cases = [
            ('point-process', ['--sigma-x', '.2'], 'bm_1_s3_uncertainty.process_forecast'),
            ('interval', [], 'bm_1_s3_interval.predict_interval'),
            ('interval', ['--sigma-x', '.2'], 'bm_1_s3_interval_process.process_interval'),
            ('residual', ['--residual-snapshot', 'r.json'], 'bm_1_s3_residual.predict_residual'),
        ]
        for variant, extra, function in cases:
            with self.subTest(variant=variant, extra=extra), ExitStack() as stack:
                events = []
                snapshot = {'unchanged': 'saved snapshot'}
                inputs = object()
                result = {'unchanged': 'numerical result'}
                args = parser().parse_args([variant, 'predict', '--snapshot', 's.json',
                    '--inputs-csv', 'inputs.csv', '--output', 'out',
                    '--observations-csv', 'missing.csv', *extra])
                stack.enter_context(patch('src.tools.bm_1_deterministic_data.raw_csv', return_value=inputs))
                stack.enter_context(patch('src.method.bm_1_s3_forecast.load_snapshot', return_value=snapshot))
                stack.enter_context(patch('src.method.bm_1_s3_interval.load_snapshot', return_value=snapshot))
                stack.enter_context(patch('pathlib.Path.read_text', return_value='{}'))
                predict = stack.enter_context(patch('src.method.'+function, return_value=result))
                write = stack.enter_context(patch('src.exe.run_model.bm_1.s3_forecast.write_forecast',
                    side_effect=lambda *a, **kw: events.append('forecast persisted')))
                def open_targets(*a, **kw):
                    self.assertEqual(events, ['forecast persisted'])
                    raise FileNotFoundError('missing target file')
                stack.enter_context(patch('pandas.read_csv', side_effect=open_targets))
                with self.assertRaises(FileNotFoundError):
                    _predict(args, Path('out'))
                self.assertIs(predict.call_args.args[0], snapshot)
                self.assertIs(predict.call_args.args[-1], inputs)
                write.assert_called_once_with(Path('out'), result)


if __name__=='__main__':unittest.main()

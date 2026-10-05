"""Focused G regressions: numerical reference, estimation, budgets and cutoff."""
import unittest
import numpy as np
import pandas as pd
from src.model.bm_1 import bm_1
from src.params.bm_1_deterministic import BM1DeterministicConfig
from src.method.bm_1_deterministic_fit import (BM1DeterministicInputs, fit_deterministic,
                                            exact_linear_interval, linear_demand, BudgetStop)
from src.tools.bm_1_deterministic_data import prepare_frame, common_inputs
from src.exe.run_model.wm_4.output_model import statistical_output


def inputs(hours=48, steps=12, contract='hourly-point'):
    time = np.arange(hours*60+1)/60
    baseline = .45+.12*np.sin(time/4)+.03*np.cos(time/1.3)
    price = np.random.default_rng(421).uniform(.05,.95,hours)
    return BM1DeterministicInputs(time,baseline,price,steps_per_hour=steps,observation_contract=contract)


class DeterministicFitTests(unittest.TestCase):
    def test_scalar_kernel_matches_existing_model(self):
        model = bm_1(lambda t:.4,lambda t:0.,(0.,1.),2)
        for x,u,b in np.random.default_rng(9).uniform(0,1,(100,3)):
            actual = linear_demand(x,u,b,3.,.8)
            expected = model.D(b,.8,model.l(1-2*x,1-2*u,3.))
            self.assertAlmostEqual(actual,float(expected),places=14)

    def test_price_switches_and_linear_baseline_against_adaptive_ode(self):
        data = inputs(hours=8)
        p = np.array([2.,4.,.8,.2])
        B = lambda t:np.interp(t,data.time,data.baseline)
        def U(t):
            return data.prices[np.minimum(np.floor(np.asarray(t)).astype(int),7)]
        model = bm_1(B,lambda t:0.,(0.,8.),9)
        reference = model.deterministic_solver([p[3]],(U,lambda x:1-2*x,lambda u:1-2*u,*p[:3]),
                                               t_eval=np.arange(9.),breakpoints=np.arange(1.,8.),rtol=1e-10,atol=1e-12)
        actual = data.solve(p)
        np.testing.assert_allclose(actual['state'],reference[1],atol=2e-6,rtol=0)
        np.testing.assert_allclose(actual['point_demand'],reference[-1][:-1],atol=2e-6,rtol=0)

    def test_exact_constant_interval_and_energy(self):
        data = BM1DeterministicInputs(np.arange(5.),np.full(5,.4),np.full(4,.7),observation_contract='hourly-mean')
        actual = data.solve((6.,2.,.7,.1))
        x = .1
        for i in range(4):
            x, demand = exact_linear_interval(x,.4,.7,1.,6.,2.,.7)
            self.assertAlmostEqual(actual['state'][i+1],x,places=8)
            self.assertAlmostEqual(actual['demand'][i],demand,places=8)
        np.testing.assert_allclose(actual['mean_demand']-.4,6*np.diff(actual['state']),atol=1e-14)

    def test_point_and_mean_are_distinct(self):
        data = inputs(8)
        out = data.solve((2.,4.,.8,.2))
        self.assertGreater(np.max(np.abs(out['point_demand']-out['mean_demand'])),.01)

    def test_no_flexibility_and_bounds(self):
        data = inputs()
        out = data.solve((12.,2.,0.,0.))
        np.testing.assert_array_equal(out['state'],0.)
        np.testing.assert_allclose(out['demand'],np.interp(np.arange(48),data.time,data.baseline))
        for x0 in (0.,1.):
            x = data.solve((1.,10.,1.,x0))['state']
            self.assertTrue(np.all((x>=0)&(x<=1)))

    def test_bad_domains_and_unstable_grid_rejected(self):
        with self.assertRaises(ValueError):
            BM1DeterministicInputs([0,1],[.4,.4],[1.01])
        with self.assertRaises(ValueError):
            inputs(8,steps=1).solve((1.,10.,1.,.5))
        with self.assertRaises(ValueError):
            BM1DeterministicConfig(starts=((1.,2.,3.,.5),))

    def test_synthetic_parameter_fit(self):
        data = inputs(hours=120)
        truth = (8.,2.,.6,.3)
        data.observations = data.solve(truth)['demand']
        config = BM1DeterministicConfig(starts=((12.,1.,.4,.4),),max_calls=250)
        fit = fit_deterministic(data,config)
        self.assertEqual(fit['status'],'converged')
        p = fit['best_converged']['parameters']
        np.testing.assert_allclose(p,truth,rtol=.01,atol=.001)
        self.assertLess(fit['best_converged']['mse'],1e-10)
        self.assertGreater(fit['counts']['residual_calls'],5)
        self.assertEqual(fit['counts']['forward_solves'],fit['counts']['residual_calls'])

    def test_unidentifiable_zero_xi_reports_low_rank(self):
        data = inputs(24)
        data.observations = np.interp(np.arange(24),data.time,data.baseline)
        fit = fit_deterministic(data,BM1DeterministicConfig(starts=((12.,2.,0.,.5),)))
        self.assertEqual(fit['status'],'converged')
        self.assertLess(fit['best_converged']['local_rank'],4)

    def test_budget_includes_jacobian_and_saves_unfinished(self):
        data = inputs(8)
        data.observations = data.solve((8.,2.,.6,.3))['demand']
        saved = []
        fit = fit_deterministic(data,BM1DeterministicConfig(starts=((12.,1.,.4,.4),),max_calls=3),checkpoint=saved.append)
        self.assertEqual(fit['status'],'unfinished')
        self.assertEqual(fit['counts']['residual_calls'],3)
        self.assertIsNone(fit['best_converged'])
        self.assertIsNotNone(fit['best_provisional'])
        self.assertEqual(saved[-1]['status'],'unfinished')

    def test_expired_solve_budget(self):
        with self.assertRaises(BudgetStop):
            inputs().solve((12.,2.,.5,.5),deadline=0.)


class DataCutoffTests(unittest.TestCase):
    def raw(self):
        idx = pd.date_range('2022-01-01',periods=240,freq='h')
        t = np.arange(len(idx))
        return pd.DataFrame({'Demand':2+.3*np.sin(t/5)+.01*np.cos(t/2),
                             'Price':100+30*np.sin(t/4),'Temp':5+np.cos(t/8)},index=idx)

    def test_scales_and_baseline_do_not_use_test_demand(self):
        raw = self.raw(); changed = raw.copy()
        cutoff = pd.Timestamp('2022-01-09')
        changed.loc[changed.index>=cutoff,'Demand'] += 1000
        frames = [prepare_frame(r,'2022-01-01',cutoff) for r in (raw,changed)]
        self.assertEqual(frames[0][1],frames[1][1])
        prepared = []
        for frame,_ in frames:
            model = statistical_output().fit(frame,cutoff)
            prepared.append(common_inputs(frame,model,cutoff,'2022-01-11'))
        np.testing.assert_array_equal(prepared[0]['baseline'],prepared[1]['baseline'])
        self.assertNotIn('observed',prepared[0])

    def test_missing_future_demand_does_not_block_prediction_setup(self):
        raw = self.raw()
        frame,_ = prepare_frame(raw,'2022-01-01','2022-01-09')
        model = statistical_output().fit(frame,'2022-01-09')
        frame.loc[frame.index>=pd.Timestamp('2022-01-09'),'I'] = np.nan
        self.assertEqual(len(common_inputs(frame,model,'2022-01-09','2022-01-11')['index']),48)

    def test_gaps_and_future_price_domain_fail_without_clipping(self):
        raw = self.raw()
        with self.assertRaises(ValueError):
            prepare_frame(raw.drop(raw.index[4]),'2022-01-01','2022-01-09')
        frame,_ = prepare_frame(raw,'2022-01-01','2022-01-09')
        model = statistical_output().fit(frame,'2022-01-09')
        frame.loc[frame.index>=pd.Timestamp('2022-01-09'),'U'] = 1.1
        with self.assertRaises(ValueError):
            common_inputs(frame,model,'2022-01-09','2022-01-11')

    def test_declared_ex_ante_price_bounds(self):
        frame,scales = prepare_frame(self.raw(),'2022-01-01','2022-01-09',price_bounds=(0,200))
        self.assertEqual(scales['price_scale_source'],'declared-ex-ante')
        self.assertTrue(np.all(np.abs(frame.U)<1))

    def test_weather_requests_do_not_cross_training_cutoff(self):
        from unittest.mock import patch
        from src.tools.bm_1_deterministic_data import load_raw_frame
        raw = self.raw()
        module = 'src.tools.bm_1_deterministic_data.'
        requests = []
        def temperatures(weather,start,end):
            requests.append((pd.Timestamp(start),pd.Timestamp(end)))
            idx = pd.date_range(start,end,freq='h',inclusive='left')
            return pd.Series(5.,index=idx)
        cutoff = pd.Timestamp('2022-01-09')
        with patch(module+'input_data'),patch(module+'spot_price_data'),patch(module+'weather_data'), \
             patch(module+'consumption_series',return_value=raw.Demand), \
             patch(module+'price_series',return_value=raw.Price),patch(module+'temperature_series',side_effect=temperatures):
            load_raw_frame('.',raw.index[0],'2022-01-11',cutoff=cutoff)
        self.assertEqual(requests,[(raw.index[0],cutoff),(cutoff,pd.Timestamp('2022-01-11'))])

    def test_end_to_end_cutoff_and_wm_failure_preserves_bm(self):
        from unittest.mock import patch
        from src.method.bm_1_deterministic_benchmark import run_benchmark
        raw = self.raw()
        changed = raw.copy()
        changed.loc[changed.index>=pd.Timestamp('2022-01-09'),'Demand'] += 10
        config = BM1DeterministicConfig(starts=((12.,2.,0.,.5),),max_calls=150)
        with patch('src.method.bm_1_deterministic_benchmark.matched_wm_fit',side_effect=ValueError('WM rejection')):
            results = [run_benchmark(r,'2022-01-01','2022-01-09',forecast_hours=48,config=config) for r in (raw,changed)]
        self.assertEqual(results[0]['status'],'wm-rejected')
        self.assertNotIn('wm4',results[0]['metrics'])
        self.assertEqual(results[0]['fit']['best_converged']['parameters'],results[1]['fit']['best_converged']['parameters'])
        np.testing.assert_array_equal(results[0]['predictions'].bm1,results[1]['predictions'].bm1)


if __name__=='__main__': unittest.main()

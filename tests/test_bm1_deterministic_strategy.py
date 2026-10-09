"""The basic G fit estimates C, k, Xi and a without curve penalties."""
import unittest
import numpy as np
import pandas as pd
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_deterministic_weekday import BM1PooledInputs
from src.method.bm_1_deterministic_strategy import fit_strategy
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_g_contract import calendar_inputs

class BasicFitTests(unittest.TestCase):
    def inputs(self):
        parameters=np.array([4.,2.,.65,.5])
        curve=BM1PiecewiseResponse.from_inner_values([.3],pieces=2)
        histories=[]
        for day in range(2):
            time=np.arange(49.)
            baseline=.45+.15*np.sin(time/4+day)
            prices=np.random.default_rng(91+day).uniform(.05,.95,48)
            data=BM1ResponseInputs(time,baseline,prices,warmup_hours=24,steps_per_hour=12)
            observed=data.solve(parameters,price_response=curve)['demand']
            mask=np.ones(48,dtype=bool);mask[30]=False;observed[30]=1e6
            histories.append(BM1ResponseInputs(time,baseline,prices,observed,warmup_hours=24,
                steps_per_hour=12,observation_mask=mask))
        return BM1PooledInputs(histories),parameters

    def test_recovers_the_four_parameters_and_excludes_missing_measurements(self):
        inputs,truth=self.inputs()
        result=fit_strategy(inputs,[6.,1.5,.4,.5],max_calls=350,seconds=30)
        self.assertEqual(result['status'],'converged')
        best=result['best_converged']
        np.testing.assert_allclose(best['parameters'],truth,rtol=.002,atol=.001)
        self.assertAlmostEqual(best['g_mid'],.3,places=3)
        self.assertLess(best['mse'],1e-10)
        self.assertEqual(best['mse'],best['objective'])
        self.assertEqual(result['observed_samples'],46)
        self.assertEqual(result['lambda'],0.)
        self.assertEqual(best['data_local_rank'],4)

    def test_budget_stop_preserves_only_an_unfinished_fit(self):
        inputs,_=self.inputs();saved=[]
        result=fit_strategy(inputs,[6.,1.5,.4,.5],max_calls=1,seconds=30,checkpoint=saved.append)
        self.assertEqual(result['status'],'unfinished')
        self.assertIsNone(result['best_converged'])
        self.assertEqual(result['counts']['residual_calls'],1)
        self.assertEqual(saved[-1]['status'],'unfinished')

    def test_spring_day_has_23_physical_hours_without_target_demand(self):
        class Baseline:
            def predict(self,index,temperature):return np.full(len(index),.4)
        index=pd.date_range('2022-03-26','2022-03-29',freq='h',inclusive='left')
        frame=pd.DataFrame({'U':0.,'Temp':5.,'I':1e9},index=index)
        data,times=calendar_inputs(frame[['U','Temp']],Baseline(),'2022-03-27')
        self.assertEqual(data.n_hours,23)
        self.assertIsNone(data.observations)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'),times)
        history,_=calendar_inputs(frame,Baseline(),'2022-03-27',history=True)
        self.assertEqual(history.n_hours,47)
        self.assertEqual(history.warmup,24)

if __name__=='__main__':unittest.main()

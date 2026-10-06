"""Small controls of free coordinates and the new DST scenario interface."""
import unittest
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1.deterministic_strategies import calendar_inputs, input_fingerprint
from src.exe.run_model.bm_1.deterministic_validation import week_job
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_deterministic_weekday import BM1PooledInputs
from src.method.bm_1_deterministic_strategy import fit_strategy
from src.params.bm_1_response import BM1PiecewiseResponse


class StrategyTests(unittest.TestCase):
    def test_g_only_holds_physical_parameters_and_penalty_changes_fit(self):
        p=np.array([10.,.6,.5,.5]);g=BM1PiecewiseResponse.from_inner_values([.4],pieces=2)
        histories=[]
        for j in range(5):
            t=np.arange(49.)
            inp=BM1ResponseInputs(t,.4+.1*np.sin(t/9+j),
                .5+.35*np.sin(np.arange(48)/7+j),warmup_hours=24,steps_per_hour=2)
            y=inp.solve(p,price_response=g)['demand']
            histories.append(BM1ResponseInputs(t,inp.baseline,inp.prices,y,
                warmup_hours=24,steps_per_hour=2))
        pool=BM1PooledInputs(histories)
        a=fit_strategy(pool,p,'g-only',0.)['best_converged']
        b=fit_strategy(pool,p,'g-only',.001)['best_converged']
        np.testing.assert_array_equal(a['parameters'],p)
        np.testing.assert_array_equal(b['parameters'],p)
        self.assertAlmostEqual(a['g_mid'],.4,places=4)
        self.assertLess(abs(b['g_mid']),abs(a['g_mid']))
        self.assertGreater(b['mse'],a['mse'])
        continued=fit_strategy(pool,p,'g-only',.001,initial_records=[b])
        self.assertEqual(len(continued['starts']),1)
        self.assertAlmostEqual(continued['best_converged']['g_mid'],b['g_mid'],places=5)

    def test_spring_day_uses_23_hours_without_target_demand(self):
        class Baseline:
            def predict(self,index,temperature):
                return np.full(len(index),.4)
        index=pd.date_range('2022-03-26','2022-03-29',freq='h',inclusive='left')
        frame=pd.DataFrame({'U':0.,'Temp':5.,'I':1e9},index=index)
        inp,times=calendar_inputs(frame[['U','Temp']],Baseline(),'2022-03-27')
        self.assertEqual(inp.n_hours,23)
        self.assertIsNone(inp.observations)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'),times)
        history,_=calendar_inputs(frame,Baseline(),'2022-03-27',history=True)
        self.assertEqual(history.n_hours,47)
        self.assertEqual(history.warmup,24)
        after_dst,times=calendar_inputs(frame,Baseline(),'2022-03-28',history=True)
        self.assertEqual(after_dst.n_hours,48)
        self.assertEqual(times[0],pd.Timestamp('2022-03-26 23:00'))
        self.assertEqual(times[24],pd.Timestamp('2022-03-28 00:00'))

    def test_resume_fingerprint_binds_index_and_all_three_inputs(self):
        frame=pd.DataFrame({'Demand':[1.,np.nan],'Price':[2.,3.],'Temp':[4.,5.]},
                           index=pd.date_range('2022-01-03',periods=2,freq='h'))
        original=input_fingerprint(frame)
        self.assertEqual(original,input_fingerprint(frame.copy()))
        for column in frame:
            changed=frame.copy();changed.loc[frame.index[0],column]+=1
            self.assertNotEqual(original,input_fingerprint(changed))
        shifted=frame.copy();shifted.index+=pd.Timedelta(hours=1)
        self.assertNotEqual(original,input_fingerprint(shifted))

    def test_validation_week_has_shared_35_histories_and_no_future_observations(self):
        class Baseline:
            def predict(self,index,temperature):return np.full(len(index),.4)
        index=pd.date_range('2022-02-20','2022-04-04',freq='h',inclusive='left')
        frame=pd.DataFrame({'I':.4,'U':.1,'Temp':5.},index=index)
        origin=pd.Timestamp('2022-03-28');p=[10.,.6,.5,.5]
        job,baseline=week_job(frame,Baseline(),origin,p,'unused')
        changed=frame.copy();changed.loc[changed.index>=origin,'I']=1e9
        alternate,_=week_job(changed,Baseline(),origin,p,'unused')
        self.assertEqual(len(job['histories']),35)
        self.assertEqual(len(job['forecasts']),7)
        self.assertEqual(sorted(sum(job['state_positions'],[])),list(range(35)))
        self.assertEqual(BM1PooledInputs(job['histories']).observation_mask.sum(),839)
        for a,b in zip(job['histories'],alternate['histories']):
            np.testing.assert_array_equal(a.observations,b.observations)
        for a,_ in job['forecasts']:self.assertIsNone(a.observations)
        self.assertEqual(sum(len(x) for x in baseline),168)


if __name__=='__main__':
    unittest.main()

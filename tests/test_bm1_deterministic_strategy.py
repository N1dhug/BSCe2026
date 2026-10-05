"""Small controls of free coordinates and the new DST scenario interface."""
import unittest
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1.deterministic_strategies import calendar_inputs, input_fingerprint
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
        index=pd.date_range('2022-03-26','2022-03-28',freq='h',inclusive='left')
        frame=pd.DataFrame({'U':0.,'Temp':5.,'I':1e9},index=index)
        inp,times=calendar_inputs(frame[['U','Temp']],Baseline(),'2022-03-27')
        self.assertEqual(inp.n_hours,23)
        self.assertIsNone(inp.observations)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'),times)
        history,_=calendar_inputs(frame,Baseline(),'2022-03-27',history=True)
        self.assertEqual(history.n_hours,47)
        self.assertEqual(history.warmup,24)

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


if __name__=='__main__':
    unittest.main()

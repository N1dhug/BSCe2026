"""Interval semantics, analytic time covariance, conservation and cutoff."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.linalg import expm

from tests import test_bm1_s3_forecast as fixture
from src.method import bm_1_s3_interval as interval
from src.method import bm_1_s3_interval_process as process
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.params.bm_1_response import BM1PiecewiseResponse


class S3IntervalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.S3ForecastTests.setUpClass();cls.raw=fixture.S3ForecastTests.raw
        cls.origin=fixture.S3ForecastTests.origin
        def fit(inputs,*args,**kwargs):
            assert all(t.contract=='hourly-mean' for t in inputs.trajectories)
            return {'status':'converged','best_converged':deepcopy(fixture.S3ForecastTests.best),
                'counts':{'residual_calls':0},'observed_samples':int(inputs.observation_mask.sum())}
        cls.mock_fit=staticmethod(fit)
        with patch('src.method.bm_1_s3_interval.fit_strategy',side_effect=fit):
            cls.snapshot=interval.fit_interval(cls.raw,cls.origin,demand_label='hour_start')['snapshot']

    def test_own_interval_fit_and_snapshot_roundtrip_without_refit(self):
        s=self.snapshot
        self.assertEqual(s['fit']['objective_operator'],'hourly-mean')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'s.json';interval.save_snapshot(path,s)
            with patch('src.exe.run_model.wm_4.output_model.statistical_output.fit',side_effect=AssertionError('refit')):
                a=interval.predict_interval(interval.load_snapshot(path),self.raw[['Price','Temp']])
        self.assertEqual(a['status'],'complete')
        self.assertGreater(float(abs(a['predictions'].prediction-a['predictions'].point_prediction).max()),.001)
        with self.assertRaises(ValueError):interval.validate_snapshot(fixture.S3ForecastTests.snapshot)
        with self.assertRaises(ValueError):interval.point.validate_snapshot(s)

    def test_fit_and_forecast_ignore_unavailable_future_values(self):
        changed=self.raw.copy();changed.loc[changed.index>=self.origin,'Demand']=1e20
        with patch('src.method.bm_1_s3_interval.fit_strategy',side_effect=self.mock_fit):
            other=interval.fit_interval(changed,self.origin,demand_label='hour_start')['snapshot']
        self.assertEqual(self.snapshot,other)
        pd.testing.assert_frame_equal(interval.predict_interval(self.snapshot,self.raw)['predictions'],
            interval.predict_interval(self.snapshot,changed)['predictions'])

    def test_end_label_availability_at_cutoff_and_scoring_alignment(self):
        index=pd.date_range('2022-02-06 22:00',periods=4,freq='h')
        raw=pd.DataFrame({'Demand':[1.,2.,3.,1e9],'Price':[10.,20.,30.,1e9],'Temp':[1.,2.,3.,1e9]},index=index)
        train=interval.training_frame(raw,self.origin,'hour_end')
        self.assertEqual(train.Demand.tolist(),[2.,3.]);self.assertEqual(train.Price.tolist(),[10.,20.])
        raw.loc[raw.index>self.origin,'Demand']=-1e9
        pd.testing.assert_frame_equal(train,interval.training_frame(raw,self.origin,'hour_end'))
        a=interval.predict_interval(self.snapshot,self.raw);a['manifest']['demand_timestamp_label']='hour_end'
        y=a['predictions'].prediction.copy();y.index=y.index+pd.Timedelta(hours=1)
        b=interval.score_interval(a,y)
        self.assertAlmostEqual(b['scoring']['metrics']['prediction']['mse'],0.)

    def test_spring_intervals_have_one_physical_hour_and_missing_label_is_not_fabricated(self):
        s=deepcopy(self.snapshot);origin=pd.Timestamp('2022-03-21');s['origin']=str(origin)
        s['history']['days_by_weekday']=[[str((origin+pd.Timedelta(days=d)-pd.Timedelta(weeks=w)).date())
            for w in range(1,6)] for d in range(7)]
        index=pd.date_range(origin,origin+pd.Timedelta(days=7),freq='h',inclusive='left')
        a=interval.predict_interval(s,pd.DataFrame({'Price':100.,'Temp':5.},index=index))
        p=a['predictions'];self.assertEqual(len(p),167)
        duration=pd.to_datetime(p.interval_end_utc)-pd.to_datetime(p.interval_start_utc)
        self.assertTrue((duration==pd.Timedelta(hours=1)).all())
        ends=pd.Series([1.,2.],index=pd.DatetimeIndex(['2022-03-27 01:00','2022-03-27 03:00']))
        self.assertEqual(interval.align_demand(ends,'hour_end').index.tolist(),
            [pd.Timestamp('2022-03-27 00:00'),pd.Timestamp('2022-03-27 01:00')])
        with self.assertRaises(ValueError):
            interval.align_demand(pd.Series([1.],index=pd.DatetimeIndex(['2022-03-27 02:00'])),'hour_end')

    def test_frozen_linear_covariance_matches_independent_van_loan_reference(self):
        # Integrated Brownian motion has Var(Z)=c^2 b^2 T^3/3, not c^2 Var(X_T).
        actual=process.covariance_step(np.zeros((2,2)),0.,2.,3.,1.)
        np.testing.assert_allclose(actual,[[9.,9.],[9.,12.]],atol=1e-13)
        a=np.array([[-.4,0.],[-.7,0.]]);q=np.diag([.2**2,0.]);h=.25
        van_loan=expm(h*np.block([[a,q],[np.zeros((2,2)),-a.T]]))
        exact=van_loan[:2,2:]@van_loan[:2,:2].T
        np.testing.assert_allclose(process.covariance_step(np.zeros((2,2)),-.4,-.7,.2,h),exact,atol=1e-11)

    def test_reward_moments_match_analytic_two_state_integrated_process(self):
        # Stationary +/-d telegraph signal with jump rate lambda=1:
        # Var(int_0^1 signal dt)=d^2*(1-(1-exp(-2))/2).
        exact=.2**2*(1-(1-np.exp(-2))/2);errors=[]
        for count in (128,256):
            h=1/count;transition=np.linalg.inv(np.eye(2)-h*np.array([[-1.,1.],[1.,-1.]]))
            fields=np.column_stack(([.5,.5],[0.,0.],[0.,0.]))
            for j in range(count):
                fields,_=process.reward_step(fields,.5*h*np.array([.2,-.2]),
                    lambda f,t,s:(transition@f,{}),j*h,h)
            variance=fields[:,2].sum()-fields[:,1].sum()**2;errors.append(abs(variance-exact))
            self.assertLess(abs(fields[:,0].sum()-1),1e-12)
        self.assertLess(errors[1],errors[0]);self.assertLess(errors[1],1e-4)

    def test_real_integrated_fv_first_hour_variance_and_no_flexibility_limit(self):
        t=np.arange(121)/60;curve=BM1PiecewiseResponse.linear(pieces=2)
        inp=BM1ResponseInputs(t,np.full_like(t,.5),[.5,.5],steps_per_hour=12,observation_contract='hourly-mean')
        out=process.interval_day_moments(inp,[4.,2.,.5,.5],curve,.5,.2,64,1/24,perf_counter()+10)
        self.assertGreater(out['process_variance'][0],0.)
        self.assertLess(out['diagnostics']['maximum_abs_mass_error'],1e-10)
        noflex=process.interval_day_moments(inp,[4.,2.,0.,.5],curve,.5,.2,64,1/24,perf_counter()+10)
        np.testing.assert_array_equal(noflex['process_variance'],0.)
        np.testing.assert_array_equal(noflex['process_mean'],inp.b_mean)
        unit=process.unit_interval_variance(inp,[4.,2.,.5,.5],curve,reset_hour=0,deadline=perf_counter()+10)
        self.assertGreater(unit['unit_interval_variance'][0],0.)

    def test_zero_process_exact_interval_mean_raw_variance_scaling_and_no_fake_bands(self):
        a=process.process_interval(self.snapshot,self.raw,sigma_x=0.)
        np.testing.assert_array_equal(a['predictions'].process_mean,a['predictions'].prediction)
        np.testing.assert_array_equal(a['predictions'].process_variance,0.)
        self.assertNotIn('process_lower95',a['predictions'])
        self.assertEqual(a['uncertainty']['bands_status'],'not-computed; interval moments do not determine exact quantiles')
        scored=process.score_interval_process(a,self.raw.Demand)
        self.assertIn('process_mean',scored['scoring']['metrics'])


if __name__=='__main__':unittest.main()

"""Hourly-mean fit, frozen snapshots, integrated baseline and causal cutoff."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests import test_bm1_s3_forecast as fixture
from src.method import bm_1_s3_interval as interval
from src.method.bm_1_s3_interval_baseline import IntervalStatisticalBaseline,mean_features


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
        changed.loc[changed.index>=self.origin,['Price','Temp']]=1e20
        with patch('src.method.bm_1_s3_interval.fit_strategy',side_effect=self.mock_fit):
            no_future=interval.fit_interval(changed,self.origin,demand_label='hour_start')['snapshot']
        self.assertEqual(self.snapshot,no_future)

    def test_interval_snapshot_rejects_changed_sources_and_incompatible_data(self):
        edits={
            'changed source':lambda s:s['source_sha256'].update({'src/method/bm_1_s3_interval.py':'wrong'}),
            'point operator':lambda s:s['fit'].update(objective_operator='hourly-point'),
            'unknown labels':lambda s:s.update(demand_timestamp_label='unspecified'),
            'point baseline':lambda s:s['baseline'].update(implementation='statistical_output'),
            'future baseline':lambda s:s['baseline']['metadata'].update(fit_end=s['origin']),
            'invalid daily starts':lambda s:s.update(forecast_states=[float('nan')]*7),
            'inconsistent curve':lambda s:s['fit']['best_converged'].update(g_mid=-.9),
            'failed refinement':lambda s:s['numerics'].update(passed=False),
            'missing history':lambda s:s.pop('history'),
        }
        for name,edit in edits.items():
            with self.subTest(name=name):
                snapshot=deepcopy(self.snapshot);edit(snapshot)
                with self.assertRaises(ValueError):interval.validate_snapshot(snapshot)

    def test_baseline_design_integrates_harmonics_over_the_observed_hour(self):
        model=IntervalStatisticalBaseline(hour_harmonics=1,season_harmonics=0)
        source=pd.Series([5.,5.],index=pd.date_range('2022-02-07 10:00',periods=2,freq='h'))
        row=mean_features(model,source.index[:1],source)[0]
        omega=2*np.pi/24
        exact=np.array([1.,(np.cos(10*omega)-np.cos(11*omega))/omega,
            (np.sin(11*omega)-np.sin(10*omega))/omega])
        # Composite trapezoid bound: max|f''|/(12*N^2), N=60 on one hour.
        np.testing.assert_allclose(row[:3],exact,atol=omega**2/(12*60**2)+1e-12,rtol=0)
        self.assertGreater(abs(row[1]-np.sin(10*omega)),.01)

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



if __name__=='__main__':unittest.main()

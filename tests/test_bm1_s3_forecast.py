"""S3 interface risks: frozen starts, no refit/future-demand leakage, durable I/O."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.method.bm_1_s3_forecast import (
    fit_s3, load_snapshot, save_snapshot, predict_s3, score_s3, validate_snapshot)
from src.exe.run_model.bm_1.s3_forecast import main


class S3ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin = pd.Timestamp('2022-02-07')
        index = pd.date_range('2022-01-01','2022-02-14',freq='h',inclusive='left')
        t = np.arange(len(index))
        cls.raw = pd.DataFrame({'Demand':2+.3*np.sin(t/5)+.01*np.cos(t/2),
            'Price':100+30*np.sin(t/4),'Temp':5+np.cos(t/8)}, index=index)
        cls.best = {'parameters':[4.,2.,.6,.5], 'g_mid':.2, 'converged':True, 'mse':.1}
        def fixed_fit(inputs, *args, **kwargs):
            return {'status':'converged','best_converged':deepcopy(cls.best),
                'counts':{'residual_calls':0},'observed_samples':int(inputs.observation_mask.sum())}
        # Existing optimizer tests cover LS. This nonzero response fixture uses
        # actual baseline, all 35 histories, coarse/fine RK4 and forecast starts.
        with patch('src.method.bm_1_s3_forecast.fit_strategy', side_effect=fixed_fit):
            cls.result = fit_s3(cls.raw, cls.origin)
        cls.snapshot = cls.result['snapshot']

    def test_roundtrip_preserves_starts_and_forecast_without_fit(self):
        before = predict_s3(self.snapshot, self.raw[['Price','Temp']])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'snapshot.json'; save_snapshot(path,self.snapshot)
            with patch('src.method.bm_1_s3_forecast.fit_strategy', side_effect=AssertionError('refit')), \
                 patch('src.exe.run_model.wm_4.output_model.statistical_output.fit', side_effect=AssertionError('baseline refit')):
                loaded = load_snapshot(path)
                after = predict_s3(loaded,self.raw[['Price','Temp']])
        pd.testing.assert_frame_equal(before['predictions'],after['predictions'])
        self.assertEqual(after['status'],'complete')
        self.assertEqual(len(after['predictions']),168)
        np.testing.assert_array_equal(after['predictions'].groupby('weekday').state.first(),self.snapshot['forecast_states'])
        self.assertNotEqual(after['predictions'].groupby('weekday').state.first().iloc[1],
                            after['predictions'].state.iloc[23])

    def test_future_demand_is_not_used_and_scoring_does_not_mutate(self):
        a = predict_s3(self.snapshot,self.raw)
        changed=self.raw.copy();changed.loc[changed.index>=self.origin,'Demand']=1e12
        b = predict_s3(self.snapshot,changed)
        pd.testing.assert_frame_equal(a['predictions'],b['predictions'])
        self.assertEqual(a['manifest']['scenario_fingerprint'],b['manifest']['scenario_fingerprint'])
        observations=self.raw.Demand.copy();observations.loc[self.origin]=np.nan
        scored=score_s3(a,observations)
        self.assertEqual(scored['scoring']['scored_samples'],167)
        self.assertNotIn('observed',a['predictions'])
        pd.testing.assert_series_equal(a['predictions'].prediction,scored['predictions'].prediction)
        e=scored['predictions'].prediction-scored['predictions'].observed
        self.assertAlmostEqual(scored['scoring']['metrics']['prediction']['mse'],float((e*e).mean()))

    def test_snapshot_rejects_incomplete_or_inconsistent_fitted_data(self):
        edits=[lambda s:s.update(schema_version=2),lambda s:s.update(forecast_states=[.5]*6),
            lambda s:s['numerics'].update(fine_steps_per_hour=13),
            lambda s:s['numerics'].update(history_max_demand_error=.1),
            lambda s:s['baseline']['metadata'].update(fit_end=s['origin']),
            lambda s:s['source_sha256'].clear(), lambda s:s.pop('scales')]
        for edit in edits:
            with self.subTest(edit=edit):
                snapshot=deepcopy(self.snapshot);edit(snapshot)
                with self.assertRaises(ValueError): validate_snapshot(snapshot)

    def test_input_domain_and_missing_hour_fail_without_repair(self):
        for column,value in [('Price',1e8),('Temp',np.nan)]:
            changed=self.raw.copy();changed.loc[self.origin,column]=value
            with self.assertRaises(ValueError):predict_s3(self.snapshot,changed)
        with self.assertRaises(ValueError):predict_s3(self.snapshot,self.raw.drop(self.origin))

    def test_spring_week_and_explicit_autumn_rejection(self):
        for date,expected in [('2022-03-21',167),('2022-10-24',None)]:
            s=deepcopy(self.snapshot);origin=pd.Timestamp(date);s['origin']=str(origin)
            s['history']['days_by_weekday']=[[str((origin+pd.Timedelta(days=d)-pd.Timedelta(weeks=w)).date())
                for w in range(1,6)] for d in range(7)]
            index=pd.date_range(origin,origin+pd.Timedelta(days=7),freq='h',inclusive='left')
            scenario=pd.DataFrame({'Price':100.,'Temp':5.},index=index)
            if expected is None:
                with self.assertRaisesRegex(ValueError,'Autumn DST'):predict_s3(s,scenario)
            else:
                result=predict_s3(s,scenario)
                self.assertEqual(len(result['predictions']),expected)
                self.assertNotIn(pd.Timestamp('2022-03-27 02:00'),result['predictions'].index)

    def test_cli_saves_forecast_before_missing_observation_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);snapshot=root/'snapshot.json';save_snapshot(snapshot,self.snapshot)
            inputs=root/'inputs.csv';self.raw[['Price','Temp']].to_csv(inputs,index_label='Datetime')
            out=root/'output'
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=main(['predict','--snapshot',str(snapshot),'--inputs-csv',str(inputs),
                    '--observations-csv',str(root/'missing.csv'),'--output',str(out)])
            self.assertEqual(code,2)
            self.assertTrue((out/'forecast.csv').exists())
            receipt=json.loads((out/'forecast.json').read_text())
            self.assertEqual(receipt['status'],'complete')
            self.assertEqual(receipt['scoring']['status'],'not-scored')
            self.assertTrue((out/'failure.json').exists())


if __name__ == '__main__': unittest.main()

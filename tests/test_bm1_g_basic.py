"""The pruned G entry point retains interval predictions and safe outputs."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from src.exe.run_model import bm_1_g
from src.exe.run_model.bm_1 import s3
from src.method import bm_1_s3_interval as interval
from tests import test_bm1_s3_interval as fixture
from tests.g_fixtures import synthetic_history
from src.exe.run_model.bm_1.s3_data import prepare_frame


class GBasicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.S3IntervalTests.setUpClass()
        cls.snapshot=fixture.S3IntervalTests.snapshot
        cls.raw=fixture.S3IntervalTests.raw

    def test_main_alias_and_removed_options(self):
        self.assertIs(bm_1_g.main,s3.main)
        predict=['predict','--snapshot','saved.json','--inputs-csv','inputs.csv','--output','new-run']
        self.assertEqual(s3.parser().parse_args(predict).command,'predict')
        for argv in (['residual'],['point-process'],['diagnose'],['bands']):
            with self.subTest(argv=argv),redirect_stderr(StringIO()),self.assertRaises(SystemExit) as error:
                s3.parser().parse_args(argv)
            self.assertEqual(error.exception.code,2)
        stderr=StringIO()
        with redirect_stderr(stderr),self.assertRaises(SystemExit) as error:
            s3.parser().parse_args([*predict,'--sigma-x','.2'])
        self.assertEqual(error.exception.code,2)
        self.assertIn('unrecognized arguments: --sigma-x',stderr.getvalue())

    def test_future_measurements_do_not_change_training_or_scales(self):
        raw,origin,_=synthetic_history()
        start=origin-pd.Timedelta(weeks=5)
        before,scales=prepare_frame(raw,start,origin)
        changed=raw.copy()
        changed.loc[changed.index>=origin,['Demand','Price','Temp']]=1e8
        after,new_scales=prepare_frame(changed,start,origin)
        self.assertEqual(scales,new_scales)
        pd.testing.assert_frame_equal(before.loc[start:origin-pd.Timedelta(hours=1)],
                                      after.loc[start:origin-pd.Timedelta(hours=1)])

    def test_training_gaps_preserve_missing_demand_and_require_prices(self):
        raw,origin,_=synthetic_history()
        start=origin-pd.Timedelta(weeks=5)
        last=origin-pd.Timedelta(hours=1)
        raw.loc[[start,last],'Temp']=np.nan
        raw.loc[start,'Demand']=np.nan
        raw.loc[origin,'Temp']=1e8
        frame,_=prepare_frame(raw,start,origin)
        self.assertTrue(np.isnan(frame.loc[start,'I']))
        self.assertEqual(frame.loc[start,'Temp'],raw.loc[start+pd.Timedelta(hours=1),'Temp'])
        self.assertEqual(frame.loc[last,'Temp'],raw.loc[last-pd.Timedelta(hours=1),'Temp'])
        self.assertTrue(frame.loc[[start,last],'temperature_imputed'].all())
        raw.loc[start,'Price']=np.nan
        with self.assertRaisesRegex(ValueError,'complete prices'):
            prepare_frame(raw,start,origin)

    def test_input_only_prediction_completes_without_refit_and_creates_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=root/'snapshot.json';inputs=root/'inputs.csv';out=root/'run'
            interval.save_snapshot(snapshot,self.snapshot)
            self.raw[['Price','Temp']].to_csv(inputs,index_label='Datetime')
            with patch.object(interval,'fit_strategy',side_effect=AssertionError('refit')), \
                 patch.object(interval.IntervalStatisticalBaseline,'fit',side_effect=AssertionError('baseline refit')), \
                 redirect_stdout(StringIO()):
                code=bm_1_g.main(['predict','--snapshot',str(snapshot),'--inputs-csv',str(inputs),'--output',str(out)])
            self.assertEqual(code,0)
            receipt=json.loads((out/'forecast.json').read_text())
            self.assertEqual(receipt['status'],'complete')
            self.assertEqual(receipt['scoring']['status'],'not-scored')
            self.assertFalse(receipt['manifest']['demand_used'])
            self.assertFalse(receipt['manifest']['model_refitted'])
            self.assertFalse(receipt['manifest']['baseline_refitted'])
            actual=pd.read_csv(out/'forecast.csv')
            self.assertEqual(len(actual),168)
            self.assertNotIn('observed',actual)
            self.assertTrue((out/'interval_week.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertFalse((out/'failure.json').exists())

    def test_missing_demand_column_is_reported_after_forecast_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=root/'snapshot.json';inputs=root/'inputs.csv'
            observations=root/'measurements.csv';out=root/'run'
            interval.save_snapshot(snapshot,self.snapshot)
            self.raw[['Price','Temp']].to_csv(inputs,index_label='Datetime')
            self.raw[['Price']].to_csv(observations,index_label='Datetime')
            with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
                code=bm_1_g.main(['predict','--snapshot',str(snapshot),'--inputs-csv',str(inputs),
                    '--observations-csv',str(observations),'--output',str(out)])
            self.assertEqual(code,2)
            self.assertEqual(json.loads((out/'forecast.json').read_text())['status'],'complete')
            failure=json.loads((out/'failure.json').read_text())
            self.assertIn('Demand',failure['error'])
            self.assertFalse((out/'scored.json').exists())

    def test_scoring_uses_only_measured_hours_and_preserves_frozen_forecast(self):
        expected=interval.predict_interval(self.snapshot,self.raw)['predictions'].prediction
        observed=expected.copy();observed.iloc[0]=float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=root/'snapshot.json';inputs=root/'inputs.csv'
            observations=root/'measurements.csv';out=root/'run'
            interval.save_snapshot(snapshot,self.snapshot)
            self.raw[['Price','Temp']].to_csv(inputs,index_label='Datetime')
            observed.rename('Demand').to_csv(observations,index_label='Datetime')
            with patch.object(s3,'plot'),redirect_stdout(StringIO()):
                code=bm_1_g.main(['predict','--snapshot',str(snapshot),'--inputs-csv',str(inputs),
                    '--observations-csv',str(observations),'--output',str(out)])
            self.assertEqual(code,0)
            frozen=pd.read_csv(out/'forecast.csv')
            scored=pd.read_csv(out/'scored.csv')
            pd.testing.assert_series_equal(frozen.prediction,scored.prediction)
            self.assertNotIn('observed',frozen)
            receipt=json.loads((out/'scored.json').read_text())
            self.assertEqual(receipt['scoring']['status'],'partial')
            self.assertEqual(receipt['scoring']['scored_samples'],167)
            self.assertLess(receipt['scoring']['metrics']['prediction']['mse'],1e-24)

    def test_completed_fit_exports_a_loadable_interval_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'history.csv';out=root/'fit'
            self.raw.to_csv(raw,index_label='Datetime')
            with patch.object(interval,'fit_strategy',side_effect=fixture.S3IntervalTests.mock_fit), \
                 redirect_stdout(StringIO()):
                code=bm_1_g.main(['fit','--raw-csv',str(raw),'--origin',self.snapshot['origin'],
                    '--demand-label','hour_start','--population','test houses','--demand-unit','kW','--output',str(out)])
            self.assertEqual(code,0)
            snapshot=interval.load_snapshot(out/'interval_snapshot.json')
            self.assertEqual(snapshot['population'],'test houses')
            self.assertEqual(snapshot['demand_unit'],'kW')
            self.assertEqual(snapshot['parameters'],self.snapshot['parameters'])
            self.assertEqual(snapshot['price_response'],self.snapshot['price_response'])
            self.assertEqual(json.loads((out/'fit_result.json').read_text())['status'],'complete')

    def test_exhausted_real_fit_budget_never_exports_completed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'history.csv';out=root/'fit'
            self.raw.to_csv(raw,index_label='Datetime')
            with redirect_stdout(StringIO()):
                code=bm_1_g.main(['fit','--raw-csv',str(raw),'--origin',self.snapshot['origin'],
                    '--demand-label','hour_start','--max-calls','1','--seconds','30','--output',str(out)])
            self.assertEqual(code,2)
            receipt=json.loads((out/'fit_result.json').read_text())
            self.assertEqual(receipt['status'],'unfinished')
            self.assertEqual(receipt['calls_used'],1)
            self.assertFalse((out/'interval_snapshot.json').exists())
            self.assertTrue((out/'fit_checkpoint.json').exists())

    def test_failed_refinement_is_saved_and_returns_failure(self):
        snapshot=deepcopy(self.snapshot)
        snapshot['fine_forecast_states']=[0.]*7
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);saved=root/'snapshot.json';inputs=root/'inputs.csv';out=root/'run'
            interval.save_snapshot(saved,snapshot)
            self.raw[['Price','Temp']].to_csv(inputs,index_label='Datetime')
            with patch.object(s3,'plot'),redirect_stdout(StringIO()):
                code=bm_1_g.main(['predict','--snapshot',str(saved),'--inputs-csv',str(inputs),'--output',str(out)])
            self.assertEqual(code,2)
            receipt=json.loads((out/'forecast.json').read_text())
            self.assertEqual(receipt['status'],'numerics-failed')
            self.assertFalse(receipt['numerics']['passed'])

    def test_real_prediction_preserved_and_saved_before_target_failure(self):
        expected=interval.predict_interval(self.snapshot,self.raw)['predictions']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=root/'snapshot.json';inputs=root/'inputs.csv';out=root/'run'
            interval.save_snapshot(snapshot,self.snapshot)
            self.raw.to_csv(inputs,index_label='Datetime')
            with patch.object(s3,'plot'),redirect_stderr(StringIO()):
                code=bm_1_g.main(['predict','--snapshot',str(snapshot),'--inputs-csv',str(inputs),
                    '--observations-csv',str(root/'missing.csv'),'--output',str(out)])
            self.assertEqual(code,2)
            self.assertTrue((out/'forecast.json').is_file())
            actual=pd.read_csv(out/'forecast.csv',index_col='Datetime',parse_dates=['Datetime'])
            self.assertEqual(len(actual),len(expected))
            for name in ('prediction','baseline','state','state_end'):
                self.assertLess(abs(actual[name]-expected[name]).max(),1e-12)
            self.assertTrue((out/'failure.json').is_file())

    def test_completed_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'finished';out.mkdir();sentinel=out/'keep.txt';sentinel.write_text('original')
            with redirect_stderr(StringIO()),patch.object(interval,'load_snapshot') as load:
                result=bm_1_g.main(['predict','--snapshot','unused','--inputs-csv','unused','--output',str(out)])
            self.assertEqual(result,2);load.assert_not_called()
            self.assertEqual(sentinel.read_text(),'original')
            self.assertFalse((out/'failure.json').exists())


if __name__=='__main__':unittest.main()

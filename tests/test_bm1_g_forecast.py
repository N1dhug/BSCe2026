"""Frozen G deployment: target isolation, calendar, provenance and forward parity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.method import bm_1_g_forecast as model
from src.exe.run_model import bm_1_g as runner
from src.exe.run_model.wm_4.output_model import statistical_output


def fixture():
    baseline = statistical_output()
    size = baseline._features(pd.DatetimeIndex(['2022-01-01']), [5.]).shape[1]
    beta = np.zeros(size); beta[0] = .5; beta[1] = .04
    baseline_record = {'implementation':'weekday-hourly-mean-statistical-output-v1',
        'settings':{key:getattr(baseline,key) for key in ('base_temperature','hour_harmonics','season_harmonics','ridge','huber_delta','max_iter')},
        'coef':beta.tolist(), 'feature_mean':np.zeros(size).tolist(),
        'feature_scale':np.ones(size).tolist(), 'metadata':{}}
    days = [{'weekday':d, 'baseline':deepcopy(baseline_record),
        'G':{'parameters':[12., 1., .4, .5],
             'price_response':model.BM1PiecewiseResponse.from_inner_values([.25],pieces=2).to_record(),
             'forecast_state':.3+.05*d, 'fine_forecast_state':.3+.05*d, 'steps_per_hour':60},
        'g_residual_samples_normalized':[-.2, .1, .3]} for d in range(7)]
    fitted = {'origin':'2022-02-28 00:00:00', 'training_start':'2022-01-24 00:00:00',
        'training_fingerprint':'synthetic-no-fit', 'days':days,
        'scales':{'demand_min':-2., 'demand_scale':4., 'price_min':-500., 'price_scale':6500.},
        'population':'MED mean over available measured meters', 'demand_unit':'synthetic raw units'}
    original = {'refined_view':fitted}
    with patch('src.method.g_weekday_export.validate_export', return_value=original) as validate:
        record = model.export_model(original)
        validate.assert_called_once_with(original)
    return record


def inputs(origin='2022-03-21'):
    window = model.calendar.WeekWindow(origin)
    index = window.hours(window.origin, window.end).tz_localize(None)
    n = np.arange(len(index))
    return pd.DataFrame({'Price':100.+70*np.sin(n/7), 'Temp':5.+4*np.sin(n/19),
                         'Demand':np.nan}, index=index)


def save_model(directory, record):
    directory.mkdir()
    (directory/'model.json').write_text(json.dumps(record), encoding='utf-8')
    (directory/'original_snapshot.json').write_text(json.dumps(record['original_snapshot']), encoding='utf-8')
    (directory/'conversion_freeze.json').write_text(json.dumps({'fit_executed':False,
        'historical_replay_executed':False,
        'artifact_sha256':{name:runner.digest(directory/name) for name in ('model.json','original_snapshot.json')}}), encoding='utf-8')
    return directory/'model.json'


class FrozenGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = fixture()

    def test_later_forecast_retains_daily_starts_and_never_reads_demand(self):
        data = inputs(); data['Demand'] = object()
        out = model.forecast(self.record, data, '2022-03-21')
        self.assertEqual(len(out['predictions']), 167)
        self.assertEqual(out['manifest']['fit_cutoff'], '2022-02-28 00:00:00')
        self.assertEqual(out['manifest']['forecast_origin'], '2022-03-21 00:00:00')
        self.assertFalse(out['manifest']['demand_used'])
        for d, day in out['predictions'].groupby('weekday'):
            self.assertEqual(day.state.iloc[0], self.record['days'][d]['state'])
        self.assertLess(out['predictions'].g_prediction_lower.min(), 0)
        self.assertEqual(out['predictions'].forecast_horizon_hours.iloc[-1], 166)
        np.testing.assert_array_equal(out['predictions'].horizon_start_hours,np.arange(167))
        np.testing.assert_array_equal(out['predictions'].horizon_end_hours,np.arange(1,168))

    def test_direct_day_matches_original_calendar_baseline_and_rk4(self):
        from src.method import bm_1_s3_interval as original
        data = inputs(); day = self.record['days'][6]
        frame = pd.DataFrame({'U':2*(data.Price+500)/6500-1,'Temp':data.Temp},index=data.index)
        baseline = model._baseline(day); date = pd.Timestamp('2022-03-27')
        old, old_index = original.day_inputs(frame, baseline, date, steps=60)
        new, new_index = model._day_inputs(data[['Price','Temp']], date, baseline, self.record['scales'])
        self.assertTrue(old_index.equals(new_index)); self.assertEqual(len(new_index),23)
        np.testing.assert_allclose(new.baseline,old.baseline,rtol=0,atol=0)
        np.testing.assert_allclose(new.prices,old.prices,rtol=0,atol=2e-16)
        p=day['parameters'].copy();p[3]=day['state']
        curve=model.BM1PiecewiseResponse.from_record(day['price_response'])
        a=new.solve(p,price_response=curve);b=old.solve(p,price_response=curve)
        np.testing.assert_allclose(a['state'],b['state'],rtol=0,atol=2e-14)
        np.testing.assert_allclose(a['demand'],b['demand'],rtol=0,atol=2e-14)

    def test_sources_runtime_content_and_population_changes_rejected(self):
        for mutation in ('source','runtime','content','group'):
            changed=deepcopy(self.record)
            if mutation=='source':changed['source_sha256']['src/method/bm_1_g_forecast.py']='0'*64
            if mutation=='runtime':changed['runtime']['numpy']='changed'
            if mutation=='content':changed['days'][0]['state']=.9
            if mutation=='group':changed['population']['group']='UDEN'
            if mutation=='group':
                changed['model_sha256']=model.fingerprint({k:v for k,v in changed.items() if k!='model_sha256'})
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):model.validate_model(changed)

    def test_cutoff_prices_missing_dst_and_hours_are_checked(self):
        data=inputs()
        for origin,hours in [('2022-02-21',None),('2022-03-22',None),('2022-03-21',168),('2022-03-21',0)]:
            with self.subTest(origin=origin,hours=hours),self.assertRaises(ValueError):
                model.prepare(self.record,data,origin,hours=hours)
        for bad in (data.iloc[1:],data.assign(Price=6001.)):
            with self.assertRaises(ValueError):model.prepare(self.record,bad,'2022-03-21')
        bad=data.copy();bad.loc[pd.Timestamp('2022-03-27 02:00')]=1
        with self.assertRaises(ValueError):model.prepare(self.record,bad.sort_index(),'2022-03-21')
        autumn=inputs('2022-10-24')
        with self.assertRaises(ValueError):model.prepare(self.record,autumn,'2022-10-24')

    def test_wrong_group_and_early_origin_fail_before_input_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=save_model(Path(tmp)/'model',self.record)
            base=['forecast','--model',str(path),'--origin','2022-03-21','--inputs-csv','never-open.csv','--check']
            with patch.object(runner.pd,'read_csv',side_effect=AssertionError('targets/exogenous inputs opened')) as read:
                self.assertEqual(runner.main(base+['--group','UDEN']),2)
                early=base.copy();early[early.index('2022-03-21')]='2022-02-21'
                self.assertEqual(runner.main(early),2)
                read.assert_not_called()

    def test_conversion_receipt_binds_original_bytes_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=save_model(Path(tmp)/'model',self.record)
            self.assertEqual(runner.load_model(path)['model_sha256'],self.record['model_sha256'])
            with (path.parent/'original_snapshot.json').open('a') as handle:handle.write(' ')
            with self.assertRaises(ValueError):runner.load_model(path)

    def test_source_change_during_forecast_is_rejected(self):
        with patch.object(model,'source_hashes',side_effect=[self.record['source_sha256'],{'changed':'source'}]):
            with self.assertRaisesRegex(ValueError,'changed during forecast'):
                model.forecast(self.record,inputs(),'2022-03-21')

    def test_input_change_during_forecast_prevents_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=save_model(root/'model',self.record)
            csv=root/'inputs.csv';inputs().to_csv(csv,index_label='Datetime')
            out=root/'run'
            def changed(*args,**kwargs):
                with csv.open('a') as handle:handle.write('\n')
                return {}
            with patch.object(model,'forecast',side_effect=changed):
                self.assertEqual(runner.main(['forecast','--model',str(path),'--origin','2022-03-21',
                    '--inputs-csv',str(csv),'--output',str(out)]),2)
            self.assertFalse((out/'forecast_freeze.json').exists())
            self.assertTrue((out/'failure.json').exists())

    def test_show_requires_every_forecast_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'forecast_freeze.json').write_text(json.dumps({'status':'frozen-input-only',
                'target_observation_file_opened':False,'artifact_sha256':{}}))
            (root/'forecast.json').write_text('{"unverified":true}')
            self.assertEqual(runner.main(['show','--result',str(root)]),2)

    def test_forecast_is_frozen_before_scoring_and_show_executes_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=save_model(root/'model',self.record)
            data=inputs();csv=root/'scenario.csv';data.to_csv(csv,index_label='Datetime')
            obs=root/'observations.csv';data.assign(Demand=-.1)[['Demand']].to_csv(obs,index_label='Datetime')
            out=root/'result';read_csv=pd.read_csv
            def read(path,*args,**kwargs):
                if Path(path)==csv:self.assertEqual(kwargs['usecols'],['Datetime','Price','Temp'])
                if Path(path)==obs:
                    receipt=json.loads((out/'forecast_freeze.json').read_text())
                    self.assertFalse(receipt['target_observation_file_opened'])
                    runner._verify_files(out,receipt['artifact_sha256'])
                return read_csv(path,*args,**kwargs)
            with patch.object(runner.pd,'read_csv',side_effect=read),patch.object(runner,'plot'):
                self.assertEqual(runner.main(['forecast','--model',str(path),'--origin','2022-03-21',
                    '--inputs-csv',str(csv),'--output',str(out),'--hours','1','--score',str(obs)]),0)
            with patch.object(model,'forecast',side_effect=AssertionError('unexpected propagation')):
                self.assertEqual(runner.main(['show','--result',str(out)]),0)
            self.assertTrue((out/'scored.csv').exists())


if __name__=='__main__':unittest.main()

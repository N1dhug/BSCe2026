"""Shared calendar, isolated model callbacks, freeze-before-score and G wiring."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.tools.bm_1_reports import _atomic_report
from src.tools.weekday_backtest import (WeekWindow, WeekdayCallbacks, rolling_windows,
    split_week, split_weekday, run_backtest, score_predictions)
from src.exe.run_model.weekday_backtest import main


def raw_frame():
    physical = pd.date_range('2022-01-01', '2022-05-01', freq='h', inclusive='left', tz='Europe/Copenhagen')
    index = physical.tz_localize(None); n = np.arange(len(index))
    return pd.DataFrame({'Demand':10+index.weekday+np.sin(n/9), 'Price':100+np.cos(n), 'Temp':n/100}, index=index)


def callbacks(name, calls, *, fail=False):
    """Synthetic F/G interface fixture, not either physical model's estimator."""
    def baseline(day):
        calls.append((name, 'baseline', day))
        assert len(day.records) == 5
        assert all(r.index.weekday[0] == day.weekday for r in day.records)
        assert day.context.loc[day.context.index.weekday != day.weekday, 'Demand'].isna().all()
        return float(pd.concat(day.records).Demand.mean())
    def fit(day, baseline):
        calls.append((name, 'fit', day))
        if fail:
            raise RuntimeError('Synthetic own-model budget exhausted')
        residuals = pd.concat(day.records).Demand.dropna().to_numpy() - baseline
        return {'variance':float(np.mean(residuals**2))} if name == 'F' else {'offset':0.}
    def predict(day, baseline, parameters):
        assert not day.records and 'Demand' not in day.context and 'Demand' not in day.inputs
        assert day.context.index[-1] == day.inputs.index[-1]
        assert day.context.loc[day.inputs.index[0]].Temp == day.inputs.iloc[0].Temp
        calls.append((name, 'predict', day))
        result = pd.DataFrame({'prediction':baseline}, index=day.inputs.index)
        if name == 'F':
            result['variance'] = parameters['variance']
        return result
    def save(day, baseline, parameters, directory):
        _atomic_report(directory/f'model_{day.weekday}.json', {'baseline':baseline, 'parameters':parameters})
    return WeekdayCallbacks(name=name, metadata={'fit_budget':'synthetic fixture only',
        'initialization_policy':'synthetic no-state fixture', 'population':'synthetic', 'target_unit':'synthetic',
        'source_sha256':{'fixture':'synthetic'}}, prediction_columns={name:'prediction'},
        fit_baseline=baseline, fit_parameters=fit, predict=predict, save_fit=save, warmup_hours=24)


class CalendarTests(unittest.TestCase):
    def test_consecutive_windows_and_five_matching_weekdays(self):
        windows = rolling_windows('2022-02-28', '2022-03-14')
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[1].start, windows[0].start + pd.Timedelta(weeks=1))
        for w in windows:
            for weekday in range(7):
                days = w.training_days(weekday)
                self.assertEqual(len(days), 5)
                self.assertEqual(days[0], w.start + pd.Timedelta(days=weekday))
                self.assertTrue(all(w.start <= d < w.origin and d.weekday() == weekday for d in days))
                self.assertEqual(days[-1] + pd.Timedelta(weeks=1), w.origin + pd.Timedelta(days=weekday))
        for first,last in [('2022-03-01','2022-03-14'), ('2022-03-14','2022-03-07'),
                           ('2022-03-07 01:00','2022-03-14')]:
            with self.assertRaises(ValueError): rolling_windows(first,last)

    def test_spring_dst_physical_week_and_local_weekday(self):
        raw = raw_frame(); window = WeekWindow('2022-03-21')
        data = split_week(raw, raw, window, warmup_hours=24)
        self.assertEqual(len(data.inputs), 167)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'), data.inputs.index)
        sunday = split_weekday(data, 6)
        self.assertEqual(len(sunday.inputs), 23)
        self.assertEqual([len(r) for r in sunday.records], [24]*5)
        after = split_week(raw,raw,WeekWindow('2022-04-04'))
        self.assertEqual(len(after.history),839)
        self.assertEqual([len(r) for r in split_weekday(after,6).records], [24,24,24,23,24])

    def test_aware_autumn_preserves_both_hours_naive_rejected(self):
        idx = pd.date_range('2022-09-01','2022-11-01', freq='h', inclusive='left', tz='Europe/Copenhagen')
        raw = pd.DataFrame({'Demand':1.,'Price':1.,'Temp':1.}, index=idx)
        data = split_week(raw,raw,WeekWindow('2022-10-24'))
        self.assertEqual(len(data.inputs),169)
        self.assertEqual(len(split_weekday(data,6).inputs),25)
        self.assertTrue(data.inputs.index.is_unique)
        naive = raw.copy(); naive.index = naive.index.tz_localize(None)
        with self.assertRaises(ValueError): split_week(naive,naive,WeekWindow('2022-10-24'))

    def test_no_older_future_or_other_weekday_targets_leak(self):
        raw = raw_frame(); w = WeekWindow('2022-03-21')
        original = split_week(raw,raw,w,warmup_hours=24)
        changed = raw.copy(); changed.loc[(changed.index<w.start)|(changed.index>=w.origin),'Demand']=1e30
        mutated = split_week(changed,changed,w,warmup_hours=24)
        pd.testing.assert_frame_equal(original.history,mutated.history)
        pd.testing.assert_frame_equal(original.inputs,mutated.inputs)
        changed = raw.copy(); changed.loc[changed.index.weekday!=0,'Demand']=1e20
        monday = split_weekday(original,0)
        mutated_monday = split_weekday(split_week(changed,changed,w,warmup_hours=24),0)
        pd.testing.assert_frame_equal(monday.context,mutated_monday.context)
        for a,b in zip(monday.records,mutated_monday.records): pd.testing.assert_frame_equal(a,b)

    def test_missing_targets_retained_missing_day_or_inputs_fail(self):
        raw = raw_frame(); w = WeekWindow('2022-03-21'); raw.loc['2022-03-07 11:00','Demand']=np.nan
        monday = split_weekday(split_week(raw,raw,w),0)
        self.assertEqual(sum(r.Demand.count() for r in monday.records),119)
        raw.loc['2022-03-07':'2022-03-07 23:00','Demand']=np.nan
        with self.assertRaises(ValueError): split_weekday(split_week(raw,raw,w),0)
        raw.loc['2022-03-25 11:00','Temp']=np.nan
        with self.assertRaises(ValueError): split_week(raw,raw,w)
        with self.assertRaises(ValueError): split_week(raw,raw,w,input_columns=('Demand',))

    def test_finite_spring_placeholder_is_not_discarded(self):
        raw=raw_frame(); raw.loc[pd.Timestamp('2022-03-27 02:00')]=[1.,1.,1.]; raw=raw.sort_index()
        with self.assertRaisesRegex(ValueError,'nonexistent physical hours'):
            split_week(raw,raw,WeekWindow('2022-03-21'))


class RunnerTests(unittest.TestCase):
    def test_F_and_G_callbacks_are_separate_and_all_predictions_precede_scoring(self):
        raw = raw_frame(); windows = rolling_windows('2022-03-21','2022-03-28')
        for name in ('F','G'):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temp:
                out = Path(temp)/'run'; calls=[]
                def load():
                    self.assertTrue((out/'all_forecasts_frozen.json').exists())
                    for w in windows:
                        directory = out/w.origin.strftime('%Y%m%d')
                        freeze=json.loads((directory/'forecast_freeze.json').read_text())
                        for p,h in freeze['artifact_sha256'].items():
                            self.assertEqual(hashlib.sha256((directory/p).read_bytes()).hexdigest(),h)
                    self.assertEqual(len([c for c in calls if c[1]=='fit']),14)
                    return raw.Demand
                result = run_backtest(raw,raw,windows,callbacks(name,calls),out,load_observations=load)
                self.assertEqual(result['status'],'complete'); self.assertEqual(result['pooled'][name]['samples'],335)
                self.assertEqual(len(calls),42)
                # Next window legitimately uses last week's Monday after it is historical.
                next_monday=[c[2] for c in calls if c[1]=='fit' and c[2].weekday==0][1]
                self.assertIn(pd.Timestamp('2022-03-21'),next_monday.training_days)
                for c in calls:
                    if c[1]=='fit':
                        self.assertLess(c[2].context.index.max(),c[2].inputs.index.min())
                first=pd.read_csv(out/'20220321'/'predictions.csv')
                self.assertEqual('variance' in first,name=='F')
                with self.assertRaises(FileExistsError): run_backtest(raw,raw,windows,callbacks(name,[]),out)

    def test_failure_preserves_artifacts_does_not_retry_or_open_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=raw_frame(); out=Path(temp)/'run'; calls=[]; opened=[]
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                run_backtest(raw,raw,rolling_windows('2022-03-21','2022-03-28'),callbacks('F',calls,fail=True),out,
                             load_observations=lambda:opened.append(True))
            self.assertFalse(opened); self.assertEqual([c[1] for c in calls],['baseline','fit'])
            self.assertTrue((out/'20220321'/'window.json').exists())
            self.assertFalse((out/'20220321'/'forecast_freeze.json').exists())
            self.assertFalse((out/'20220328').exists())
            self.assertFalse(json.loads((out/'failure.json').read_text())['automatic_retry'])

    def test_nonfinite_or_incomplete_forecast_is_not_frozen(self):
        for bad in ('missing','nan','target'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temp:
                raw=raw_frame(); adapter=callbacks('G',[])
                def broken(data,directory):
                    df=pd.DataFrame({'prediction':1.},index=data.inputs.index)
                    if bad=='missing': return df.iloc[:-1]
                    if bad=='nan': df.iloc[0,0]=np.nan
                    if bad=='target': df['Demand']=0.
                    return df
                adapter.forecast_week=broken
                with self.assertRaises(ValueError):
                    run_backtest(raw,raw,rolling_windows('2022-03-21','2022-03-21'),adapter,Path(temp)/'run')
                self.assertFalse((Path(temp)/'run'/'20220321'/'forecast_freeze.json').exists())

    def test_pooling_uses_observation_counts_not_mean_week_RMSE(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=raw_frame(); adapter=callbacks('G',[])
            adapter.forecast_week=lambda data,directory: pd.DataFrame({'prediction':0.},index=data.inputs.index)
            truth=raw.Demand.copy(); truth.loc['2022-03-21':'2022-03-27 23:00']=1.
            truth.loc['2022-03-28':'2022-04-03 23:00']=3.
            truth.loc['2022-03-28 00:00']=np.nan
            report=run_backtest(raw,raw,rolling_windows('2022-03-21','2022-03-28'),adapter,Path(temp)/'run',load_observations=lambda:truth)
            self.assertEqual(report['pooled']['G']['samples'],334)
            self.assertAlmostEqual(report['pooled']['G']['rmse'],np.sqrt(5.))
            idx=pd.date_range('2022-01-01',periods=2,freq='h')
            scores=score_predictions(pd.DataFrame({'p':1.},index=idx),pd.Series(np.nan,index=idx),{'G':'p'})
            self.assertIsNone(scores['G']['rmse']); self.assertEqual(scores['G']['samples'],0)
            scores=score_predictions(pd.DataFrame({'p':1.},index=idx),pd.Series(2.,index=idx),{'G':'p'})
            self.assertEqual(scores['G']['bias'],-1.)

    def test_modified_frozen_artifact_blocks_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=raw_frame(); out=Path(temp)/'run'
            def load():
                path=out/'20220321'/'predictions.csv'; path.write_text('modified',encoding='utf-8')
                return raw.Demand
            with self.assertRaisesRegex(ValueError,'artifacts changed'):
                run_backtest(raw,raw,rolling_windows('2022-03-21','2022-03-21'),callbacks('F',[]),out,load_observations=load)
            self.assertFalse((out/'20220321'/'scores.json').exists())

    def test_plan_only_requires_no_data_and_does_not_load_G(self):
        with tempfile.TemporaryDirectory() as temp, patch('src.exe.run_model.weekday_backtest._csv',side_effect=AssertionError('no data')):
            out=Path(temp)/'plan'
            self.assertEqual(main(['--first-origin','2023-01-02','--last-origin','2023-01-16',
                                  '--output',str(out),'--plan-only']),0)
            plan=json.loads((out/'calendar_plan.json').read_text())
            self.assertEqual(len(plan['windows']),3); self.assertFalse(plan['model_executed'])


if __name__ == '__main__':
    unittest.main()

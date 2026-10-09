"""Shared calendar, isolated model callbacks, freeze-before-score and G wiring."""
import hashlib
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.tools.bm_1_reports import _atomic_report
from src.tools.weekday_backtest import (WeekWindow, WeekdayData, rolling_windows,
    split_week, split_weekday, run_backtest, score_predictions)
from src.tools.g_weekday_backtest_adapter import GWeekdayAdapter, make_adapter
from src.exe.run_model.weekday_backtest import main


def raw_frame():
    physical = pd.date_range('2022-01-01', '2022-05-01', freq='h', inclusive='left', tz='Europe/Copenhagen')
    index = physical.tz_localize(None); n = np.arange(len(index))
    return pd.DataFrame({'Demand':10+index.weekday+np.sin(n/9), 'Price':100+np.cos(n), 'Temp':n/100}, index=index)


def g_manifest(**changes):
    return {'schema':'bm1-g-weekday-input-v1', 'group':'MED', 'population_id':'MED_synthetic_available',
            'measurement_columns':['synthetic_MED'], 'aggregation':'mean over available measured meters',
            'demand_timestamp_label':'hour_start', 'history_csv_sha256':'a'*64,
            'target_unit':'synthetic raw units', 'input_scenario_policy':'synthetic declared scenario', **changes}


def callbacks(name, calls, *, fail=False):
    """Synthetic adapter; production adapters own their fit and prediction stages."""
    def forecast_week(data, directory):
        forecasts = []
        for weekday in range(7):
            day = split_weekday(data, weekday)
            calls.append((name, 'baseline', day))
            assert len(day.records) == 5
            assert all(r.index.weekday[0] == weekday for r in day.records)
            assert day.context.loc[day.context.index.weekday != weekday, 'Demand'].isna().all()
            values = pd.concat(day.records).Demand.dropna().to_numpy()
            baseline = float(pd.concat(day.records).Demand.mean())
            calls.append((name, 'fit', day))
            if fail:
                raise RuntimeError('Synthetic own-model budget exhausted')
            parameters = {'variance':float(np.mean((values-baseline)**2))} if name == 'F' else {'offset':0.}
            _atomic_report(directory/f'model_{weekday}.json', {'baseline':baseline, 'parameters':parameters})
            prefix = data.inputs.loc[data.inputs.index <= day.inputs.index[-1]]
            context = pd.concat([day.context.drop(columns=['Demand']), prefix])
            prediction_day = WeekdayData(weekday, day.training_days, (), context, day.inputs.copy(), 'Demand')
            assert 'Demand' not in prediction_day.context and 'Demand' not in prediction_day.inputs
            assert context.index[-1] == day.inputs.index[-1]
            assert context.loc[day.inputs.index[0]].Temp == day.inputs.iloc[0].Temp
            calls.append((name, 'predict', prediction_day))
            result = pd.DataFrame({'prediction':baseline}, index=day.inputs.index)
            if name == 'F':
                result['variance'] = parameters['variance']
            forecasts.append(result)
        return pd.concat(forecasts).sort_index()
    return SimpleNamespace(name=name, metadata={'fit_budget':'synthetic fixture only',
        'initialization_policy':'synthetic no-state fixture', 'population':'synthetic', 'target_unit':'synthetic',
        'source_sha256':{'fixture':'synthetic'}}, prediction_columns={name:'prediction'},
        warmup_hours=24, input_columns=('Price','Temp'), target_column='Demand', forecast_week=forecast_week)


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
        g = GWeekdayAdapter(g_manifest())
        with self.assertRaises(ValueError): g.forecast_week(data,Path('unused'))

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

    def test_G_adapter_reuses_existing_math_and_budget_stop(self):
        raw=raw_frame(); data=split_week(raw,raw,WeekWindow('2022-04-11'),warmup_hours=24)
        g=GWeekdayAdapter(g_manifest())
        with tempfile.TemporaryDirectory() as temp, patch('src.tools.g_weekday_backtest_adapter.model.fit_rolling') as fit:
            fit.return_value={'status':'unfinished','reports':[],'snapshot':None,'seconds':0.}
            with self.assertRaisesRegex(RuntimeError,'no automatic retry'):
                g.forecast_week(data,Path(temp))
            self.assertEqual(fit.call_count,1)
            self.assertEqual(fit.call_args.kwargs['baseline_policy'],'weekday')
            pd.testing.assert_frame_equal(fit.call_args.args[0],data.history)
            self.assertEqual(g.metadata['fit_budget']['per_weekday']['G_calls'],750)
            binding=json.loads((Path(temp)/'population_binding.json').read_text())
            self.assertEqual(binding['metadata']['measurement_columns'],['synthetic_MED'])
            self.assertEqual(binding['membership_status'],'declared-unverified')
            self.assertFalse((Path(temp)/'fit_population_binding.json').exists())

    def test_G_success_writes_own_snapshots_and_exports_before_return(self):
        from copy import deepcopy
        from src.tools.g_weekday_backtest_adapter import _fingerprint, model
        raw=raw_frame(); data=split_week(raw,raw,WeekWindow('2022-04-11'),warmup_hours=24)
        table=pd.DataFrame({'baseline':1.,'g_prediction_output_mean':2.,'wm_prediction_output_mean':3.},index=data.inputs.index)
        snapshot={'own':'parent', 'population':'MED mean over available measured meters',
                  'training_start':str(data.window.start), 'training_end_exclusive':str(data.window.origin),
                  'origin':str(data.window.origin), 'demand_timestamp_label':'hour_start',
                  'training_fingerprint':model.interval.point._fingerprint(model.training_data(data.history,data.window.origin)[0]),
                  'scales':{'demand_min':1.,'demand_scale':2.},
                  'days':[{'weekday':d,'baseline':{'synthetic':d}} for d in range(7)]}
        fitted={'status':'complete','snapshot':snapshot,'historical_replay':data.history,'seconds':0.}
        refined={}
        def export(snapshot,history):
            refined.update(snapshot={'parent':deepcopy(snapshot),'refined_view':deepcopy(snapshot),
                'parent_snapshot_sha256':model.residual.snapshot_fingerprint(snapshot)},historical_replay=history)
            return refined
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.tools.g_weekday_backtest_adapter.model.fit_rolling',return_value=fitted) as fit, \
             patch('src.tools.g_weekday_backtest_adapter.numerical.refine_for_export',side_effect=export) as refine, \
             patch('src.tools.g_weekday_backtest_adapter.numerical.predict_comparison',return_value={'predictions':table,'manifest':{}}) as predict, \
             patch('src.tools.g_weekday_backtest_adapter.write_forecast') as write:
            g=GWeekdayAdapter(g_manifest())
            result=g.forecast_week(data,Path(temp))
            pd.testing.assert_frame_equal(result,table)
            self.assertTrue((Path(temp)/'weekday_snapshot.json').exists())
            self.assertTrue((Path(temp)/'weekday_export_snapshot.json').exists())
            self.assertEqual(refine.call_args.args[0],fitted['snapshot'])
            self.assertEqual(predict.call_args.args[0],refined['snapshot'])
            pd.testing.assert_frame_equal(predict.call_args.args[1],data.inputs)
            write.assert_called_once()
            binding=json.loads((Path(temp)/'fit_population_binding.json').read_text())
            self.assertEqual(binding['metadata']['input_manifest'],g_manifest())
            self.assertFalse(binding['metadata']['history_file_sha256_verified'])
            self.assertEqual(binding['training_start'],str(data.window.start))
            self.assertEqual(binding['training_end_exclusive'],str(data.window.origin))
            self.assertEqual(binding['scales'],snapshot['scales'])
            self.assertEqual(binding['fitted_training_fingerprint'],snapshot['training_fingerprint'])
            self.assertEqual(len(binding['baseline_sha256']),7)
            self.assertEqual(binding['history_sha256'],_fingerprint(data.history))
            self.assertEqual(binding['snapshot_sha256'],model.residual.snapshot_fingerprint(snapshot))
            self.assertEqual(binding['export_snapshot_sha256'],model.residual.snapshot_fingerprint(refined['snapshot']))
            initial=json.loads((Path(temp)/'population_binding.json').read_text())
            self.assertEqual(snapshot['population_binding_sha256'],model.residual.snapshot_fingerprint(initial))
            self.assertEqual(write.call_args.args[1]['manifest']['population']['meters'],['synthetic_MED'])

    def test_G_manifest_rejects_unknown_or_mismatched_population_before_fit(self):
        invalid=[None, {}, g_manifest(group='UDEN'), g_manifest(measurement_columns=[]),
                 g_manifest(measurement_columns=['a_MED','b_UDEN']), g_manifest(measurement_columns=['a_MED','a_MED']),
                 g_manifest(measurement_columns=['_MED']), g_manifest(population_id='UDEN_wrong'),
                 g_manifest(aggregation='sum'), g_manifest(history_csv_sha256='unknown')]
        with patch('src.tools.g_weekday_backtest_adapter.model.fit_rolling') as fit:
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    GWeekdayAdapter(config)
            fit.assert_not_called()
        config=g_manifest(); g=GWeekdayAdapter(config); config['measurement_columns'].append('changed_MED')
        self.assertEqual(g.metadata['measurement_columns'],['synthetic_MED'])

    def test_G_fit_population_and_window_mismatch_prevents_export(self):
        raw=raw_frame(); data=split_week(raw,raw,WeekWindow('2022-04-11'),warmup_hours=24)
        expected={'population':'MED mean over available measured meters',
                  'training_start':str(data.window.start), 'training_end_exclusive':str(data.window.origin)}
        for key,value in [('population','UDEN'),('training_start','2021-01-01'),('training_end_exclusive','2023-01-01')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp, \
                 patch('src.tools.g_weekday_backtest_adapter.model.fit_rolling') as fit, \
                 patch('src.tools.g_weekday_backtest_adapter.numerical.refine_for_export') as refine:
                fit.return_value={'status':'complete','snapshot':{**expected,key:value},'historical_replay':data.history}
                with self.assertRaisesRegex(ValueError,'population or training window'):
                    GWeekdayAdapter(g_manifest()).forecast_week(data,Path(temp))
                refine.assert_not_called()
                self.assertFalse((Path(temp)/'weekday_snapshot.json').exists())

    def test_G_factory_binds_exact_csv_bytes_and_rejects_later_change(self):
        with tempfile.TemporaryDirectory() as temp:
            history=Path(temp)/'history.csv'; history.write_bytes(b'Datetime,Demand,Price,Temp\n')
            config=g_manifest(history_csv_sha256=hashlib.sha256(history.read_bytes()).hexdigest())
            manifest=Path(temp)/'manifest.json'; manifest.write_text(json.dumps(config))
            g=make_adapter(manifest,history)
            self.assertTrue(g.metadata['history_file_sha256_verified'])
            history.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'): g.verify_history_file()
            with self.assertRaisesRegex(ValueError,'SHA256'): make_adapter(manifest,history)
            manifest.write_text(json.dumps(g_manifest(group='UDEN')))
            with patch('src.tools.g_weekday_backtest_adapter._sha',side_effect=AssertionError('history must remain unopened')):
                with self.assertRaisesRegex(ValueError,'MED only'): make_adapter(manifest,history)

    def test_G_fitted_training_identity_must_match_actual_historical_frame(self):
        raw=raw_frame();data=split_week(raw,raw,WeekWindow('2022-04-11'),warmup_hours=24)
        snapshot={'population':'MED mean over available measured meters','training_start':str(data.window.start),
                  'training_end_exclusive':str(data.window.origin),'training_fingerprint':'0'*64}
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.tools.g_weekday_backtest_adapter.model.fit_rolling',return_value={'status':'complete','snapshot':snapshot}), \
             patch('src.tools.g_weekday_backtest_adapter.numerical.refine_for_export') as refine:
            with self.assertRaisesRegex(ValueError,'actual isolated training data'):
                GWeekdayAdapter(g_manifest()).forecast_week(data,Path(temp))
            refine.assert_not_called()

    def test_G_cli_requires_manifest_and_rejects_UDEN_before_input(self):
        args=['--first-origin','2022-04-11','--last-origin','2022-04-11','--output','unused',
              '--history-csv','unopened.csv','--inputs-csv','unopened-inputs.csv']
        with patch('src.exe.run_model.weekday_backtest._csv',side_effect=AssertionError('no data')), \
             patch('src.tools.g_weekday_backtest_adapter.make_adapter') as factory, redirect_stderr(io.StringIO()):
            for extra in ([], ['--group','UDEN','--adapter-config','unopened.json']):
                with self.subTest(extra=extra), self.assertRaises(SystemExit) as stopped: main(args+extra)
                self.assertEqual(stopped.exception.code,2)
            factory.assert_not_called()

    def test_F_cli_group_is_assertion_before_input(self):
        args=['--first-origin','2022-04-11','--last-origin','2022-04-11','--output','unused',
              '--history-csv','unopened.csv','--inputs-csv','unopened-inputs.csv',
              '--adapter','F','--adapter-config','model.json','--group','UDEN']
        with patch('src.exe.run_model.weekday_backtest._csv',side_effect=AssertionError('no data')), \
             patch('src.tools.f_weekday_backtest_adapter.make_adapter') as factory, redirect_stderr(io.StringIO()):
            factory.return_value.group='MED'
            with self.assertRaises(SystemExit) as stopped: main(args)
            self.assertEqual(stopped.exception.code,2)

    def test_plan_only_requires_no_data_and_does_not_load_G(self):
        with tempfile.TemporaryDirectory() as temp, patch('src.exe.run_model.weekday_backtest._csv',side_effect=AssertionError('no data')):
            out=Path(temp)/'plan'
            self.assertEqual(main(['--first-origin','2023-01-02','--last-origin','2023-01-16',
                                  '--output',str(out),'--plan-only']),0)
            plan=json.loads((out/'calendar_plan.json').read_text())
            self.assertEqual(len(plan['windows']),3); self.assertFalse(plan['model_executed'])


if __name__ == '__main__':
    unittest.main()

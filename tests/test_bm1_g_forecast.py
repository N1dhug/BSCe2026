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
        record = model.export_model(original,legacy=True)
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


def declared_binding(original, *, marked=True):
    """Synthetic declaration only: never claim the preserved aggregate is verified."""
    from tests.test_weekday_backtest import g_manifest
    original=deepcopy(original)
    view=original['refined_view']
    view.update(training_end_exclusive=view['origin'],demand_timestamp_label='hour_start')
    original.setdefault('parent',deepcopy(view))
    parent=original['parent']
    config=g_manifest()
    metadata={'input_manifest':config,'input_manifest_sha256':model.fingerprint(config),
        'group':config['group'],'population':config['population_id'],'measurement_columns':config['measurement_columns'],
        'aggregation':config['aggregation'],'target_unit':config['target_unit'],
        'input_scenario_policy':config['input_scenario_policy'],'membership_status':'declared-unverified',
        'history_file_sha256_verified':False}
    initial={'metadata':metadata,'history_sha256':'b'*64,'training_start':parent['training_start'],
        'training_end_exclusive':parent['origin'],'membership_status':'declared-unverified'}
    for snapshot in (parent,view):
        if marked:snapshot['population_binding_sha256']=model.fingerprint(initial)
        else:snapshot.pop('population_binding_sha256',None)
    original['parent_snapshot_sha256']=model.fingerprint(parent)
    binding={**initial,'snapshot_sha256':model.fingerprint(parent),'export_snapshot_sha256':model.fingerprint(original),
        'fitted_training_fingerprint':parent['training_fingerprint'],'scales':deepcopy(parent['scales']),
        'baseline_sha256':{str(day['weekday']):model.fingerprint(day['baseline']) for day in parent['days']}}
    return original,binding


def save_fit(directory, original, binding):
    directory.mkdir()
    final_fields=('snapshot_sha256','export_snapshot_sha256','fitted_training_fingerprint','scales','baseline_sha256')
    artifacts={'weekday_snapshot.json':original['parent'],'weekday_export_snapshot.json':original,
        'population_binding.json':{k:v for k,v in binding.items() if k not in final_fields},
        'fit_population_binding.json':binding}
    for name,record in artifacts.items():
        (directory/name).write_text(json.dumps(record),encoding='utf-8')
    (directory/'forecast_freeze.json').write_text(json.dumps({'status':'frozen-input-only',
        'artifact_sha256':{name:runner.digest(directory/name) for name in artifacts}}),encoding='utf-8')
    return directory/'weekday_export_snapshot.json'


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

    def test_declared_population_flows_from_new_fit_to_export_forecast_and_result(self):
        from src.tools.g_weekday_backtest_adapter import GWeekdayAdapter, model as rolling
        from src.tools.weekday_backtest import run_backtest,rolling_windows
        from tests.test_weekday_backtest import raw_frame,g_manifest
        raw=raw_frame();origin='2022-02-28'
        parent=deepcopy(self.record['original_snapshot']['refined_view'])
        parent.update(training_end_exclusive=parent['origin'],demand_timestamp_label='hour_start',
            training_fingerprint=rolling.interval.point._fingerprint(rolling.training_data(raw,origin)[0]))
        def refine(snapshot,history):
            return {'snapshot':{'parent':deepcopy(snapshot),'refined_view':deepcopy(snapshot),
                'parent_snapshot_sha256':model.fingerprint(snapshot)},'historical_replay':history}
        def compare(snapshot,scenario):
            table=pd.DataFrame({'baseline':1.,'g_prediction_output_mean':1.,'wm_prediction_output_mean':1.},index=scenario.index)
            return {'predictions':table,'normalized_predictions':table.copy(),'manifest':{}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'fit'
            with patch.object(rolling,'fit_rolling',side_effect=lambda history,*a,**k:{'status':'complete','snapshot':parent,'historical_replay':history}), \
                 patch('src.tools.g_weekday_backtest_adapter.numerical.refine_for_export',side_effect=refine), \
                 patch('src.tools.g_weekday_backtest_adapter.numerical.predict_comparison',side_effect=compare):
                run_backtest(raw,raw,rolling_windows(origin,origin),GWeekdayAdapter(g_manifest()),out)
            fit_dir=out/'20220228';saved=fit_dir/'weekday_export_snapshot.json';package=root/'model'
            with patch('src.method.g_weekday_export.validate_export',side_effect=lambda value:value):
                self.assertEqual(runner.main(['export-model','--snapshot',str(saved),'--output',str(package)]),0)
            record=runner.load_model(package/'model.json')
            self.assertEqual(record['population']['verification'],'declared-unverified')
            self.assertEqual(record['population']['meters'],['synthetic_MED'])
            self.assertEqual(record['population']['declaration'],g_manifest())
            self.assertEqual((package/'fit_population_binding.json').read_bytes(),(fit_dir/'fit_population_binding.json').read_bytes())
            csv=root/'scenario.csv';inputs().to_csv(csv,index_label='Datetime');result=root/'forecast'
            with patch.object(runner,'plot'):
                self.assertEqual(runner.main(['forecast','--model',str(package/'model.json'),'--origin','2022-03-21',
                    '--inputs-csv',str(csv),'--hours','1','--output',str(result)]),0)
            metadata=json.loads((result/'forecast.json').read_text())['manifest']
            self.assertEqual(metadata['population'],record['population'])
            self.assertEqual(metadata['population']['declaration']['target_unit'],g_manifest()['target_unit'])
            self.assertFalse(metadata['demand_used']);self.assertEqual(metadata['physical_hours'],1)
            with (package/'fit_population_binding.json').open('a') as handle:handle.write(' ')
            with self.assertRaises(ValueError):runner.load_model(package/'model.json')
            (package/'fit_population_binding.json').unlink()
            with self.assertRaises((ValueError,OSError)):runner.load_model(package/'model.json')

    def test_binding_rejects_wrong_snapshots_history_scales_baselines_and_units(self):
        original,binding=declared_binding(self.record['original_snapshot'])
        self.assertEqual(model.population_contract(original,binding)['verification'],'declared-unverified')
        for changed_key in ('snapshot_sha256','export_snapshot_sha256','fitted_training_fingerprint','history_sha256',
                            'scales','baseline_sha256','target_unit','group'):
            changed=deepcopy(binding)
            if changed_key in ('scales','baseline_sha256'):changed[changed_key]={}
            elif changed_key in ('target_unit','group'):changed['metadata'][changed_key]='changed'
            else:changed[changed_key]='c'*64
            with self.subTest(key=changed_key),self.assertRaises(ValueError):model.population_contract(original,changed)
        for key in ('training_fingerprint','scales'):
            changed=deepcopy(original);changed['refined_view'][key]='changed'
            receipt=deepcopy(binding);receipt['export_snapshot_sha256']=model.fingerprint(changed)
            with self.subTest(refined=key),self.assertRaises(ValueError):model.population_contract(changed,receipt)
        with self.assertRaisesRegex(ValueError,'required'):model.population_contract(original,None,legacy=True)

    def test_missing_changed_and_old_unmarked_bindings_fail_closed(self):
        for marked in (False,True):
            original,binding=declared_binding(self.record['original_snapshot'],marked=marked)
            with self.subTest(marked=marked),tempfile.TemporaryDirectory() as temp:
                directory=Path(temp)/'fit';path=save_fit(directory,original,binding)
                self.assertEqual(model.load_fit_binding(path,original)[0],binding)
                sidecar=directory/'fit_population_binding.json'
                sidecar.write_text(json.dumps({**binding,'history_sha256':'changed'}))
                with self.assertRaises(ValueError):model.load_fit_binding(path,original,legacy=True)
                sidecar.unlink()
                with self.assertRaisesRegex(ValueError,'Missing'):model.load_fit_binding(path,original,legacy=True)
                out=directory/'rejected'
                self.assertEqual(runner.main(['export-model','--snapshot',str(path),'--legacy-unverified','--output',str(out)]),2)
                self.assertFalse(out.exists())
        original,binding=declared_binding(self.record['original_snapshot'],marked=False)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'fit';path=save_fit(directory,original,binding)
            (directory/'forecast_freeze.json').unlink()
            with self.assertRaisesRegex(ValueError,'original forecast_freeze'):model.load_fit_binding(path,original)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'legacy.json';path.write_text(json.dumps(self.record['original_snapshot']))
            with self.assertRaises(ValueError):model.load_fit_binding(path,self.record['original_snapshot'])
            self.assertEqual(model.load_fit_binding(path,self.record['original_snapshot'],legacy=True),(None,None,None))

    def test_shared_artifact_validator_rejects_unsafe_missing_and_changed_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);artifact=root/'ok.json';artifact.write_text('{}')
            digest=runner.digest(artifact)
            runner._verify_files(root,{'ok.json':digest})
            for name in (str(artifact.resolve()),'child/../ok.json','../ok.json','missing.json'):
                with self.subTest(name=name),self.assertRaises(ValueError):runner._verify_files(root,{name:digest})
            with self.assertRaises(ValueError):runner._verify_files(root,{'ok.json':'0'*64})

    def test_unmarked_api_requires_original_fit_proof_and_package_is_portable(self):
        original,binding=declared_binding(self.record['original_snapshot'],marked=False)
        with patch('src.method.g_weekday_export.validate_export',side_effect=lambda value:value), tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fit=root/'fit';path=save_fit(fit,original,binding)
            with self.assertRaisesRegex(ValueError,'original frozen fit directory'):
                model.export_model(original,binding)
            changed=deepcopy(binding);changed['history_sha256']='c'*64
            with self.assertRaisesRegex(ValueError,'differs from the original'):
                model.export_model(original,changed,fit_directory=fit)
            package=root/'package'
            self.assertEqual(runner.main(['export-model','--snapshot',str(path),'--output',str(package)]),0)
            fit.rename(root/'no-longer-original-fit')
            record=runner.load_model(package/'model.json')
            self.assertEqual(record['population']['verification'],'declared-unverified')
            self.assertIn('fit_population_freeze',record)
            for replace_proof in (False,True):
                changed=deepcopy(record);receipt=changed['fit_population_binding']
                metadata=receipt['metadata'];metadata['measurement_columns']=['different_MED']
                metadata['input_manifest']['measurement_columns']=['different_MED']
                metadata['input_manifest_sha256']=model.fingerprint(metadata['input_manifest'])
                receipt['history_sha256']='d'*64
                changed['population']=model.population_contract(original,receipt)
                if replace_proof:changed['fit_population_binding_json']=json.dumps(receipt)
                changed['model_sha256']=model.fingerprint({k:v for k,v in changed.items() if k!='model_sha256'})
                with self.subTest(replace_proof=replace_proof),self.assertRaisesRegex(ValueError,'original binding proof'):
                    model.forecast(changed,inputs(),'2022-03-21',hours=1)
            # Even a newly self-consistent conversion receipt cannot replace the
            # preserved byte identity from the original unmarked fit freeze.
            with (package/'fit_population_binding.json').open('a') as handle:handle.write(' ')
            freeze=json.loads((package/'conversion_freeze.json').read_text())
            freeze['artifact_sha256']['fit_population_binding.json']=runner.digest(package/'fit_population_binding.json')
            (package/'conversion_freeze.json').write_text(json.dumps(freeze))
            with self.assertRaises(ValueError):runner.load_model(package/'model.json')

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

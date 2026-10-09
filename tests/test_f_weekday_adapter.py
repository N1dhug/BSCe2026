"""Cheap native integration, continuity, chronology and failure-isolation checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.method.bm_1_interval_native import replay_native_candidate
from src.method.bm_1_interval_precise import replay_precise_candidate, validate_checkpoint
from src.tools.f_weekday_backtest_adapter import FWeekdayAdapter, make_adapter, _plain
from src.tools.weekday_backtest import WeekWindow, WeekdayData, split_week, _fingerprint, run_backtest
from src.tools.bm_1_reports import _atomic_report


def fixture():
    model = dict(schema='bm1-f4-fixed-model-v1', timezone='Europe/Copenhagen',
        population_id='MED_synthetic-single-meter', measurement_columns=['synthetic_MED'],
        baseline_information_cutoff='2021-12-01T00:00:00+01:00',
        baseline_weekday_hour_kW=[.4]*168, scales=dict(power_kW=1., price_low_DKK_MWh=0., price_high_DKK_MWh=1.),
        physical=dict(C=1., k=2., Xi=0., sigma_x=0., sigma_y=.2, time_unit='day'),
        prior=dict(alpha=1., beta=1.), numerics=dict(base_cells=3, fine_cells=5,
            base_max_step=1/24, fine_max_step=1/48, frequencies=64, span=4.,
            relative_guard=1e-8, overlap_log_tolerance=1e-7),
        candidates={'fitted':dict(pieces=2, logits=[-2., -2.])})
    config = dict(schema='bm1-f-weekday-adapter-v1', population_id=model['population_id'],
        measurement_columns=model['measurement_columns'],
        aggregation='fixed-population mean; any missing member makes Demand NaN',
        target_unit='synthetic raw interval mean', input_scenario_policy='explicit synthetic scenario',
        starts=[[0., 0.]], regularization=2., logit_bound=4., max_evaluations=12, max_iterations=4,
        fit_seconds=15., validation_seconds=15., forecast_seconds=5.)
    return config, model


def short_day():
    index = pd.date_range('2022-01-24', periods=3, freq='h')
    history = pd.DataFrame({'Demand':[.4, np.nan, .4], 'Price':.5}, index=index)
    return WeekdayData(0, (), (), history, history[['Price']].copy(), 'Demand')


class NativeTrainingTests(unittest.TestCase):
    def test_native_precise_agree_with_nonzero_flexibility_and_diffusion(self):
        spec=dict(record_id='nonzero',horizon=[23/24,27/24],input_edges=(np.arange(23,28)/24).tolist(),
            baseline=[.3,.65,.5,.4],price=[.8,.2,.7,.3],physical=dict(C=1.,k=2.,Xi=.4,sigma_x=.15,sigma_y=.2),
            starts=[23/24,25/24,26/24],ends=[1.,26/24,27/24],values=[.4,None,.35],observed=[True,False,True],
            initial_log_mass=[float(-np.log(7.))]*7,time_unit='day',midnight_resets=0,
            settings=dict(cells=7,max_step=1/96,frequencies=64,span=4.,
                centers=[.5,.25,.75,0.,1.],relative_guard=1e-8,overlap_log_tolerance=1e-7))
        candidate=dict(pieces=2,logits=[.2,-.3])
        prefix=replay_native_candidate(spec,candidate,limit_units=2)
        native=replay_native_candidate(spec,candidate,saved=prefix)['payload']
        precise=replay_precise_candidate(spec,candidate)['payload']
        self.assertEqual(native['status'],'completed')
        self.assertEqual(precise['status'],'completed')
        self.assertEqual([row['observed'] for row in native['rows']],[True,False,False,True])
        self.assertAlmostEqual(native['log_likelihood'],precise['log_likelihood'],delta=1e-7)
        np.testing.assert_allclose(native['log_mass'],precise['log_mass'],atol=1e-7,rtol=0.)
        self.assertGreater(np.max(np.abs(np.array(native['log_mass'])-spec['initial_log_mass'])),.001)

    def test_precise_equivalence_missing_gap_and_candidate_owned_resume(self):
        spec = dict(record_id='synthetic', horizon=[23/24, 27/24], input_edges=(np.arange(23,28)/24).tolist(),
            baseline=[.3,.4,.5,.6], price=[.5]*4,
            physical=dict(C=1.,k=2.,Xi=0.,sigma_x=0.,sigma_y=.2),
            starts=[23/24,25/24,26/24], ends=[1.,26/24,27/24],
            values=[.35,None,.55], observed=[True,False,True], initial_log_mass=[0.,-900.,-950.],
            time_unit='day', midnight_resets=0, settings=dict(cells=3,max_step=1/48,
            frequencies=64,span=4.,centers=[.5,.25,.75,0.,1.],relative_guard=1e-8,overlap_log_tolerance=1e-7))
        candidate = dict(pieces=2, logits=[0.,0.])
        initial = replay_native_candidate(spec,candidate,limit_units=1)
        self.assertEqual(initial['payload']['status'],'prefix_complete')
        resumed = replay_native_candidate(spec,candidate,saved=initial)
        precise = replay_precise_candidate(spec,candidate)
        with self.assertRaisesRegex(ValueError,'backend/version/source'):
            replay_native_candidate(spec,candidate,saved=precise)
        tampered=deepcopy(initial);tampered['native_source_sha256']['src/method/bm_1_response.py']='changed'
        with self.assertRaisesRegex(ValueError,'backend/version/source'):
            replay_native_candidate(spec,candidate,saved=tampered)
        tampered=deepcopy(initial);tampered['native_runtime']['scipy']='changed'
        with self.assertRaisesRegex(ValueError,'runtime'):
            replay_native_candidate(spec,candidate,saved=tampered)
        tampered=deepcopy(initial);tampered['candidate']['logits'][0]=.3
        with self.assertRaisesRegex(ValueError,'candidate/spec identity'):
            replay_native_candidate(spec,candidate,saved=tampered)
        self.assertEqual(initial['spec'],spec)
        self.assertEqual(initial['candidate'],candidate)
        self.assertEqual(resumed['payload']['status'],'completed')
        self.assertEqual(len(resumed['payload']['rows']),4)
        self.assertEqual([r['observed'] for r in resumed['payload']['rows']],[True,False,False,True])
        self.assertAlmostEqual(resumed['payload']['log_likelihood'],precise['payload']['log_likelihood'],delta=1e-7)
        np.testing.assert_allclose(resumed['payload']['log_mass'],precise['payload']['log_mass'],atol=1e-7,rtol=0)
        changed = dict(pieces=2,logits=[.1,0.])
        with self.assertRaisesRegex(ValueError,'identity'):
            validate_checkpoint(initial,spec,changed)
        paused = replay_native_candidate(spec,candidate,should_pause=lambda:True)
        self.assertEqual(paused['payload']['rows'],[])
        self.assertEqual(paused['payload']['status'],'paused')
        zero=deepcopy(spec);zero['initial_log_mass']=[0.,None,None]
        zero_result=replay_native_candidate(zero,candidate)
        self.assertEqual(zero_result['payload']['log_mass'],[0.,None,None])
        self.assertEqual(json.loads(json.dumps(_plain(np.array([0.,-np.inf])),allow_nan=False)),[0.,None])

    def test_actual_optimizer_uses_native_train_and_returns_new_fit(self):
        config,model = fixture()
        adapter=FWeekdayAdapter(config,model)
        with tempfile.TemporaryDirectory() as directory:
            result=adapter._fit(short_day(),Path(directory))
            self.assertEqual(result['pieces'],2)
            np.testing.assert_array_equal(result['logits'],[0.,0.])
            self.assertNotEqual(result['logits'],model['candidates']['fitted']['logits'])
            self.assertTrue(json.loads((Path(directory)/'frozen_fit.json').read_text())['new_curve_fit'])
            self.assertEqual(json.loads((Path(directory)/'numerical_gate.json').read_text())['compared_hours'],3)
            self.assertGreater(json.loads((Path(directory)/'fit_status.json').read_text())['distinct_evaluations'],0)
            forecast=adapter._predict(short_day().inputs,result,Path(directory))
            np.testing.assert_allclose(forecast.f_prediction,.4,atol=1e-10)
            np.testing.assert_allclose(forecast.f_observation_variance,.04,atol=1e-10)
            self.assertEqual(forecast.forecast_horizon_hours.tolist(),[1,2,3])


class FAdapterContractTests(unittest.TestCase):
    def test_later_baseline_cutoff_and_invalid_population_are_rejected_before_fit(self):
        config,model=fixture()
        model['baseline_information_cutoff']='2022-01-25T00:00:00+01:00'
        adapter=FWeekdayAdapter(config,model)
        with patch('src.tools.f_weekday_backtest_adapter.fit_shared_curve_pair',side_effect=AssertionError('fit')):
            with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,'cutoff'):
                adapter._fit(short_day(),Path(directory))
        config['measurement_columns']=['different']
        with self.assertRaisesRegex(ValueError,'population'):
            FWeekdayAdapter(config,model)

    def test_unfinished_optimizer_never_reaches_validation_or_forecast(self):
        config,model=fixture();adapter=FWeekdayAdapter(config,model)
        result=dict(logits=np.zeros(2),objective=0.,log_likelihood=0.,budget_exhausted=True,paused=False,
            optimizer_results=[],distinct_evaluations=1,elapsed_seconds=.01)
        with patch('src.tools.f_weekday_backtest_adapter.fit_shared_curve_pair',return_value=result), \
             patch('src.tools.f_weekday_backtest_adapter.replay_native_record',side_effect=AssertionError('validation')):
            with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(RuntimeError,'Incomplete'):
                adapter._fit(short_day(),Path(directory))

    def test_equal_objective_does_not_accept_different_nonconverged_best_curve(self):
        config,model=fixture();adapter=FWeekdayAdapter(config,model)
        result=dict(logits=np.zeros(2),objective=0.,log_likelihood=0.,budget_exhausted=False,paused=False,
            optimizer_results=[dict(success=True,objective=0.,logits=[.2,.2])],
            distinct_evaluations=1,elapsed_seconds=.01)
        with patch('src.tools.f_weekday_backtest_adapter.fit_shared_curve_pair',return_value=result), \
             patch('src.tools.f_weekday_backtest_adapter.replay_native_record',side_effect=AssertionError('validation')):
            with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(RuntimeError,'Incomplete'):
                adapter._fit(short_day(),Path(directory))

    def test_seven_fits_keep_continuous_masked_history_and_target_free_prefixes(self):
        config,model=fixture();adapter=FWeekdayAdapter(config,model)
        window=WeekWindow('2022-02-28')
        history=pd.DataFrame({'Demand':.4,'Price':.5},index=window.hours(window.start,window.origin))
        inputs=pd.DataFrame({'Demand':999999.,'Price':.5},index=window.hours(window.origin,window.end))
        data=split_week(history,inputs,window,input_columns=('Price',))
        lengths=[];weekdays=[]
        def fit(day,directory):
            weekdays.append(day.weekday)
            self.assertEqual(len(day.context),840)
            mask=day.context.index.weekday==day.weekday
            self.assertTrue(day.context.loc[~mask,'Demand'].isna().all())
            self.assertTrue(day.context.loc[mask,'Demand'].notna().all())
            candidate = dict(weekday=day.weekday)
            _atomic_report(directory/'frozen_fit.json', candidate)
            return candidate
        def predict(prefix,candidate,directory):
            self.assertNotIn('Demand',prefix)
            lengths.append(len(prefix))
            return pd.DataFrame({'f_prediction':np.arange(len(prefix)),'baseline':.4},index=prefix.index)
        with patch.object(adapter,'_fit',side_effect=fit),patch.object(adapter,'_predict',side_effect=predict):
            with tempfile.TemporaryDirectory() as directory:
                predictions=adapter.forecast_week(data,Path(directory))
        self.assertEqual(weekdays,list(range(7)))
        self.assertEqual(lengths,[24,48,72,96,120,144,168])
        self.assertEqual(predictions.f_prediction.tolist(),list(range(168)))
        self.assertEqual(inputs.Demand.iloc[0],999999.)

    def test_model_hash_and_physical_hour_gaps_fail_closed(self):
        config,model=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'model.json').write_text(json.dumps(model))
            config.update(model_file='model.json',model_sha256='wrong')
            (path/'adapter.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                make_adapter(path/'adapter.json')
        adapter=FWeekdayAdapter(config,model)
        frame=short_day().inputs.iloc[[0,2]]
        with self.assertRaisesRegex(ValueError,'physical input hour'):
            adapter._signals(frame)
        spring=pd.DataFrame({'Price':.5},index=pd.date_range('2022-03-27',periods=4,freq='h',tz='Europe/Copenhagen'))
        edges,_,_,local=adapter._signals(spring)
        np.testing.assert_allclose(np.diff(edges),1/24)
        self.assertEqual(local.hour.tolist(),[0,1,3,4])


class CompletedFitResumeTests(unittest.TestCase):
    """Synthetic fit callbacks isolate recovery; native numerics are tested above."""
    def data(self):
        window = WeekWindow('2022-02-28')
        history = pd.DataFrame({'Demand':.4,'Price':.5}, index=window.hours(window.start, window.origin))
        inputs = pd.DataFrame({'Price':.5}, index=window.hours(window.origin, window.end))
        return split_week(history, inputs, window, input_columns=('Price',))

    def fake_fit(self, calls, fail=None):
        def fit(day, directory):
            calls.append(day.weekday)
            if fail == day.weekday:
                _atomic_report(directory/'partial.json', {'status':'incomplete'})
                raise TimeoutError('synthetic interruption within fit')
            candidate = dict(pieces=2, logits=[0.,0.], base_log_state=[0.,None,None],
                             fine_log_state=[0.,None,None,None,None])
            _atomic_report(directory/'frozen_fit.json', dict(candidate,
                history_sha256=_fingerprint(day.context), new_curve_fit=True))
            return candidate
        return fit

    def fake_prediction(self, prefix, candidate, directory):
        self.assertNotIn('Demand', prefix)
        return pd.DataFrame({'f_prediction':.4, 'baseline':.4}, index=prefix.index)

    def test_resume_reuses_complete_fits_and_forecasts_without_refitting(self):
        adapter=FWeekdayAdapter(*fixture());data=self.data();calls=[];predicted=[]
        def interrupt(prefix, candidate, directory):
            predicted.append(len(prefix))
            if len(prefix)==48:
                raise KeyboardInterrupt()
            return self.fake_prediction(prefix,candidate,directory)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            with patch.object(adapter,'_fit',side_effect=self.fake_fit(calls)), patch.object(adapter,'_predict',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_backtest(data.history,data.inputs,[data.window],adapter,root)
            self.assertEqual(calls,[0,1])
            self.assertEqual(json.loads((root/'run_status.json').read_text())['status'],'paused')
            with patch.object(adapter,'_fit',side_effect=self.fake_fit(calls)), patch.object(adapter,'_predict',side_effect=self.fake_prediction):
                result=run_backtest(data.history,data.inputs,[data.window],adapter,root,resume=True)
            self.assertEqual(calls,list(range(7)))
            self.assertEqual(result['status'],'complete')
            with patch.object(adapter,'forecast_week',side_effect=AssertionError('repeated model work')):
                again=run_backtest(data.history,data.inputs,[data.window],adapter,root,resume=True)
            self.assertEqual(result,again)

    def test_explicit_resume_preserves_partial_attempt_and_retries_only_unfinished_fit(self):
        adapter=FWeekdayAdapter(*fixture());data=self.data();calls=[]
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            with patch.object(adapter,'_fit',side_effect=self.fake_fit(calls,fail=1)),patch.object(adapter,'_predict',side_effect=self.fake_prediction):
                with self.assertRaises(TimeoutError):adapter.forecast_week(data,directory)
            partial=directory/'weekday_1/attempt_0001/partial.json';original=partial.read_bytes()
            with patch.object(adapter,'_fit',side_effect=self.fake_fit(calls)),patch.object(adapter,'_predict',side_effect=self.fake_prediction):
                prediction=adapter.forecast_week(data,directory,resume=True)
            self.assertEqual(calls,[0,1,1,2,3,4,5,6])
            self.assertEqual(partial.read_bytes(),original)
            self.assertTrue((directory/'weekday_1/attempt_0002/fit_complete.json').is_file())
            self.assertEqual(len(prediction),168)

    def test_changed_frozen_fit_input_or_config_is_rejected_before_model_work(self):
        adapter=FWeekdayAdapter(*fixture());data=self.data();calls=[]
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            with patch.object(adapter,'_fit',side_effect=self.fake_fit(calls,fail=1)),patch.object(adapter,'_predict',side_effect=self.fake_prediction):
                with self.assertRaises(TimeoutError):adapter.forecast_week(data,directory)
            changed=deepcopy(data);changed.inputs.iloc[0,0]=.6
            with patch.object(adapter,'_fit',side_effect=AssertionError('fit')):
                with self.assertRaisesRegex(ValueError,'identity'):adapter.forecast_week(changed,directory,resume=True)
                adapter.config['fit_seconds']+=1
                with self.assertRaisesRegex(ValueError,'identity'):adapter.forecast_week(data,directory,resume=True)
                adapter.config['fit_seconds']-=1
                fit=directory/'weekday_0/attempt_0001/frozen_fit.json';fit.write_bytes(fit.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'changed'):adapter.forecast_week(data,directory,resume=True)

    def test_frozen_week_reuse_rejects_changed_source_and_inputs_and_never_opens_targets(self):
        adapter=FWeekdayAdapter(*fixture());data=self.data()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            with patch.object(adapter,'_fit',side_effect=self.fake_fit([])),patch.object(adapter,'_predict',side_effect=self.fake_prediction):
                run_backtest(data.history,data.inputs,[data.window],adapter,root)
            changed=data.inputs.copy();changed.iloc[0,0]=.7
            with patch.object(adapter,'forecast_week',side_effect=AssertionError('model')):
                with self.assertRaisesRegex(ValueError,'identity'):
                    run_backtest(data.history,changed,[data.window],adapter,root,resume=True)
                plan=json.loads((root/'plan.json').read_text());plan['runner_sha256']='changed'
                (root/'plan.json').write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError,'identity'):
                    run_backtest(data.history,data.inputs,[data.window],adapter,root,resume=True)

    def test_pre_model_crashes_recover_only_when_no_unbound_artifacts_exist(self):
        adapter=FWeekdayAdapter(*fixture());data=self.data()
        for crash in ('plan', 'window', 'adapter'):
            with self.subTest(crash=crash),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'run'
                root.mkdir()
                if crash!='plan':
                    # Stop after the runner has atomically published its plan/window.
                    with patch.object(adapter,'forecast_week',side_effect=KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            run_backtest(data.history,data.inputs,[data.window],adapter,root,resume=True)
                    if crash=='window':(root/'20220228/window.json').unlink()
                with patch.object(adapter,'_fit',side_effect=self.fake_fit([])),patch.object(adapter,'_predict',side_effect=self.fake_prediction):
                    result=run_backtest(data.history,data.inputs,[data.window],adapter,root,resume=True)
                self.assertEqual(result['status'],'complete')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'unbound.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Missing run plan'):
                run_backtest(data.history,data.inputs,[data.window],adapter,root,resume=True)

    def test_invalid_meter_lists_and_mixed_groups_fail_before_fit(self):
        for meters in ([],['one_MED','one_MED'],['one_MED','two_UDEN'],['unknown']):
            config,model=fixture();config['measurement_columns']=meters;model['measurement_columns']=meters
            with self.subTest(meters=meters),self.assertRaisesRegex(ValueError,'population'):
                FWeekdayAdapter(config,model)

    def test_group_cannot_relabel_frozen_measurement_population(self):
        config,model=fixture();model['measurement_columns']=['1_MED'];config['measurement_columns']=['1_MED']
        config['group']='UDEN'
        with self.assertRaisesRegex(ValueError,'group'):FWeekdayAdapter(config,model)


if __name__=='__main__':unittest.main()

"""Existing bounded G/WM weekday mathematics behind the shared backtest API."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from src.method import g_weekday_rolling as model
from src.method import g_weekday_export as numerical
from src.method import bm_1_g_forecast as deployment
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1_g import write_forecast
from src.tools.weekday_backtest import _fingerprint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GWeekdayAdapter:
    name = 'G/WM'
    warmup_hours = 24
    input_columns = ('Price', 'Temp')
    target_column = 'Demand'
    prediction_columns = {'G':'g_prediction_output_mean', 'WM':'wm_prediction_output_mean', 'baseline':'baseline'}

    def __init__(self, config=None):
        self.config = deployment.validate_input_manifest(config)
        self.group = 'MED'
        self.history_path = None

    def verify_history_file(self):
        if self.history_path is not None and _sha(self.history_path)!=self.config['history_csv_sha256']:
            raise ValueError('Historical aggregate CSV differs from its declared SHA256.')

    @property
    def metadata(self):
        sources = numerical.source_hashes()
        sources['src/tools/g_weekday_backtest_adapter.py'] = _sha(__file__)
        sources['src/exe/run_model/weekday_backtest.py'] = _sha(model.interval.ROOT/'src/exe/run_model/weekday_backtest.py')
        sources['src/method/bm_1_g_forecast.py'] = _sha(deployment.__file__)
        sources['src/exe/run_model/bm_1_g.py'] = _sha(model.interval.ROOT/'src/exe/run_model/bm_1_g.py')
        return {'fit_budget':{'per_weekday':{'G_seconds':30, 'G_calls':750, 'WM_seconds':10, 'WM_calls':400},
                              'numerical_export_seconds':15},
                'initialization_policy':'existing G input-only warmup / WM variable projection; separate weekday median states',
                'population':self.config['population_id'], 'group':self.group,
                'measurement_columns':deepcopy(self.config['measurement_columns']),
                'aggregation':self.config['aggregation'], 'membership_status':'declared-unverified',
                'membership_limit':'An aggregate CSV and its byte hash do not prove the declared meter membership or aggregation.',
                'input_manifest':deepcopy(self.config), 'input_manifest_sha256':model.residual.snapshot_fingerprint(self.config),
                'history_file_sha256_verified':self.history_path is not None,
                'target_unit':self.config['target_unit'],
                'source_sha256':sources, 'baseline_policy':'weekday',
                'uncertainty_policy':'existing separate weekday empirical residuals; no automatic band recalibration',
                'input_scenario_policy':self.config['input_scenario_policy']}

    def forecast_week(self, data, directory):
        if data.window.zone != model.interval.ZONE or data.history.index.tz is not None or data.inputs.index.tz is not None:
            raise ValueError('Existing G adapter requires unique local-naive Copenhagen labels; autumn ambiguity unsupported.')
        self.verify_history_file()
        metadata = self.metadata
        binding = {'metadata':metadata, 'history_sha256':_fingerprint(data.history),
            'training_start':str(data.window.start), 'training_end_exclusive':str(data.window.origin),
            'observations_passed_to_fit':'historical aggregate only; future Demand absent',
            'membership_status':'declared-unverified'}
        _atomic_report(directory/'population_binding.json', binding)

        def checkpoint(record):
            _atomic_report(directory/f"weekday_{record['weekday']}_{record['stage']}.json",
                           model.interval.point._plain(record), replace_existing=True)

        fitted = model.fit_rolling(data.history, data.window.origin, baseline_policy='weekday', checkpoint=checkpoint)
        _atomic_report(directory/'fit_report.json', model.interval.point._plain(
            {k:v for k,v in fitted.items() if k not in ('snapshot', 'historical_replay')}))
        if fitted['status'] != 'complete':
            raise RuntimeError('Incomplete G/WM weekday fit; checkpoints retained, no automatic retry or extension.')
        snapshot = fitted['snapshot']
        if (snapshot['population']!='MED mean over available measured meters'
                or snapshot['training_start']!=binding['training_start']
                or snapshot['training_end_exclusive']!=binding['training_end_exclusive']):
            raise ValueError('Completed G fit disagrees with its declared population or training window.')
        train = model.training_data(data.history,data.window.origin)[0]
        if snapshot['training_fingerprint']!=model.interval.point._fingerprint(train):
            raise ValueError('Completed G fit disagrees with the actual isolated training data.')
        snapshot['population_binding_sha256'] = deployment.fingerprint(binding)
        _atomic_report(directory/'weekday_snapshot.json', fitted['snapshot'])
        fitted['historical_replay'].to_csv(directory/'historical_replay_normalized.csv', index_label='Datetime')
        exported = numerical.refine_for_export(fitted['snapshot'], data.history)
        _atomic_report(directory/'weekday_export_snapshot.json', exported['snapshot'])
        exported['historical_replay'].to_csv(directory/'export_replay_normalized.csv', index_label='Datetime')
        fit_binding = {**binding,
            'snapshot_sha256':model.residual.snapshot_fingerprint(snapshot),
            'export_snapshot_sha256':model.residual.snapshot_fingerprint(exported['snapshot']),
            'fitted_training_fingerprint':snapshot['training_fingerprint'], 'scales':deepcopy(snapshot['scales']),
            'baseline_sha256':{str(day['weekday']):model.residual.snapshot_fingerprint(day['baseline']) for day in snapshot['days']}}
        population = deployment.population_contract(exported['snapshot'],fit_binding)
        _atomic_report(directory/'fit_population_binding.json',fit_binding)
        result = numerical.predict_comparison(exported['snapshot'], data.inputs)
        result['manifest']['population'] = population
        self.verify_history_file()
        if metadata!=self.metadata:
            raise ValueError('G adapter sources or population manifest changed during fitting/forecast.')
        write_forecast(directory, result)
        return result['predictions']


def make_adapter(config_path, history_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    adapter = GWeekdayAdapter(config)
    adapter.history_path = Path(history_path).resolve()
    adapter.verify_history_file()
    return adapter

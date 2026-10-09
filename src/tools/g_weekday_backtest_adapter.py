"""Existing bounded G/WM weekday mathematics behind the shared backtest API."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from src.method import g_weekday_rolling as model
from src.method import g_weekday_export as numerical
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast
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
        if not isinstance(config, dict) or config.get('schema')!='bm1-g-weekday-input-v1':
            raise ValueError('New G fits require an explicit population/input manifest.')
        config = deepcopy(config)
        if config.get('group')!='MED':
            raise ValueError('This preserved G fitting kernel supports declared MED only; UDEN needs its own reviewed fitting contract.')
        columns = config.get('measurement_columns')
        if (not isinstance(columns, list) or not columns or any(not isinstance(c,str) or not c.endswith('_MED') or not c[:-4] for c in columns)
                or len(set(columns))!=len(columns)):
            raise ValueError('Declare a nonempty unique MED meter list for the aggregate.')
        if (not isinstance(config.get('population_id'),str) or not config['population_id'].startswith('MED_')
                or config.get('aggregation')!='mean over available measured meters'
                or config.get('demand_timestamp_label')!='hour_start'):
            raise ValueError('Declare the MED population, available-meter aggregation and hour-start observation contract.')
        digest = config.get('history_csv_sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Declare the lowercase SHA256 of the exact historical aggregate CSV.')
        if any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('target_unit','input_scenario_policy')):
            raise ValueError('Declare raw-unit assumptions and scenario availability.')
        self.config = config
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
        _atomic_report(directory/'weekday_snapshot.json', fitted['snapshot'])
        fitted['historical_replay'].to_csv(directory/'historical_replay_normalized.csv', index_label='Datetime')
        exported = numerical.refine_for_export(fitted['snapshot'], data.history)
        _atomic_report(directory/'weekday_export_snapshot.json', exported['snapshot'])
        exported['historical_replay'].to_csv(directory/'export_replay_normalized.csv', index_label='Datetime')
        _atomic_report(directory/'fit_population_binding.json', {**binding,
            'snapshot_sha256':model.residual.snapshot_fingerprint(snapshot),
            'export_snapshot_sha256':model.residual.snapshot_fingerprint(exported['snapshot']),
            'fitted_training_fingerprint':snapshot['training_fingerprint'], 'scales':deepcopy(snapshot['scales']),
            'baseline_sha256':{str(day['weekday']):model.residual.snapshot_fingerprint(day['baseline']) for day in snapshot['days']}})
        result = numerical.predict_comparison(exported['snapshot'], data.inputs)
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

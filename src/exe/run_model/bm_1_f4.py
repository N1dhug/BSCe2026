"""Portable F4 development runner using the existing native interval kernel.

Fixed shared curves, one fixed-population hourly mean and one open forecast.
X is filling degree; in days dX=(D-B)/C dt + sigma_x X(1-X) dW, with
D=B+Xi*tanh(k*(f(X)+g(U))/2)*(1-B if response>=0 else B).
Y is the hourly mean of D plus nominal Gaussian observation noise, not an
individual meter. prepare freezes history-only input; advance_hour predicts
and optionally conditions one interval; run checkpoints; _export reads rows.
This does not continue the historical campaign or change its acceptance.
Run from the repository root: python -m src.exe.run_model.bm_1_f4 --help
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]


def read_result(directory):
    """Inspect saved output without loading model code or current input files."""
    directory = Path(directory)
    directory = directory if directory.is_absolute() else ROOT/directory
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if not summary.get('identity') or summary['identity'] != manifest.get('identity'):
        raise ValueError('Saved summary/manifest identities disagree.')
    for name, expected in summary.get('output_sha256', {}).items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError('Invalid saved output path.')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise ValueError(f'Saved output checksum differs: {name}.')
    return dict(action='read_saved_result', directory=str(directory.resolve()),
        new_forecast=False, summary=summary, population=manifest.get('population'),
        fit_binding=manifest.get('fit_binding'),
        output_hashes_verified=bool(summary.get('output_sha256')),
        source_identity_policy='Original saved identity retained; no claim that current code produced this result.')


if __name__ == '__main__' and '--read-result' in sys.argv[1:]:
    saved_parser = argparse.ArgumentParser(description='Read saved F4 results without model or input work.')
    saved_parser.add_argument('--read-result', required=True, type=Path)
    saved_args = saved_parser.parse_args()
    try:
        print(json.dumps(read_result(saved_args.read_result), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        saved_parser.error(str(exc))
    raise SystemExit(0)


# Only the executable parent reports boot: imported modules and spawn workers
# must stay silent. This measures imports after the initial standard library.
_BOOT_STARTED = perf_counter() if __name__ == '__main__' else None
_BOOT_UTC = datetime.now(timezone.utc).isoformat() if _BOOT_STARTED is not None else None
if _BOOT_STARTED is not None:
    print(f'F4 {_BOOT_UTC}: imports started; no input or model work.', file=sys.stderr, flush=True)

# Applied before NumPy import, including Windows spawn workers.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
from scipy.special import betainc, logsumexp

from src.exe.run_model.progress import runtime_text
from src.method.bm_1_interval_likelihood import EXTENDED_CENTERS
from src.method.bm_1_interval_forecast import (
    interval_cdf_plan, native_interval_distribution, native_interval_moments,
)
from src.method.bm_1_interval_native import (
    advance_frequency_block, freeze_native_operator, native_terminal_update,
)
from src.method.bm_1_response import curve_pair_from_logits
from src.tools.bm_1_history_io import read_raw_power_window, read_raw_spot_window
from src.tools.bm_1_reports import _atomic_report, _read_array_checkpoint, _RunLock

_IMPORT_PHASE = None
if _BOOT_STARTED is not None:
    _IMPORT_PHASE = dict(phase='imports', started_at_utc=_BOOT_UTC,
                         seconds=perf_counter()-_BOOT_STARTED, status='complete')
    print(f'F4 imports complete in {_IMPORT_PHASE["seconds"]:.3f}s.', file=sys.stderr, flush=True)

SCHEMA = 'bm1-f4-runnable-v1'
OBSERVATION_CONTRACT = {
    'operator': 'Mean demand over [timestamp, interval_end), one elapsed hour.',
    'unit_status': 'Working assumption; consumption units and hour-start labels are not independently verified.',
    'observed_unit_assumed': 'kWh per hour; numerically equal to interval-average kW for this one-hour interval.',
    'prediction_unit_assumed': 'kW',
    'population_aggregation': 'Fixed arithmetic mean; one missing meter masks the entire hour.',
    'raw_negative_policy': 'Reject under the saved consumption contract; Gaussian Y itself is not restricted to nonnegative values. Net/gross metering is unverified.',
    'group_labels': 'Workbook installation metadata maps MED/UDEN to with/without electric heating; solar equipment is unknown.',
    'quantile_probabilities': [.05, .50, .95],
    'uncertainty': 'Marginal initial-state/process and nominal Gaussian observation uncertainty; no parameter/input uncertainty or coverage calibration.',
}


def _progress(event):
    detail = ''.join(f' {key}={event[key]}' for key in ('model', 'hour') if key in event)
    elapsed = f' in {event["seconds"]:.3f}s' if 'seconds' in event else ''
    print(f'F4 {event["at_utc"]}: {event["phase"]}{detail} {event["event"]}{elapsed}.',
          file=sys.stderr, flush=True)


@contextmanager
def _phase(name, records, progress=None, **details):
    started = perf_counter()
    record = dict(phase=name, started_at_utc=datetime.now(timezone.utc).isoformat(), **details)
    if progress is not None:
        progress(dict(record, at_utc=record['started_at_utc'], event='started'))
    status = 'complete'
    try:
        yield
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else 'failed'
        raise
    finally:
        record.update(seconds=perf_counter()-started, status=status)
        records.append(record)
        if progress is not None:
            progress(dict(record, at_utc=datetime.now(timezone.utc).isoformat(), event=status))


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _timestamp(value, timezone):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError('A valid timestamp is required.')
    try:
        result = (result.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
                  if result.tzinfo is None else result.tz_convert(timezone))
    except Exception as exc:
        raise ValueError(f'Invalid or ambiguous local timestamp {value!r}; supply an explicit UTC offset. {exc}') from exc
    if result.minute or result.second or result.microsecond or result.nanosecond:
        raise ValueError('Use whole-hour interval boundaries.')
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _execution_limits(max_seconds, workers):
    if isinstance(max_seconds, bool) or not np.isfinite(max_seconds) or max_seconds < 15:
        raise ValueError('max_seconds must be finite and at least 15.')
    _positive_integer(workers, 'workers')
    if workers > 8:
        raise ValueError('Use at most eight frequency workers.')


def _state(logs, cells):
    logs = np.asarray(logs)
    if (logs.shape != (cells,) or logs.dtype != np.float64 or np.isnan(logs).any()
            or np.isposinf(logs).any() or abs(float(logsumexp(logs))) > 1e-10):
        raise ValueError('Invalid normalized full log state in F4 checkpoint.')


def _path(config, key):
    path = Path(config[key]).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _consumption_values(frame, *, phase):
    """Keep the historical raw-data contract distinct from Gaussian Y support.

    Nonnegativity is checked per meter before aggregation so cancellation cannot
    silently admit an unclassified import/export channel. This is a conservative
    domain restriction, not a mathematical consequence of Gaussian likelihood.
    """
    raw = frame.to_numpy(dtype=float)
    invalid = np.isinf(raw) | (np.isfinite(raw) & (raw < 0))
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise ValueError('Consumption must be finite/nonnegative or explicitly missing. '
            f'First invalid {phase} cell: timestamp={frame.index[row].isoformat()}, '
            f'column={frame.columns[column]!r}, value={raw[row, column]!r}.')
    return raw


def fixed_population_mean(frame, *, phase):
    """One shared aggregation and missing-data rule for filter and evaluation."""
    raw = _consumption_values(frame, phase=phase)
    observed = np.isfinite(raw).all(axis=1)
    values = np.full(len(raw), np.nan)
    values[observed] = raw[observed].mean(axis=1)
    return values, observed


def model_population(model, requested_group=None):
    """Bind a saved fit to its exact named group; selecting a group is not refit."""
    columns = model['measurement_columns']
    if (not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)
            or any(not isinstance(c, str) or '_' not in c for c in columns)):
        raise ValueError('Saved model requires a nonempty unique measurement-column list.')
    groups = {column.rsplit('_', 1)[1] for column in columns}
    if len(groups) != 1 or not groups <= {'MED', 'UDEN'}:
        raise ValueError('A saved model must bind one separate MED or UDEN group.')
    group = groups.pop()
    if not model.get('population_id', '').startswith(group+'_'):
        raise ValueError('Saved population_id and measurement-column group disagree.')
    if requested_group is not None and requested_group != group:
        raise ValueError(f'Requested group {requested_group} differs from saved fit group {group}; provide a separately fitted model.')
    return dict(group=group, population_id=model['population_id'], measurement_columns=columns,
        aggregation=OBSERVATION_CONTRACT['population_aggregation'],
        group_meaning='electric heating' if group == 'MED' else 'without electric heating',
        solar_equipment='unknown')


def prepare(config, *, progress=None):
    """Freeze inputs before numerical work; forecast targets are read later."""
    phases = []
    if config.get('schema') != SCHEMA:
        raise ValueError('Unknown F4 configuration schema.')
    model_path = _path(config, 'model_file')
    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if model.get('schema') != 'bm1-f4-fixed-model-v1':
        raise ValueError('Unsupported fixed-model snapshot.')
    population = model_population(model, config.get('group'))
    # Validate the declared physical condition and prior before any propagation.
    physical = model['physical']
    for key in ('C', 'k', 'sigma_y'):
        if not np.isfinite(physical[key]) or physical[key] <= 0:
            raise ValueError(f'Invalid physical parameter {key}.')
    if not np.isfinite(physical['Xi']) or not 0 <= physical['Xi'] <= 1:
        raise ValueError('Relative flexibility Xi must lie in [0,1].')
    if physical['time_unit'] != 'day' or not np.isfinite(physical['sigma_x']) or physical['sigma_x'] < 0:
        raise ValueError('Physical parameters must use days with nonnegative process noise.')
    prior = model['prior']
    if not np.isfinite([prior['alpha'], prior['beta']]).all() or min(prior['alpha'], prior['beta']) <= 0:
        raise ValueError('Invalid declared Beta prior.')
    for candidate in model['candidates'].values():
        curve_pair_from_logits(candidate['logits'], candidate['pieces'])
    zone = model['timezone']
    start = _timestamp(config['filter_start'], zone)
    cutoff = _timestamp(config['forecast_origin'], zone)
    hours = _positive_integer(config['forecast_hours'], 'forecast_hours')
    if not start < cutoff or hours > 168:
        raise ValueError('Require filter_start < forecast_origin and 1..168 forecast hours.')
    if cutoff < _timestamp(model['curve_information_cutoff'], zone):
        raise ValueError('Forecast origin precedes the fixed curve fit/selection cutoff.')
    if start < _timestamp(model['baseline_information_cutoff'], zone):
        raise ValueError('Filtering starts before the fixed baseline/scale information cutoff.')
    if config['price_policy'] not in ('previous_week', 'supplied'):
        raise ValueError('price_policy must be previous_week or supplied.')
    grid = config['grid']
    if grid not in ('base', 'fine'):
        raise ValueError('grid must be base or fine; original numerical settings are preserved.')
    models = config['models']
    if (not isinstance(models, list) or not models or len(set(models)) != len(models)
            or any(name not in ('fitted', 'linear') for name in models)):
        raise ValueError('Select fitted, linear or both, once each.')
    horizon = cutoff + pd.Timedelta(hours=hours)
    index = pd.date_range(start, horizon, freq='h', inclusive='left')
    count = int((cutoff - start) / pd.Timedelta(hours=1))
    requested = config.get('quantile_hours', [1, 6, 12, 24, 48, 72, 168])
    if (not isinstance(requested, list) or len(set(requested)) != len(requested)
            or any(isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= 168 for h in requested)):
        raise ValueError('quantile_hours must contain unique integers in 1..168.')
    quantiles = sorted(h for h in requested if h <= hours)
    paths = {key: _path(config, key) for key in ('input_workbook', 'price_workbook')}
    with _phase('input_hash_before', phases, progress):
        hashes = {key: file_hash(path) for key, path in paths.items()}
    with _phase('demand_input', phases, progress):
        measurements = read_raw_power_window(paths['input_workbook'],
            columns=model['measurement_columns'], start=start, end=cutoff,
            timezone=zone, sheet=model['input_sheet'])
    values, observed = fixed_population_mean(measurements, phase='filter')
    prices_start = min(start, cutoff-pd.Timedelta(days=7)) if config['price_policy'] == 'previous_week' else start
    with _phase('price_input', phases, progress):
        prices = read_raw_spot_window(paths['price_workbook'], start=prices_start, end=cutoff,
            price_area=model['price_area'], sheet=model['price_sheet'])
    historical = prices.reindex(index[:count]).to_numpy(dtype=float)
    future_index = index[count:]
    if config['price_policy'] == 'previous_week':
        future = prices.reindex(future_index-pd.Timedelta(days=7)).to_numpy(dtype=float)
        price_source = 'Previous seven elapsed days, fixed at forecast origin; no future realized prices.'
    else:
        supplied_path = _path(config, 'future_prices_csv')
        hashes['future_prices_csv'] = file_hash(supplied_path)
        supplied = pd.read_csv(supplied_path)
        if list(supplied.columns) != ['timestamp', 'spot_price_DKK_MWh']:
            raise ValueError('Future price CSV requires timestamp,spot_price_DKK_MWh columns.')
        stamps = pd.DatetimeIndex([_timestamp(t, zone) for t in supplied['timestamp']])
        if not stamps.equals(future_index):
            raise ValueError('Supplied future prices must match every forecast hour, in order.')
        future = supplied['spot_price_DKK_MWh'].to_numpy(dtype=float)
        price_source = 'Caller-supplied scenario; availability is the caller\'s declared assumption.'
    spot = np.r_[historical, future]
    scales = model['scales']
    power = float(scales['power_kW'])
    low, high = scales['price_low_DKK_MWh'], scales['price_high_DKK_MWh']
    if not np.isfinite(power) or power <= 0 or not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('Invalid frozen power/price scales.')
    normalized_price = (spot-low)/(high-low)
    if not np.isfinite(normalized_price).all() or np.any((normalized_price < 0) | (normalized_price > 1)):
        raise ValueError('Prices are missing or outside the frozen [0,1] price scale. No fill, clipping or automatic retuning.')
    profile = np.asarray(model['baseline_weekday_hour_kW'], dtype=float)
    if profile.shape != (168,) or not np.isfinite(profile).all() or np.any((profile < 0) | (profile > power)):
        raise ValueError('Invalid fixed weekday/hour baseline.')
    baseline = profile[index.dayofweek*24+index.hour]
    n = model['numerics']
    settings = dict(cells=n[grid+'_cells'], max_step=n[grid+'_max_step'],
        frequencies=n['frequencies'], span=n['span'], centers=list(EXTENDED_CENTERS),
        relative_guard=n['relative_guard'], overlap_log_tolerance=n['overlap_log_tolerance'])
    with _phase('input_hash_after', phases, progress):
        if hashes != {key: file_hash(paths[key] if key in paths else supplied_path) for key in hashes}:
            raise ValueError('Input files changed during preparation.')
    edges = np.arange(len(index)+1, dtype=float)/24.
    spec = dict(settings=settings, physical=physical, input_edges=edges.tolist(),
        baseline=(baseline/power).tolist(), price=normalized_price.tolist(),
        horizon=[0., float(edges[-1])], time_unit='day', midnight_resets=0)
    with _phase('source_model_hash', phases, progress):
        if file_hash(model_path) != model_sha256:
            raise ValueError('Model file changed during preparation.')
        source = {path.relative_to(ROOT).as_posix(): file_hash(path)
                  for path in sorted((ROOT/'src').rglob('*.py'))}
        numerical_identity = dict(schema=SCHEMA, source_sha256=source,
            config={k:v for k,v in config.items() if k not in ('output_dir', 'workers', 'max_seconds')},
            model_sha256=model_sha256, input_sha256=hashes)
    identity = digest(numerical_identity)
    manifest = dict(numerical_identity, identity=identity, model=model,
        start=start.isoformat(), origin=cutoff.isoformat(), end=horizon.isoformat(),
        filter_hours=count, forecast_hours=hours, quantile_hours=quantiles,
        missing_filter_hours=int((~observed).sum()), price_source=price_source,
        observation_contract=OBSERVATION_CONTRACT, population=population,
        fit_binding=dict(**population, model_sha256=numerical_identity['model_sha256'],
            curve_information_cutoff=model['curve_information_cutoff'],
            baseline_information_cutoff=model['baseline_information_cutoff'],
            scales=model['scales'], baseline_sha256=digest(model['baseline_weekday_hour_kW']),
            training_period=model.get('training_period', 'Not embedded in this portable snapshot; retained packet provenance binds the original TRAIN split.'),
            provenance=model.get('provenance', {})),
        scientific_acceptance=False, initial_state_policy='Fresh declared Beta prior at filter_start; continuous assimilation, no daily reset.',
        includes_parameter_uncertainty=False, includes_input_uncertainty=False)
    return dict(config=config, manifest=manifest, model=model, paths=paths, preparation_phases=phases,
        spec=spec, index=index, filter_hours=count, observed=observed,
        values=values/power, baseline=baseline, prices=spot, quantiles=quantiles)


def _worker_init():
    import numba
    numba.set_num_threads(1)


class FrequencyPool:
    """Persistent spawn workers with bounded shutdown, including on errors."""
    def __init__(self, workers, deadline):
        self.workers, self.deadline, self.pool = workers, deadline, None

    def check(self):
        if perf_counter() >= self.deadline:
            raise TimeoutError('Time limit reached; resume from the last complete hour.')

    def mapper(self, jobs):
        self.check()
        if self.pool is None:
            self.pool = mp.get_context('spawn').Pool(self.workers, initializer=_worker_init)
        pending = self.pool.map_async(advance_frequency_block, jobs)
        while True:
            self.check()
            try:
                return pending.get(timeout=.25)
            except mp.TimeoutError:
                pass

    def close(self):
        if self.pool is not None:
            pool = self.pool
            workers = list(pool._pool)
            # Same bounded-close principle as the saved H2 recovery: worker
            # queue/finalizer shutdown must not strand an otherwise saved run.
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
            closer = threading.Thread(target=pool.terminate, daemon=True)
            closer.start()
            closer.join(timeout=2.)
            for worker in workers:
                worker.join(timeout=.25)
            if closer.is_alive():
                pool._terminate.cancel()
            self.pool = None


def _checkpoint(directory, saved):
    _atomic_report(directory/'checkpoint.npz', saved, replace_existing=True)


def _export(directory, data, saved, *, status, elapsed):
    """Write useful partial results too; NPZ remains the resume authority."""
    all_rows = [dict(model=name, **row) for name, stream in saved['streams'].items()
                for row in stream['rows']]
    frame = pd.DataFrame(all_rows)
    if len(frame):
        temporary = directory/'predictions.csv.tmp'
        frame.to_csv(temporary, index=False)
        temporary.replace(directory/'predictions.csv')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for name, group in frame.groupby('model', sort=False):
            t = pd.to_datetime(group['timestamp'], utc=True).dt.tz_convert(data['model']['timezone'])
            axes[0].plot(t, group['prediction_kW'], label=name)
            axes[1].plot(t, group['state_mean'], label=name)
        first = frame[frame['model'] == data['config']['models'][0]]
        t = pd.to_datetime(first['timestamp'], utc=True).dt.tz_convert(data['model']['timezone'])
        axes[0].plot(t, first['baseline_kW'], '--', label='baseline')
        axes[0].scatter(t, first['observed_kWh'], color='black', s=12, label='observed hourly consumption (assumed kWh)')
        for ax in axes:
            ax.axvline(pd.Timestamp(data['manifest']['origin']), color='grey', linestyle=':')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0].set_ylabel('Hourly average power [assumed kW]')
        axes[0].set_title('Unit and hour-start interpretation are working assumptions', fontsize=9)
        axes[1].set_ylabel('Endpoint state X'); axes[1].set_ylim(0, 1)
        fig.autofmt_xdate(); fig.tight_layout()
        temporary = directory/'comparison.png.tmp'
        fig.savefig(temporary, format='png', dpi=140); plt.close(fig)
        temporary.replace(directory/'comparison.png')
    metrics = {}
    for name in data['config']['models']:
        rows = [r for r in all_rows if r['model'] == name]
        metrics[name] = {}
        for phase in ('filter', 'forecast'):
            seen = [r for r in rows if r['phase'] == phase and r['observed_kWh'] is not None]
            if seen:
                error = np.array([r['prediction_kW']-r['observed_kWh'] for r in seen])
                baseline_error = np.array([r['baseline_kW']-r['observed_kWh'] for r in seen])
                metrics[name][phase] = dict(observed_hours=len(seen), rmse_kW=float(np.sqrt(np.mean(error**2))),
                    mse_kW2=float(np.mean(error**2)), mae_kW=float(np.mean(np.abs(error))),
                    bias_kW=float(np.mean(error)), baseline_rmse_kW=float(np.sqrt(np.mean(baseline_error**2))),
                    baseline_mse_kW2=float(np.mean(baseline_error**2)),
                    baseline_mae_kW=float(np.mean(np.abs(baseline_error))), baseline_bias_kW=float(np.mean(baseline_error)))
    summary = dict(schema=SCHEMA, identity=saved['identity'], status=status,
        elapsed_seconds=elapsed, completed_hours={k:len(v['rows']) for k,v in saved['streams'].items()},
        planned_hours_per_model=len(data['index']), metrics=metrics,
        missing_filter_hours=data['manifest']['missing_filter_hours'], scientific_acceptance=False,
        observation_contract=OBSERVATION_CONTRACT,
        metric_mask='Same finite fixed-population observations for model and baseline within each phase.',
        bias_definition='prediction minus observation',
        meaning='Fixed nominal physical parameters and shared fitted response. Development replay; no refit or population calibration.',
        forecast_assimilated_observations=0,
        forecast_targets_read=saved.get('evaluation_complete', False),
        output_sha256={name:file_hash(directory/name) for name in ('predictions.csv', 'comparison.png')
                       if (directory/name).is_file()})
    _atomic_report(directory/'summary.json', summary, replace_existing=True)
    return summary


def _origin(directory, data, saved):
    model = data['model']; n = data['spec']['settings']['cells']
    origins = {}
    for name, stream in saved['streams'].items():
        if stream['origin_log_state'] is None:
            continue
        f, g = curve_pair_from_logits(**model['candidates'][name])
        origins[name] = dict(log_state=stream['origin_log_state'],
            x_edges=np.linspace(0., 1., n+1), f=f.to_record(), g=g.to_record())
    if origins:
        _atomic_report(directory/'origin.npz', dict(schema='bm1-f4-origin-v1',
            identity=saved['identity'], origin_timestamp=data['manifest']['origin'],
            origin_time=data['filter_hours']/24., time_unit='day',
            physical=model['physical'], origins=origins), replace_existing=True)


def load_origin(directory, *, model='fitted'):
    """Actual filtered posterior bridge to the existing F3 control interface."""
    from src.control.bm_1 import BM1Parameters
    from src.control.bm_1_handoff import BM1ForecastOrigin
    from src.params.bm_1_response import BM1PiecewiseResponse
    directory = Path(directory)
    artifact = _read_array_checkpoint((directory/'origin.npz').read_bytes())
    saved = _read_array_checkpoint((directory/'checkpoint.npz').read_bytes())
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if artifact['schema'] != 'bm1-f4-origin-v1' or not artifact['identity'] == saved['identity'] == manifest['identity']:
        raise ValueError('Origin/manifest/checkpoint identities disagree.')
    item = artifact['origins'][model]; p = artifact['physical']
    _state(item['log_state'], len(item['x_edges'])-1)
    if not np.array_equal(item['log_state'], saved['streams'][model]['origin_log_state']):
        raise ValueError('Origin differs from its saved filtered posterior.')
    parameters = BM1Parameters(C=p['C'], k=p['k'], Xi=p['Xi'], sigma_x=p['sigma_x'],
        state_response=BM1PiecewiseResponse.from_record(item['f']),
        price_response=BM1PiecewiseResponse.from_record(item['g']))
    return BM1ForecastOrigin(artifact['origin_time'], 'day', parameters, p['sigma_y'],
        item['x_edges'], np.exp(item['log_state']))


def advance_hour(data, name, hour, logs, plan, pool):
    """Predict an interval, condition only on past Y, then form one output row.

    Predictions are prior to that hour's measurement; reported X is its ending
    posterior (or open predicted state). No future observation enters this step.
    """
    cells = data['spec']['settings']['cells']
    split = data['filter_hours']
    candidate = data['model']['candidates'][name]
    edges = data['spec']['input_edges']
    operator = freeze_native_operator(data['spec'], candidate,
        dict(start=edges[hour], end=edges[hour+1]))
    predicted = native_interval_moments(operator, logs, plan.sigma_y)
    forecast_hour = hour-split+1
    q = None
    if hour >= split and forecast_hour in data['quantiles']:
        full = native_interval_distribution(operator, logs, plan, mapper=pool.mapper)
        active = np.isfinite(predicted['log_state'])
        if (not np.array_equal(active, np.isfinite(full['log_state']))
                or np.max(np.abs(predicted['log_state'][active]-full['log_state'][active])) > 1e-7):
            raise ValueError('Moment/characteristic forecast state disagreement.')
        q = full['quantiles']
        predicted['log_state'] = full['log_state']
    seen = hour < split and bool(data['observed'][hour])
    updated = native_terminal_update(operator, logs, data['values'][hour],
        plan.sigma_y, data['spec']['settings'], mapper=pool.mapper) if seen else None
    next_logs = updated['log_posterior_mass'] if seen else predicted['log_state']
    _state(next_logs, cells)
    p = np.exp(next_logs); x = operator['x']
    power = data['model']['scales']['power_kW']
    row = dict(timestamp=data['index'][hour].isoformat(),
        timestamp_utc=data['index'][hour].tz_convert('UTC').isoformat(),
        interval_end=(data['index'][hour]+pd.Timedelta(hours=1)).isoformat(),
        interval_end_utc=(data['index'][hour]+pd.Timedelta(hours=1)).tz_convert('UTC').isoformat(),
        interval_hours=1.,
        phase='filter' if hour < split else 'forecast', horizon_hour=None if hour < split else forecast_hour,
        baseline_kW=float(data['baseline'][hour]), spot_price_DKK_MWh=float(data['prices'][hour]),
        prediction_kW=predicted['interval_mean']*power,
        demand_variance_kW2=predicted['interval_variance']*power**2,
        observation_variance_kW2=predicted['observation_variance']*power**2,
        observed_kWh=float(data['values'][hour]*power) if seen else None,
        assimilated=bool(seen), log_evidence=float(updated['log_evidence']) if seen else None,
        state_mean=float(p@x), state_variance=float(p@((x-p@x)**2)),
        maximum_abs_mass_error=max(predicted['maximum_abs_mass_error'],
            updated['maximum_abs_mass_error'] if seen else 0.),
        q05_kW=q[0]['value']*power if q else None,
        q50_kW=q[1]['value']*power if q else None,
        q95_kW=q[2]['value']*power if q else None)
    return next_logs, row


def run(data, *, directory, resume=False, max_seconds=600., workers=4, invocation_started=None, progress=None):
    begun = perf_counter()
    _execution_limits(max_seconds, workers)
    timing_start = begun if invocation_started is None else invocation_started
    phases = list(data.get('preparation_phases', [])) if invocation_started is not None else []
    directory = Path(directory)
    with _RunLock(directory):
        if (directory/'FAILED.json').exists():
            raise ValueError('This run has a preserved failure. Use a new output directory after correcting its cause.')
        identity = data['manifest']['identity']
        if (directory/'manifest.json').exists():
            old = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
            if not resume:
                raise FileExistsError('Output directory already contains a run. Use --resume or a new directory.')
            if old['identity'] != identity:
                raise ValueError('Config, model, inputs or source changed; cannot resume this run.')
            saved = _read_array_checkpoint((directory/'checkpoint.npz').read_bytes())
            if saved['identity'] != identity:
                raise ValueError('Checkpoint identity mismatch.')
        else:
            if resume:
                raise FileNotFoundError('No run exists to resume.')
            _atomic_report(directory/'manifest.json', data['manifest'])
            cells = data['spec']['settings']['cells']; prior = data['model']['prior']
            mass = np.diff(betainc(prior['alpha'], prior['beta'], np.linspace(0., 1., cells+1)))
            with np.errstate(divide='ignore'):
                logs = np.log(mass)
            _state(logs, cells)
            saved = dict(identity=identity, streams={name:dict(log_state=logs.copy(),
                origin_log_state=None, rows=[]) for name in data['config']['models']}, evaluation_complete=False)
            _checkpoint(directory, saved)
        cells = data['spec']['settings']['cells']; split = data['filter_hours']
        for stream in saved['streams'].values():
            _state(stream['log_state'], cells)
            if len(stream['rows']) > len(data['index']):
                raise ValueError('Checkpoint exceeds the declared period.')
        summary_path = directory/'summary.json'
        if resume and summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding='utf-8'))
            if previous.get('status') == 'complete':
                counts = {name:len(stream['rows']) for name, stream in saved['streams'].items()}
                if (previous.get('identity') != identity
                        or set(counts) != set(data['config']['models'])
                        or previous.get('completed_hours') != counts
                        or any(count != len(data['index']) for count in counts.values())
                        or (data['config'].get('evaluate_forecast', True) and not saved.get('evaluation_complete'))):
                    raise ValueError('Completed summary/checkpoint disagree; preserve this run and use a new directory.')
                for stream in saved['streams'].values():
                    _state(stream['origin_log_state'], cells)
                if any(not (directory/name).is_file() for name in ('predictions.csv', 'comparison.png', 'origin.npz')):
                    raise ValueError('Completed run has missing exports; restore its original package or use a new directory.')
                print('F4 already complete; saved results and timing receipts are unchanged.', flush=True)
                return 'complete'
        pool = FrequencyPool(workers, begun+max_seconds-5.)
        plan = interval_cdf_plan(data['model']['physical']['sigma_y'])
        initial_completed = sum(len(s['rows']) for s in saved['streams'].values())
        total = len(data['index'])*len(saved['streams'])
        status = 'complete'
        try:
            for name, stream in saved['streams'].items():
                for hour in range(len(stream['rows']), len(data['index'])):
                    pool.check()
                    if (directory/'STOP').exists():
                        raise TimeoutError('STOP file found; remove it before --resume.')
                    with _phase('filtering' if hour < split else 'forecast', phases, progress, model=name, hour=hour+1):
                        next_logs, row = advance_hour(data, name, hour, stream['log_state'], plan, pool)
                        stream['log_state'] = next_logs
                        stream['rows'].append(row)
                        if hour+1 == split:
                            stream['origin_log_state'] = next_logs.copy()
                        _checkpoint(directory, saved)
                    done = sum(len(s['rows']) for s in saved['streams'].values())
                    print(f'{name}: {hour+1}/{len(data["index"])} hours; '+runtime_text(
                        begun, done, total, initial_completed=initial_completed), flush=True)
            # Evaluation observations cannot affect any forecast or state.
            if data['config'].get('evaluate_forecast', True) and not saved['evaluation_complete']:
                with _phase('evaluation_input', phases, progress):
                    measured = read_raw_power_window(data['paths']['input_workbook'],
                        columns=data['model']['measurement_columns'], start=pd.Timestamp(data['manifest']['origin']),
                        end=pd.Timestamp(data['manifest']['end']), timezone=data['model']['timezone'],
                        sheet=data['model']['input_sheet'])
                    means, _ = fixed_population_mean(measured, phase='forecast evaluation')
                    for stream in saved['streams'].values():
                        for row, value in zip(stream['rows'][split:], means):
                            row['observed_kWh'] = float(value) if np.isfinite(value) else None
                    saved['evaluation_complete'] = True
                    _checkpoint(directory, saved)
        except (TimeoutError, KeyboardInterrupt) as exc:
            status = 'paused'
            print(str(exc) or 'Interrupted after the last saved hour.', flush=True)
        except Exception as exc:
            status = 'failed'
            _atomic_report(directory/'FAILED.json', dict(error=type(exc).__name__, message=str(exc)))
            raise
        finally:
            with _phase('worker_shutdown', phases, progress):
                pool.close()
            with _phase('export', phases, progress):
                _origin(directory, data, saved)
                summary = _export(directory, data, saved, status=status, elapsed=perf_counter()-begun)
            # Measure through finished CSV/figure/origin export, then persist the
            # timing receipt. Abruptly killed invocations cannot report their time.
            elapsed = perf_counter()-timing_start
            timings = saved.setdefault('invocation_timings', [])
            timings.append(dict(seconds=elapsed, includes_prepare=invocation_started is not None,
                                status=status, phases=phases))
            _checkpoint(directory, saved)
            summary['timing'] = dict(last_invocation_seconds=elapsed,
                recorded_invocations=len(timings),
                cumulative_recorded_seconds=sum(t['seconds'] for t in timings),
                includes_prepare_all_invocations=all(t['includes_prepare'] for t in timings),
                last_invocation_phases=phases,
                import_phase=data.get('import_phase') if invocation_started is not None else None,
                phase_scope='Elapsed phase spans; filtering/forecast include first-use JIT compilation and worker startup where incurred. Per-hour spans include checkpoint writes. Gaps are validation/orchestration. Import time is separate from invocation totals.',
                scope='CLI main (or direct run call) through completed result export; excludes interpreter/module imports and final timing writes. Killed invocations without a timing receipt are not included.',
                legacy_elapsed_seconds_scope='Current run call through origin export, before CSV/figure export.')
            _atomic_report(directory/'summary.json', summary, replace_existing=True)
        return status


def main(argv=None):
    invocation_started = perf_counter()
    parser = argparse.ArgumentParser(description='Reuse the saved F4 curve fit, filter from a fresh Beta prior, and forecast hourly averages. No parameter refit.')
    parser.add_argument('--read-result', type=Path, help='Read saved summary and verify available output hashes; no new forecast or input loading.')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bm/f4_quick.json')
    parser.add_argument('--filter-start', help='First assimilated hour (local time or explicit UTC offset); overrides config.')
    parser.add_argument('--forecast-origin', help='First forecast hour, after the saved fit cutoff; overrides config.')
    parser.add_argument('--forecast-hours', type=int, help='Number of elapsed forecast hours (1..168); overrides config.')
    parser.add_argument('--model', nargs='+', choices=('fitted', 'linear'), help='Saved response model(s); overrides config models.')
    parser.add_argument('--group', choices=('MED', 'UDEN'), help='Must match the saved model population; does not refit.')
    parser.add_argument('--grid', choices=('base', 'fine'), help='Original numerical grid; overrides config.')
    parser.add_argument('--output', type=Path, help='New output directory, relative to repository root or absolute.')
    parser.add_argument('--resume', action='store_true', help='Continue the exact same saved run.')
    parser.add_argument('--check', action='store_true', help='Validate inputs and show planned work; no numerical propagation.')
    parser.add_argument('--max-seconds', type=float, help='Pause between complete hours after this invocation time limit.')
    parser.add_argument('--workers', type=int, help='Frequency workers (1..8).')
    args = parser.parse_args(argv)
    if args.read_result is not None:
        print(json.dumps(read_result(args.read_result), indent=2))
        return 0
    input_started = perf_counter()
    print('F4: validating input/workbook metadata; no model propagation.', file=sys.stderr, flush=True)
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        if not isinstance(config, dict):
            raise ValueError('The configuration must be a JSON object.')
        for key in ('filter_start', 'forecast_origin', 'forecast_hours', 'grid', 'group'):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if args.model is not None:
            config['models'] = args.model
        max_seconds = args.max_seconds if args.max_seconds is not None else config.get('max_seconds', 600)
        workers = args.workers if args.workers is not None else config.get('workers', 4)
        _execution_limits(max_seconds, workers)
        data = prepare(config, progress=_progress)
        data['import_phase'] = _IMPORT_PHASE
    except KeyboardInterrupt:
        print(f'F4 input control stopped after {perf_counter()-input_started:.3f}s; no run created.',
              file=sys.stderr, flush=True)
        return 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(f'F4 input check failed: {exc}')
    print(f'F4 input control complete in {perf_counter()-input_started:.3f}s; no model propagation yet.',
          file=sys.stderr, flush=True)
    print(json.dumps(dict(filter_hours=data['filter_hours'], forecast_hours=config['forecast_hours'],
        models=config['models'], grid=config['grid'], missing_filter_hours=data['manifest']['missing_filter_hours'],
        filter_start=data['manifest']['start'], forecast_origin=data['manifest']['origin'],
        forecast_end_exclusive=data['manifest']['end'],
        population=data['manifest']['population'],
        fit_policy='Reused fixed fit; no parameter estimation.',
        curve_information_cutoff=data['model']['curve_information_cutoff'],
        observation_contract=OBSERVATION_CONTRACT,
        identity=data['manifest']['identity'],
        import_phase=_IMPORT_PHASE, preparation_phases=data['preparation_phases']), indent=2), flush=True)
    if args.check:
        return 0
    directory = args.output or _path(config, 'output_dir')
    if not directory.is_absolute():
        directory = ROOT/directory
    status = run(data, directory=directory, resume=args.resume,
        max_seconds=max_seconds, workers=workers, invocation_started=invocation_started, progress=_progress)
    print(f'F4 {status}: {directory.resolve()}', flush=True)
    return 0 if status == 'complete' else 2


if __name__ == '__main__':
    mp.freeze_support()
    result = main()
    # All reports/checkpoints are already closed. Avoid unbounded pool
    # finalizers after a timeout on platforms with multiprocessing queue hangs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)

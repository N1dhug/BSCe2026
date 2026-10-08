"""Portable F4 development runner using the existing native interval kernel.

Fixed shared curves, hourly energy observations and one open forecast. This
does not continue the historical research campaign or change its acceptance.
Run from the repository root: python -m src.exe.run_model.bm_1_f4 --help
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
from time import perf_counter

# Applied before NumPy import, including Windows spawn workers.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
from scipy.special import betainc, logsumexp

from src.exe.run_model.progress import runtime_text
from src.method.bm_1_interval_contour_extension import EXTENDED_CENTERS
from src.method.bm_1_interval_forecast import (
    interval_cdf_plan, native_interval_distribution, native_interval_moments,
)
from src.method.bm_1_interval_native import (
    advance_frequency_block, freeze_native_operator, native_terminal_update,
)
from src.method.bm_1_response import curve_pair_from_logits
from src.tools.bm_1_history_io import read_raw_power_window, read_raw_spot_window
from src.tools.bm_1_reports import _atomic_report, _read_array_checkpoint, _RunLock

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'bm1-f4-runnable-v1'


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
    result = (result.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
              if result.tzinfo is None else result.tz_convert(timezone))
    if result.minute or result.second or result.microsecond or result.nanosecond:
        raise ValueError('Use whole-hour interval boundaries.')
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _state(logs, cells):
    logs = np.asarray(logs)
    if (logs.shape != (cells,) or logs.dtype != np.float64 or np.isnan(logs).any()
            or np.isposinf(logs).any() or abs(float(logsumexp(logs))) > 1e-10):
        raise ValueError('Invalid normalized full log state in F4 checkpoint.')


def _path(config, key):
    path = Path(config[key]).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def prepare(config):
    """Freeze inputs before numerical work; forecast targets are read later."""
    if config.get('schema') != SCHEMA:
        raise ValueError('Unknown F4 configuration schema.')
    model_path = _path(config, 'model_file')
    model = json.loads(model_path.read_text(encoding='utf-8'))
    if model.get('schema') != 'bm1-f4-fixed-model-v1':
        raise ValueError('Unsupported fixed-model snapshot.')
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
    hashes = {key: file_hash(path) for key, path in paths.items()}
    measurements = read_raw_power_window(paths['input_workbook'],
        columns=model['measurement_columns'], start=start, end=cutoff,
        timezone=zone, sheet=model['input_sheet'])
    raw = measurements.to_numpy(dtype=float)
    if np.isinf(raw).any() or np.any(raw[np.isfinite(raw)] < 0):
        raise ValueError('Consumption must be finite/nonnegative or explicitly missing.')
    # Fixed-population mean: any missing house masks the entire hour.
    observed = np.isfinite(raw).all(axis=1)
    values = np.full(count, np.nan)
    values[observed] = np.mean(raw[observed], axis=1)
    prices_start = min(start, cutoff-pd.Timedelta(days=7)) if config['price_policy'] == 'previous_week' else start
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
    # Validate the declared physical condition and prior before any propagation.
    physical = model['physical']
    for key in ('C', 'k', 'Xi', 'sigma_y'):
        if not np.isfinite(physical[key]) or physical[key] <= 0:
            raise ValueError(f'Invalid physical parameter {key}.')
    if physical['time_unit'] != 'day' or not np.isfinite(physical['sigma_x']) or physical['sigma_x'] < 0:
        raise ValueError('Physical parameters must use days with nonnegative process noise.')
    prior = model['prior']
    if not np.isfinite([prior['alpha'], prior['beta']]).all() or min(prior['alpha'], prior['beta']) <= 0:
        raise ValueError('Invalid declared Beta prior.')
    for candidate in model['candidates'].values():
        curve_pair_from_logits(candidate['logits'], candidate['pieces'])
    if hashes != {key: file_hash(paths[key] if key in paths else supplied_path) for key in hashes}:
        raise ValueError('Input files changed during preparation.')
    edges = np.arange(len(index)+1, dtype=float)/24.
    spec = dict(settings=settings, physical=physical, input_edges=edges.tolist(),
        baseline=(baseline/power).tolist(), price=normalized_price.tolist(),
        horizon=[0., float(edges[-1])], time_unit='day', midnight_resets=0)
    source = {path.relative_to(ROOT).as_posix(): file_hash(path)
              for path in sorted((ROOT/'src').rglob('*.py'))}
    numerical_identity = dict(schema=SCHEMA, source_sha256=source,
        config={k:v for k,v in config.items() if k not in ('output_dir', 'workers', 'max_seconds')},
        model_sha256=file_hash(model_path), input_sha256=hashes)
    identity = digest(numerical_identity)
    manifest = dict(numerical_identity, identity=identity, model=model,
        start=start.isoformat(), origin=cutoff.isoformat(), end=horizon.isoformat(),
        filter_hours=count, forecast_hours=hours, quantile_hours=quantiles,
        missing_filter_hours=int((~observed).sum()), price_source=price_source,
        scientific_acceptance=False, initial_state_policy='Fresh declared Beta prior at filter_start; continuous assimilation, no daily reset.',
        includes_parameter_uncertainty=False, includes_input_uncertainty=False)
    return dict(config=config, manifest=manifest, model=model, paths=paths,
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
        frame.to_csv(directory/'predictions.csv', index=False)
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
        axes[0].scatter(t, first['observed_kWh'], color='black', s=12, label='observed hourly kWh (= average kW over 1h)')
        for ax in axes:
            ax.axvline(pd.Timestamp(data['manifest']['origin']), color='grey', linestyle=':')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0].set_ylabel('Hourly average power [kW]')
        axes[1].set_ylabel('Endpoint state X'); axes[1].set_ylim(0, 1)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(directory/'comparison.png', dpi=140); plt.close(fig)
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
                    mae_kW=float(np.mean(np.abs(error))), baseline_rmse_kW=float(np.sqrt(np.mean(baseline_error**2))))
    summary = dict(schema=SCHEMA, identity=saved['identity'], status=status,
        elapsed_seconds=elapsed, completed_hours={k:len(v['rows']) for k,v in saved['streams'].items()},
        planned_hours_per_model=len(data['index']), metrics=metrics,
        missing_filter_hours=data['manifest']['missing_filter_hours'], scientific_acceptance=False,
        meaning='Fixed nominal physical parameters and shared fitted response. Development replay; no refit or population calibration.',
        forecast_assimilated_observations=0,
        forecast_targets_read=saved.get('evaluation_complete', False))
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


def run(data, *, directory, resume=False, max_seconds=600., workers=4):
    begun = perf_counter()
    if not np.isfinite(max_seconds) or max_seconds < 15:
        raise ValueError('max_seconds must be finite and at least 15.')
    _positive_integer(workers, 'workers')
    if workers > 8:
        raise ValueError('Use at most eight frequency workers.')
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
        pool = FrequencyPool(workers, begun+max_seconds-5.)
        plan = interval_cdf_plan(data['model']['physical']['sigma_y'])
        initial_completed = sum(len(s['rows']) for s in saved['streams'].values())
        total = len(data['index'])*len(saved['streams'])
        status = 'complete'
        try:
            for name, stream in saved['streams'].items():
                candidate = data['model']['candidates'][name]
                for hour in range(len(stream['rows']), len(data['index'])):
                    pool.check()
                    if (directory/'STOP').exists():
                        raise TimeoutError('STOP file found; remove it before --resume.')
                    logs = stream['log_state']
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
                measured = read_raw_power_window(data['paths']['input_workbook'],
                    columns=data['model']['measurement_columns'], start=pd.Timestamp(data['manifest']['origin']),
                    end=pd.Timestamp(data['manifest']['end']), timezone=data['model']['timezone'],
                    sheet=data['model']['input_sheet']).to_numpy(dtype=float)
                if np.isinf(measured).any() or np.any(measured[np.isfinite(measured)] < 0):
                    raise ValueError('Invalid forecast evaluation observations.')
                seen = np.isfinite(measured).all(axis=1)
                means = np.full(len(measured), np.nan)
                means[seen] = measured[seen].mean(axis=1)
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
            pool.close()
            _origin(directory, data, saved)
            _export(directory, data, saved, status=status, elapsed=perf_counter()-begun)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run fixed-curve F4 filtering and an open hourly forecast.')
    parser.add_argument('--config', type=Path, default=ROOT/'root/f4_quick.json')
    parser.add_argument('--output', type=Path, help='New output directory, relative to repository root or absolute.')
    parser.add_argument('--resume', action='store_true', help='Continue the exact same saved run.')
    parser.add_argument('--check', action='store_true', help='Validate inputs and show planned work; no numerical propagation.')
    parser.add_argument('--max-seconds', type=float, help='Pause between complete hours after this invocation time limit.')
    parser.add_argument('--workers', type=int, help='Frequency workers (1..8).')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    data = prepare(config)
    print(json.dumps(dict(filter_hours=data['filter_hours'], forecast_hours=config['forecast_hours'],
        models=config['models'], grid=config['grid'], missing_filter_hours=data['manifest']['missing_filter_hours'],
        identity=data['manifest']['identity']), indent=2), flush=True)
    if args.check:
        return 0
    directory = args.output or _path(config, 'output_dir')
    if not directory.is_absolute():
        directory = ROOT/directory
    status = run(data, directory=directory, resume=args.resume,
        max_seconds=args.max_seconds if args.max_seconds is not None else config.get('max_seconds', 600),
        workers=args.workers if args.workers is not None else config.get('workers', 4))
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

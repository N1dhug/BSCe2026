"""Forecast G and the WM hourly adapter from a saved weekday export, without fit.

The seven daily states stay frozen at the snapshot's Monday cutoff. --start
and --hours select output inside that week; propagation still checks the full
week. Empirical residual bands describe development errors, not a 95% guarantee.
For declared chronological band calibration use: bm_1_g bands --help.
"""
from argparse import ArgumentParser
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from src.method import g_weekday_export as model
from src.tools import weekday_backtest as calendar
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(snapshot, inputs, start=None, hours=None):
    """Validate the saved fit, physical input week and requested output window."""
    snapshot = model.validate_export(snapshot)
    window = calendar.WeekWindow(snapshot['parent']['origin'])
    calendar._frame(inputs, ('Price', 'Temp'))
    if inputs.index.tz is not None:
        raise ValueError('G requires unique local-naive Copenhagen hour-start labels.')
    scenario = calendar._grid(inputs[['Price', 'Temp']], window.hours(window.origin, window.end))
    scenario = scenario.apply(pd.to_numeric, errors='raise')
    if np.iscomplexobj(scenario.to_numpy()) or not np.isfinite(scenario.to_numpy(float)).all():
        raise ValueError('Supply complete finite Price/Temp for the saved forecast week; no filling.')
    scale = snapshot['parent']['scales']
    if not scenario.Price.between(scale['price_min'], scale['price_min']+scale['price_scale']).all():
        raise ValueError('Prices are outside the saved declared price domain; no clipping.')
    begin = window.origin if start is None else pd.Timestamp(start)
    if pd.isna(begin) or begin.tzinfo is not None or begin != begin.floor('h') or begin not in scenario.index:
        raise ValueError('--start must be an existing hour within the saved snapshot week; a new origin requires a new fit.')
    offset = scenario.index.get_loc(begin)
    count = len(scenario)-offset if hours is None else hours
    if type(count) is not int or count < 1 or offset+count > len(scenario):
        raise ValueError('--hours must be positive and stay inside the saved snapshot week.')
    return snapshot, scenario, scenario.index[offset:offset+count]


def select_output(result, scenario, selected):
    result = deepcopy(result)
    cutoff = pd.Timestamp(result['manifest']['cutoff']).tz_localize('Europe/Copenhagen').tz_convert('UTC')
    for key in ('predictions', 'normalized_predictions'):
        table = result[key].loc[selected].copy()
        physical = selected.tz_localize('Europe/Copenhagen').tz_convert('UTC')
        table['temperature'] = scenario.loc[selected, 'Temp'].to_numpy()
        table['forecast_horizon_hours'] = (physical-cutoff).total_seconds()/3600
        result[key] = table
    end = (selected[-1].tz_localize('Europe/Copenhagen')+pd.Timedelta(hours=1)).tz_localize(None)
    result['manifest'].update(output_start=str(selected[0]), end_exclusive=str(end),
        physical_hours=len(selected), propagation='full saved week; selected output only',
        scenario='caller-supplied Price/Temp; information availability at origin not independently verified',
        wm_reference='WM4-equation hourly adapter with its own fit; not original WM4 CLI',
        interval_definition='one physical hour [interval_start_utc, interval_end_utc)',
        runner_source_sha256=digest(__file__))
    return result


def plot(directory, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = result['predictions']
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for name, label in (('g_prediction', 'G'), ('wm_prediction', 'WM hourly adapter')):
        line, = axes[0].plot(table.index, table[name+'_output_mean'], label=label+' empirical mean')
        axes[0].fill_between(table.index, table[name+'_lower'], table[name+'_upper'],
                             alpha=.12, color=line.get_color())
    axes[0].plot(table.index, table.baseline, label='Baseline', linestyle='--')
    if 'observed' in table:
        axes[0].scatter(table.index, table.observed, s=7, color='black', label='Observed')
    axes[0].set_ylabel('Hourly demand\n(raw unit unverified)'); axes[0].legend()
    axes[1].step(table.index, table.price, where='post', label='Supplied price')
    axes[1].set_ylabel('Supplied raw price'); axes[1].set_xlabel('Europe/Copenhagen interval start')
    figure.suptitle('Frozen weekday fit; conditional inputs; empirical bands are uncalibrated')
    figure.tight_layout(); figure.savefig(directory/'comparison.png', dpi=130); plt.close(figure)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == 'bands':
        from src.method import bm_1_bands
        return bm_1_bands.main(argv[1:])
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, help='Saved weekday_export_snapshot.json; source hashes must match.')
    parser.add_argument('--inputs-csv', required=True, help='Datetime,Price,Temp for the full saved week; Demand ignored.')
    parser.add_argument('--start', help='Output start inside the saved week; default snapshot origin.')
    parser.add_argument('--hours', type=int, help='Physical output hours; default remainder of saved week.')
    parser.add_argument('--output', help='New output directory; completed runs are never overwritten.')
    parser.add_argument('--check', action='store_true', help='Validate snapshot, dates and inputs without forecast or output.')
    parser.add_argument('--score', metavar='OBSERVATIONS_CSV', help='Datetime,Demand; opened only after forecast is frozen.')
    args = parser.parse_args(argv)
    if not args.check and args.output is None:
        parser.error('--output is required unless --check is used.')
    begun = perf_counter(); directory = None
    try:
        snapshot = json.loads(Path(args.snapshot).read_text(encoding='utf-8'))
        inputs = pd.read_csv(args.inputs_csv, usecols=['Datetime', 'Price', 'Temp'],
                             index_col='Datetime', parse_dates=['Datetime'], float_precision='round_trip')
        snapshot, scenario, selected = prepare(snapshot, inputs, args.start, args.hours)
        origin = snapshot['parent']['origin']
        print(f'G cutoff={origin}; selected {selected[0]} through {selected[-1]} ({len(selected)} physical hours).', flush=True)
        print('Saved weekday fit; no optimizer or historical replay. WM is the separate hourly adapter.', flush=True)
        if args.check:
            print(f'Validated; no model executed ({perf_counter()-begun:.3f} s).', flush=True)
            return 0
        target = Path(args.output); target.mkdir(parents=True, exist_ok=False); directory = target
        _atomic_report(directory/'invocation.json', {'status':'running', 'snapshot':str(args.snapshot),
            'snapshot_sha256':digest(args.snapshot), 'inputs_sha256':digest(args.inputs_csv),
            'runner_source_sha256':digest(__file__), 'fit_cutoff':str(origin),
            'start':str(selected[0]), 'hours':len(selected), 'fit_executed':False, 'historical_replay_executed':False})
        _atomic_report(directory/'saved_snapshot.json', snapshot)
        result = select_output(model.predict_comparison(snapshot, scenario), scenario, selected)
        write_forecast(directory, result)
        files = ('invocation.json', 'saved_snapshot.json', 'forecast.json', 'forecast.csv', 'forecast_normalized.csv')
        _atomic_report(directory/'forecast_freeze.json', {'status':'frozen-input-only',
            'origin':str(pd.Timestamp(origin).date()), 'target_observation_file_opened':False,
            'artifact_sha256':{name:digest(directory/name) for name in files}})
        if args.score:
            from src.method.bm_1_bands import observations_for_forecast
            observations = pd.read_csv(args.score, index_col='Datetime', parse_dates=['Datetime'],
                                       float_precision='round_trip')['Demand']
            observations = observations_for_forecast(observations, result)
            result = model.score_comparison(result, observations)
            write_forecast(directory, result, prefix='scored')
        plot(directory, result)
        elapsed = perf_counter()-begun
        _atomic_report(directory/'summary.json', {'status':'complete', 'seconds_including_load_forecast_export_plot':elapsed,
            'forecast_seconds':result['seconds'], 'physical_hours':len(selected), 'fit_executed':False,
            'historical_replay_executed':False, 'scoring':result['scoring'],
            'cold_import_seconds_included':False, 'uncertainty':'uncalibrated empirical output residuals; no process/parameter guarantee'})
        print(f'Complete: {len(selected)} hours, {elapsed:.3f} s after imports; {directory}', flush=True)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, TypeError) as exc:
        if directory is not None:
            _atomic_report(directory/'failure.json', {'status':'failed', 'error':str(exc), 'seconds':perf_counter()-begun})
        print(f'G forecast failed: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

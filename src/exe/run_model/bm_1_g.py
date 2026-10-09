"""G: export-model once, forecast a new week, or show an existing result.

`forecast` uses frozen weekday parameters and seven separate daily states.
The legacy --snapshot form (also `replay`) stays inside the saved origin week
and includes the WM hourly adapter. `bands` retains the existing comparison
protocol. Empirical residual bands do not guarantee 95% future coverage.
New fits use the separate weekday_backtest entry point and explicit budgets.
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

from src.tools import weekday_backtest as calendar
from src.tools.weekday_backtest import _verify_artifacts as _verify_files
from src.tools.bm_1_reports import _atomic_report


def _legacy():
    from src.method import g_weekday_export
    return g_weekday_export


def write_forecast(directory, result, *, prefix='forecast'):
    """Atomic files owned by this fresh run; no historical study CLI import."""
    from src.tools.checkpoint_io import replace_file
    for key, suffix in (('predictions', ''), ('normalized_predictions', '_normalized')):
        path = directory/f'{prefix}{suffix}.csv'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}.')
        temporary = path.with_suffix('.csv.partial')
        result[key].to_csv(temporary, index_label='Datetime')
        replace_file(temporary, path)
    _atomic_report(directory/f'{prefix}.json',
                   {k:v for k,v in result.items() if k not in ('predictions', 'normalized_predictions')})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(snapshot, inputs, start=None, hours=None):
    """Validate the saved fit, physical input week and requested output window."""
    snapshot = _legacy().validate_export(snapshot)
    window = calendar.WeekWindow(snapshot['parent']['origin'])
    from src.method.bm_1_g_forecast import prepare_inputs
    scenario = prepare_inputs(inputs,window,snapshot['parent']['scales'])
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
        if name not in table:
            continue
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
    if argv and argv[0] in ('export-model', 'forecast', 'show'):
        return deployment_main(argv)
    if argv and argv[0] == 'replay':
        argv = argv[1:]
    if argv and argv[0] == 'bands':
        return bands_main(argv[1:])
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
        result = select_output(_legacy().predict_comparison(snapshot, scenario), scenario, selected)
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
            result = _legacy().score_comparison(result, observations)
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


def load_model(path, group=None):
    from src.method import bm_1_g_forecast as deployment
    path = Path(path)
    if path.name!='model.json':
        raise ValueError('Use model.json beside its conversion_freeze.json receipt.')
    freeze = json.loads((path.parent/'conversion_freeze.json').read_text(encoding='utf-8'))
    if freeze.get('fit_executed') is not False or freeze.get('historical_replay_executed') is not False:
        raise ValueError('Require an explicit no-fit/no-replay conversion receipt.')
    if not {'model.json', 'original_snapshot.json'} <= set(freeze['artifact_sha256']):
        raise ValueError('Conversion must preserve and bind the original snapshot bytes.')
    _verify_files(path.parent, freeze['artifact_sha256'])
    model = deployment.validate_model(json.loads(path.read_text(encoding='utf-8')), group)
    original = json.loads((path.parent/'original_snapshot.json').read_text(encoding='utf-8'))
    if deployment.fingerprint(original)!=model['original_snapshot_sha256']:
        raise ValueError('Model and saved original snapshot disagree.')
    binding = model.get('fit_population_binding')
    if binding is not None or 'fit_population_binding.json' in freeze['artifact_sha256']:
        if ('fit_population_binding.json' not in freeze['artifact_sha256']
                or json.loads((path.parent/'fit_population_binding.json').read_text(encoding='utf-8'))!=binding):
            raise ValueError('Model package is missing its unchanged fit population binding.')
    if 'fit_population_freeze' in model:
        hashes = model['fit_population_freeze']['artifact_sha256']
        _verify_files(path.parent,{'original_snapshot.json':hashes['weekday_export_snapshot.json'],
                                 'fit_population_binding.json':hashes['fit_population_binding.json']})
    return model


def deployment_main(argv):
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export-model', help='Validate and convert a saved G export; no fit or replay.')
    export.add_argument('--snapshot', required=True)
    export.add_argument('--population-binding', help='Default: fit_population_binding.json beside the snapshot; checked against its fit.')
    export.add_argument('--legacy-unverified', action='store_true', help='Explicitly export an unmarked historical fit without population information; cannot bypass missing/invalid declared bindings.')
    export.add_argument('--output', required=True, help='New directory containing model.json and its conversion receipt.')
    forecast = commands.add_parser('forecast', help='Compute a new G-only week from frozen model parameters.')
    forecast.add_argument('--model', required=True, help='model.json from export-model; sources/runtime must match.')
    forecast.add_argument('--origin', required=True, help='Monday midnight at or after fit cutoff; may be a later week.')
    forecast.add_argument('--inputs-csv', required=True, help='Datetime,Price,Temp for that full week; Demand never decoded.')
    forecast.add_argument('--group', choices=('MED', 'UDEN'), help='Assert saved group; cannot change or verify meter membership.')
    forecast.add_argument('--hours', type=int, help='Physical prefix hours to export; default full calendar week.')
    forecast.add_argument('--output', help='New output directory; completed runs are never overwritten.')
    forecast.add_argument('--check', action='store_true', help='Validate group/cutoff/model/inputs; no solver or output.')
    forecast.add_argument('--score', help='Datetime,Demand observations opened only after durable forecast freeze.')
    show = commands.add_parser('show', help='Read and verify an existing forecast result; no model execution.')
    show.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    if args.command=='forecast' and not args.check and args.output is None:
        parser.error('--output is required unless --check is used.')
    begun = perf_counter(); directory = None
    try:
        if args.command=='show':
            target = Path(args.result)
            receipt = json.loads((target/'forecast_freeze.json').read_text(encoding='utf-8'))
            if receipt.get('status')!='frozen-input-only' or receipt.get('target_observation_file_opened') is not False:
                raise ValueError('Require an input-only frozen forecast receipt.')
            if not {'forecast.json', 'forecast.csv', 'forecast_normalized.csv'} <= set(receipt['artifact_sha256']):
                raise ValueError('Receipt must bind the forecast manifest and both tables.')
            _verify_files(target, receipt['artifact_sha256'])
            print((target/'forecast.json').read_text(encoding='utf-8'), flush=True)
            print('Verified saved result; no forecast, replay or fit executed.', flush=True)
            return 0
        from src.method import bm_1_g_forecast as deployment
        if args.command=='export-model':
            original = Path(args.snapshot).read_bytes()
            snapshot = json.loads(original)
            binding, binding_bytes, _ = deployment.load_fit_binding(args.snapshot,snapshot,args.population_binding,legacy=args.legacy_unverified)
            fit_directory = Path(args.population_binding).parent if args.population_binding else Path(args.snapshot).parent
            record = deployment.export_model(snapshot,binding,legacy=args.legacy_unverified,fit_directory=fit_directory)
            target = Path(args.output); target.mkdir(parents=True, exist_ok=False); directory = target
            (directory/'original_snapshot.json').write_bytes(original)
            files = ['model.json','original_snapshot.json']
            if binding_bytes is not None:
                (directory/'fit_population_binding.json').write_bytes(binding_bytes)
                files.append('fit_population_binding.json')
            _atomic_report(directory/'model.json', record)
            _atomic_report(directory/'conversion_freeze.json', {'status':'complete-validated-conversion',
                'fit_executed':False, 'historical_replay_executed':False,
                'original_snapshot_sha256':record['original_snapshot_sha256'],
                'artifact_sha256':{name:digest(directory/name) for name in files}})
            print(f'Model exported: {directory / "model.json"}; group={record["population"]}; no fit/replay.', flush=True)
            return 0
        # Group and temporal metadata fail before the exogenous CSV is opened.
        model_sha256 = digest(args.model)
        record = load_model(args.model, args.group)
        origin = calendar.WeekWindow(args.origin).origin
        if origin<pd.Timestamp(record['fit_cutoff']):
            raise ValueError('Forecast origin precedes the saved fit cutoff.')
        inputs_sha256 = digest(args.inputs_csv)
        inputs = pd.read_csv(args.inputs_csv, usecols=['Datetime', 'Price', 'Temp'],
            index_col='Datetime', parse_dates=['Datetime'], float_precision='round_trip')
        if inputs_sha256!=digest(args.inputs_csv) or model_sha256!=digest(args.model):
            raise ValueError('Model or input file changed while loading.')
        scenario, selected = deployment.prepare(record, inputs, origin, group=args.group, hours=args.hours)
        print(f'G fit cutoff={record["fit_cutoff"]}; new forecast origin={origin}; {len(selected)} output hours.', flush=True)
        print(f'Group={record["population"]["group"]}, membership={record["population"]["verification"]}; seven separate daily paths.', flush=True)
        if args.check:
            print('Validated; no model executed.', flush=True)
            return 0
        target = Path(args.output); target.mkdir(parents=True, exist_ok=False); directory = target
        _atomic_report(directory/'invocation.json', {'status':'running', 'model_sha256':model_sha256,
            'inputs_sha256':inputs_sha256, 'runner_source_sha256':digest(__file__),
            'fit_executed':False, 'historical_replay_executed':False, 'fit_cutoff':record['fit_cutoff'],
            'forecast_origin':str(origin), 'hours':len(selected), 'population':record['population']})
        result = deployment.forecast(record, scenario, origin, group=args.group, hours=args.hours)
        if inputs_sha256!=digest(args.inputs_csv) or model_sha256!=digest(args.model):
            raise ValueError('Model or input file changed during forecast; refusing to publish.')
        write_forecast(directory, result)
        files = ('invocation.json', 'forecast.json', 'forecast.csv', 'forecast_normalized.csv')
        _atomic_report(directory/'forecast_freeze.json', {'status':'frozen-input-only', 'origin':str(origin.date()),
            'target_observation_file_opened':False, 'artifact_sha256':{name:digest(directory/name) for name in files}})
        if args.score:
            observed = pd.read_csv(args.score, usecols=['Datetime', 'Demand'], index_col='Datetime',
                parse_dates=['Datetime'], float_precision='round_trip')
            calendar._frame(observed, ('Demand',))
            observed['Demand'] = pd.to_numeric(observed.Demand, errors='raise')
            if np.iscomplexobj(observed.to_numpy()) or np.isinf(observed.to_numpy(float)).any():
                raise ValueError('Targets must be real finite values or NaN; never infinite.')
            observed = calendar._grid(observed, selected.tz_localize('Europe/Copenhagen')).Demand
            result['scoring'] = {'status':'scored-after-freeze', 'metrics':calendar.score_predictions(
                result['predictions'], observed, {'G':'g_prediction_output_mean', 'pure_G':'g_prediction', 'baseline':'baseline'})}
            result['predictions']['observed'] = observed
            result['normalized_predictions']['observed'] = (observed-record['scales']['demand_min'])/record['scales']['demand_scale']
            write_forecast(directory, result, prefix='scored')
        plot(directory, result)
        elapsed = perf_counter()-begun
        _atomic_report(directory/'summary.json', {'status':'complete', 'seconds_including_load_forecast_export_plot':elapsed,
            'forecast_seconds':result['seconds'], 'physical_hours':len(selected),
            'fit_executed':False, 'historical_replay_executed':False, 'cold_import_seconds_included':False,
            'fit_cutoff':record['fit_cutoff'], 'forecast_origin':str(origin), 'scoring':result['scoring']})
        print(f'Complete: new G forecast, {len(selected)} hours, {elapsed:.3f} s after imports; {directory}', flush=True)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, TypeError) as exc:
        if directory is not None:
            _atomic_report(directory/'failure.json', {'status':'failed', 'error':str(exc), 'seconds':perf_counter()-begun})
        print(f'G {args.command} failed: {exc}', file=sys.stderr, flush=True)
        return 2


def bands_main(argv=None):
    from src.method.bm_1_bands import (
        PROTOCOL, MODELS, source_hashes, load_forecast, calibration_plan, calibrate,
        load_calibration, apply, observations_for_forecast, original,
    )
    parser = ArgumentParser(description='Declared-date empirical band calibration/evaluation.'); commands = parser.add_subparsers(dest='command', required=True)
    for command in ('calibrate', 'evaluate'):
        part = commands.add_parser(command)
        part.add_argument('--forecasts', nargs='+', required=True)
        part.add_argument('--observations', nargs='+', required=True)
        part.add_argument('--output', required=True)
        if command == 'calibrate': part.add_argument('--evaluation-origins', nargs='+', required=True)
        else: part.add_argument('--calibration', required=True)
    args = parser.parse_args(argv); directory = None
    try:
        if len(args.forecasts) != len(args.observations):
            raise ValueError('Provide one observations file for each forecast week.')
        forecasts = [load_forecast(path) for path in args.forecasts]
        if args.command == 'calibrate':
            calibration_plan(forecasts, args.evaluation_origins)
        def observations(path):
            return pd.read_csv(path, index_col='Datetime', parse_dates=['Datetime'], float_precision='round_trip')['Demand']
        target = Path(args.output); target.mkdir(parents=True, exist_ok=False); directory = target
        _atomic_report(directory/'invocation.json', {'command':args.command, 'protocol':PROTOCOL,
            'forecasts':args.forecasts, 'observations':args.observations, 'source_sha256':source_hashes()})
        if args.command == 'calibrate':
            result = calibrate([{'forecast':f, 'observations':observations(path)} for f, path in zip(forecasts, args.observations)], args.evaluation_origins)
            _atomic_report(directory/'calibration_snapshot.json', result['snapshot'])
            from src.exe.run_model.bm_1.deterministic import _atomic_csv
            _atomic_csv(directory/'calibration_scores.csv', result['scores'])
            _atomic_report(directory/'calibration_freeze.json', {'evaluation_observations_used':False,
                'artifact_sha256':{name:digest(directory/name) for name in ('calibration_snapshot.json', 'calibration_scores.csv')}})
        else:
            record = load_calibration(args.calibration)
            if [str(pd.Timestamp(f['manifest']['cutoff']).date()) for f in forecasts] != record['evaluation_origins']:
                raise ValueError('Supply every declared evaluation week in chronological order.')
            calibrated = [apply(f, record) for f in forecasts]; receipts = {}
            for result in calibrated:
                name = pd.Timestamp(result['manifest']['cutoff']).strftime('%Y%m%d')
                week = directory/name; week.mkdir(); write_forecast(week, result)
                receipts[name] = {file:digest(week/file) for file in ('forecast.json', 'forecast.csv', 'forecast_normalized.csv')}
            _atomic_report(directory/'evaluation_forecasts_freeze.json', {'target_observation_files_opened':False,
                'calibration_sha256':original.residual.snapshot_fingerprint(record), 'forecasts':receipts})
            scored = []
            for result, path in zip(calibrated, args.observations):
                result = original.score_calibrated(result, observations_for_forecast(observations(path), result)); scored.append(result)
                write_forecast(directory/pd.Timestamp(result['manifest']['cutoff']).strftime('%Y%m%d'), result, prefix='scored')
            table = pd.concat([result['predictions'] for result in scored]); y = table.observed.to_numpy(float)
            _atomic_report(directory/'evaluation_report.json', {'protocol':PROTOCOL, 'hours':len(table),
                'scored_samples':int(np.isfinite(y).sum()), 'coverage_status':record['coverage_status'],
                'metrics':{name:original.comparison.metrics(y, table[name].to_numpy(float))
                           for name in ('baseline', *(model+'_output_mean' for model in MODELS))},
                'calibrated_bands':{model:original.band_metrics(table, model+'_calibrated_lower', model+'_calibrated_upper') for model in MODELS},
                'weekly':[{'origin':result['manifest']['cutoff'], 'scoring':result['scoring']} for result in scored]})
        print('Complete: '+args.command+'; chronological development bands only.', flush=True); return 0
    except (ValueError, OSError, KeyError, RuntimeError, TypeError) as exc:
        if directory is not None:
            _atomic_report(directory/'failure.json', {'status':'failed', 'error':str(exc)})
        print(str(exc), file=sys.stderr, flush=True); return 2


if __name__ == '__main__':
    raise SystemExit(main())

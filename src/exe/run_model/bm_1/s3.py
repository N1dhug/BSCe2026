"""Pooled S3 development tools: interval, residual, or point-process semantics.

Select a variant explicitly, then fit/predict/diagnose. For the current saved
weekday G model use src.exe.run_model.bm_1_g; these are separate older studies.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys


def parser():
    root = ArgumentParser(description=__doc__)
    variants = root.add_subparsers(dest='variant', required=True)
    for variant in ('interval', 'residual', 'point-process'):
        commands = variants.add_parser(variant).add_subparsers(dest='command', required=True)
        actions = ('fit', 'predict') if variant == 'residual' else ('predict', 'diagnose')
        if variant == 'interval':
            actions = ('fit', *actions)
        for action in actions:
            command = commands.add_parser(action)
            command.add_argument('--output', required=True)
            if (variant, action) != ('interval', 'fit'):
                command.add_argument('--snapshot', required=True)
            if action == 'predict':
                command.add_argument('--inputs-csv', required=True)
                command.add_argument('--observations-csv')
                if variant == 'residual':
                    command.add_argument('--residual-snapshot', required=True)
                else:
                    command.add_argument('--sigma-x', type=float, required=variant == 'point-process',
                                         help='Explicit amplitude per sqrt(hour); not fitted.')
                    command.add_argument('--cells', type=int, default=128)
                    command.add_argument('--max-step', type=float, default=1/24)
                    command.add_argument('--seconds', type=float, default=120.)
            else:
                command.add_argument('--raw-csv', required=True)
                seconds = 60. if action == 'diagnose' else (30. if variant == 'residual' else 600.)
                command.add_argument('--seconds', type=float, default=seconds)
            if (variant, action) == ('interval', 'fit'):
                command.add_argument('--origin', required=True)
                command.add_argument('--demand-label', required=True, choices=['hour_start', 'hour_end'])
                command.add_argument('--max-calls', type=int, default=3000)
                command.add_argument('--reference', nargs=4, type=float, default=[12., 2., .5, .5])
                command.add_argument('--population', default='declared raw aggregate')
                command.add_argument('--demand-unit', default='raw average-power units; physical unit unverified')
    return root


def _predict(args, directory):
    from src.exe.run_model.bm_1.s3_forecast import write_forecast
    from src.tools.bm_1_deterministic_data import raw_csv

    inputs = raw_csv(args.inputs_csv, require_demand=False)
    if args.variant == 'point-process':
        from src.method.bm_1_s3_forecast import load_snapshot
        from src.method.bm_1_s3_uncertainty import process_forecast, score_process_forecast
        result = process_forecast(load_snapshot(args.snapshot), inputs, sigma_x=args.sigma_x,
                                  cells=args.cells, max_step=args.max_step, seconds=args.seconds)
        score = score_process_forecast
    else:
        from src.method import bm_1_s3_interval as interval
        snapshot = interval.load_snapshot(args.snapshot)
        if args.variant == 'residual':
            from src.method import bm_1_s3_residual as residual
            companion = json.loads(Path(args.residual_snapshot).read_text(encoding='utf-8'))
            result = residual.predict_residual(snapshot, companion, inputs)
            score = residual.score_residual
        elif args.sigma_x is None:
            result = interval.predict_interval(snapshot, inputs)
            score = interval.score_interval
        else:
            from src.method.bm_1_s3_interval_process import process_interval, score_interval_process
            result = process_interval(snapshot, inputs, sigma_x=args.sigma_x,
                                      cells=args.cells, max_step=args.max_step, seconds=args.seconds)
            score = score_interval_process
    # The durable conditional forecast always precedes opening target observations.
    write_forecast(directory, result)
    if args.observations_csv:
        import pandas as pd
        observed = pd.read_csv(args.observations_csv, index_col='Datetime', parse_dates=['Datetime']).Demand
        result = score(result, observed)
        write_forecast(directory, result, prefix='scored')
    plot(directory, result, args.variant)
    return result


def _fit(args, directory):
    from src.method import bm_1_s3_interval as interval
    from src.tools.bm_1_deterministic_data import raw_csv
    from src.tools.bm_1_reports import _atomic_report

    if args.variant == 'residual':
        from src.method import bm_1_s3_residual as residual
        result = residual.fit_residual(interval.load_snapshot(args.snapshot), raw_csv(args.raw_csv),
                                       seconds=args.seconds)
        _atomic_report(directory/'residual_snapshot.json', result['snapshot'])
        result['historical_replay'].to_csv(directory/'historical_replay_normalized.csv', index_label='Datetime')
        _atomic_report(directory/'fit_report.json',
                       {k: v for k, v in result.items() if k not in ('snapshot', 'historical_replay')})
    else:
        result = interval.fit_interval(raw_csv(args.raw_csv), args.origin, demand_label=args.demand_label,
            seconds=args.seconds, max_calls=args.max_calls, reference=args.reference,
            population=args.population, demand_unit=args.demand_unit,
            checkpoint=lambda value: _atomic_report(directory/'fit_checkpoint.json',
                interval.point._plain(value), replace_existing=True))
        _atomic_report(directory/'fit_result.json',
                       interval.point._plain({k: v for k, v in result.items() if k != 'snapshot'}))
        if result['snapshot'] is not None:
            interval.save_snapshot(directory/'interval_snapshot.json', result['snapshot'])
    return result


def _diagnose(args, directory):
    from src.method.bm_1_s3_forecast import _plain
    from src.tools.bm_1_deterministic_data import raw_csv
    from src.tools.bm_1_reports import _atomic_report

    if args.variant == 'interval':
        from src.method.bm_1_s3_interval import load_snapshot
        from src.method.bm_1_s3_interval_process import diagnose_interval as diagnose
    else:
        from src.method.bm_1_s3_forecast import load_snapshot
        from src.method.bm_1_s3_uncertainty import diagnose_calibration as diagnose
    result = diagnose(load_snapshot(args.snapshot), raw_csv(args.raw_csv), seconds=args.seconds)
    _atomic_report(directory/'calibration_audit.json', _plain(result))
    return result


def plot(directory, result, variant):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = result['predictions']
    fig, axes = plt.subplots(2 if variant == 'interval' else 1, 1,
                             figsize=(12, 6 if variant == 'interval' else 4.5), squeeze=False, sharex=True)
    ax = axes[0, 0]
    ax.plot(p.index, p.prediction, label='G-S3 deterministic prediction')
    if variant == 'interval':
        ax.plot(p.index, p.baseline, label='Integrated baseline', alpha=.7)
        if 'process_mean' in p:
            ax.plot(p.index, p.process_mean, ls='--', label='Process E[hourly average]')
        lower = axes[1, 0]
        if 'process_variance' in p:
            lower.plot(p.index, p.process_variance**.5)
            lower.set_ylabel('Process SD (raw units)')
        else:
            lower.step(p.index, p.price, where='post')
            lower.set_ylabel('Declared raw price')
        title = 'G-S3 hourly-average observation operator; conditional development'
    elif variant == 'residual':
        ax.fill_between(p.index, p.observed_lower, p.observed_upper, alpha=.2,
                        label='95% empirical output-error band (development)')
        ax.plot(p.index, p.observed_mean, label='G + historical mean error')
        title = 'Historical empirical errors; coverage not independently calibrated'
    else:
        ax.fill_between(p.index, p.process_lower95, p.process_upper95, alpha=.3,
                        label='Process-only pointwise 95% band')
        ax.plot(p.index, p.process_mean, label='E[D(X)]')
        title = 'G-S3 process scenario: explicit sigma, no historical calibration acceptance'
    if 'observed' in p:
        ax.scatter(p.index, p.observed, s=8, label='Measured')
    ax.set_ylabel(result['manifest']['demand_unit'] if variant == 'point-process'
                  else 'Mean power (raw units; unit unverified)')
    axes[-1, 0].set_xlabel('Europe/Copenhagen local time' if variant == 'point-process'
                           else 'Physical hourly intervals, Europe/Copenhagen start labels')
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    stem = 'process' if variant == 'point-process' else variant
    fig.savefig(directory/f'{stem}_week.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parser().parse_args(argv)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    from src.tools.bm_1_reports import _atomic_report
    try:
        action = {'fit': _fit, 'predict': _predict, 'diagnose': _diagnose}[args.command]
        result = action(args, directory)
        print(f"G-S3 {args.variant} {args.command}: {result['status']}; {result['seconds']:.2f} s", flush=True)
        accepted = ('complete', 'complete-interval-process', 'complete-process-scenario',
                    'candidate-needs-nonlinear-validation')
        return 0 if args.variant == 'residual' or result['status'] in accepted else 2
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        _atomic_report(directory/'failure.json',
                       {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        print(f'G-S3 {args.variant} {args.command} failed: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

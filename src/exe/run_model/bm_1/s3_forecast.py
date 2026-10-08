"""G-S3 fit/save/predict/score. Run with --help for the two small entry points."""
from argparse import ArgumentParser
from pathlib import Path
import sys

from src.method.bm_1_s3_forecast import (
    fit_s3, load_snapshot, save_snapshot, predict_s3, score_s3, _plain)
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report


def write_forecast(directory, result, *, prefix='forecast'):
    from src.exe.run_model.bm_1.deterministic import _atomic_csv
    _atomic_csv(directory/f'{prefix}.csv', result['predictions'])
    _atomic_csv(directory/f'{prefix}_normalized.csv', result['normalized_predictions'])
    plain = {k:v for k,v in result.items() if k not in ('predictions','normalized_predictions')}
    _atomic_report(directory/f'{prefix}.json', _plain(plain))


def plot_forecast(directory, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = result['predictions']
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(p.index, p.prediction, label='G-S3')
    axes[0].plot(p.index, p.baseline, label='Baseline', alpha=.7)
    if 'observed' in p: axes[0].scatter(p.index, p.observed, s=8, label='Measured')
    axes[0].set_ylabel(result['manifest']['demand_unit']); axes[0].legend()
    axes[1].step(p.index, p.price, where='post'); axes[1].set_ylabel('Declared raw price')
    axes[1].set_xlabel('Europe/Copenhagen local time')
    fig.suptitle('G-S3: seven daily forecasts frozen Monday; conditional price/temperature')
    fig.tight_layout(); fig.savefig(directory/'week.png', dpi=150); plt.close(fig)


def main(argv=None):
    parser = ArgumentParser(description='Standalone G-S3: one fit, reusable snapshot, input-only forecast.')
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Fit one Monday origin from historical raw-unit CSV.')
    fit.add_argument('--raw-csv', required=True)
    fit.add_argument('--origin', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--reference', type=float, nargs=4, default=[12.,2.,.5,.5])
    fit.add_argument('--calibration-start')
    fit.add_argument('--calibration-end')
    fit.add_argument('--price-bounds', type=float, nargs=2, default=[-500.,6000.])
    fit.add_argument('--seconds', type=float, default=600.)
    fit.add_argument('--max-calls', type=int, default=3000)
    fit.add_argument('--population', default='declared raw aggregate')
    fit.add_argument('--demand-unit', default='original raw units; physical unit unverified')
    predict = commands.add_parser('predict', help='Forecast a week from saved fit, without fitting.')
    predict.add_argument('--snapshot', required=True)
    predict.add_argument('--inputs-csv', required=True, help='Datetime,Price,Temp; future demand is unnecessary.')
    predict.add_argument('--observations-csv', help='Optional Datetime,Demand for scoring after forecast export.')
    predict.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        if args.command == 'fit':
            raw = raw_csv(args.raw_csv)
            result = fit_s3(raw, args.origin, reference=args.reference,
                calibration_start=args.calibration_start, calibration_end=args.calibration_end,
                price_bounds=args.price_bounds, seconds=args.seconds, max_calls=args.max_calls,
                population=args.population, demand_unit=args.demand_unit,
                checkpoint=lambda x:_atomic_report(directory/'fit_checkpoint.json', _plain(x), replace_existing=True))
            _atomic_report(directory/'fit_result.json', _plain({k:v for k,v in result.items() if k != 'snapshot'}))
            if result['snapshot'] is not None: save_snapshot(directory/'s3_snapshot.json', result['snapshot'])
            print(f"S3 fit: {result['status']}; {result['calls_used']} residual calls; {result['seconds']:.2f} s", flush=True)
            return 0 if result['status'] == 'complete' else 2
        result = predict_s3(load_snapshot(args.snapshot), raw_csv(args.inputs_csv, require_demand=False))
        # Publish the input-only prediction before even opening observation files.
        write_forecast(directory, result)
        if args.observations_csv:
            import pandas as pd
            observations = pd.read_csv(args.observations_csv, index_col='Datetime', parse_dates=['Datetime'])
            result = score_s3(result, observations['Demand'])
            write_forecast(directory, result, prefix='scored')
        plot_forecast(directory, result)
        print(f"S3 forecast: {result['status']}; {len(result['predictions'])} physical hours; scoring={result['scoring']['status']}", flush=True)
        return 0 if result['status'] == 'complete' else 2
    except (ValueError, OSError, KeyError) as exc:
        _atomic_report(directory/'failure.json', {'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        print(f'S3 {args.command} failed: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__': raise SystemExit(main())

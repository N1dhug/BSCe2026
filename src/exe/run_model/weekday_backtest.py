"""Arbitrary rolling five-week / sixth-week tests; G and explicit native F fits."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

from argparse import ArgumentParser
import importlib
import json
from pathlib import Path

import pandas as pd

from src.tools.bm_1_reports import _atomic_report
from src.tools.weekday_backtest import rolling_windows, run_backtest


def _csv(path, columns=None):
    # Aware CSVs require a uniform explicit offset or upstream UTC conversion.
    # Never discard offsets or infer ambiguous local autumn labels.
    return pd.read_csv(path, index_col='Datetime', parse_dates=['Datetime'], float_precision='round_trip',
                       usecols=None if columns is None else ['Datetime', *columns])


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    for name in ('first-origin', 'last-origin', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--history-csv')
    parser.add_argument('--inputs-csv')
    parser.add_argument('--observations-csv', help='Optional scoring only, opened after all forecasts are frozen.')
    parser.add_argument('--zone', default='Europe/Copenhagen')
    parser.add_argument('--plan-only', action='store_true', help='Save calendar plan without loading data or fitting.')
    parser.add_argument('--adapter', default='G', help='G, F, or module:zero_argument_factory.')
    parser.add_argument('--adapter-config', help='Required for F: explicit earlier frozen baseline, population and new-fit budgets.')
    args = parser.parse_args(argv)
    windows = rolling_windows(args.first_origin, args.last_origin, zone=args.zone)
    if args.plan_only:
        directory = Path(args.output); directory.mkdir(parents=True, exist_ok=False)
        _atomic_report(directory/'calendar_plan.json', {'windows':[w.record() for w in windows], 'model_executed':False})
        return 0
    if not args.history_csv or not args.inputs_csv:
        parser.error('--history-csv and --inputs-csv are required unless --plan-only is used.')
    if args.adapter == 'G':
        from src.tools.g_weekday_backtest_adapter import make_adapter
        if args.adapter_config:
            parser.error('--adapter-config belongs to the explicit F adapter.')
        adapter = make_adapter()
    elif args.adapter == 'F':
        from src.tools.f_weekday_backtest_adapter import make_adapter
        if not args.adapter_config:
            parser.error('--adapter F requires --adapter-config; saved curves are never a weekday fit.')
        adapter = make_adapter(args.adapter_config)
    else:
        module, separator, name = args.adapter.partition(':')
        if not separator or not module or not name:
            parser.error('Custom adapter must be module:zero_argument_factory; F has no implicit fixed-fit fallback.')
        make_adapter = getattr(importlib.import_module(module), name)
        if args.adapter_config:
            parser.error('Custom adapter factories retain their zero-argument contract.')
        adapter = make_adapter()
    load_targets = None
    if args.observations_csv:
        load_targets = lambda: _csv(args.observations_csv)[adapter.target_column]
    summary = run_backtest(_csv(args.history_csv), _csv(args.inputs_csv, adapter.input_columns), windows, adapter,
                           args.output, load_observations=load_targets)
    print(json.dumps(summary, allow_nan=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())

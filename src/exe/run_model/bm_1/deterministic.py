"""Run G0-G3: python -m src.exe.run_model.bm_1.deterministic --month 2022-01."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import os


def _jsonable(value):
    import numpy as np
    import pandas as pd
    if isinstance(value, dict): return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    return value


def write_result(directory, result):
    import matplotlib.pyplot as plt
    from src.tools.bm_1_reports import _atomic_report
    from src.exe.run_model.wm_4.analyze_performance import summary_table, weekday_summary
    plain = {k:v for k,v in result.items() if k not in ('predictions','normalized_predictions','daily_metrics','performance')}
    _atomic_report(directory/'result.json', _jsonable(plain))
    _atomic_report(directory/'manifest.json', _jsonable(result['manifest']))
    _atomic_report(directory/'fit.json', _jsonable(result['fit']))
    if 'predictions' not in result: return
    p = result['predictions']
    p.to_csv(directory/'predictions.csv', index_label='Datetime')
    result['normalized_predictions'].to_csv(directory/'predictions_normalized.csv', index_label='Datetime')
    result['daily_metrics'].to_csv(directory/'daily_metrics.csv', index=False)
    for name, data in result['performance'].items():
        data.to_csv(directory/f'{name}_daily_performance.csv', index=False)
        adapted = data.rename(columns=lambda key: key.replace('bm1_', 'wm4_')) if name == 'bm1' else data
        for suffix, function in (('summary',summary_table), ('weekday',weekday_summary)):
            table = function(adapted)
            if name == 'bm1': table = table.rename(columns=lambda key: key.replace('wm4_', 'bm1_'))
            table.to_csv(directory/f'{name}_{suffix}.csv', index=False)
    fig, axes = plt.subplots(4,1,figsize=(15,11),sharex=True)
    for key in ('observed','O','bm1','wm4'):
        if key in p: axes[0].plot(p.index,p[key],label=key)
    axes[0].legend()
    axes[0].scatter(p.index,p.observed,s=8)
    axes[0].set_ylabel('Workbook consumption units')
    axes[1].plot(p.index,p.bm1-p.O,label='BM D-B')
    if 'R' in p: axes[1].plot(p.index,p.R,label='WM I-O=R')
    axes[1].legend(); axes[1].set_ylabel('Baseline correction')
    axes[2].plot(p.index,p.U,label='Signed price U'); axes[2].legend(); axes[2].set_ylabel('Signed price')
    for key in ('x','F'):
        if key in p: axes[3].plot(p.index,p[key],label='BM x' if key=='x' else 'WM F')
    axes[3].legend(); axes[3].set_ylabel('Separate BM/WM states')
    for axis in axes: axis.grid(True,alpha=.25)
    fig.suptitle('G0-G3 conditional development forecast | fit '+result['manifest']['training_start'][:10]+' to '+result['manifest']['training_end_exclusive'][:10])
    fig.tight_layout(); fig.savefig(directory/'week.png',dpi=150); plt.close(fig)


def main(argv=None):
    parser = ArgumentParser(description='Budgeted deterministic BM fit with matched WM4; January is training, never test.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--month', help='Entire fit month, default 2022-01; predicts next 168 hours.')
    group.add_argument('--week', help='Monday fit start, matched one-week protocol with 24h warm-up.')
    parser.add_argument('--data-directory', type=Path, default=Path(__file__).resolve().parents[4]/'files/input/data')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input-csv', type=Path, help='Raw Datetime,Demand,Price,Temp CSV instead of workbook/DMI.')
    source.add_argument('--synthetic',action='store_true',help='Reproducible synthetic January demonstration; no historical claims.')
    parser.add_argument('--tag', default='MED')
    parser.add_argument('--price-bounds',type=float,nargs=2,metavar=('LOW','HIGH'),
                        help='Optional ex-ante DK1 DKK/MWh bounds shared by BM/WM; never choose from test results.')
    parser.add_argument('--warmup-hours', type=int, help='Default 0 for whole month, 24 for week.')
    parser.add_argument('--starts', type=int, choices=range(1,5), default=4)
    parser.add_argument('--max-calls', type=int, default=1000)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--bm-steps-per-hour', type=int, default=60)
    parser.add_argument('--wm-steps-per-hour', type=int, default=60)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    os.environ.setdefault('MPLBACKEND','Agg')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ[name] = '1'
    import hashlib
    import pandas as pd
    from contextlib import nullcontext
    from src.params.bm_1_deterministic import BM1DeterministicConfig
    from src.method.bm_1_deterministic_benchmark import run_benchmark
    from src.tools.bm_1_deterministic_data import load_raw_frame, raw_csv, prepare_frame
    from src.tools.bm_1_reports import _RunLock, _atomic_report
    from src.exe.run_model.progress import runtime_text
    from time import perf_counter
    try:
        from threadpoolctl import threadpool_limits
        threads = threadpool_limits(limits=1)
    except ImportError:
        threads = nullcontext()
    if args.week:
        start = pd.Timestamp(args.week).normalize()
        if start.weekday() != 0: parser.error('--week must be Monday.')
        end, warmup = start+pd.Timedelta(hours=168), 24
    else:
        start = pd.Timestamp((args.month or '2022-01')+'-01')
        end, warmup = start+pd.offsets.MonthBegin(1), 0
    if args.warmup_hours is not None: warmup = args.warmup_hours
    if warmup < 0: parser.error('--warmup-hours must be nonnegative.')
    root = Path(__file__).resolve().parents[4]
    directory = args.output or root/'files/output/bm_1/runs/deterministic'/datetime.now(timezone.utc).strftime('g_january_%Y%m%dT%H%M%SZ')
    if directory.exists() and any(directory.iterdir()):
        parser.error('Output directory must be new/empty; checkpoints are not an exact optimizer resume.')
    directory.mkdir(parents=True,exist_ok=True)
    config = BM1DeterministicConfig(starts=BM1DeterministicConfig().starts[:args.starts],
                                   max_calls=args.max_calls, seconds=args.seconds,
                                   steps_per_hour=args.bm_steps_per_hour)
    print(f'TRAIN={start} -> {end}; PREDICT={end} -> {end+pd.Timedelta(hours=168)}; BLAS=1',flush=True)
    began = perf_counter()
    def checkpoint(state):
        _atomic_report(directory/'checkpoint.json',_jsonable(state),replace_existing=True)
    def progress(state):
        print('G2 calls='+str(state['counts']['residual_calls'])+'; '+runtime_text(began,state['counts']['residual_calls'],config.max_calls),flush=True)
    with _RunLock(directory), threads:
        try:
            if args.synthetic:
                from src.validation.bm_1.deterministic_fit import synthetic_january
                raw = synthetic_january()
            else:
                raw = raw_csv(args.input_csv) if args.input_csv else load_raw_frame(args.data_directory,start-pd.Timedelta(hours=warmup),end+pd.Timedelta(hours=169),args.tag,cutoff=end)
            frame, scales = prepare_frame(raw,start,end,price_bounds=args.price_bounds)
            audit = {'training_start':str(start),'training_end_exclusive':str(end),'scales':scales,
                     'samples':int((end-start).total_seconds()/3600),'tag':args.tag,
                     'contract':'hourly-point-approximation, metadata unresolved',
                     'raw_fingerprint':hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).values.tobytes()).hexdigest()}
            _atomic_report(directory/'audit.json',audit)
            if args.audit_only:
                print(f'Audit saved: {directory}',flush=True)
                return 0
            result = run_benchmark(raw,start,end,config=config,warmup_hours=warmup,
                                   wm_steps_per_hour=args.wm_steps_per_hour,price_bounds=args.price_bounds,
                                   checkpoint=checkpoint,progress=progress)
            result['manifest']['data_audit'] = audit
            result['manifest']['data_source'] = 'synthetic, seed 43' if args.synthetic else 'raw CSV' if args.input_csv else 'historical workbook/DMI'
            source_paths = ['src/method/bm_1_deterministic_fit.py','src/method/bm_1_deterministic_benchmark.py',
                            'src/params/bm_1_deterministic.py','src/tools/bm_1_deterministic_data.py',
                            'src/model/base/bm.py','src/model/wm_4.py','src/params/wm_4.py',
                            'src/exe/run_model/wm_4/output_model.py']
            result['manifest']['source_sha256'] = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths}
            write_result(directory,result)
            print(f"status={result['status']}; output={directory}",flush=True)
            if result['status']=='complete':
                print(result['metrics'],flush=True)
            return 0 if result['status']=='complete' else 2
        except Exception as exc:
            _atomic_report(directory/'failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc)},replace_existing=True)
            print(f'{type(exc).__name__}: {exc}; output={directory}',flush=True)
            return 2


if __name__ == '__main__':
    raise SystemExit(main())

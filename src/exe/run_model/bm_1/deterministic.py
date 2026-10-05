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


def write_performance(directory, prefix, daily, performance):
    from src.exe.run_model.wm_4.analyze_performance import summary_table, weekday_summary
    daily.to_csv(directory/f'{prefix}daily_metrics.csv', index=False)
    for name, data in performance.items():
        data.to_csv(directory/f'{prefix}{name}_daily_performance.csv', index=False)
        adapted = data.rename(columns=lambda key: key.replace('bm1_', 'wm4_')) if name == 'bm1' else data
        for suffix, function in (('summary',summary_table), ('weekday',weekday_summary)):
            table = function(adapted)
            if name == 'bm1': table = table.rename(columns=lambda key: key.replace('wm4_', 'bm1_'))
            table.to_csv(directory/f'{prefix}{name}_{suffix}.csv', index=False)


def save_fit_plots(directory, result):
    """WM-style normalized signals, hourly observations and fine model curves."""
    import matplotlib.pyplot as plt
    import pandas as pd
    observed, fine = result['fit_normalized'], result['fit_signals']
    start = pd.Timestamp(result['manifest']['training_start'])
    end = pd.Timestamp(result['manifest']['training_end_exclusive'])
    def plot(a, b, filename):
        points = observed.loc[(observed.index >= a)&(observed.index < b)]
        curves = fine.loc[(fine.index >= a)&(fine.index <= b)]
        fig, axes = plt.subplots(4,1,figsize=(15,12),sharex=True)
        axes[0].plot(points.index,points.observed,color='black',alpha=.55,label='Observed hourly aggregate input I')
        axes[0].scatter(points.index,points.observed,s=10,color='black')
        filled = points.loc[points.demand_imputed]
        if len(filled):
            axes[0].scatter(filled.index,filled.demand_input,s=20,marker='x',color='gray',
                            label='Interpolated demand: excluded from fit scores')
        axes[0].plot(curves.index,curves.bm1,label='Fitted BM1 input D')
        if 'wm4' in curves: axes[0].plot(curves.index,curves.wm4,label='Fitted WM4 input I=O+R')
        axes[0].plot(curves.index,curves.O,alpha=.7,label='Shared weather-aware baseline O=B')
        axes[0].set_ylabel('Normalized input')
        axes[1].plot(curves.index,curves.O,label='Expected output O(time, weekday, temperature)')
        axes[1].step(curves.index,curves.U,where='post',label='Realized DK1 spot price U, normalized')
        axes[1].set_ylabel('Signals / output')
        axes[2].plot(curves.index,curves.bm_correction,label='BM1 baseline correction D-B')
        if 'R' in curves: axes[2].plot(curves.index,curves.R,label='WM4 baseline correction R=I-O')
        axes[2].set_ylabel('Input correction')
        axes[3].plot(curves.index,curves.bm_signed_state,label='BM1 signed coordinate 1-2x')
        if 'F' in curves: axes[3].plot(curves.index,curves.F,label='WM4 flexibility state F')
        axes[3].set_ylabel('Separate model states')
        axes[3].set_xlabel('Physical time')
        for axis in axes: axis.legend(); axis.grid(True,alpha=.25)
        source = result['manifest'].get('data_source','raw data')
        x0 = result['fit']['best_converged']['parameters'][3]
        fig.suptitle(f'G January in-sample fit | {a:%Y-%m-%d} to {b:%Y-%m-%d} | x0={x0:.4f}\n{source}; realized prices; common baseline and scaling')
        fig.tight_layout(); fig.savefig(directory/filename,dpi=150); plt.close(fig)
    plot(start,end,'fit.png')
    for number,a in enumerate(pd.date_range(start,end,freq='7D',inclusive='left'),1):
        plot(a,min(a+pd.Timedelta(days=7),end),f'fit_week_{number:02d}.png')


def print_fit_result(result):
    best = result['fit'].get('best_converged')
    if best is None: return
    C,k,xi,x0 = best['parameters']
    print(f'BM1: C_hours={C:.8f}, k={k:.8f}, Xi={xi:.8f}, x0={x0:.8f}; fixed_x0={result["fit"]["fixed_x0"]}',flush=True)
    if 'wm_fit' in result:
        wm = result['wm_fit']; r = wm['r']
        print(f'WM4: r_F={r[0]:.8f}, r_U={r[1]:.8f}, r_R={r[2]:.8f}; initial [F0,R0]={wm["init_cond"]}',flush=True)
    for units,key in (('normalized, same WM metrics','fit_metrics_normalized'),('raw workbook units','fit_metrics')):
        if key not in result: continue
        print(f'January in-sample fit ({units})',flush=True)
        print(f'{"":<8}{"MSE":>14}{"RMSE":>14}{"MAE":>14}{"bias":>14}',flush=True)
        for name in ('bm1','wm4','O'):
            if name not in result[key]: continue
            m=result[key][name]
            print(f'{name:<8}{m["mse"]:>14.10f}{m["rmse"]:>14.10f}{m["mae"]:>14.10f}{m["bias"]:>14.10f}',flush=True)


def write_result(directory, result):
    import matplotlib.pyplot as plt
    from src.tools.bm_1_reports import _atomic_report
    tables = {'predictions','normalized_predictions','daily_metrics','performance',
              'fit_predictions','fit_normalized','fit_signals','fit_daily_metrics','fit_performance'}
    plain = {k:v for k,v in result.items() if k not in tables}
    _atomic_report(directory/'manifest.json', _jsonable(result['manifest']))
    _atomic_report(directory/'fit.json', _jsonable(result['fit']))
    if 'fit_predictions' in result:
        result['fit_predictions'].to_csv(directory/'fit_predictions.csv',index_label='Datetime')
        result['fit_normalized'].to_csv(directory/'fit_predictions_normalized.csv',index_label='Datetime')
        result['fit_signals'].to_csv(directory/'fit_signals_normalized.csv',index_label='Datetime')
        write_performance(directory,'fit_',result['fit_daily_metrics'],result['fit_performance'])
        save_fit_plots(directory,result)
    if 'predictions' not in result:
        _atomic_report(directory/'result.json',_jsonable(plain))
        return
    p = result['predictions']
    p.to_csv(directory/'predictions.csv', index_label='Datetime')
    result['normalized_predictions'].to_csv(directory/'predictions_normalized.csv', index_label='Datetime')
    write_performance(directory,'',result['daily_metrics'],result['performance'])
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
    _atomic_report(directory/'result.json',_jsonable(plain))


def main(argv=None):
    parser = ArgumentParser(description='Budgeted deterministic BM fit with matched WM4; January is training, never test.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--month', help='Entire fit month, default 2022-01; predicts next 168 hours.')
    group.add_argument('--week', help='Monday fit start, matched one-week protocol with 24h warm-up.')
    parser.add_argument('--fit-only',action='store_true',help='Plot/score the training fit; no next-week inputs or February data needed.')
    parser.add_argument('--initial-state',choices=('wm-neutral','fit'),default='wm-neutral',
                        help='Default: fixed neutral x0=(1-INIT_F)/2 from WM4; fit jointly estimates x0 on training data.')
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
    from src.exe.run_model.wm_4.fit_daily import INIT_F
    from src.method.bm_1_deterministic_benchmark import run_benchmark
    from src.tools.bm_1_deterministic_data import (load_raw_frame,raw_csv,prepare_frame,
                                                 training_data_audit,training_audit_text)
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
                                   steps_per_hour=args.bm_steps_per_hour,
                                   fixed_x0=(1-float(INIT_F))/2 if args.initial_state=='wm-neutral' else None)
    forecast_hours = 0 if args.fit_only else 168
    data_end = end if args.fit_only else end+pd.Timedelta(hours=169)
    print(f'TRAIN={start} -> {end}; '+('FIT ONLY' if args.fit_only else f'PREDICT={end} -> {end+pd.Timedelta(hours=168)}')+'; BLAS=1',flush=True)
    print(f'BM initial-state mode={args.initial_state}; fixed x0={config.fixed_x0}; WM INIT_F={INIT_F}',flush=True)
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
                raw = raw_csv(args.input_csv) if args.input_csv else load_raw_frame(args.data_directory,start-pd.Timedelta(hours=warmup),data_end,args.tag,cutoff=end)
            coverage = training_data_audit(raw,start,end)
            _atomic_report(directory/'data_audit.json',coverage)
            print(training_audit_text(coverage),flush=True)
            if args.audit_only:
                print(f'Data audit saved: {directory}; no parameter fit performed.',flush=True)
                return 0
            frame, scales = prepare_frame(raw,start,end,price_bounds=args.price_bounds)
            audit = {'training_start':str(start),'training_end_exclusive':str(end),'scales':scales,
                     'samples':int((end-start).total_seconds()/3600),'tag':args.tag,'coverage':coverage,
                     'observed_demand_samples':coverage['columns']['Demand']['valid_hours'],
                     'contract':'hourly-point-approximation, metadata unresolved',
                     'raw_fingerprint':hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).values.tobytes()).hexdigest()}
            _atomic_report(directory/'audit.json',audit)
            result = run_benchmark(raw,start,end,config=config,warmup_hours=warmup,
                                   forecast_hours=forecast_hours,
                                   wm_steps_per_hour=args.wm_steps_per_hour,price_bounds=args.price_bounds,
                                   checkpoint=checkpoint,progress=progress)
            result['manifest']['data_audit'] = audit
            result['manifest']['data_source'] = 'synthetic, seed 43' if args.synthetic else 'raw CSV' if args.input_csv else 'historical workbook/DMI'
            result['manifest']['initialization']['cli_mode'] = args.initial_state
            result['manifest']['initialization']['wm_INIT_F'] = float(INIT_F)
            result['manifest']['initialization']['neutral_coordinate'] = '1-2x=F0; neutral coordinate convention, distinct physical states'
            source_paths = ['src/method/bm_1_deterministic_fit.py','src/method/bm_1_deterministic_benchmark.py',
                            'src/params/bm_1_deterministic.py','src/tools/bm_1_deterministic_data.py',
                            'src/model/base/bm.py','src/model/wm_4.py','src/params/wm_4.py',
                            'src/exe/run_model/wm_4/output_model.py','src/exe/run_model/wm_4/fit_daily.py',
                            'src/exe/run_model/bm_1/deterministic.py','src/tools/import_data/import_input.py']
            result['manifest']['source_sha256'] = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths}
            write_result(directory,result)
            print(f"status={result['status']}; output={directory}",flush=True)
            print_fit_result(result)
            if 'metrics' in result:
                print(result['metrics'],flush=True)
            if 'fit_predictions' in result:
                print(f'January fit plot={directory/"fit.png"}; details=fit_week_*.png',flush=True)
            return 0 if result['status']=='complete' else 2
        except Exception as exc:
            _atomic_report(directory/'failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc)},replace_existing=True)
            print(f'{type(exc).__name__}: {exc}; output={directory}',flush=True)
            return 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Small G4 pilot from original training exports, then frozen development forecasts."""
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from time import perf_counter


def save_response_plot(directory, result):
    import matplotlib.pyplot as plt
    import numpy as np
    from src.params.bm_1_deterministic_response import BM1ResponseModel
    model = BM1ResponseModel.from_record(result['fit_snapshot']['model'])
    query = np.linspace(0,1,201)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(query,1-2*query,label='Linear reference')
    ax.plot(query,model.price_response(query),label='Shared G4 price response g')
    ax.scatter(model.price_response.knots,model.price_response.values)
    ax.set(xlabel='Normalized price u',ylabel='g(u)',title='Unselected DEVELOPMENT candidate')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(directory/'response.png',dpi=150); plt.close(fig)
    if 'predictions' in result:
        p = result['predictions']
        fig, ax = plt.subplots(figsize=(14,5))
        for name in ('observed','O','linear_bm1','bm1','wm4'):
            if name in p: ax.plot(p.index,p[name],label=name)
        ax.set(ylabel='Original workbook units',title='G4 conditional DEVELOPMENT forecast: frozen references')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(directory/'comparison.png',dpi=150); plt.close(fig)


def main(argv=None):
    parser = ArgumentParser(description=__doc__,allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference-fit',type=Path,help='Original zero-warmup linear fit folder, including training CSVs.')
    mode.add_argument('--from-fit',type=Path,help='One checked G4 variant folder; no parameter optimization.')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--regularizations',type=float,nargs='+',default=[0.,1e-3])
    parser.add_argument('--starts',type=float,nargs='+',default=[0.,.25],help='Initial g(0.5) values; at most two.')
    parser.add_argument('--max-calls',type=int,default=3000)
    parser.add_argument('--calls-per-start',type=int,default=750)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--data-directory',type=Path,default=Path(__file__).resolve().parents[4]/'files/input/data')
    parser.add_argument('--input-csv',type=Path,help='Forecast raw Datetime,Price,Temp and optional Demand.')
    actual = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(actual)
    if args.from_fit:
        forbidden = {'--regularizations','--starts','--max-calls','--calls-per-start','--seconds'}
        if any(t.split('=')[0] in forbidden for t in actual): parser.error('G4 --from-fit freezes the candidate; remove fit options.')
    elif args.input_csv or '--data-directory' in [t.split('=')[0] for t in actual]:
        parser.error('G4 fitting reads only original TRAIN exports; forecast input options are not applicable.')
    os.environ.setdefault('MPLBACKEND','Agg')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[name] = '1'
    from src.exe.run_model.bm_1.deterministic import _jsonable, write_result, write_forecast
    from src.tools.bm_1_reports import _RunLock, _atomic_report
    from src.exe.run_model.progress import runtime_text
    from src.tools.bm_1_deterministic_response import (fit_training_archive,load_response_snapshot,
        forecast_response,score_response_forecast)
    try:
        from threadpoolctl import threadpool_limits
        threads = threadpool_limits(limits=1)
    except ImportError:
        threads = nullcontext()
    root = Path(__file__).resolve().parents[4]
    directory = args.output or root/'files/output/bm_1/runs/deterministic'/datetime.now(timezone.utc).strftime('g4_%Y%m%dT%H%M%SZ')
    if directory.exists() and any(directory.iterdir()): parser.error('Use a new/empty output directory; original artifacts are preserved.')
    directory.mkdir(parents=True,exist_ok=True)
    with _RunLock(directory), threads:
        try:
            if args.reference_fit:
                print('G4: original TRAIN only; baseline/WM/scales/k frozen; no candidate selection; BLAS=1',flush=True)
                began = perf_counter()
                def checkpoint(state): _atomic_report(directory/'pilot.json',_jsonable(state),replace_existing=True)
                def progress(state):
                    calls = state['counts']['residual_calls']
                    print(f'G4 residual calls={calls}; '+runtime_text(began,calls,args.max_calls)
                          +f'; call-budget estimate; hard compute cap={args.seconds:g}s',flush=True)
                report, results = fit_training_archive(args.reference_fit,seconds=args.seconds,
                    regularizations=args.regularizations,starts=args.starts,max_calls=args.max_calls,
                    calls_per_start=args.calls_per_start,checkpoint=checkpoint,progress=progress)
                for result in results:
                    variant = directory/'variants'/result['variant']; variant.mkdir(parents=True)
                    write_result(variant,result); save_response_plot(variant,result)
                    best = result['fit']['best_converged']
                    print(f"{result['variant']}: lambda={result['fit']['regularization']}; parameters={best['parameters']}; g_mid={best['g_mid']}; data MSE={best['mse']}; penalty={best['penalty']}; refinement={result['numerics']['max_demand_refinement_error']}",flush=True)
                print(f"status={report['status']}; exported={len(results)}; output={directory}; compare candidates on DEVELOPMENT",flush=True)
                return 0 if report['status']=='complete' else 2
            from src.tools.bm_1_deterministic_data import raw_csv,load_forecast_inputs,load_forecast_observations
            import pandas as pd
            snapshot = load_response_snapshot(args.from_fit)
            origin = pd.Timestamp(snapshot['origin']); end = origin+pd.Timedelta(hours=168)
            print(f'G4 FIT REUSED: {origin} -> {end}; no parameter optimization; BLAS=1',flush=True)
            raw = raw_csv(args.input_csv,require_demand=False) if args.input_csv else load_forecast_inputs(args.data_directory,origin,end)
            result = forecast_response(snapshot,raw)
            _atomic_report(directory/'fit_snapshot.json',_jsonable(snapshot))
            write_forecast(directory,result)
            try:
                tag = snapshot['reference']['manifest'].get('data_audit',{}).get('tag')
                if not args.input_csv and tag is None: raise ValueError('Reference population tag is missing for workbook scoring.')
                observed = raw.get('Demand') if args.input_csv else load_forecast_observations(args.data_directory,origin,end,tag)
                result = score_response_forecast(result,observed)
            except Exception as exc:
                result = score_response_forecast(result,None)
                result['scoring']['status'] = 'failed'
                result['scoring_failure'] = {'type':type(exc).__name__,'message':str(exc)}
            write_result(directory,result); save_response_plot(directory,result)
            print(f"status={result['status']}; scoring={result['scoring']['status']}; output={directory}",flush=True)
            print(result['scoring'],flush=True); print(result['metrics'],flush=True)
            return 0 if result['status']=='complete' and result['scoring']['status']!='failed' else 2
        except Exception as exc:
            _atomic_report(directory/'failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc)},replace_existing=True)
            print(f'{type(exc).__name__}: {exc}; output={directory}',flush=True)
            return 2


if __name__ == '__main__':
    raise SystemExit(main())

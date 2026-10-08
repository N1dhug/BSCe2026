"""Separate shared-baseline hourly G/WM4 development comparison."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import pandas as pd

from src.method import bm_1_s3_interval as interval
from src.method import g_wm_hourly as comparison
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


def plot(directory,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=result['predictions'];fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for c,label in (('g_prediction','G hourly mean'),('wm_prediction','WM4 adapted hourly mean'),('baseline','Shared baseline')):
        axes[0].plot(p.index,p[c],label=label)
    if 'observed' in p:axes[0].scatter(p.index,p.observed,s=8,label='Measured')
    axes[0].set_ylabel('Mean power (raw units; unit unverified)');axes[0].legend()
    for c,label in (('g_prediction','G empirical output-error'),('wm_prediction','WM empirical output-error')):
        if c+'_lower' in p:
            axes[1].fill_between(p.index,p[c+'_lower'],p[c+'_upper'],alpha=.18,label=label+' 95% band')
            axes[1].plot(p.index,p[c+'_output_mean'],label=label+' mean')
    if 'observed' in p:axes[1].scatter(p.index,p.observed,s=8,label='Measured')
    axes[1].legend();axes[1].set_ylabel('Development output distribution')
    axes[1].set_xlabel('Physical hour start, Europe/Copenhagen')
    fig.suptitle('Shared baseline/scenario/cutoff; adapted WM interval fit; coverage not calibrated')
    fig.tight_layout();fig.savefig(directory/'comparison_week.png',dpi=150);plt.close(fig)


def main(argv=None):
    parser=ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('fit');fit.add_argument('--raw-csv',required=True)
    fit.add_argument('--seconds',type=float,default=60.);fit.add_argument('--max-calls',type=int,default=400)
    predict=commands.add_parser('predict');predict.add_argument('--wm-snapshot',required=True)
    predict.add_argument('--g-residual-snapshot');predict.add_argument('--inputs-csv',required=True)
    predict.add_argument('--observations-csv')
    for p in (fit,predict):p.add_argument('--g-snapshot',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    try:
        parent=interval.load_snapshot(args.g_snapshot)
        if args.command=='fit':
            result=comparison.fit_wm(parent,raw_csv(args.raw_csv),seconds=args.seconds,max_calls=args.max_calls,
                checkpoint=lambda value:_atomic_report(directory/'fit_checkpoint.json',value,replace_existing=True))
            _atomic_report(directory/'fit_report.json',{k:v for k,v in result.items() if k not in ('snapshot','historical_replay')})
            if result['snapshot'] is not None:
                _atomic_report(directory/'wm_snapshot.json',comparison.validate_wm(result['snapshot'],parent))
                result['historical_replay'].to_csv(directory/'historical_replay_normalized.csv',index_label='Datetime')
        else:
            s=json.loads(Path(args.wm_snapshot).read_text(encoding='utf-8'))
            r=None if args.g_residual_snapshot is None else json.loads(Path(args.g_residual_snapshot).read_text(encoding='utf-8'))
            result=comparison.predict_comparison(parent,s,raw_csv(args.inputs_csv,require_demand=False),g_residual=r)
            write_forecast(directory,result)
            if args.observations_csv:
                y=pd.read_csv(args.observations_csv,index_col='Datetime',parse_dates=['Datetime']).Demand
                result=comparison.score_comparison(result,y);write_forecast(directory,result,prefix='scored')
            plot(directory,result)
        print(f"G/WM {args.command}: {result['status']}; {result['seconds']:.2f} s",flush=True)
        return 0 if result['status'] in ('complete','complete-development-comparison') else 2
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        print(f'G/WM failed: {exc}',file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())

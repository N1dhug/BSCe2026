"""Explicit empirical output-error companion; deterministic G is unchanged."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import pandas as pd

from src.method import bm_1_s3_interval as interval
from src.method import bm_1_s3_residual as residual
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


def plot(directory,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=result['predictions'];fig,ax=plt.subplots(figsize=(12,4.5))
    ax.fill_between(p.index,p.observed_lower,p.observed_upper,alpha=.2,label='95% empirical output-error band (development)')
    ax.plot(p.index,p.prediction,label='G physical hourly mean')
    ax.plot(p.index,p.observed_mean,label='G + historical mean error')
    if 'observed' in p:ax.scatter(p.index,p.observed,s=8,label='Measured')
    ax.set_ylabel('Mean power (raw units; unit unverified)')
    ax.set_xlabel('Europe/Copenhagen physical hour start')
    ax.set_title('Historical empirical errors; coverage not independently calibrated')
    ax.legend();fig.tight_layout();fig.savefig(directory/'residual_week.png',dpi=150);plt.close(fig)


def main(argv=None):
    parser=ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('fit');fit.add_argument('--raw-csv',required=True)
    fit.add_argument('--seconds',type=float,default=30.)
    predict=commands.add_parser('predict');predict.add_argument('--residual-snapshot',required=True)
    predict.add_argument('--inputs-csv',required=True);predict.add_argument('--observations-csv')
    for p in (fit,predict):p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    try:
        s=interval.load_snapshot(args.snapshot)
        if args.command=='fit':
            result=residual.fit_residual(s,raw_csv(args.raw_csv),seconds=args.seconds)
            _atomic_report(directory/'residual_snapshot.json',result['snapshot'])
            result['historical_replay'].to_csv(directory/'historical_replay_normalized.csv',index_label='Datetime')
            _atomic_report(directory/'fit_report.json',{k:v for k,v in result.items() if k not in ('snapshot','historical_replay')})
        else:
            r=json.loads(Path(args.residual_snapshot).read_text(encoding='utf-8'))
            result=residual.predict_residual(s,r,raw_csv(args.inputs_csv,require_demand=False))
            # Persist predictions before reading any target observations.
            write_forecast(directory,result)
            if args.observations_csv:
                y=pd.read_csv(args.observations_csv,index_col='Datetime',parse_dates=['Datetime']).Demand
                result=residual.score_residual(result,y);write_forecast(directory,result,prefix='scored')
            plot(directory,result)
        print(f"G-S3 residual {args.command}: {result['status']}; {result['seconds']:.2f} s",flush=True)
        return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        print(f'G residual failed: {exc}',file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())

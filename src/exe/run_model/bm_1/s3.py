"""Deterministic G: hourly-average fit and forecast with C, k, Xi and g.

The state, response, observation operator and fitted baseline are the existing
S3 kernels. Each forecast day starts from its own frozen weekday median.
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

from src.method import bm_1_s3_interval as interval
from src.exe.run_model.bm_1.s3_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.tools.bm_1_g_reports import write_forecast


def parser():
    root=ArgumentParser(description=__doc__)
    commands=root.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('fit',help='Fit the deterministic model on historical hourly means.')
    fit.add_argument('--raw-csv',required=True)
    fit.add_argument('--origin',required=True)
    fit.add_argument('--demand-label',required=True,choices=['hour_start','hour_end'])
    fit.add_argument('--seconds',type=float,default=600.)
    fit.add_argument('--max-calls',type=int,default=3000)
    fit.add_argument('--reference',nargs=4,type=float,default=[12.,2.,.5,.5])
    fit.add_argument('--population',default='declared raw aggregate')
    fit.add_argument('--demand-unit',default='raw average-power units; physical unit unverified')
    predict=commands.add_parser('predict',help='Forecast from an own S3 interval snapshot without fit.')
    predict.add_argument('--snapshot',required=True)
    predict.add_argument('--inputs-csv',required=True)
    predict.add_argument('--observations-csv')
    for command in (fit,predict):command.add_argument('--output',required=True)
    return root


def plot(directory,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=result['predictions']
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    axes[0].plot(p.index,p.prediction,label='G hourly average')
    axes[0].plot(p.index,p.baseline,label='Integrated baseline',alpha=.7)
    if 'observed' in p:axes[0].scatter(p.index,p.observed,s=8,label='Measured')
    axes[0].set_ylabel('Mean power (raw unit unverified)');axes[0].legend()
    axes[1].step(p.index,p.price,where='post')
    axes[1].set_ylabel('Supplied raw price')
    axes[1].set_xlabel('Europe/Copenhagen interval start')
    fig.tight_layout();fig.savefig(directory/'interval_week.png',dpi=150);plt.close(fig)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    # Preserve the deterministic spelling of the former combined CLI.
    if argv and argv[0]=='interval':argv=argv[1:]
    args=parser().parse_args(argv)
    directory=Path(args.output)
    try:
        directory.mkdir(parents=True,exist_ok=False)
    except OSError as exc:
        print(f'G output directory unavailable: {exc}',file=sys.stderr,flush=True)
        return 2
    try:
        if args.command=='fit':
            result=interval.fit_interval(raw_csv(args.raw_csv),args.origin,demand_label=args.demand_label,
                seconds=args.seconds,max_calls=args.max_calls,reference=args.reference,
                population=args.population,demand_unit=args.demand_unit,
                checkpoint=lambda value:_atomic_report(directory/'fit_checkpoint.json',
                    interval.contract._plain(value),replace_existing=True))
            _atomic_report(directory/'fit_result.json',
                interval.contract._plain({k:v for k,v in result.items() if k!='snapshot'}))
            if result['snapshot'] is not None:
                interval.save_snapshot(directory/'interval_snapshot.json',result['snapshot'])
        else:
            result=interval.predict_interval(interval.load_snapshot(args.snapshot),
                raw_csv(args.inputs_csv,require_demand=False))
            write_forecast(directory,result)
            if args.observations_csv:
                import pandas as pd
                observations=pd.read_csv(args.observations_csv,index_col='Datetime',parse_dates=['Datetime'])
                if 'Demand' not in observations:
                    raise ValueError('Observation CSV needs Datetime,Demand.')
                y=observations['Demand']
                result=interval.score_interval(result,y)
                write_forecast(directory,result,prefix='scored')
            plot(directory,result)
        print(f"G {args.command}: {result['status']}; {result['seconds']:.2f} s",flush=True)
        return 0 if result['status']=='complete' else 2
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',
            {'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        print(f'G {args.command} failed: {exc}',file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())

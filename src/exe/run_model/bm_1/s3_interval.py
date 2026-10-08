"""G-S3 hourly-average fit, forecast, integrated process moments and diagnosis."""
from argparse import ArgumentParser
from pathlib import Path
import sys

from src.method import bm_1_s3_interval as interval
from src.method.bm_1_s3_interval_process import process_interval,score_interval_process,diagnose_interval
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


def plot_interval(directory,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=result['predictions'];fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    axes[0].plot(p.index,p.prediction,label='G-S3 hourly average')
    axes[0].plot(p.index,p.baseline,label='Integrated baseline',alpha=.7)
    if 'process_mean' in p:axes[0].plot(p.index,p.process_mean,ls='--',label='Process E[hourly average]')
    if 'observed' in p:axes[0].scatter(p.index,p.observed,s=8,label='Measured hourly consumption')
    axes[0].set_ylabel(result['manifest']['demand_unit']);axes[0].legend()
    if 'process_variance' in p:
        axes[1].plot(p.index,p.process_variance**.5)
        axes[1].set_ylabel('Process-only interval standard deviation')
    else:axes[1].step(p.index,p.price,where='post');axes[1].set_ylabel('Declared raw price')
    axes[1].set_xlabel('Physical hourly intervals, Europe/Copenhagen start labels')
    fig.suptitle('G-S3: hourly-average observation operator; conditional development')
    fig.tight_layout();fig.savefig(directory/'interval_week.png',dpi=150);plt.close(fig)


def main(argv=None):
    parser=ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('fit');fit.add_argument('--raw-csv',required=True);fit.add_argument('--origin',required=True)
    fit.add_argument('--demand-label',required=True,choices=['hour_start','hour_end'])
    fit.add_argument('--seconds',type=float,default=600.);fit.add_argument('--max-calls',type=int,default=3000)
    fit.add_argument('--reference',nargs=4,type=float,default=[12.,2.,.5,.5])
    fit.add_argument('--population',default='declared raw aggregate')
    fit.add_argument('--demand-unit',default='raw average-power units; physical unit unverified')
    predict=commands.add_parser('predict');predict.add_argument('--inputs-csv',required=True)
    predict.add_argument('--observations-csv');predict.add_argument('--sigma-x',type=float)
    predict.add_argument('--cells',type=int,default=128);predict.add_argument('--max-step',type=float,default=1/24)
    predict.add_argument('--seconds',type=float,default=120.)
    audit=commands.add_parser('diagnose');audit.add_argument('--raw-csv',required=True)
    audit.add_argument('--seconds',type=float,default=60.)
    for command in (predict,audit):command.add_argument('--snapshot',required=True)
    for command in (fit,predict,audit):command.add_argument('--output',required=True)
    args=parser.parse_args(argv);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    try:
        if args.command=='fit':
            result=interval.fit_interval(raw_csv(args.raw_csv),args.origin,demand_label=args.demand_label,
                seconds=args.seconds,max_calls=args.max_calls,reference=args.reference,
                population=args.population,demand_unit=args.demand_unit,
                checkpoint=lambda value:_atomic_report(directory/'fit_checkpoint.json',interval.point._plain(value),replace_existing=True))
            _atomic_report(directory/'fit_result.json',interval.point._plain({k:v for k,v in result.items() if k!='snapshot'}))
            if result['snapshot'] is not None:interval.save_snapshot(directory/'interval_snapshot.json',result['snapshot'])
        elif args.command=='diagnose':
            result=diagnose_interval(interval.load_snapshot(args.snapshot),raw_csv(args.raw_csv),seconds=args.seconds)
            _atomic_report(directory/'calibration_audit.json',interval.point._plain(result))
        else:
            s=interval.load_snapshot(args.snapshot);inputs=raw_csv(args.inputs_csv,require_demand=False)
            result=interval.predict_interval(s,inputs) if args.sigma_x is None else process_interval(s,inputs,
                sigma_x=args.sigma_x,cells=args.cells,max_step=args.max_step,seconds=args.seconds)
            write_forecast(directory,result)
            if args.observations_csv:
                import pandas as pd
                y=pd.read_csv(args.observations_csv,index_col='Datetime',parse_dates=['Datetime']).Demand
                result=interval.score_interval(result,y) if args.sigma_x is None else score_interval_process(result,y)
                write_forecast(directory,result,prefix='scored')
            plot_interval(directory,result)
        print(f"G-S3 interval {args.command}: {result['status']}; {result['seconds']:.2f} s",flush=True)
        return 0 if result['status'] in ('complete','complete-interval-process','candidate-needs-nonlinear-validation') else 2
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        print(f'G-S3 interval failed: {exc}',file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())

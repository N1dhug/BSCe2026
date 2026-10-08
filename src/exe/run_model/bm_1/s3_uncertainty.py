"""Fixed-S3 process scenarios and a separate training-only calibration audit."""
from argparse import ArgumentParser
from pathlib import Path
import sys

from src.method.bm_1_s3_forecast import load_snapshot, _plain
from src.method.bm_1_s3_uncertainty import (
    diagnose_calibration, process_forecast, score_process_forecast)
from src.exe.run_model.bm_1.s3_forecast import write_forecast
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report


def plot_process(directory, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=result['predictions']
    fig,ax=plt.subplots(figsize=(12,4.5))
    ax.fill_between(p.index,p.process_lower95,p.process_upper95,alpha=.3,label='Process-only pointwise 95% band')
    ax.plot(p.index,p.process_mean,label='E[D(X)]')
    ax.plot(p.index,p.prediction,ls='--',lw=1,label='Deterministic S3')
    if 'observed' in p:ax.scatter(p.index,p.observed,s=8,label='Measured')
    ax.set_ylabel(result['manifest']['demand_unit']);ax.set_xlabel('Europe/Copenhagen local time')
    ax.set_title('G-S3 conditional process scenario: explicit sigma, no historical calibration acceptance')
    ax.legend();fig.tight_layout();fig.savefig(directory/'process_week.png',dpi=150);plt.close(fig)


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    audit=commands.add_parser('diagnose',help='Audit historical variance/support; never supplies forecast sigma.')
    audit.add_argument('--raw-csv',required=True)
    audit.add_argument('--seconds',type=float,default=60.)
    predict=commands.add_parser('predict',help='Input-only, numerically checked process scenario.')
    predict.add_argument('--inputs-csv',required=True)
    predict.add_argument('--sigma-x',required=True,type=float,help='Explicit amplitude per sqrt(hour); not fitted.')
    predict.add_argument('--cells',type=int,default=128)
    predict.add_argument('--max-step',type=float,default=1/24)
    predict.add_argument('--seconds',type=float,default=120.)
    predict.add_argument('--observations-csv')
    for command in (audit,predict):
        command.add_argument('--snapshot',required=True)
        command.add_argument('--output',required=True)
    args=parser.parse_args(argv);directory=Path(args.output)
    directory.mkdir(parents=True,exist_ok=False)
    try:
        snapshot=load_snapshot(args.snapshot)
        if args.command=='diagnose':
            result=diagnose_calibration(snapshot,raw_csv(args.raw_csv),seconds=args.seconds)
            _atomic_report(directory/'calibration_audit.json',_plain(result))
            print(f"S3 calibration audit: {result['status']}; {result['seconds']:.2f} s",flush=True)
            return 0 if result['status']=='candidate-needs-nonlinear-validation' else 2
        result=process_forecast(snapshot,raw_csv(args.inputs_csv,require_demand=False),
            sigma_x=args.sigma_x,cells=args.cells,max_step=args.max_step,seconds=args.seconds)
        # Observations are opened only after the conditional forecast is durable.
        write_forecast(directory,result)
        if args.observations_csv:
            import pandas as pd
            observed=pd.read_csv(args.observations_csv,index_col='Datetime',parse_dates=['Datetime'])
            result=score_process_forecast(result,observed.Demand)
            write_forecast(directory,result,prefix='scored')
        plot_process(directory,result)
        print(f"S3 process: {result['status']}; {result['seconds']:.2f} s; scoring={result['scoring']['status']}",flush=True)
        return 0 if result['status']=='complete-process-scenario' else 2
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        print(f'S3 uncertainty {args.command} failed: {exc}',file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())

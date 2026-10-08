"""One fixed G/WM origin: own fits and input-only frozen forecast, no target score."""
from argparse import ArgumentParser
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
from time import perf_counter

from src.method import bm_1_s3_interval as interval
from src.method import bm_1_s3_residual as residual
from src.method import g_wm_hourly as comparison
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast

CALIBRATION_ORIGINS=('2022-02-14','2022-02-21','2022-02-28')
EVALUATION_ORIGINS=('2022-03-07','2022-03-14','2022-03-21')


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--raw-csv',required=True);parser.add_argument('--inputs-csv',required=True)
    parser.add_argument('--origin',choices=CALIBRATION_ORIGINS+EVALUATION_ORIGINS,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    started=perf_counter();phase='calibration' if args.origin in CALIBRATION_ORIGINS else 'evaluation'
    _atomic_report(directory/'invocation.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'origin':args.origin,'phase':phase,'protocol':'G-multiperiod-prequential-bands-v1',
        'budgets':{'G_seconds':120,'G_calls':1000,'residual_seconds':30,'WM_seconds':60,'WM_calls':400},
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(args.raw_csv),Path(args.inputs_csv))},
        'target_observation_file_opened':False,'automatic_retry':False})
    try:
        raw=raw_csv(args.raw_csv)
        print(f'{args.origin}: own G fit',flush=True)
        g=interval.fit_interval(raw,args.origin,demand_label='hour_start',seconds=120.,max_calls=1000,
            population='MED hourly mean over available measured meters',
            demand_unit='raw average-power units; physical unit unverified',
            checkpoint=lambda x:_atomic_report(directory/'g_fit_checkpoint.json',interval.point._plain(x),replace_existing=True))
        _atomic_report(directory/'g_fit_report.json',interval.point._plain({k:v for k,v in g.items() if k!='snapshot'}))
        if g['status']!='complete':raise RuntimeError('G origin fit unfinished; no forecast or automatic retry.')
        parent=g['snapshot'];interval.save_snapshot(directory/'g_snapshot.json',parent)
        print(f'{args.origin}: own output-error replay',flush=True)
        errors=residual.fit_residual(parent,raw,seconds=30.)
        _atomic_report(directory/'g_residual_snapshot.json',errors['snapshot'])
        _atomic_report(directory/'g_residual_report.json',{k:v for k,v in errors.items() if k not in ('snapshot','historical_replay')})
        errors['historical_replay'].to_csv(directory/'g_historical_replay_normalized.csv',index_label='Datetime')
        print(f'{args.origin}: own WM interval fit',flush=True)
        wm=comparison.fit_wm(parent,raw,seconds=60.,max_calls=400,
            checkpoint=lambda x:_atomic_report(directory/'wm_fit_checkpoint.json',x,replace_existing=True))
        _atomic_report(directory/'wm_fit_report.json',{k:v for k,v in wm.items() if k not in ('snapshot','historical_replay')})
        if wm['status']!='complete':raise RuntimeError('WM origin fit unfinished; no forecast or automatic retry.')
        snapshot=comparison.validate_wm(wm['snapshot'],parent);_atomic_report(directory/'wm_snapshot.json',snapshot)
        wm['historical_replay'].to_csv(directory/'wm_historical_replay_normalized.csv',index_label='Datetime')
        inputs=raw_csv(args.inputs_csv,require_demand=False)
        forecast=comparison.predict_comparison(parent,snapshot,inputs,g_residual=errors['snapshot'])
        write_forecast(directory,forecast)
        frozen_files=('g_snapshot.json','g_residual_snapshot.json','wm_snapshot.json','forecast.json','forecast.csv','forecast_normalized.csv')
        _atomic_report(directory/'forecast_freeze.json',{'status':'frozen-input-only','origin':args.origin,'phase':phase,
            'utc':datetime.now(timezone.utc).isoformat(),'total_seconds':perf_counter()-started,
            'target_observation_file_opened':False,'G_calls':g['calls_used'],'WM_calls':wm['calls'],
            'model_seconds':{'G':g['seconds'],'residual':errors['seconds'],'WM':wm['seconds'],'forecast':forecast['seconds']},
            'artifact_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in frozen_files}})
        print(f"{args.origin}: frozen {len(forecast['predictions'])} physical hours; {perf_counter()-started:.2f}s",flush=True)
        return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','origin':args.origin,'phase':phase,
            'seconds':perf_counter()-started,'error_type':type(exc).__name__,'error':str(exc),'retry':False})
        print(f'G period failed: {exc}',file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())

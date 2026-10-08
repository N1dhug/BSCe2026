"""Fit five preceding matching weekdays and freeze a sixth-week G/WM forecast."""
from argparse import ArgumentParser
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from time import perf_counter
import sys

from src.method import g_weekday_rolling as model
from src.method.g_prequential_bands import CALIBRATION_ORIGINS,EVALUATION_ORIGINS
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    for name in ('raw-csv','inputs-csv','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--origin',choices=CALIBRATION_ORIGINS+EVALUATION_ORIGINS,required=True)
    parser.add_argument('--baseline-policy',choices=('weekday','shared'),default='weekday')
    args=parser.parse_args(argv);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    begun=perf_counter();phase='calibration' if args.origin in CALIBRATION_ORIGINS else 'evaluation'
    _atomic_report(directory/'invocation.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'origin':args.origin,'phase':phase,'protocol':model.PROTOCOL,'baseline_policy':args.baseline_policy,
        'budgets_per_weekday':{'G_seconds':30,'G_calls':750,'WM_seconds':10,'WM_calls':400},
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(args.raw_csv),Path(args.inputs_csv))},
        'target_observation_file_opened':False,'automatic_retry':False})
    try:
        def checkpoint(record):
            _atomic_report(directory/f"weekday_{record['weekday']}_{record['stage']}.json",model.interval.point._plain(record),replace_existing=True)
            if record['stage']=='complete':print(f"{args.origin}: weekday {record['weekday']} frozen",flush=True)
        fitted=model.fit_rolling(raw_csv(args.raw_csv),args.origin,baseline_policy=args.baseline_policy,checkpoint=checkpoint)
        _atomic_report(directory/'fit_report.json',model.interval.point._plain({k:v for k,v in fitted.items() if k not in ('snapshot','historical_replay')}))
        if fitted['status']!='complete':raise RuntimeError('Weekday fit incomplete; no automatic retry or pooled fallback.')
        _atomic_report(directory/'weekday_snapshot.json',fitted['snapshot'])
        fitted['historical_replay'].to_csv(directory/'historical_replay_normalized.csv',index_label='Datetime')
        forecast=model.predict_comparison(fitted['snapshot'],raw_csv(args.inputs_csv,require_demand=False));write_forecast(directory,forecast)
        files=('weekday_snapshot.json','forecast.json','forecast.csv','forecast_normalized.csv')
        _atomic_report(directory/'forecast_freeze.json',{'status':'frozen-input-only','origin':args.origin,'phase':phase,
            'utc':datetime.now(timezone.utc).isoformat(),'total_seconds':perf_counter()-begun,
            'target_observation_file_opened':False,'G_calls':sum(d['G']['calls'] for d in fitted['snapshot']['days']),
            'WM_calls':sum(d['WM']['calls'] for d in fitted['snapshot']['days']),
            'artifact_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files}})
        print(f"{args.origin}: frozen {len(forecast['predictions'])} hours, {perf_counter()-begun:.2f}s",flush=True);return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','origin':args.origin,'seconds':perf_counter()-begun,
            'error':str(exc),'error_type':type(exc).__name__,'automatic_retry':False})
        print(str(exc),file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())

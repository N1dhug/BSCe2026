"""Freeze earlier-week band calibration, then score three later rolling weeks."""
from argparse import ArgumentParser
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from src.method import g_prequential_bands as bands
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


def load_forecast(directory):
    directory=Path(directory);receipt=json.loads((directory/'forecast_freeze.json').read_text(encoding='utf-8'))
    if receipt['status']!='frozen-input-only' or receipt['target_observation_file_opened']:
        raise ValueError('Require input-only frozen forecast receipt.')
    for name,digest in receipt['artifact_sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen forecast artefact changed: '+name)
    result=json.loads((directory/'forecast.json').read_text(encoding='utf-8'))
    for key,name in (('predictions','forecast.csv'),('normalized_predictions','forecast_normalized.csv')):
        result[key]=pd.read_csv(directory/name,index_col=0,parse_dates=[0],float_precision='round_trip')
    origin,_=bands.validate_forecast(result)
    if str(origin.date())!=receipt['origin']:raise ValueError('Receipt origin disagrees with forecast.')
    return result


def observations(path):
    return pd.read_csv(path,index_col='Datetime',parse_dates=['Datetime'],float_precision='round_trip')['Demand']


def main(argv=None):
    parser=ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    for command in ('calibrate','evaluate'):
        part=commands.add_parser(command);part.add_argument('--forecasts',nargs=3,required=True)
        part.add_argument('--observations',nargs=3,required=True);part.add_argument('--output',required=True)
        if command=='evaluate':part.add_argument('--calibration',required=True)
    args=parser.parse_args(argv);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    _atomic_report(directory/'invocation.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'command':args.command,'protocol':bands.PROTOCOL,'source_sha256':bands.source_hashes(),
        'forecasts':args.forecasts,'observations':args.observations,'automatic_retry':False})
    try:
        forecasts=[load_forecast(path) for path in args.forecasts]
        if args.command=='calibrate':
            fitted=bands.calibrate([{'forecast':f,'observations':observations(p)} for f,p in zip(forecasts,args.observations)])
            _atomic_report(directory/'calibration_snapshot.json',fitted['snapshot'])
            fitted['scores'].to_csv(directory/'calibration_scores.csv',index_label='Datetime')
            _atomic_report(directory/'calibration_freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
                'evaluation_observations_used':False,'artifact_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                    for name in ('calibration_snapshot.json','calibration_scores.csv')},
                'observation_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.observations}})
            print('Three earlier forecast weeks calibrated; record frozen.',flush=True);return 0
        record=bands.validate_calibration(json.loads(Path(args.calibration).read_text(encoding='utf-8')))
        calibrated=[bands.apply_calibration(f,record) for f in forecasts]
        receipt={}
        # Save ALL three mean/band forecasts before opening ANY target Y file.
        for forecast in calibrated:
            name=pd.Timestamp(forecast['manifest']['cutoff']).strftime('%Y%m%d');target=directory/name;target.mkdir()
            write_forecast(target,forecast)
            receipt[name]={file:hashlib.sha256((target/file).read_bytes()).hexdigest()
                for file in ('forecast.json','forecast.csv','forecast_normalized.csv')}
        _atomic_report(directory/'evaluation_forecasts_freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
            'target_observation_files_opened':False,'calibration_sha256':bands.residual.snapshot_fingerprint(record),'forecasts':receipt})
        scored=[]
        for forecast,path in zip(calibrated,args.observations):
            result=bands.score_calibrated(forecast,observations(path));scored.append(result)
            name=pd.Timestamp(result['manifest']['cutoff']).strftime('%Y%m%d');write_forecast(directory/name,result,prefix='scored')
        report=bands.pooled_report(scored);_atomic_report(directory/'evaluation_report.json',report)
        pd.concat([r['predictions'] for r in scored]).to_csv(directory/'evaluation_scored.csv',index_label='Datetime')
        print('Three later weeks scored with one unchanged calibration record.',flush=True);return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','error':str(exc),'error_type':type(exc).__name__,'automatic_retry':False})
        print(str(exc),file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())

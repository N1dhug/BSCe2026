"""Recover a saved complete weekday fit via explicit numerical export, no refit."""
from argparse import ArgumentParser
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

from src.method import g_weekday_export as model
from src.method.g_prequential_bands import CALIBRATION_ORIGINS
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    for name in ('snapshot','raw-csv','inputs-csv','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    _atomic_report(directory/'invocation.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'snapshot':args.snapshot,'snapshot_file_sha256':hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(),
        'automatic_retry':False,'parameter_refit':False,'source_sha256':model.source_hashes(),'seconds_cap':15})
    try:
        parent=model.rolling.validate_snapshot(json.loads(Path(args.snapshot).read_text(encoding='utf-8')))
        exported=model.refine_for_export(parent,raw_csv(args.raw_csv))
        _atomic_report(directory/'weekday_snapshot.json',parent)
        _atomic_report(directory/'weekday_export_snapshot.json',exported['snapshot'])
        exported['historical_replay'].to_csv(directory/'export_replay_normalized.csv',index_label='Datetime')
        forecast=model.predict_comparison(exported['snapshot'],raw_csv(args.inputs_csv,require_demand=False));write_forecast(directory,forecast)
        origin=str(model.pd.Timestamp(parent['origin']).date());files=('weekday_snapshot.json','weekday_export_snapshot.json','forecast.json','forecast.csv','forecast_normalized.csv')
        _atomic_report(directory/'forecast_freeze.json',{'status':'frozen-input-only','origin':origin,
            'phase':'calibration' if origin in CALIBRATION_ORIGINS else 'evaluation','utc':datetime.now(timezone.utc).isoformat(),
            'target_observation_file_opened':False,'parameter_refit':False,'G_calls':0,'WM_calls':0,
            'numerical_export_seconds':exported['snapshot']['seconds'],'original_fit_snapshot':args.snapshot,
            'artifact_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files}})
        print('Saved weekday fit exported on60/120grid; zero optimizer calls.',flush=True);return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','error':str(exc),'parameter_refit':False})
        print(str(exc),file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())

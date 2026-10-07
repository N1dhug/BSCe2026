"""Data-only PC export after extension forecast freeze, using the v2 campaign."""
from time import perf_counter
BEGUN=perf_counter()-.25
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from root.f4 import run_h2_same_day as s
from root.f4.export_h2_final_observations import export
c=s.c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-workbook',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--launcher-seconds',type=float,default=0.)
    a=p.parse_args()
    c.require(0.<=a.launcher_seconds<=60.,'Invalid launcher charge.')
    archive=a.output.with_name(a.output.name+'_review.zip')
    c.require(not archive.exists(),'Preserve earlier exports.')
    decision=c.load(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    outer=c.load(a.frozen_run/'extension_execution.json')
    selection=c.load(a.frozen_run/'selection.json')
    c.require(outer['identity']==c.digest({k:v for k,v in outer.items() if k!='identity'})
              and selection['execution_identity']==outer['identity']
              and selection['execution_manifest_sha256']==c.sha(a.frozen_run/'extension_execution.json'),
              'Selection does not belong to the frozen extension.')
    with s.reservation(a.campaign,decision,'final-data-export',begun=BEGUN-a.launcher_seconds,requested=180.):
        report=export(a.frozen_run,a.output,a.input_workbook)
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('export_report.json','final_observations.npz','manifest.json'):
                z.write(a.output/name,name)
            z.write(__file__,'export_h2_cap_extension_final.py')
    with zipfile.ZipFile(archive,'a') as z:z.write(a.campaign,'ledger.json')
    print(json.dumps(dict(status='exported',review_zip=str(archive),report=report)),flush=True)


if __name__=='__main__':main()

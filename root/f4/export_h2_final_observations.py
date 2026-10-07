"""PC data-only bridge after a sealed open forecast; never executes a model.

Use when the fitted run lives on another machine. Cross-platform data export
does not relax the numerical runner's same-source/environment resume rule.
"""
from time import perf_counter
BEGUN=perf_counter()-.25
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'


def checked_frozen_run(directory):
    from src.exe.run_model import bm_1_h2_compact as c
    directory=Path(directory); manifest=c.load(directory/'manifest.json')
    c.require(manifest['identity']==c.digest({k:v for k,v in manifest.items() if k!='identity'}), 'Manifest identity changed.')
    packet=c.load(directory/'packet.json'); selection=c.load(directory/'selection.json')
    c.require(packet['identity']==manifest['packet_identity']
        and packet['identity']==c.digest({k:v for k,v in packet.items() if k!='identity'}), 'Packet identity changed.')
    c.require(selection['identity']==manifest['identity'] and selection['final_observations_decoded']==0,
        'Selection was not frozen before final observations.')
    frozen=c._read_array_checkpoint((directory/'forecast_frozen.npz').read_bytes())
    c.require(frozen['identity']==manifest['identity'] and frozen['selection_sha256']==c.sha(directory/'selection.json')
        and frozen['checksum']==c.fingerprint({k:v for k,v in frozen.items() if k!='checksum'})
        and frozen['conditional_forecast_precision_passed'] is True
        and frozen['final_observations_decoded']==frozen['assimilated_observations']==0,
        'A complete validated open forecast must exist before final data export.')
    # Source can have an exact Windows text checkout; this operation does not
    # resume a floating-point state or use the PC's runtime to price a model.
    for p,h in manifest['source_sha256'].items():
        raw=(ROOT/p).read_bytes(); lf=raw.replace(b'\r\n',b'\n')
        c.require(h in {hashlib.sha256(v).hexdigest() for v in (raw,lf,lf.replace(b'\n',b'\r\n'))},
            'Export adapter source differs from the frozen run: '+p)
    return manifest,packet,frozen


def export(directory,output,workbook):
    from src.exe.run_model import bm_1_h2_compact as c
    manifest,packet,frozen=checked_frozen_run(directory)
    output=Path(output)
    c.require(not output.exists(),'Preserve every final-data export; no overwrite.')
    output.mkdir(parents=True)
    for name in ('manifest.json','packet.json','selection.json','forecast_frozen.npz'):
        shutil.copyfile(Path(directory)/name,output/name)
    engine=SimpleNamespace(directory=output,manifest=manifest,data={'packet':packet})
    observations,_=c.load_final_observations(engine,workbook)
    report=dict(schema='f4-h2-final-data-export-v1',identity=manifest['identity'],
        forecast_sha256=c.sha(output/'forecast_frozen.npz'),
        observation_sha256=c.sha(output/'final_observations.npz'),
        input_sha256=packet['input_sha256']['input'],selected_hours=168,
        observed_hours=int(observations['observed'].sum()),fixed_households=13,
        model_propagations=0,future_realized_prices_decoded=0,
        fitted_curve_changed=False,scientific_acceptance=False,blind_h3_acceptance=False,
        adapter_sha256=c.sha(Path(__file__)))
    c.write(output/'export_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-workbook',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--launcher-seconds',type=float,default=0.)
    args=parser.parse_args()
    from src.exe.run_model import bm_1_h2_compact as c
    archive=args.output.with_name(args.output.name+'_review.zip')
    c.require(not archive.exists(),'Preserve prior export receipt.')
    c.require(0.<=args.launcher_seconds<=60.,'Invalid launcher charge.')
    # Declared final validation I/O, already in the production auxiliary graph.
    with c.reservation(args.campaign,180.,'final-data-export',begun=BEGUN-args.launcher_seconds):
        report=export(args.frozen_run,args.output,args.input_workbook)
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('export_report.json','final_observations.npz','manifest.json'):
                z.write(args.output/name,name)
            z.write(__file__,'export_h2_final_observations.py')
    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_STORED) as z:
        z.write(args.campaign,'ledger.json')
    print(json.dumps(dict(status='exported',review_zip=str(archive),report=report)),flush=True)


if __name__=='__main__':
    main()

"""Immutable-source F4 compact stages; no automatic retry or final-data access."""
from time import perf_counter
BEGUN = perf_counter()-.25
import argparse
import json
import os
from pathlib import Path
import re
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','check','probe','run','final'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=ROOT/'files/output/bm_1/runs/estimation/f4_h2_remaining_budget.json')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--max-seconds',type=float,default=180.)
    parser.add_argument('--launcher-seconds',type=float,default=0.)
    parser.add_argument('--input-workbook',type=Path)
    parser.add_argument('--resume',action='store_true',help='Explicitly resume a paused/crashed stage; never a numerical failure.')
    args = parser.parse_args()
    from src.exe.run_model import bm_1_h2_compact as c
    c.require(re.fullmatch('[0-9a-f]{40}',args.source_commit) is not None,'Declare exact immutable Git source.')
    c.require(0. <= args.launcher_seconds <= 60.,'Invalid launcher startup time.')
    c.require(args.stage != 'final' or args.input_workbook is not None,'Final stage requires the existing private workbook.')
    c.require(args.stage == 'final' or args.input_workbook is None,'Pre-final stages must not receive a final-data loader.')
    output = args.output.resolve(); campaign = args.campaign.resolve()
    archive = output.with_name(output.name+'_'+args.stage+'_review.zip')
    if args.resume:
        prior_report = output/(args.stage+'_report.json')
        c.require(not prior_report.exists() or c.load(prior_report).get('status') == 'paused',
            'Only a paused/crashed stage can resume; completed/failed stages are preserved.')
        attempt = 1
        while archive.exists():
            attempt += 1
            archive = output.with_name(output.name+'_'+args.stage+f'_resume{attempt}_review.zip')
    else:
        c.require(not archive.exists(),'Previous receipt exists; preserve it. Resume only explicitly after a pause.')
    # Distinct stage receipts preserve failed work. An interrupted invocation
    # without a receipt can resume. A completed/paused receipt is never overwritten.
    result = None; engine = None; failed = False
    native = args.stage in ('prepare','check','probe')
    with c.reservation(campaign,args.max_seconds,'staged-'+args.stage,native=native,
            begun=BEGUN-args.launcher_seconds) as allocation, c._RunLock(output):
        try:
            c.require(not (output/'FAILED.json').exists(),'Preserve failed run; no automatic retry.')
            if args.stage == 'check':
                from root.f4.check_h2_compact_runner import control
                result = control(output)
                c.write(output/'check_source.json',c.source_hashes())
            else:
                data = c.receipt()
                manifest = c.prepare(output,data,args.source_commit)
                engine = c.Engine(output,data,manifest,allocation)
                if args.stage == 'prepare':
                    result = dict(status='prepared', identity=manifest['identity'],
                        historical_fit_ready=False, next_stage='probe', final_observations_decoded=0)
                elif args.stage == 'probe':
                    result = c.probe(engine)
                elif args.stage == 'run':
                    families, selected = c.fit_and_select(engine)
                    frozen = c.forecast_week(engine,families,selected)
                    result = dict(status='prefinal_completed', selected_family=selected['selected_family'],
                        conditional_forecast_precision_passed=True, next_stage='final', final_observations_decoded=0)
                else:
                    c.load_probe(engine)
                    result = c.final_evaluation(engine,args.input_workbook)
                c.require(c.source_hashes() == manifest['source_sha256'],'Source changed during invocation.')
            result = dict(result, scientific_acceptance=False, blind_h3_acceptance=False, automatic_retry=False)
        except (Exception,KeyboardInterrupt) as exc:
            failed = True; paused = isinstance(exc,(TimeoutError,KeyboardInterrupt))
            result = dict(status='paused' if paused else 'failed', reason=str(exc), traceback=traceback.format_exc(),
                scientific_acceptance=False, blind_h3_acceptance=False, automatic_retry=False)
            if not paused:
                c.write(output/'FAILED.json',result,replace=(output/'FAILED.json').exists())
        finally:
            if engine is not None:
                engine.close()
        c.write(output/(args.stage+'_report.json'),result,replace=(output/(args.stage+'_report.json')).exists())
        # Inner .npz files are already compressed; avoid another costly pass.
        with zipfile.ZipFile(archive.with_suffix('.zip.partial'),'w',compression=zipfile.ZIP_STORED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.name != 'run.lock' and not p.name.endswith('.partial'):
                    z.write(p,p.relative_to(output).as_posix())
        os.replace(archive.with_suffix('.zip.partial'),archive)
    raw = campaign.read_bytes(); (output/'ledger.json').write_bytes(raw)
    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_STORED) as z:
        z.writestr('closed_ledger.json',raw)
    ledger = json.loads(raw)
    print(json.dumps(dict(status=result.get('status','completed'),review_zip=str(archive),
        charged_seconds=ledger['charged_seconds'],invocation_seconds=ledger['entries'][-1]['charged_seconds'])),flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

"""Explicit time-budgeted search experiment, preserving the frozen numeric engine.

The original 49-candidate worst-case graph is rejected, not silently repriced.
This outer controller freezes its own source/protocol and admission decision;
native jobs keep the unchanged numerical manifest identity and their bytes.
"""
from time import perf_counter
BEGUN=perf_counter()-.25
import argparse
import json
import os
from pathlib import Path
import re
import sys
import traceback
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'

EXTRA=('root/f4/run_h2_budgeted_search.py','root/H2_COMPACT_BUDGETED_SEARCH.md',
    'root/f4/audit_h2_linear_window.py')


def execution_manifest(engine, source_commit, audit_path):
    from src.exe.run_model import bm_1_h2_compact as c
    audit=c.load(audit_path)
    c.require(audit['status']=='passed' and audit['identity']==engine.manifest['identity']
        and audit['source_commit']==engine.manifest['source_commit'] and audit['optimizer_started'] is False,
        'Require the completed, audited pre-optimizer linear reference.')
    c.require(audit['comparison']==c.load(engine.directory/'linear_full_window.json')['comparison'],
        'Linear window evidence differs.')
    for name,h in audit['source_job_sha256'].items():
        c.require(c.sha(engine.directory/'jobs'/name)==h,'Verified baseline job changed.')
    outer=dict(schema='f4-h2-budgeted-search-execution-v1',source_commit=source_commit,
        numerical_manifest_identity=engine.manifest['identity'],
        numerical_source_commit=engine.manifest['source_commit'],
        source_sha256={p:c.sha(ROOT/p) for p in EXTRA},
        audit_sha256=c.sha(audit_path),rates=audit['rates'],
        original49_worst_case_seconds=audit['remaining49_global_max_total_seconds'],
        original49_worst_case_admitted=False,conditional_time_bounded_attempt=True,
        candidate_cap=49,all_three_families_required=True,
        time_stop_is_convergence=False,scientific_acceptance=False,blind_h3_acceptance=False)
    outer['identity']=c.digest(outer)
    path=engine.directory/'budgeted_execution.json'
    if path.exists():
        c.require(c.load(path)==outer,'Budgeted resume requires the identical controller and evidence.')
    else:
        c.write(path,outer)
        with zipfile.ZipFile(engine.directory/'budgeted_source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in outer['source_sha256'].items():
                raw=(ROOT/p).read_bytes();c.require(c.hashlib.sha256(raw).hexdigest()==h,'Controller source changed.')
                archive.writestr(p,raw)
    return outer,audit


def attempt(engine, outer, audit):
    from src.exe.run_model import bm_1_h2_compact as c
    from root.f4.measure_h2_compact_rates import work_graph
    c.load_probe(engine)
    engine.rates=audit['rates']
    charged=engine.allocation['before']+perf_counter()-engine.allocation['begun']
    graph=work_graph(engine.rates,48,24,charged)
    engine.validation_remaining=graph['protected_validation_seconds']
    # Reconstruct completed validation credit from its exact cached streams.
    # No completed native job is recomputed or assigned a new identity.
    families={1:c.validated_family(engine,c.LINEAR)}
    c.require(abs(engine.validation_remaining-audit['protected_remaining_validation_seconds'])<=1e-7,
        'Remaining validation graph does not match the independent raw audit.')
    engine.allocation['ledger']['validation_reserve_seconds']=max(
        engine.allocation['ledger']['validation_reserve_seconds'],engine.validation_remaining)
    funded=36000.-(engine.allocation['before']+perf_counter()-engine.allocation['begun'])-engine.validation_remaining-60.
    one_candidate=1.5*(48*engine.rates['base']['observed']+24*engine.rates['base']['prediction'])+300.
    c.require(funded>one_candidate,'No funded search remains after complete validation protection.')
    admission=dict(schema='f4-h2-funded-attempt-admission-v1',execution_identity=outer['identity'],
        remaining_validation_seconds=engine.validation_remaining,search_seconds_available=funded,
        original49_worst_case_admitted=False,conditional_attempt_admitted=True,
        convergence_guaranteed=False,ceiling_seconds=36000.,scientific_acceptance=False)
    # Preserve each invocation's budget arithmetic; never overwrite its predecessor.
    count=len(list(engine.directory.glob('budgeted_admission_*.json')))+1
    c.write(engine.directory/f'budgeted_admission_{count:03d}.json',admission)
    for pieces in (2,4):
        result=c.optimizer_family(engine,pieces)
        families[pieces]=c.validated_family(engine,result['candidate'])
    selected=c.select_families(families)
    selected.update(schema='f4-h2-frozen-selection-v1',identity=engine.manifest['identity'],
        execution_identity=outer['identity'],execution_manifest_sha256=c.sha(engine.directory/'budgeted_execution.json'),
        curves={str(p):families[p]['candidate'] for p in families},
        metrics={str(p):{g:c.selection_metrics(families[p]['streams'][g],engine.data['observations']) for g in ('base','fine')} for p in families},
        comparisons={str(p):families[p]['comparison'] for p in families},
        cutoff=engine.data['packet']['policy']['split']['holdout_start'],
        final_observations_decoded=0,scientific_acceptance=False,blind_h3_acceptance=False)
    path=engine.directory/'selection.json'
    if path.exists():c.require(c.load(path)==selected,'Frozen selection changed.')
    else:c.write(path,selected)
    c.forecast_week(engine,families,selected)
    return dict(status='prefinal_completed',selected_family=selected['selected_family'],
        execution_identity=outer['identity'],conditional_forecast_precision_passed=True,
        final_observations_decoded=0,scientific_acceptance=False,blind_h3_acceptance=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--max-seconds',type=float,default=30000.)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    from src.exe.run_model import bm_1_h2_compact as c
    c.require(re.fullmatch('[0-9a-f]{40}',args.source_commit) is not None,'Declare the immutable outer source commit.')
    directory=args.run.resolve();report_path=directory/'budgeted_report.json'
    c.require(not (directory/'FAILED.json').exists(),'Preserve numerical failure; no retry.')
    c.require(not (directory/'STOP').exists(),'A STOP marker is present; preserve/review the pause explicitly.')
    if report_path.exists():
        c.require(args.resume and c.load(report_path)['status']=='paused','Only an explicitly paused attempt may resume.')
    number=len(list(directory.parent.glob(directory.name+'_budgeted*_review.zip')))+1
    archive=directory.with_name(directory.name+f'_budgeted{number:03d}_review.zip')
    engine=None;failed=False
    with c.reservation(args.campaign,args.max_seconds,'budgeted-prefinal',begun=BEGUN) as allocation,c._RunLock(directory):
        try:
            data=c.receipt();old=c.load(directory/'manifest.json')
            manifest=c.prepare(directory,data,old['source_commit'])
            engine=c.Engine(directory,data,manifest,allocation)
            outer,audit=execution_manifest(engine,args.source_commit,args.audit)
            result=attempt(engine,outer,audit)
            c.require(c.source_hashes()==manifest['source_sha256'],'Numerical source changed during attempt.')
            c.require({p:c.sha(ROOT/p) for p in EXTRA}==outer['source_sha256'],'Outer controller source changed.')
        except (Exception,KeyboardInterrupt) as exc:
            failed=True;paused=isinstance(exc,(TimeoutError,KeyboardInterrupt))
            result=dict(status='paused' if paused else 'failed',reason=str(exc),traceback=traceback.format_exc(),
                scientific_acceptance=False,blind_h3_acceptance=False,automatic_retry=False)
            if not paused:c.write(directory/'FAILED.json',result)
        finally:
            if engine is not None:engine.close()
        c.write(report_path,result,replace=report_path.exists())
        with zipfile.ZipFile(archive.with_suffix('.zip.partial'),'w',compression=zipfile.ZIP_STORED) as z:
            for p in sorted(directory.rglob('*')):
                if p.is_file() and p.name!='run.lock' and not p.name.endswith('.partial'):
                    z.write(p,p.relative_to(directory).as_posix())
        os.replace(archive.with_suffix('.zip.partial'),archive)
    raw=args.campaign.read_bytes();(directory/'ledger.json').write_bytes(raw)
    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_STORED) as z:z.writestr('closed_ledger.json',raw)
    print(json.dumps(dict(status=result['status'],review_zip=str(archive),charged_seconds=json.loads(raw)['charged_seconds'])),flush=True)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()

"""Create a distinct continuation; never mutate or erase terminal evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(old, new, decision):
    old=Path(old);new=Path(new)
    if new.exists():raise FileExistsError('Use a new continuation directory.')
    failure=json.loads((old/'SAME_DAY_FAILED.json').read_text())
    result=json.loads((old/'optimizer_2_result.json').read_text())
    if failure['status']!='failed' or result['status']!=5 or result['converged'] or result['nfev']!=17:
        raise ValueError('Only the reviewed 17-call evaluation cap may be extended.')
    metadata={}
    new.mkdir();(new/'jobs').mkdir()
    for p in sorted(old.iterdir()):
        if not p.is_file() or p.name.endswith('.partial') or p.name in ('run.lock','STOP','campaign_at_checkpoint.json','durable_checkpoint.json'):
            continue
        name='predecessor_'+p.name if p.name in ('FAILED.json','SAME_DAY_FAILED.json') else p.name
        shutil.copy2(p,new/name)
        if p.name in ('SAME_DAY_FAILED.json','optimizer_2_calls.json','optimizer_2_result.json','candidate_registry.json'):
            metadata[name]=sha(p)
    jobs={}
    for p in sorted((old/'jobs').glob('*.npz')):
        shutil.copy2(p,new/'jobs'/p.name);jobs[p.name]=sha(p)
    if len(jobs)!=1402:raise ValueError('Predecessor job count differs from terminal review.')
    shutil.copy2(old/'optimizer_2_calls.json',new/'extension_optimizer_2_calls.json')
    shutil.copy2(old/'candidate_registry.json',new/'extension_candidate_registry.json')
    origin=dict(schema='f4-h2-cap-extension-origin-v1',predecessor_path=str(old),
        predecessor_manifest_identity=json.loads((old/'manifest.json').read_text())['identity'],
        decision_identity=decision['identity'],preserved_predecessor_metadata=metadata,
        reused_job_sha256=jobs,terminal_predecessor_unchanged=True)
    (new/'extension_origin.json').write_text(json.dumps(origin,separators=(',',':'))+'\n')
    return dict(reused_jobs=len(jobs),new_run=str(new),new_model_jobs=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('decision',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.old,a.new,json.loads(a.decision.read_text()))))

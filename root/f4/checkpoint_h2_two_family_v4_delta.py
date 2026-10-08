"""Read-only delta export: preserve audited raw jobs and frozen metadata dependencies."""
from time import perf_counter
BEGUN=perf_counter()
import argparse,json,hashlib,os,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from root.f4.run_h2_two_family_v4 import require,sha,load,write,numerical,verify,Lock,validate,reservation,SOURCE
def checkpoint(directory, decision, campaign, destination):
    original, c, manifest, terminal, oldest = verify(directory, decision)
    require(not destination.exists(), 'Never overwrite a checkpoint.')
    previous = sorted(directory.glob('two_family_checkpoint_*.json'))
    seen_jobs = {}; seen_adoptions = {}; prior_archives = {}; seen_metadata = {}; retained_metadata = {}
    for p in previous:
        old_index = load(p)
        require(sha(Path(old_index['archive_path'])) == old_index['archive_sha256'], 'Previous archive changed.')
        seen_jobs.update(old_index['new_jobs']); seen_adoptions.update(old_index['new_adoptions']); seen_metadata.update(old_index['metadata_sha256'])
        prior_archives[Path(old_index['archive_path']).name] = old_index['archive_sha256']
    files = {}; jobs = {}; adoption_hashes = {}
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or p.name.endswith(('.lock', '.partial')):
            continue
        name = str(p.relative_to(directory)); raw = p.read_bytes()
        if name.startswith('jobs/') and p.name in seen_jobs:
            require(hashlib.sha256(raw).hexdigest() == seen_jobs[p.name], 'Previously audited job changed.')
            continue
        if name.startswith('jobs/') and p.suffix == '.npz':
            saved = c._read_array_checkpoint(raw)
            require(saved['token']['identity'] == manifest['identity'] and c.digest(saved['token']) == p.stem
                and saved['checksum'] == c.fingerprint({k:v for k,v in saved.items() if k != 'checksum'}),
                'Recovery raw token/checksum changed.')
            c.state_check(saved['prior_log_state']); c.state_check(saved['result']['log_state'])
            jobs[p.name] = hashlib.sha256(raw).hexdigest()
        if name in seen_metadata and hashlib.sha256(raw).hexdigest() == seen_metadata[name]:
            retained_metadata[name] = seen_metadata[name]
        else:
            files[name] = raw
    for p in sorted((directory/'adopted').glob('*.json')):
        if p.name in seen_adoptions:
            require(sha(p) == seen_adoptions[p.name], 'Previously audited adoption changed.')
            continue
        adoption_hashes[p.name] = sha(p)
        receipt = c.load(p); token = receipt['original_token']
        predecessor = terminal if token['identity'] == load(terminal/'manifest.json')['identity'] else oldest
        key = receipt['original_key']; raw_path = predecessor/'jobs'/f'{key}.npz'
        saved = c._read_array_checkpoint(raw_path.read_bytes())
        require(c.sha(raw_path) == receipt['predecessor_raw_sha256']
            and saved['checksum'] == receipt['original_checksum']
            and saved['token'] == token and c.digest(token) == key
            and c.digest(receipt['requested_token']) == p.stem
            and receipt['requested_token']['identity'] == manifest['identity']
            and saved['checksum'] == c.fingerprint({k:v for k,v in saved.items() if k != 'checksum'})
            and receipt['exact_prior_fingerprint'] == c.fingerprint(saved['prior_log_state'])
            and receipt['old_checkpoint_relabelled'] is False, 'Adoption audit failed.')
    files['source/root/f4/checkpoint_h2_two_family_v4_delta.py'] = Path(__file__).read_bytes()
    files['campaign_at_checkpoint.json'] = campaign.read_bytes()
    files['approved_decision.json'] = json.dumps(decision, separators=(',', ':')).encode()
    for name in (*manifest['source_sha256'], *manifest['continuation_source_sha256'], SOURCE, decision['manifest_source_path']):
        files['source/'+name] = (ROOT/name).read_bytes()
    index = dict(schema='f4-h2-durable-filesystem-recovery-v1', new_jobs=jobs,
        metadata_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in files.items()},
        terminal_archive_dependency=decision['terminal_checkpoint_archive'],
        full_predecessor_archive_chain_required=True, previous_delta_archives=prior_archives,
        new_adoptions=adoption_hashes, previously_audited_jobs=len(seen_jobs),
        retained_metadata_dependencies=retained_metadata,
        campaign_is_snapshot=True, scientific_acceptance=False)
    destination.parent.mkdir(parents=True, exist_ok=True); partial = destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, raw in files.items():
            z.writestr(name, raw)
        z.writestr('durable_recovery.json', json.dumps(index, separators=(',', ':')))
    with zipfile.ZipFile(partial) as z:
        require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())), 'Archive corrupted.')
    with partial.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    record = dict(index,archive_path=str(destination),archive_sha256=sha(destination))
    write(directory/f'two_family_checkpoint_{len(previous)+1:03d}.json',record)
    print(json.dumps(dict(path=str(destination), observed_final_hours=(None if not (directory/'final_observations.npz').exists() else int(c._read_array_checkpoint((directory/'final_observations.npz').read_bytes())['observed'].sum())), sha256=sha(destination),
        bytes=destination.stat().st_size, new_jobs=len(jobs), numerical_propagations=0)), flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    decision=load(ROOT/'root/f4/h2_two_family_v4_approval_20261008.json')
    registration=load(ROOT/'root/f4/h2_two_family_v4_delta_registration_20261008.json')
    require(sha(Path(__file__))==registration['source_sha256'],'Exporter changed.')
    require(sha(ROOT/SOURCE)==decision['source_sha256'][SOURCE],'Controller changed.')
    campaign=ROOT/'files/output/bm_1/runs/estimation/f4_h2_two_family_v4_campaign_20261008.json'
    directory=ROOT/decision['new_run_path']
    with Lock(directory/'run.lock'):
        with reservation(campaign,decision,'two-family-efficient-delta-checkpoint',8,native=True):
            checkpoint(directory,decision,campaign,a.destination)
    return 0
if __name__=='__main__':
    result=main();sys.stdout.flush();sys.stderr.flush();os._exit(result)

"""Close a charged, read-only audit and durable continuation checkpoint.

Run only between numerical blocks. Preserve all predecessor artifacts.
"""
from time import perf_counter, time
BEGUN = perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.exe.run_model.bm_1_h2_continuation import c, verify_origin
from root.f4.run_h2_continuation import validate_campaign


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--previous', type=Path, nargs='*', default=[])
    a = p.parse_args(); run = a.run.resolve(); out = a.destination.resolve()
    decision = c.load(a.decision)
    c.require(not out.exists(), 'Never overwrite a durable checkpoint.')
    with c._RunLock(a.campaign.with_suffix('.lock')), c._RunLock(run):
        ledger = validate_campaign(c.load(a.campaign), decision)
        native = c.ordered_sum(e['charged_seconds'] for e in ledger['entries']
            if e['label'].startswith('native-development|'))
        c.require(native+30 <= 1080 and ledger['charged_seconds']+30 <= 64800,
            'Checkpoint reservation does not fit preserved ceilings.')
        c.require(ledger['entries'][-1]['status'] == 'closed', 'Close production block before export.')
        entry = dict(label='native-development|continuation-checkpoint-'+out.stem,
            status='reserved', charged_seconds=30., started_utc_seconds=time())
        ledger['entries'].append(entry)
        ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        c.write(a.campaign, ledger, replace=True)
        watchdog = threading.Timer(max(.01, 29-(perf_counter()-BEGUN)), lambda: os._exit(124))
        watchdog.daemon = True; watchdog.start()
        try:
            manifest, origin, old = verify_origin(run)
            prior_index = c.load(run/'predecessor_job_index.json')
            seen = set(); previous = {}
            for prior in a.previous:
                with zipfile.ZipFile(prior) as z:
                    idx = json.loads(z.read('durable_continuation.json'))
                    seen.update(idx['new_jobs'])
                previous[prior.name] = c.sha(prior)
            new = {}; adopted = 0
            for path in sorted((run/'jobs').glob('*.npz')):
                if path.name in seen: continue
                raw = path.read_bytes(); saved = c._read_array_checkpoint(raw)
                c.require(saved['token']['identity'] == manifest['identity']
                    and c.digest(saved['token']) == path.stem, 'New job namespace/key mismatch.')
                c.require(saved['checksum'] == c.fingerprint({k:v for k,v in saved.items() if k!='checksum'}),
                    'New job checksum mismatch.')
                c.state_check(saved['prior_log_state']); c.state_check(saved['result']['log_state'])
                new[path.name] = hashlib.sha256(raw).hexdigest()
            for path in sorted((run/'adopted').glob('*.json')):
                receipt = c.load(path); key = receipt['original_key']
                saved_path = old/'jobs'/f'{key}.npz'; saved = c._read_array_checkpoint(saved_path.read_bytes())
                c.require(c.sha(saved_path) == prior_index[key] == receipt['predecessor_raw_sha256']
                    and receipt['requested_token']['identity'] == manifest['identity']
                    and c.digest(receipt['requested_token']) == path.stem
                    and receipt['original_token'] == saved['token']
                    and c.digest(saved['token']) == key
                    and receipt['original_checksum'] == saved['checksum']
                    and saved['checksum'] == c.fingerprint({k:v for k,v in saved.items() if k!='checksum'})
                    and receipt['exact_prior_fingerprint'] == c.fingerprint(saved['prior_log_state'])
                    and receipt['old_checkpoint_relabelled'] is False,
                    'Adoption provenance mismatch.')
                adopted += 1
            if (run/'conditional_handoff.npz').exists() and not (run/'I_origin_review.npz').exists():
                from root.f4.review_h2_i_handoff import review
                c.write(run/'I_origin_review.npz', review(run))
            report = dict(schema='f4-h2-continuation-raw-checkpoint-review-v1', passed=True,
                manifest_identity=manifest['identity'], new_jobs_verified=len(new),
                preserved_jobs_in_previous_archives=len(seen), adoptions_verified=adopted,
                native_propagations=0, scientific_acceptance=False,
                source_sha256=manifest['source_sha256'], continuation_source_sha256=manifest['continuation_source_sha256'])
            c.write(run/('raw_review_'+out.stem+'.json'), report)
            entry.update(status='closed', charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            c.write(a.campaign, ledger, replace=True)
            files = {str(q.relative_to(run)):q.read_bytes() for q in sorted(run.rglob('*'))
                if q.is_file() and not q.name.endswith(('.partial','.lock'))
                and (not str(q.relative_to(run)).startswith('jobs/') or q.name in new or q.suffix == '.json')}
            files['campaign_at_checkpoint.json'] = a.campaign.read_bytes()
            files['approved_decision.json'] = a.decision.read_bytes()
            for name in (*manifest['source_sha256'], *manifest['continuation_source_sha256'],
                         'root/f4/checkpoint_h2_continuation.py', 'root/f4/review_h2_i_handoff.py'):
                files['source/'+name] = (ROOT/name).read_bytes()
            index = dict(schema='f4-h2-durable-continuation-v1', previous=previous,
                predecessor_run=origin['predecessor_path'], predecessor_job_index_sha256=manifest['job_index_sha256'],
                new_jobs=new, metadata_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
                preserved_predecessor_archive_chain_required=True,
                campaign_is_snapshot_latest_committed_ledger_required=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            partial = out.with_suffix('.zip.partial')
            with zipfile.ZipFile(partial, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                for name, raw in files.items(): z.writestr(name, raw)
                z.writestr('durable_continuation.json', json.dumps(index, separators=(',',':')))
            with zipfile.ZipFile(partial) as z:
                c.require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())), 'ZIP corruption/duplicate member.')
            os.replace(partial, out)
            report.update(path=str(out), archive_sha256=c.sha(out), bytes=out.stat().st_size)
            print(json.dumps(report), flush=True)
        finally:
            entry.update(status='closed', charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            c.write(a.campaign, ledger, replace=True); watchdog.cancel()


if __name__ == '__main__': main()

"""Copy completed immutable jobs during a live run; never touch its source/ledger.

Metadata is read before listing jobs so an included optimizer call cannot
refer to a completed job that was published after the listing. Each snapshot
contains all current metadata and only jobs not covered by earlier snapshots.
Restore numbered snapshots in order into a new directory; retain the latest
authoritative campaign ledger and source/environment checks before resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import zipfile


def snapshot(directory, destination, campaign, previous=()):
    begun=perf_counter(); directory=Path(directory); destination=Path(destination)
    if destination.exists():
        raise FileExistsError('Never overwrite a durable checkpoint.')
    seen=set(); previous_hashes={}
    for path in map(Path,previous):
        with zipfile.ZipFile(path) as z:
            index=json.loads(z.read('durable_checkpoint.json'))
            seen.update(index['new_jobs'])
        previous_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    # Read atomic metadata first; completed job files are immutable thereafter.
    metadata={}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name!='run.lock' and not path.name.endswith('.partial'):
            metadata[path.name]=path.read_bytes()
    jobs=[p for p in sorted((directory/'jobs').iterdir()) if p.is_file()
        and not p.name.endswith('.partial') and p.name not in seen]
    temporary=destination.with_suffix(destination.suffix+'.partial')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as z:
        for name,raw in metadata.items():
            z.writestr(name,raw)
        job_hashes={}
        for path in jobs:
            raw=path.read_bytes(); job_hashes[path.name]=hashlib.sha256(raw).hexdigest()
            z.writestr('jobs/'+path.name,raw)
        ledger=Path(campaign).read_bytes()
        z.writestr('campaign_at_checkpoint.json',ledger)
        index=dict(schema='f4-h2-durable-increment-v1',previous=previous_hashes,
            metadata_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in metadata.items()},
            new_jobs=job_hashes,campaign_sha256=hashlib.sha256(ledger).hexdigest(),
            interrupted_reservation_must_be_preserved=True,
            elapsed_seconds_before_archive_close=perf_counter()-begun)
        z.writestr('durable_checkpoint.json',json.dumps(index,separators=(',',':')))
    os.replace(temporary,destination)
    return dict(path=str(destination),new_jobs=len(jobs),prior_jobs=len(seen),
        bytes=destination.stat().st_size,seconds=perf_counter()-begun,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path); parser.add_argument('destination',type=Path)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--previous',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    print(json.dumps(snapshot(args.directory,args.destination,args.campaign,args.previous)))

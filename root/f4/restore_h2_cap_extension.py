"""Restore an extension from its complete verified delta chain into a NEW run.

Only the last archive's metadata defines the new run. Older terminal metadata
is retained in the original archives and in the named predecessor copies.
No original failed directory, source archive or authoritative ledger is edited.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from root.f4.restore_h2_staged_checkpoints import checked_members,sha


def restore(paths,destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Restore only into a new directory.')
    previous={};jobs={};frozen={};metadata=None
    for path in map(Path,paths):
        digest,index,contents=checked_members(path,previous)
        for name in ('manifest.json','packet.json','observations.npz','inputs.npz','source.zip'):
            current=sha(contents[name])
            if name in frozen and frozen[name]!=current:raise ValueError('Numerical origin changed.')
            frozen[name]=current
        for name in index['new_jobs']:
            if name in jobs:raise ValueError('Repeated immutable job.')
            jobs[name]=contents['jobs/'+name]
        metadata={n:raw for n,raw in contents.items() if not n.startswith('jobs/')}
        previous[path.name]=digest
    if metadata is None or 'extension_origin.json' not in metadata:raise ValueError('No extension origin.')
    origin=json.loads(metadata['extension_origin.json'])
    for name,h in origin['preserved_predecessor_metadata'].items():
        if sha(metadata[name])!=h:raise ValueError('Terminal predecessor evidence changed.')
    for name,h in origin['reused_job_sha256'].items():
        if sha(jobs[name])!=h:raise ValueError('Inherited job changed.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name+'_restore_',dir=destination.parent) as folder:
        folder=Path(folder);(folder/'jobs').mkdir()
        for name,raw in metadata.items():(folder/name).write_bytes(raw)
        for name,raw in jobs.items():(folder/'jobs'/name).write_bytes(raw)
        (folder/'STOP').write_text('Verified extension restoration. Reconcile latest campaign and source before explicit continuation. A terminal EXTENSION_FAILED must never be retried.\n')
        receipt=dict(schema='f4-h2-extension-restoration-v1',archives=previous,jobs=len(jobs),
                     latest_metadata_only=True,starts_paused=True,model_jobs=0)
        (folder/'extension_restoration.json').write_text(json.dumps(receipt))
        os.replace(folder,destination)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path);p.add_argument('archives',type=Path,nargs='+')
    a=p.parse_args();print(json.dumps(restore(a.archives,a.destination)))

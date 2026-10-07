"""Verify and restore ordered durable increments, without starting any model.

The output must be new. The last campaign snapshot remains inside it for
explicit reconciliation; an existing authoritative campaign is never replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_members(path, previous):
    raw=path.read_bytes()
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate archive member: '+path.name)
        index=json.loads(z.read('durable_checkpoint.json'))
        if index['schema']!='f4-h2-durable-increment-v1' or index['previous']!=previous:
            raise ValueError('Missing, reordered or changed preceding checkpoint: '+path.name)
        expected={'durable_checkpoint.json','campaign_at_checkpoint.json'}
        expected.update(index['metadata_sha256'])
        expected.update('jobs/'+name for name in index['new_jobs'])
        if set(names)!=expected:
            raise ValueError('Unexpected or missing archive members: '+path.name)
        hashes=dict(index['metadata_sha256'])
        hashes.update({'jobs/'+name:h for name,h in index['new_jobs'].items()})
        hashes['campaign_at_checkpoint.json']=index['campaign_sha256']
        contents={}
        for name,h in hashes.items():
            parts=PurePosixPath(name).parts
            if '\\' in name or PurePosixPath(name).is_absolute() or any(p in ('..','.') for p in parts):
                raise ValueError('Invalid archive path.')
            if len(parts)>2 or (len(parts)==2 and parts[0]!='jobs'):
                raise ValueError('Unexpected nested archive member.')
            content=z.read(name)
            if sha(content)!=h:
                raise ValueError('Archive member checksum failed: '+name)
            contents[name]=content
        return sha(raw),index,contents


def restore(paths,destination,*,verify_only=False):
    paths=list(map(Path,paths));destination=Path(destination)
    if not paths or len({p.name for p in paths})!=len(paths):
        raise ValueError('Require distinct ordered checkpoints.')
    if not verify_only and os.path.lexists(destination):
        raise FileExistsError('Restore only into a new directory.')
    previous={};job_hashes={};frozen={};identity=None;last_campaign=None
    # Verify the complete chain before creating the destination.
    for path in paths:
        digest,index,contents=checked_members(path,previous)
        current=json.loads(contents['manifest.json'])['identity']
        if identity is not None and identity!=current:
            raise ValueError('Numerical identity changed between checkpoints.')
        identity=current
        for name in ('manifest.json','packet.json','inputs.npz','observations.npz','source.zip'):
            h=sha(contents[name])
            if name in frozen and frozen[name]!=h:
                raise ValueError('Frozen preparation content changed: '+name)
            frozen[name]=h
        for name,h in index['new_jobs'].items():
            if name in job_hashes:
                raise ValueError('An immutable job was redundantly included: '+name)
            job_hashes[name]=h
        last_campaign=json.loads(contents['campaign_at_checkpoint.json'])
        previous[path.name]=digest
    receipt=dict(schema='f4-h2-restored-increments-v1',checkpoints=previous,
        numerical_manifest_identity=identity,jobs=len(job_hashes),verified_only=verify_only,
        campaign_charged_seconds=last_campaign['charged_seconds'],
        open_reservations=sum(e['status']=='reserved' for e in last_campaign['entries']),
        model_started=False,automatic_refund=False,ledger_semantics_reconciled=False,
        campaign_snapshot='campaign_at_checkpoint.json')
    if verify_only:return receipt
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name+'_restore_',dir=destination.parent) as staging:
        staging=Path(staging);previous={}
        for path in paths:
            digest,index,contents=checked_members(path,previous)
            for name,raw in contents.items():
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw)
            previous[path.name]=digest
        (staging/'restoration.json').write_text(json.dumps(receipt,separators=(',',':')),encoding='utf-8')
        # Files are complete before the restored directory becomes visible.
        if os.path.lexists(destination):raise FileExistsError('Restore destination appeared concurrently.')
        os.rename(staging,destination)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints',type=Path,nargs='+')
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(restore(args.checkpoints,args.destination,verify_only=args.verify_only)),flush=True)

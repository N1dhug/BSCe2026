"""Shared atomic BM report/checkpoint I/O; no study imports or numerical runs.

Private aliases are re-exported by estimate.py for existing callers. Formats,
exclusive directory locking and no-overwrite behavior are unchanged.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


def _atomic_report(path, payload, *, replace_existing=False):
    """Publish only closed complete JSON/gzip files under the exclusive run lock."""
    import gzip
    import tempfile
    if path.exists() and not replace_existing:
        raise FileExistsError(f'Refusing to overwrite {path}.')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if path.suffix=='.npz':
            _write_array_checkpoint(temporary, payload)
        else:
            opener = gzip.open if path.suffix=='.gz' else open
            with opener(temporary, 'wt', encoding='utf-8') as handle:
                json.dump(payload, handle, allow_nan=False, separators=(',',':'))
                handle.write('\n')
        if path.exists() and not replace_existing:
            raise FileExistsError(f'Refusing to overwrite {path}.')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_array_checkpoint(path, report):
    """Avoid millions of Python floats/text conversions for large raw arrays."""
    import zipfile
    arrays = {}
    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'biuf': raise TypeError('Checkpoint arrays must be real numeric arrays.')
            name = f'array_{len(arrays):04d}.npy'; arrays[name]=value
            return {'__npy__':name}
        if isinstance(value,dict): return {k:encode(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [encode(v) for v in value]
        if isinstance(value,np.generic): return value.item()
        return value
    metadata = dict(format='bm1-array-checkpoint-v1', report=encode(report))
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        archive.writestr('metadata.json',json.dumps(metadata,allow_nan=False,separators=(',',':')))
        for name,array in arrays.items():
            with archive.open(name,'w',force_zip64=True) as member:
                np.lib.format.write_array(member,np.asarray(array),allow_pickle=False)


def _read_array_checkpoint(raw):
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names)!=len(set(names)): raise ValueError('Duplicate checkpoint members.')
        metadata = json.loads(archive.read('metadata.json'))
        if metadata.get('format')!='bm1-array-checkpoint-v1': raise ValueError('Unknown checkpoint format.')
        used = {'metadata.json'}
        def decode(value):
            if isinstance(value,dict) and set(value)=={'__npy__'}:
                name=value['__npy__']
                if not isinstance(name,str) or not re.fullmatch(r'array_[0-9]{4}\.npy',name) or name in used:
                    raise ValueError('Invalid or duplicate checkpoint array reference.')
                used.add(name)
                with archive.open(name) as member: array=np.lib.format.read_array(member,allow_pickle=False)
                if array.dtype.kind not in 'biuf': raise ValueError('Checkpoint arrays must be real numeric arrays.')
                return array
            if isinstance(value,dict): return {k:decode(v) for k,v in value.items()}
            if isinstance(value,list): return [decode(v) for v in value]
            return value
        result=decode(metadata['report'])
        if used!=set(names): raise ValueError('Unreferenced members in checkpoint.')
        return result


class _RunLock:
    """OS-held lock releases after a crash; an old lock file never blocks resume."""
    def __init__(self, directory):
        self.path = directory / 'run.lock'

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        if self.handle.tell()==0:
            self.handle.write(b'0'); self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError('This run directory is already in use by another process.') from exc
        return self

    def __exit__(self, *args):
        self.handle.close()


def _read_likelihood_report(path):
    """Read plain/gzip evidence and identify the exact stored bytes."""
    import gzip
    raw = path.read_bytes()
    if path.suffix == '.npz':
        return _read_array_checkpoint(raw), hashlib.sha256(raw).hexdigest()
    decoded = gzip.decompress(raw) if path.suffix == '.gz' else raw
    return json.loads(decoded.decode('utf-8')), hashlib.sha256(raw).hexdigest()

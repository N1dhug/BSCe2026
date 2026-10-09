"""Atomic JSON/gzip reports for G and the shared calendar."""
import json,os
from pathlib import Path
from src.tools.checkpoint_io import discard_file,replace_file

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
        opener = gzip.open if path.suffix=='.gz' else open
        with opener(temporary, 'wt', encoding='utf-8') as handle:
            json.dump(payload, handle, allow_nan=False, separators=(',',':'))
            handle.write('\n')
        if path.exists() and not replace_existing:
            raise FileExistsError(f'Refusing to overwrite {path}.')
        replace_file(temporary, path)
    finally:
        discard_file(temporary)

"""Durable deterministic G CSV and JSON export."""
import os
from pathlib import Path
from src.tools.bm_1_reports import _atomic_report
from src.tools.checkpoint_io import replace_file,discard_file
from src.tools.bm_1_g_contract import _plain

def _atomic_csv(path, frame):
    """Replace only this run's CSV after closing a complete temporary file."""
    import tempfile
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index_label='Datetime')
        replace_file(temporary, path)
    finally:
        discard_file(temporary)

def write_forecast(directory, result, *, prefix='forecast'):
    _atomic_csv(directory/f'{prefix}.csv', result['predictions'])
    _atomic_csv(directory/f'{prefix}_normalized.csv', result['normalized_predictions'])
    plain = {k:v for k,v in result.items() if k not in ('predictions','normalized_predictions')}
    _atomic_report(directory/f'{prefix}.json', _plain(plain))

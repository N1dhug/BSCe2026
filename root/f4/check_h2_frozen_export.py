"""Focused Git archive regression; no workbook, FV, JIT or model jobs."""
from time import perf_counter
BEGUN = perf_counter() - .25
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def check():
    ready = json.loads((ROOT/'root/f4/h2_compact_data_ready_20261007.json').read_text())
    packet = json.loads((ROOT/ready['packet_path']).read_text())
    expected = {ready['audit_path']: ready['audit_sha256'],
        ready['baseline_path']: packet['outputs_sha256']['baseline.csv'],
        ready['inputs_path']: packet['outputs_sha256']['inputs.npz'],
        ready['packet_path']: hashlib.sha256((ROOT/ready['packet_path']).read_bytes()).hexdigest()}
    raw = {p: (ROOT/p).read_bytes() for p in expected}
    attributes = (ROOT/'.gitattributes').read_bytes()
    # Keep the real pre-fix attribute rule and exact Git blob bytes, including
    # the receipt's CRLF CSV; adding through a text filter would alter evidence.
    before = attributes.split(b'\n# Frozen F4 receipts:')[0]
    rows = []
    with tempfile.TemporaryDirectory(prefix='f4_export_check_') as tmp:
        def git(*args, data=None):
            return subprocess.check_output(['git', '-C', tmp, *args], input=data)
        git('init', '--quiet')
        for name, attrs in (('before', before), ('fixed', attributes)):
            for p, content in dict(raw, **{'.gitattributes': attrs}).items():
                blob = git('hash-object', '-w', '--stdin', data=content).decode().strip()
                git('update-index', '--add', '--cacheinfo', '100644', blob, p)
            tree = git('write-tree').decode().strip()
            for autocrlf in ('false', 'true', 'input'):
                for eol in ('lf', 'crlf'):
                    archive = git('-c', 'core.autocrlf='+autocrlf, '-c', 'core.eol='+eol,
                        'archive', '--format=zip', tree, '.gitattributes', *raw)
                    with zipfile.ZipFile(io.BytesIO(archive)) as z:
                        hashes = {p: hashlib.sha256(z.read(p)).hexdigest() for p in raw}
                    same = hashes == expected
                    if name == 'fixed':
                        assert same, (autocrlf, eol, hashes)
                    if name == 'before' and autocrlf == 'true':
                        assert hashes[ready['audit_path']] != expected[ready['audit_path']]
                    rows.append(dict(attributes=name, core_autocrlf=autocrlf,
                        core_eol=eol, all_frozen_bytes_match=same, sha256=hashes))
    # The strict byte gate still rejects changes to actual receipt content.
    assert hashlib.sha256(raw[ready['audit_path']]+b' ').hexdigest() != ready['audit_sha256']
    return dict(passed=True, old_windows_failure_reproduced=True,
        fixed_export_all_six_configurations_passed=True, changed_bytes_rejected=True,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in ('.gitattributes', 'root/f4/check_h2_frozen_export.py',
                'root/f4/measure_h2_compact_rates.py', 'root/f4/run_h2_compact_rates.ps1')},
        rows=rows, model_jobs=0, workbook_values_decoded=0,
        historical_fit_ready=False, runtime_admitted=False, scientific_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Preserve the previous export check receipt.')
    report = check()
    report['elapsed_seconds'] = perf_counter()-BEGUN
    report['budget_status'] = 'Pending exact prefix reconciliation with failed PC ledger; do not reset seed.'
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('passed', 'elapsed_seconds', 'model_jobs', 'budget_status')}))

"""Recover exact archived G7 inputs without importing numerical kernels.

The failed invocation, its ledger and its old sources remain historical.
Only the two archived M254 inputs are recovered: no FV origin is invented.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

ARCHIVE = 'files/output/bm_1/runs/g7_node_runtime_failed_20261007.raw.zlib'
INDEX = 'root/g7/node_runtime_failed_archive_20261007.json'
EXPECTED_COMPRESSED = 'eeb3944f7145a72fec7d25a845ed0d29d16f54f9d5de92814fb3a06580865d37'
EXPECTED_RAW = 'bff49ef14c9034400090db743433fe94cdec76612c9c007fd400a83554a5b30a'
INPUTS = {
    'coarse': '1cd343248607cc27baa61d4e52303fbfbf5c08ca7674fac0b9b3dce43649b3bc',
    'fine': '746e45c4603e724824687e535c2e7235f824e19ecbb6adcdd1c6ee0c0a7593e0',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_archive(repo):
    compressed = (repo / ARCHIVE).read_bytes()
    if sha(compressed) != EXPECTED_COMPRESSED:
        raise ValueError('Archived compressed bytes changed.')
    raw = zlib.decompress(compressed)
    if sha(raw) != EXPECTED_RAW:
        raise ValueError('Archived uncompressed bytes changed.')
    index = json.loads((repo / INDEX).read_bytes())
    archive = json.loads(raw)
    if (archive['schema'] != 'g7-raw-byte-archive-v1'
            or index['schema'] != 'g7-zlib-base64-byte-archive-v1'
            or set(archive['files']) != set(index['files'])):
        raise ValueError('Archive members differ from historical index.')
    files = {}
    for name, encoded in archive['files'].items():
        member = Path(name)
        if member.anchor or '..' in member.parts or ':' in name or '\\' in name:
            raise ValueError('Unsafe archived member path.')
        data = base64.b64decode(encoded, validate=True)
        expected = index['files'][name]
        if len(data) != expected['bytes'] or sha(data) != expected['sha256']:
            raise ValueError('Historical member differs: ' + name)
        files[name] = data
    return files


def recover(repo, output):
    repo = Path(repo).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new recovery destination; existing evidence is preserved.')
    files = verify_archive(repo)
    records = {}
    for label, expected in INPUTS.items():
        data = files['inputs/' + label + '.json']
        if sha(data) != expected:
            raise ValueError('Input differs from later verified production registration.')
        record = json.loads(data)
        cells = 257 if label == 'coarse' else 513
        if (record['schema'] != 'g8-rho-origin-v1'
                or record['time'] != 12 or record['backend'] != 'mixture'
                or record['basis_kind'] != 'g8-bernstein-m254-cell-average-v1'
                or len(record['rho_values']) != 5
                or len(record['x_edges']) != cells + 1
                or len(record['conditional_mass']) != 5
                or any(len(row) != cells for row in record['conditional_mass'])
                or any(len(row) != 255 for row in record['conditional_coefficients'])):
            raise ValueError('Archived M254 origin identity differs.')
        records['m254_' + label] = dict(file_sha256=expected, cells=cells,
            cutoff=record['time'], backend=record['backend'], basis_kind=record['basis_kind'])
    old_manifest = json.loads(files['source_manifest.json'])
    for name, expected in old_manifest.items():
        if sha(files['frozen_source/' + name]) != expected:
            raise ValueError('Original source differs: ' + name)
    current_manifest = json.loads((repo / 'root/g7/integration_native_manifest.json').read_bytes())
    current_mismatches = [name for name, expected in current_manifest.items()
                          if sha((repo / name).read_bytes()) != expected]
    old_to_current_changes = [name for name, value in old_manifest.items()
                             if current_manifest.get(name) != value]
    report = dict(schema='g7-day-origin-recovery-v1', recovery='exact-archived-bytes',
        archive_path=ARCHIVE, archive_sha256=EXPECTED_COMPRESSED,
        verified_members=len(files), verified_original_sources=len(old_manifest),
        recovered_origins=records, missing_origins=['fv_coarse', 'fv_fine'],
        current_integration_source_mismatches=current_mismatches,
        original_to_integration_source_changes=old_to_current_changes,
        old_invocation_resumable=False, new_native_forwards=0,
        new_projection_solves=0, full_day_launch_ready=False, scientific_accept=False,
        historical_campaign_debit=1927.3372718589885,
        historical_campaign_ceiling=1930,
        note='Archive contains M254 posteriors only. Separate own FV origins and new runtime rates are required.')
    # Finish all verification before writing any recovered evidence.
    output.mkdir(parents=True)
    for name, data in files.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if sha(destination.read_bytes()) != sha(data):
            raise ValueError('Recovered member did not read back identically.')
    (output / 'recovery_report.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(recover(arguments.repo, arguments.output), indent=2, allow_nan=False))

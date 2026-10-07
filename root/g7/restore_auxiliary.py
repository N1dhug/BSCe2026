"""Read-only G7 receipt verification; no native solver import or execution."""
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_json(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Owned file bytes changed: ' + str(path))
    return json.loads(payload)


def verified_receipt(path, expected_file_hash, expected_job):
    receipt = verified_json(path, expected_file_hash)
    if (receipt.get('schema') != 'g7-owned-node-v1' or receipt['job'] != expected_job
            or hashlib.sha256(canonical(receipt['payload'])).hexdigest() != receipt['payload_sha256']):
        raise ValueError('Receipt payload or full job identity changed.')
    return receipt['payload']


def read_owned(root, audit, reuse):
    """Validate the original complete ledger and numerical provenance read-only.

    Caller's registration must authenticate audit/reuse. This returns plain
    receipts only; native combine_nodes still validates all owned trajectories.
    """
    root = Path(root)
    files = audit['raw_data_archive']['files']
    if (audit['execution_commit'] != reuse['execution_commit']
            or audit['raw_data_archive']['sha256'] != reuse['data_archive_sha256']):
        raise ValueError('Different raw execution/archive.')
    sources = verified_json(root/'source_manifest.json', files['source_manifest.json']['sha256'])
    if len(sources) != 41:
        raise ValueError('Original 41-source manifest required.')
    for path, expected in sources.items():
        if sha(root/'frozen_source'/path) != expected:
            raise ValueError('Frozen numerical source changed: ' + path)
    ledger = verified_json(root/'campaign/ledger.json', files['campaign/ledger.json']['sha256'])
    if len(ledger['entries']) != 12:
        raise ValueError('Original twelve completed attempts required.')
    result = dict(sources=sources, contracts={}, nodes={}, ledger_sha256=sha(root/'campaign/ledger.json'))
    ids = set()
    for entry in ledger['entries']:
        key = entry['job_id']
        if key in ids or entry['attempt'] != 1 or entry['status'] != 'complete':
            raise ValueError('Original ledger has incomplete/repeated jobs.')
        ids.add(key)
        path = 'campaign/nodes/'+key+'.json'
        receipt = verified_json(root/path, files[path]['sha256'])
        if hashlib.sha256(canonical(receipt['job'])).hexdigest() != key:
            raise ValueError('Receipt is not the reserved ledger job.')
        verified_receipt(root/path, files[path]['sha256'], receipt['job'])
        if 'contract' in receipt['job']:
            result['nodes'].setdefault(receipt['job']['contract'], []).append(receipt)
    for label in ('fine', 'coarse'):
        key = reuse[label+'_contract']
        path = 'campaign/contracts/'+key+'.json'
        contract = verified_json(root/path, key)
        nodes = sorted(result['nodes'].get(key, []), key=lambda r: r['job']['index'])
        if len(nodes) != 5 or [r['job']['index'] for r in nodes] != list(range(5)):
            raise ValueError('Five named owned nodes required.')
        for j, receipt in enumerate(nodes):
            if receipt['job'] != dict(contract=key, index=j, rho=contract['origin']['rho_values'][j], prices=[.5,.5]):
                raise ValueError('Neutral owned node identity changed.')
        if (sha(root/(label+'_bank.json')) != reuse[label+'_bank_sha256']
                or files[label+'_bank.json']['sha256'] != reuse[label+'_bank_sha256']):
            raise ValueError('Owned bank file changed.')
        result['contracts'][label] = contract
        result['nodes'][label] = [r['payload'] for r in nodes]
    return result

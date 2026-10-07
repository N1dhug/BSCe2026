"""G-only receipt adoption and opt-in guards around the native node store.

No solver imports. Original anomalous journals are never rewritten. Callers
must pin this file, the native store and the manifest in a new registration.
"""
import hashlib
import json
import math
from pathlib import Path


class RecoveryBlocked(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RecoveryBlocked('Pinned file bytes changed: ' + str(path))
    return json.loads(raw)


def job_id(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise RecoveryBlocked('Invalid job identity.')
    return value


def finite(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise RecoveryBlocked('Invalid ledger time/budget.')
    return value


class VerifiedReceiptIndex:
    """Read-only payloads under an externally pinned, fully reviewed manifest.

    This is not a NodeJobStore and cannot create/retry jobs or resume the old
    writer. Reuse requires the exact original job, including its contract.
    """
    def __init__(self, root, manifest_path, *, manifest_sha256, eager_payload_validation=True):
        self.root = Path(root)
        m = checked_json(manifest_path, manifest_sha256)
        if (m['schema'] != 'g7-verified-receipt-index-v1'
                or m['active_original_store_resume_permitted'] is not False
                or m['source_journal_integrity_passed'] is not False):
            raise RecoveryBlocked('Wrong adoption contract.')
        for name, expected in m['source_files'].items():
            if name not in ('campaign/ledger.json', 'registration.json', 'worker_summary.json', 'invocation.json'):
                raise RecoveryBlocked('Unexpected source metadata path.')
            checked_json(self.root / name, expected)
        if set(m['source_files']) != {'campaign/ledger.json', 'registration.json', 'worker_summary.json', 'invocation.json'}:
            raise RecoveryBlocked('Incomplete source metadata identity.')
        if finite(m['conservative_receipt_charge']) > finite(m['already_charged_worker_wall']):
            raise RecoveryBlocked('Adoption is not covered by the old charge.')
        self.entries = {}
        for entry in m['entries']:
            key = job_id(entry['job_id'])
            if key in self.entries or digest(entry['job']) != key:
                raise RecoveryBlocked('Duplicate/changed adopted job.')
            if entry['job']['registration_sha256'] != m['source_files']['registration.json']:
                raise RecoveryBlocked('Receipt belongs to another registration.')
            finite(entry['conservative_charged_seconds'])
            self.entries[key] = entry
        if len(self.entries) != m['receipt_count'] or not self.entries:
            raise RecoveryBlocked('Incomplete receipt index.')
        names = {p.name for p in (self.root / 'campaign/nodes').glob('*.json')}
        if names != {key + '.json' for key in self.entries}:
            raise RecoveryBlocked('Receipt directory differs from reviewed snapshot.')
        if not math.isclose(sum(e['conservative_charged_seconds'] for e in self.entries.values()),
                            m['conservative_receipt_charge'], rel_tol=0, abs_tol=1e-12):
            raise RecoveryBlocked('Adopted charges changed.')
        self.manifest = m
        # Eager mode preserves the original interface. Byte-only prevalidation
        # authenticates every receipt against the externally reviewed full-file
        # hash, without deserializing unrelated EM/feedback arrays. payload()
        # still checks full job and canonical payload identity for every reuse.
        for entry in self.entries.values():
            if eager_payload_validation:
                self.payload(entry['job'])
            elif sha(self.root / 'campaign/nodes' / (entry['job_id']+'.json')) != entry['receipt_sha256']:
                raise RecoveryBlocked('Adopted receipt bytes changed.')

    def payload(self, exact_original_job):
        key = digest(exact_original_job)
        entry = self.entries.get(key)
        if entry is None or entry['job'] != exact_original_job:
            raise RecoveryBlocked('Job is not in the pinned adoption index.')
        value = checked_json(self.root / 'campaign/nodes' / (key + '.json'), entry['receipt_sha256'])
        if (value['schema'] != 'g7-owned-node-v1' or value['job'] != exact_original_job
                or value['payload_sha256'] != entry['payload_sha256']
                or digest(value['payload']) != value['payload_sha256']):
            raise RecoveryBlocked('Adopted receipt identity/payload changed.')
        return value['payload']


def audit_store(store, *, full=False):
    """Check every journal/file identity; full additionally hashes all payloads."""
    ledger = store.read()
    if ledger['seed'] != store.seed:
        raise RecoveryBlocked('Budget seed changed.')
    for name in ('prior_debit', 'ceiling', 'max_new_seconds'):
        finite(ledger['seed'][name], positive=name != 'prior_debit')
    latest = {}
    for entry in ledger['entries']:
        key = job_id(entry['job_id'])
        previous = latest.get(key)
        if (type(entry['attempt']) is not int or entry['attempt'] != (1 if previous is None else previous['attempt'] + 1)
                or type(entry['node']) is not int
                or (previous is not None and previous['status'] not in ('failed', 'interrupted'))):
            raise RecoveryBlocked('Invalid attempt history.')
        finite(entry['reserved_seconds'], positive=True)
        finite(entry['charged_seconds'])
        if entry['status'] not in ('running', 'complete', 'recovered-complete', 'failed', 'interrupted'):
            raise RecoveryBlocked('Unknown node status.')
        if entry['status'] in ('running', 'interrupted', 'recovered-complete') and entry['charged_seconds'] < entry['reserved_seconds']:
            raise RecoveryBlocked('Crash reservation was refunded.')
        latest[key] = entry
    paths = {p.stem: p for p in (store.root / 'nodes').glob('*.json')}
    if set(paths) - set(latest):
        raise RecoveryBlocked('Receipt without a reserved ledger attempt.')
    for key, entry in latest.items():
        path = paths.get(key)
        if entry['status'] in ('complete', 'recovered-complete') and path is None:
            raise RecoveryBlocked('Completed journal entry has no receipt.')
        if path is not None and entry['status'] not in ('running', 'complete', 'recovered-complete'):
            raise RecoveryBlocked('Receipt conflicts with failed/interrupted status.')
        if full and path is not None:
            value = json.loads(path.read_bytes())
            if (value['schema'] != 'g7-owned-node-v1' or digest(value['job']) != key
                    or value['job']['index'] != entry['node']
                    or digest(value['payload']) != value['payload_sha256']):
                raise RecoveryBlocked('Receipt does not match journal/payload.')
    return ledger


def guarded_store_class(base, *, native_source_path, native_source_sha256):
    """Return a G-local subclass; retain native lock, budget and retry rules.

    Native reservations are checked from disk before compute. Every completion
    checks the prior journal and its written bytes. An error blocks the scope;
    it must be reviewed explicitly, not automatically retried.
    """
    if sha(native_source_path) != native_source_sha256:
        raise RecoveryBlocked('Native store source differs from registration.')

    class GuardedNodeJobStore(base):
        def __init__(self, root, **kwargs):
            self.blocked_path = Path(root) / 'recovery_blocked.json'
            if self.blocked_path.exists():
                raise RecoveryBlocked('Scope requires explicit recovery review.')
            super().__init__(root, **kwargs)
            with self.lock():
                audit_store(self, full=True)

        def _finish(self, ledger, entry, seconds, status, **extra):
            if self.read() != ledger:
                raise RecoveryBlocked('Journal changed since reservation; refusing overwrite.')
            super()._finish(ledger, entry, seconds, status, **extra)
            if self.read() != ledger:
                raise RecoveryBlocked('Completion write did not persist the exact journal.')

        def node(self, job, reservation, compute, *, retry_ids=()):
            # Caller holds the inherited OS lock, as required by native node().
            if self.blocked_path.exists():
                raise RecoveryBlocked('Scope requires explicit recovery review.')
            try:
                before = audit_store(self)
                key = digest(job)

                def checked_compute(deadline):
                    current = audit_store(self)
                    rows = current['entries']
                    if (rows[:-1] != before['entries'] or current['seed'] != before['seed']
                            or rows[-1]['job_id'] != key or rows[-1]['status'] != 'running'
                            or rows[-1]['reserved_seconds'] != reservation):
                        raise RecoveryBlocked('Write-ahead reservation did not persist.')
                    return compute(deadline)

                # Existing running nodes without receipts are interrupted by
                # native node() before an explicit retry. Match that transition.
                if not (self.root / 'nodes' / (key + '.json')).exists():
                    attempts = [e for e in before['entries'] if e['job_id'] == key]
                    if attempts and attempts[-1]['status'] == 'running' and key in retry_ids:
                        attempts[-1]['status'] = 'interrupted'
                result = super().node(job, reservation, checked_compute, retry_ids=retry_ids)
                after = audit_store(self)
                if after['entries'][:len(before['entries'])] != before['entries']:
                    # Receipt recovery may change the last running attempt only.
                    old = [e for e in before['entries'] if e['job_id'] == key]
                    recovered = json.loads(canonical(before))
                    if old and old[-1]['status'] == 'running':
                        for e in recovered['entries']:
                            if e['job_id'] == key and e['attempt'] == old[-1]['attempt']:
                                e['status'] = 'recovered-complete'
                                e['charged_seconds'] = e['reserved_seconds']
                    if after['entries'] != recovered['entries']:
                        raise RecoveryBlocked('Earlier attempt history changed.')
                # Native code may serialize numpy values. Compare via its stored
                # receipt instead of our standard-library JSON encoder.
                self.receipt(self.root / 'nodes' / (key + '.json'), job)
                return result
            except Exception as exc:
                # Preserve the original journal and receipts. The marker is a
                # separate durable stop, not a guessed repair or refund.
                from tempfile import NamedTemporaryFile
                import os
                raw = canonical({'schema': 'g7-recovery-blocked-v1', 'job_id': digest(job), 'reason': repr(exc)})
                with NamedTemporaryFile(dir=self.root, delete=False) as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno()); tmp = stream.name
                os.replace(tmp, self.blocked_path)
                raise

    return GuardedNodeJobStore

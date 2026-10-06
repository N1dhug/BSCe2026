"""Bounded F4-H2 history-only packet freeze; no model replay or optimization."""
from time import perf_counter, time
BEGUN = perf_counter()-.25

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICY = ROOT/'root/f4/h2_compact_data_policy_20261006.json'
EXTRA_SOURCE = ('src/tools/bm_1_h2_data_packet.py', 'src/validation/bm_1/compact_data.py',
    'src/tools/import_data/import_input.py', 'root/f4/run_h2_compact_data.ps1',
    'root/f4/prepare_h2_compact_data.py', 'root/f4/h2_compact_data_policy_20261006.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-workbook', type=Path, default=ROOT/'files/input/data/input.xlsx')
    parser.add_argument('--price-workbook', type=Path, default=ROOT/'files/input/data/prices.xlsx')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=ROOT/'files/output/bm_1/runs/estimation/f4_h2_remaining_budget.json')
    parser.add_argument('--source-commit', required=True, help='Commit exported to the isolated source snapshot.')
    parser.add_argument('--max-seconds', type=float, default=180.)
    parser.add_argument('--check', action='store_true', help='Focused synthetic data-boundary/algebra check only.')
    args = parser.parse_args()
    args.output, args.campaign = args.output.resolve(), args.campaign.resolve()
    if args.output.exists() or args.output.with_name(args.output.name+'_review.zip').exists():
        raise FileExistsError('Preserve every attempt; use a new output directory only for a reviewed reason.')
    if not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('Record the exact exported 40-character Git commit.')
    if not 75. < args.max_seconds <= 300.:
        raise ValueError('Preserve 60-second stop reserve; maximum invocation grant 300 seconds.')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    from src.exe.run_model.bm_1_h2_native import native_reservation, _source
    from src.exe.run_model.bm_1_h2_precise import _environment
    from src.tools.bm_1_reports import _RunLock, _atomic_report
    from src.tools.bm_1_h2_data_packet import file_sha256, prepare_compact_packet
    def hashes():
        return dict(_source(), **{p:file_sha256(ROOT/p) for p in EXTRA_SOURCE})
    source = hashes()
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    # Source snapshot is separate from the user's current checkout/other runs.
    # Workbook content is checked against the independently audited hashes.
    with native_reservation(args.campaign,args.max_seconds,'compact-data-focused-control' if args.check
                            else 'compact-data-preparation',begun=BEGUN) as reservation, _RunLock(args.output):
        manifest = dict(schema='f4-h2-compact-data-execution-v1',source_commit=args.source_commit,
            source_sha256=source,environment=dict(_environment(),workers=1,blas_threads=1),
            grant_seconds=reservation['grant'],stop_reserve_seconds=60.,
            native_seconds_before=reservation['native_seconds_before'],mode='check' if args.check else 'prepare')
        _atomic_report(args.output/'manifest.json',manifest)
        try:
            if time() >= reservation['deadline']:
                raise TimeoutError('No allowance remains after import/startup.')
            if args.check:
                from src.validation.bm_1.compact_data import compact_data_control
                result = compact_data_control(policy)
            else:
                packet = prepare_compact_packet(policy,input_path=args.input_workbook,
                    price_path=args.price_workbook,output_dir=args.output)
                result = dict(packet_identity=packet['identity'], conditional_data_ready=True)
            if time() >= reservation['deadline']:
                raise TimeoutError('Data preparation exceeded its compute allowance; no automatic retry.')
            if hashes() != source:
                raise ValueError('Frozen F sources changed during preparation.')
            _atomic_report(args.output/'report.json',dict(schema='f4-h2-compact-data-report-v1',
                passed=True,result=result,elapsed_seconds_before_ledger_close=perf_counter()-BEGUN,
                historical_fit_ready=False,runtime_admitted=False,scientific_acceptance=False))
            # Freeze source bytes before closing the charged section. Preserve
            # these even if later repository edits change the recorded paths.
            with zipfile.ZipFile(args.output/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for p in source:
                    raw = (ROOT/p).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != source[p]:
                        raise ValueError('Source changed during archive export.')
                    archive.writestr(p,raw)
        except Exception:
            _atomic_report(args.output/'failure.json',dict(error=traceback.format_exc(),
                historical_fit_ready=False,automatic_retry=False))
            raise
    # Ledger is authoritative only AFTER orderly reservation close. Packaging
    # these already frozen bytes performs no further data/model calculation.
    ledger_bytes = args.campaign.read_bytes()
    (args.output/'ledger.json').write_bytes(ledger_bytes)
    archive_path = args.output.with_name(args.output.name+'_review.zip')
    if archive_path.exists():
        raise FileExistsError('Preserve an existing review archive.')
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('manifest.json','report.json','packet.json','baseline.csv','inputs.npz','source.zip','ledger.json'):
            p = args.output/name
            if p.exists():
                archive.write(p,name)
    ledger = json.loads(ledger_bytes)
    print(json.dumps(dict(passed=True,review_zip=str(archive_path),
        campaign_charged_seconds=ledger['charged_seconds'],
        invocation_charged_seconds=ledger['entries'][-1]['charged_seconds'])))


if __name__ == '__main__':
    main()

"""Bounded focused check of F4-H2 forecast identities; never fit history.

Reuse native_reservation for the shared ledger, 900-second cumulative cap
and watchdog. Old native/FV timing cases are not rerun. Imports/JIT are charged.
"""
from time import perf_counter, time
BEGUN = perf_counter()-.25

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=ROOT/'files/output/bm_1/runs/estimation/f4_h2_remaining_budget.json')
    parser.add_argument('--max-seconds', type=float, default=180.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve completed controls; choose a new output only for a material source change.')
    if not 75. < args.max_seconds <= 300.:
        raise ValueError('Focused control needs 60 seconds stop reserve; maximum grant is 300 seconds.')
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    from src.exe.run_model.bm_1_h2_native import native_reservation, _environment, _source
    from src.exe.run_model.bm_1_h2_precise import _write
    sources = ('src/method/bm_1_interval_forecast.py', 'src/method/bm_1_interval_native.py',
        'src/validation/bm_1/interval_forecast.py', 'src/validation/bm_1/interval_native.py',
        'root/f4/check_h2_interval_forecast.py')
    def source_hashes():
        return dict(_source(), **{p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    hashes = source_hashes()
    with native_reservation(args.campaign, args.max_seconds, 'interval-forecast-focused-control', begun=BEGUN) as reservation:
        from src.validation.bm_1.interval_forecast import forecast_control
        if time() >= reservation['deadline']:
            raise TimeoutError('No compute allowance remains after import/startup.')
        report = dict(schema='f4-h2-interval-forecast-control-v1',
            source_sha256=hashes, environment=_environment(), result=forecast_control(),
            historical_fit_ready=False, scientific_acceptance=False)
        if time() >= reservation['deadline']:
            raise TimeoutError('Focused control exceeded its compute allowance; preserve stop reserve.')
        if hashes != source_hashes():
            raise ValueError('Focused control source changed during execution.')
        report['elapsed_seconds_before_ledger_close'] = perf_counter()-BEGUN
        _write(args.output, report)
    # The shared ledger closes after report export; this is the authoritative charge.
    ledger = json.loads(args.campaign.read_text())
    print(json.dumps(dict(output=str(args.output), passed=True,
        campaign_charged_seconds=ledger['charged_seconds'],
        invocation_charged_seconds=ledger['entries'][-1]['charged_seconds'])))


if __name__ == '__main__':
    main()

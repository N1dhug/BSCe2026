"""Run the bounded synthetic interval characteristic-function control."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blas-threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.blas_threads < 1:
        parser.error('blas-threads must be positive')
    if args.output.exists():
        raise FileExistsError('Preserve existing evidence; use a new output path.')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_DEFAULT_NUM_THREADS'):
        os.environ[name] = str(args.blas_threads)
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.progress import runtime_text
    from src.tools.bm_1_reports import _atomic_report
    from src.validation.bm_1.interval_characteristic import run_characteristic_control
    source = source_snapshot()
    data = args.development_report.read_bytes()
    clock = perf_counter()
    def progress(done, total):
        print('Characteristic control: '+runtime_text(clock,done,total),flush=True)
    result = run_characteristic_control(json.loads(data),progress=progress)
    if source != source_snapshot() or args.development_report.read_bytes() != data:
        raise ValueError('Source or input changed during control; no evidence published.')
    result.update(source=source, development_report_sha256=hashlib.sha256(data).hexdigest())
    _atomic_report(args.output,result)
    print(f'Characteristic development control: {args.output}; no historical acceptance.',flush=True)
    return 0 if all(r['status']=='completed' for r in result['replays']) else 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Run one frozen synthetic interval curve control with atomic unit checkpoints."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _load_rows(path,manifest):
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('manifest') != manifest or saved.get('rows_sha256') != _digest(saved.get('rows')):
        raise ValueError('Checkpoint source/design/input mismatch or corrupted rows; no replay reused.')
    return saved['rows']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-report',type=Path,required=True)
    parser.add_argument('--cn-reference',type=Path,required=True)
    parser.add_argument('--case',choices=('runtime','linear-period','two-narrow','four-wide'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--blas-threads',type=int,default=1)
    args = parser.parse_args(argv)
    if args.blas_threads < 1:
        parser.error('blas-threads must be positive')
    if args.output.exists():
        raise FileExistsError('Preserve completed evidence; use another output path.')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_DEFAULT_NUM_THREADS'):
        os.environ[name] = str(args.blas_threads)
    from src.tools.bm_1_reports import _RunLock
    with _RunLock(args.output.with_suffix('.lock')):
        return _run_frozen(args)


def _run_frozen(args):
    if args.output.exists():
        raise FileExistsError('Preserve completed evidence; use another output path.')
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.progress import runtime_text
    from src.tools.bm_1_reports import _atomic_report
    from src.validation.bm_1.interval_curve_control import declared_control,run_curve_control
    import numpy as np
    import scipy
    data,reference = args.development_report.read_bytes(),args.cn_reference.read_bytes()
    report,cn = json.loads(data),json.loads(reference)
    _,design = declared_control(report,cn,args.case)
    source = source_snapshot()
    manifest = dict(source=source,design=design,blas_threads=args.blas_threads,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                         platform=platform.platform()),
        development_report_sha256=hashlib.sha256(data).hexdigest(),
        cn_reference_sha256=hashlib.sha256(reference).hexdigest())
    checkpoint_path = args.output.with_suffix('.checkpoint.json')
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError('Checkpoint exists; use --resume without changing frozen inputs/source.')
    rows = _load_rows(checkpoint_path,manifest) if args.resume else []
    clock = perf_counter()
    def unchanged():
        if (source != source_snapshot() or args.development_report.read_bytes() != data
                or args.cn_reference.read_bytes() != reference):
            raise ValueError('Source or input changed; no new evidence published.')
    def checkpoint(completed):
        unchanged()
        _atomic_report(checkpoint_path,dict(manifest=manifest,rows=completed,rows_sha256=_digest(completed)),
                       replace_existing=True)
    saved_weight = None
    def progress(done,total):
        nonlocal saved_weight
        if saved_weight is None:
            weights = ([1.]*3 if args.case == 'runtime' else
                       [r['cells']/r['max_step']*r['frequencies'] for r in design['settings']])
            saved_weight = sum(weights[:len(rows)])
        print(args.case+': '+runtime_text(clock,done-saved_weight,total-saved_weight),flush=True)
    result = run_curve_control(report,cn,args.case,saved_rows=rows,checkpoint=checkpoint,progress=progress)
    unchanged()
    result.update(**manifest,invocation_elapsed_seconds=perf_counter()-clock,resumed_units=len(rows))
    _atomic_report(args.output,result)
    print(f'{args.case}: {args.output}; development pass={result["development_numerics_pass"]}; no historical acceptance.',flush=True)
    return 0 if result['development_numerics_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

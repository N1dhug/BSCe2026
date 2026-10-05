"""Bounded F4-H2 terminal-contour references/pilot, with per-hour log checkpoints."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('references', 'hour', 'day'), required=True)
    parser.add_argument('--development-report', type=Path)
    parser.add_argument('--cn-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=900.)
    args = parser.parse_args(argv)
    if not 0. < args.seconds <= 900.:
        parser.error('Declare a positive invocation budget <= 900 seconds.')
    if args.stage != 'references' and (args.development_report is None or args.cn_reference is None):
        parser.error('Hour/day require both frozen reference files.')
    if args.output.exists():
        raise FileExistsError('Preserve completed evidence; select a new output.')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS'):
        os.environ[name] = '1'
    from src.tools.bm_1_reports import _RunLock
    with _RunLock(args.output.with_suffix('.lock')):
        return _run(args)


def _run(args):
    import numpy as np
    import scipy
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.bm_1_interval_curve_control import _digest, _load_rows
    from src.exe.run_model.progress import runtime_text
    from src.tools.bm_1_reports import _atomic_report
    from src.validation.bm_1.interval_terminal_tilt import (
        terminal_reference_control, declared_terminal_control, run_terminal_control)
    if args.output.exists():
        raise FileExistsError('Preserve completed evidence.')
    source, begun = source_snapshot(), perf_counter()
    inputs = {str(p): p.read_bytes() for p in (args.development_report, args.cn_reference) if p is not None}
    manifest = dict(source=source, stage=args.stage, blas_threads=1,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        inputs_sha256={p:hashlib.sha256(b).hexdigest() for p,b in inputs.items()})
    def unchanged():
        if source != source_snapshot() or any(Path(p).read_bytes() != b for p,b in inputs.items()):
            raise ValueError('Frozen source/input changed; no new report published.')
    if args.stage == 'references':
        if args.resume:
            raise ValueError('Small references are one complete unit; no resume mode.')
        result = terminal_reference_control()
    else:
        data, reference = (json.loads(inputs[str(p)]) for p in (args.development_report, args.cn_reference))
        _, design = declared_terminal_control(data, reference, args.stage)
        manifest['design'] = design
        checkpoint_path = args.output.with_suffix('.checkpoint.json')
        if checkpoint_path.exists() and not args.resume:
            raise FileExistsError('Checkpoint exists; use --resume with identical source and inputs.')
        saved = _load_rows(checkpoint_path, manifest) if args.resume else []
        def checkpoint(rows):
            unchanged()
            _atomic_report(checkpoint_path, dict(manifest=manifest, rows=rows, rows_sha256=_digest(rows)), replace_existing=True)
        weights = [s['cells']/s['max_step']*s['frequencies'] for s in design['settings'] for _ in range(design['intervals'])]
        offset = sum(weights[:len(saved)])
        def progress(done, total):
            print(args.stage+': '+runtime_text(begun, done-offset, total-offset), flush=True)
        result = run_terminal_control(data, reference, args.stage, saved_rows=saved,
            checkpoint=checkpoint, progress=progress, deadline=begun+args.seconds)
        result['resumed_intervals'] = len(saved)
        if result['status'] == 'paused':
            print(f'Paused at {len(result["rows"])} saved intervals; resume with identical source/inputs.', flush=True)
            return 3
    unchanged()
    result.update(**manifest, invocation_elapsed_seconds=perf_counter()-begun,
                  scientific_acceptance=False, historical_observation_values_read=0,
                  rigorous_roundoff_certificate=False)
    _atomic_report(args.output, result)
    print(f'{args.stage}: {args.output}; development checks={result["development_checks_pass"]}', flush=True)
    return 0 if result['development_checks_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

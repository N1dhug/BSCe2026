"""Bounded F4-H2 terminal-contour references/pilot, with per-hour log checkpoints."""

import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter


TERMINAL_REFERENCE_DEPENDENCIES = (
    'src/__init__.py',
    'src/method/__init__.py', 'src/method/beta_mixture.py',
    'src/method/bm_1_estimation.py', 'src/method/bm_1_response.py',
    'src/method/euler_maruyama.py', 'src/method/fokker_planc.py', 'src/method/objective.py',
    'src/model/__init__.py', 'src/model/base/__init__.py',
    'src/model/base/bm.py', 'src/model/bm_1.py',
    'src/params/bm_1.py', 'src/params/bm_1.py', 'src/params/bm_1_response.py',
    'src/solver/__init__.py', 'src/solver/finite_volume.py',
    'src/tools/__init__.py', 'src/tools/input_generator/input_builder.py',
    'src/validation/__init__.py', 'src/validation/control.py', 'src/validation/estimation.py',
    'src/validation/model/__init__.py', 'src/validation/model/bm.py', 'src/validation/model/bm_1.py',
    'src/validation/method/__init__.py', 'src/validation/method/euler_maruyama.py',
    'src/validation/method/fokker_planc.py', 'src/validation/method/objective.py',
    'src/validation/solver/__init__.py', 'src/validation/solver/finite_volume.py',
    'src/validation/tools/__init__.py', 'src/validation/tools/input_generator/input_builder.py',
    'src/validation/bm_1/interval_characteristic.py', 'src/validation/bm_1/interval_curve_control.py',
)


def _frozen_terminal_reference(raw, inputs_sha256, source, expected_design):
    """Reuse only the archived PR #81 day and its unchanged numerical kernel.

    The two edited orchestration files are explicit exceptions to whole-file
    matching. Their terminal-update kernel prefix must still match PR #81.
    """
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'b35a376d28659e221c2ee65ca9b8ddee98e5d9c0ca2a57ab3642d5cb0b45676c':
        raise ValueError('Terminal reference differs from the archived PR #81 day.')
    saved = json.loads(raw)
    normalized_inputs = {p.replace('\\', '/'): sha for p, sha in inputs_sha256.items()}
    saved_inputs = {p.replace('\\', '/'): sha for p, sha in saved['inputs_sha256'].items()}
    if (saved['design'] != expected_design or saved_inputs != normalized_inputs
            or saved['stage'] != 'day' or saved['status'] != 'completed'):
        raise ValueError('Terminal reference fixture or inputs changed.')
    exceptions = {'src/exe/run_model/bm_1_interval_terminal_tilt.py',
                  'src/validation/bm_1/interval_terminal_tilt.py'}
    changed = [p for p in TERMINAL_REFERENCE_DEPENDENCIES
               if p not in saved['source']['sha256']
               or source['sha256'].get(p) != saved['source']['sha256'][p]]
    if changed:
        raise ValueError('Terminal reference numerical dependencies changed: '+', '.join(changed))
    kernel_path = Path('src/validation/bm_1/interval_terminal_tilt.py')
    kernel_digest = hashlib.sha256(kernel_path.read_bytes().split(b'\ndef terminal_reference_control():')[0]).hexdigest()
    if kernel_digest != '351b08fa915b125b9df8b08bab031d8c1c452aa231a94d67831989bf125e8292':
        raise ValueError('Terminal reference update kernel changed.')
    return saved, dict(reference_json_sha256=digest, reference_commit=saved['source']['commit'],
        unchanged_dependency_files=len(TERMINAL_REFERENCE_DEPENDENCIES),
        unchanged_dependency_paths=list(TERMINAL_REFERENCE_DEPENDENCIES),
        unchanged_terminal_kernel_sha256=kernel_digest,
        orchestration_file_exceptions=sorted(exceptions))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('references', 'hour', 'day', 'cross', 'fine'), required=True)
    parser.add_argument('--development-report', type=Path)
    parser.add_argument('--cn-reference', type=Path)
    parser.add_argument('--terminal-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=900.)
    args = parser.parse_args(argv)
    if not 0. < args.seconds <= 900.:
        parser.error('Declare a positive invocation budget <= 900 seconds.')
    if args.stage != 'references' and (args.development_report is None or args.cn_reference is None):
        parser.error('Hour/day/cross/fine require both frozen reference files.')
    if (args.stage in ('cross', 'fine')) != (args.terminal_reference is not None):
        parser.error('Cross/fine require --terminal-reference (the archived PR #81 day).')
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
        terminal_reference = None
        if args.stage in ('cross', 'fine'):
            _, expected = declared_terminal_control(data, reference, 'day')
            raw = args.terminal_reference.read_bytes()
            terminal_reference, equivalence = _frozen_terminal_reference(raw, manifest['inputs_sha256'], source, expected)
            inputs[str(args.terminal_reference)] = raw
            manifest['inputs_sha256'][str(args.terminal_reference)] = hashlib.sha256(raw).hexdigest()
            manifest['reference_source_equivalence'] = equivalence
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
            checkpoint=checkpoint, progress=progress, deadline=begun+args.seconds,
            terminal_reference=terminal_reference)
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

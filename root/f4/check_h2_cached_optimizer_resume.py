"""Check SciPy's frozen objective-call sequence without any new model work."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'


def check(run):
    begun = perf_counter()
    import numpy as np
    from scipy.optimize import minimize
    from src.exe.run_model import bm_1_h2_compact as c
    manifest = c.load(run/'manifest.json')
    for package in ('numpy', 'scipy'):
        c.require(importlib.metadata.version(package) == manifest['environment'][package],
                  'Optimizer environment changed.')
    saved = c.load(run/'optimizer_2_calls.json')
    registry = c.load(run/'candidate_registry.json')
    c.require(saved['identity'] == registry['identity'] == manifest['identity'], 'Mixed identities.')
    calls = saved['calls']
    cursor = 0
    next_candidate = None

    class BeforeNewModel(BaseException):
        pass

    def objective(logits):
        nonlocal cursor, next_candidate
        candidate = dict(pieces=2, logits=np.asarray(logits, dtype=float).tolist())
        key = c.curve_key(candidate)
        if cursor == len(calls):
            next_candidate = dict(candidate=candidate, key=key)
            raise BeforeNewModel()
        row = calls[cursor]
        c.require(candidate['logits'] == row['logits'] and key == row['key'],
                  'Cached COBYQA sequence differs.')
        c.require(row['penalty'] == c.penalty(candidate)
                  and row['objective'] == -row['score']+row['penalty'], 'Cached algebra differs.')
        cursor += 1
        return row['objective']

    try:
        minimize(objective, np.zeros(2), method='COBYQA', bounds=[(-4.,4.)]*2,
                 options=dict(maxfev=17, initial_tr_radius=.5, final_tr_radius=.02, scale=False))
    except BeforeNewModel:
        pass
    c.require(cursor == len(calls) and next_candidate is not None, 'Unexpected optimizer termination.')
    c.require(next_candidate['key'] in registry['families']['2'],
              'Next candidate differs from the interrupted registered candidate.')
    # Confirm the four-piece probe really is one of the frozen initial design
    # points before crediting it in the remaining-work graph. The zero scores
    # below are confined to initial-design enumeration; they never enter a fit.
    initial_points = []
    class ProbeReached(BaseException):
        pass
    def initial_design(logits):
        initial_points.append(np.asarray(logits, dtype=float).tolist())
        if len(initial_points) == 2:
            raise ProbeReached()
        return 0.
    try:
        minimize(initial_design, np.zeros(6), method='COBYQA', bounds=[(-4.,4.)]*6,
                 options=dict(maxfev=33, initial_tr_radius=.5, final_tr_radius=.02, scale=False))
    except ProbeReached:
        pass
    c.require(initial_points == [[0.]*6, c.PROBE['logits']], 'Four-piece probe is not the next initial point.')
    return dict(schema='f4-h2-cached-optimizer-resume-review-v1', status='passed',
        manifest_identity=manifest['identity'], cached_calls=cursor,
        calls_sha256=c.sha(run/'optimizer_2_calls.json'), next_candidate=next_candidate,
        four_piece_probe_reuse_confirmed=True, native_propagations=0,
        run_files_modified=False, optimizer_converged=False,
        native_environment_admitted=False, elapsed_seconds=perf_counter()-begun)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier evidence.')
    result = check(args.run)
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result))

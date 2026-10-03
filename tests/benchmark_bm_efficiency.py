"""Paired delivery benchmark, not a new scientific acceptance study.

Run from the repository root with ``python -B -m tests.benchmark_bm_efficiency``.
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before starting Python.
"""

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np
import scipy

from src.control.bm_1 import FastFVStochasticPriceEvaluator
from src.method.bm_1_estimation import run_fv_filter
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation.bm_1 import estimation as core, joint_pilot as pilot


def benchmark(repeats=3):
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 2:
        raise ValueError('Use at least two paired repetitions.')
    group = pilot.make_group(pilot.units()[4])
    case, observations = group['case'], group['observations']
    model, parameters = case.build_model()
    model.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=True)
    cells, step, order = 3201, 1/1536, 3
    initial = core.beta_initial_mass(case, model, cells)

    def replay(history):
        return run_fv_filter(model, parameters, case.sigma_x, case.sigma_y, initial,
            case.horizon[0], observations, breakpoints=case.breakpoints,
            max_step=step, likelihood_quadrature_order=order, keep_history=history)

    # Equality and timing are separate from the intrusive memory measurement.
    full, compact = replay(True), replay(False)
    errors = dict(
        log_evidence=float(np.nanmax(abs(full.log_evidence-compact.log_evidence))),
        final_posterior=float(np.max(abs(full.posterior_mass[-1]-compact.final_step.posterior_mass))),
        final_prediction=float(np.max(abs(full.predicted_mass[-1]-compact.final_step.predicted_mass))),
        total_log_likelihood=abs(full.total_log_likelihood-compact.total_log_likelihood),
        mass_error=abs(full.maximum_abs_mass_error-compact.maximum_abs_mass_error),
        minimum_cell_mass=abs(full.minimum_cell_mass-compact.minimum_cell_mass),
        actual_max_step=abs(float(np.max(np.diff(full.integration_t)))-compact.actual_max_step),
        n_steps=abs(full.integration_t.size-1-compact.n_steps),
    )
    if any(errors.values()):
        raise AssertionError(f'Likelihood-only replay changed results: {errors}')
    del full, compact
    timings = {'full': [], 'likelihood_only': []}
    for repetition in range(repeats):
        modes = (True, False) if repetition % 2 == 0 else (False, True)
        for history in modes:
            started = perf_counter()
            result = replay(history)
            timings['full' if history else 'likelihood_only'].append(perf_counter()-started)
            del result
    peaks = {}
    for history in (True, False):
        gc.collect()
        tracemalloc.start()
        result = replay(history)
        peaks['full' if history else 'likelihood_only'] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        del result

    # A fixed U determines demand; only the tracking target changes here.
    control_case = core.build_f4_b_case()
    control_model, template = control_case.build_model()
    control_initial = core.beta_initial_mass(control_case, control_model, 201)
    evaluator = FastFVStochasticPriceEvaluator(control_model, template, [0., .5, 1.],
        .5, control_initial, objective_max_step=1/96, solver_max_step=1/768,
        terminal_reference=.5, terminal_weight=.2, price_smoothing_weight=.3)
    targets = np.repeat(np.linspace(.2, .8, 21)[:, None], len(evaluator.midpoints), axis=1)
    prices = [.7, .3]

    def target_sweep(shared):
        if shared:
            return evaluator.evaluate_targets(prices, targets)
        results = []
        for target in targets:
            evaluator.target = target.copy()
            results.append(evaluator.evaluate_cost(prices))
        return tuple(results)

    independent, reused = target_sweep(False), target_sweep(True)
    target_error = max(abs(a.total-b.total) for a, b in zip(independent, reused))
    if target_error != 0.:
        raise AssertionError(f'Target-only sweep changed costs: {target_error}')
    target_times = {'independent': [], 'reused': []}
    for repetition in range(repeats):
        for shared in ((False, True) if repetition % 2 == 0 else (True, False)):
            started = perf_counter()
            target_sweep(shared)
            target_times['reused' if shared else 'independent'].append(perf_counter()-started)
    root = Path(__file__).resolve().parents[1]
    sources = ('src/method/bm_1_estimation.py', 'src/control/bm_1.py',
        'src/control/objective.py', 'src/validation/bm_1/estimation.py',
        'src/validation/bm_1/joint_pilot.py', 'src/model/bm_1.py',
        'src/model/base/bm.py', 'src/solver/finite_volume.py',
        'src/validation/solver/finite_volume.py', 'src/params/bm_1.py',
        'tests/benchmark_bm_efficiency.py')
    return dict(
        benchmark='F4-F6 avoid redundant storage/propagation', repeats=repeats,
        scope='Implementation equality and local timing only; not H-f/g, F5 or F6 validation.',
        environment=dict(platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__, scipy=scipy.__version__),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        likelihood=dict(sequence_id=group['sequence_id'], record_days=7,
            observations=len(observations.times), cells=cells, max_step=step, quadrature_order=order,
            seconds=timings, median_seconds={k: float(np.median(v)) for k, v in timings.items()},
            peak_tracemalloc_bytes=peaks, maximum_absolute_differences=errors),
        target_sweep=dict(candidates=len(targets), horizon_days=1, cells=201,
            objective_max_step=1/96, solver_max_step=1/768,
            propagations=dict(independent=len(targets), reused=1),
            seconds=target_times, median_seconds={k: float(np.median(v)) for k, v in target_times.items()},
            maximum_absolute_cost_difference=target_error,
            scope='Same U, baseline, parameters, prior, time grid and quadratic tracking; A alone changes.'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('Output already exists; preserve earlier benchmark evidence.')
    result = benchmark(args.repeats)
    content = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(content)
    print(content)

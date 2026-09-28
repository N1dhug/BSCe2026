"""F3-B fixed-policy numerical refinement; no re-optimization and no file I/O."""

import numpy as np

from src.control.bm_1 import (
    BM1ControlNumerics,
    evaluate_mc_candidates_common_random_numbers,
    paired_mc_improvement,
    sample_beta_initial_states,
    multistart_powell,
)
from src.control import objective as tracking
from src.validation.bm_1 import control as cases


F3_A_STOCHASTIC_PRICES = np.array([0.730609, 0.264415], dtype=float)
# Spatial resolution changes alone.
SPATIAL_X_STEPS = (51, 101, 201, 401, 801)
SPATIAL_OBJECTIVE_MAX_STEP = 1.0 / 96.0
SPATIAL_SOLVER_MAX_STEP = 1.0 / 192.0

# Time integration changes on a fixed spatial/objective grid.
TIME_X_STEPS = 401
TIME_OBJECTIVE_MAX_STEP = 1.0 / 24.0
TIME_SOLVER_MAX_STEPS = (1.0 / 48.0, 1.0 / 96.0, 1.0 / 192.0, 1.0 / 384.0)

# Quadrature rules sample one common propagated distribution.
QUADRATURE_X_STEPS = 401
QUADRATURE_MAX_STEPS = (1.0 / 24.0, 1.0 / 48.0, 1.0 / 96.0, 1.0 / 192.0)
QUADRATURE_SOLVER_MAX_STEP = 1.0 / 384.0
REFERENCE_NUMERICS = BM1ControlNumerics(
    x_steps=801,
    objective_max_step=1.0 / 192.0,
    solver_max_step=1.0 / 384.0,
)
MC_INITIAL_SEED = 2061
MC_BROWNIAN_SEED = 2062
MC_TIME_INITIAL_SEED = 2063
MC_TIME_BROWNIAN_SEED = 2064
MC_PREFIX_COUNTS = (1000, 5000, 10000, 20000)


def fixed_candidates(problem):
    """Return the three fixed F3-B price vectors."""
    return {
        'deterministic_reference': np.asarray(problem.observed_prices, dtype=float).copy(),
        'f3_a_stochastic': F3_A_STOCHASTIC_PRICES.copy(),
        'neutral': np.full(problem.n_intervals, 0.5, dtype=float),
    }


def _validate_candidates(problem, candidates):
    validated = {}
    for name, prices in candidates.items():
        values = np.asarray(prices, dtype=float).reshape(-1)
        if (
            not name
            or values.size != problem.n_intervals
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(
                'every candidate must have a non-empty name and one finite price in [0,1] per problem interval.',
            )
        validated[str(name)] = values.copy()
    if not validated:
        raise ValueError('at least one fixed candidate is required.')
    return validated


def _problem_breakpoints(problem):
    values = np.concatenate(
        (np.asarray(problem.price_edges[1:-1], dtype=float), np.asarray(problem.breakpoints, dtype=float)),
    )
    return np.unique(values)


def _objective_record(name, prices, evaluation, numerics, axis, level):
    diagnostics = evaluation.diagnostics
    return {
        'candidate': name,
        'prices': np.asarray(prices, dtype=float).tolist(),
        'axis': axis,
        'level': float(level),
        'x_steps': int(numerics.x_steps),
        'objective_max_step': float(numerics.objective_max_step),
        'solver_max_step': float(numerics.solver_max_step),
        'total': float(evaluation.objective.total),
        'bias': float(evaluation.objective.bias),
        'variance': float(evaluation.objective.variance),
        'fv_steps': int(diagnostics.get('n_steps', -1)),
        'maximum_abs_mass_error': float(diagnostics.get('maximum_abs_mass_error', np.nan)),
        'minimum_cell_mass': float(diagnostics.get('minimum_cell_mass', np.nan)),
    }


def _add_distance_to_finest(rows, *, axis, finest_level):
    by_candidate = {}
    for row in rows:
        by_candidate.setdefault(row['candidate'], []).append(row)
    for candidate_rows in by_candidate.values():
        finest = [
            row
             for row in candidate_rows if np.isclose(row['level'], finest_level, atol=0.0, rtol=0.0)
        ]
        if len(finest) != 1:
            raise RuntimeError(f'expected one {axis} finest row per candidate.')
        reference = finest[0]['total']
        scale = max(abs(reference), np.finfo(float).tiny)
        for row in candidate_rows:
            row['abs_total_difference_to_finest'] = abs(row['total'] - reference)
            row['rel_total_difference_to_finest'] = row['abs_total_difference_to_finest'] / scale
    return rows


def _locate_times(common_t, requested):
    common_t = np.asarray(common_t, dtype=float)
    requested = np.asarray(requested, dtype=float)
    indices = np.searchsorted(common_t, requested)
    if (
        np.any(indices >= common_t.size)
        or not np.allclose(common_t[indices], requested, atol=2e-14, rtol=0.0)
    ):
        raise RuntimeError('failed to locate quadrature points in common FV output.')
    return indices


def _prefix_statistics(path_costs, counts):
    path_costs = np.asarray(path_costs, dtype=float).reshape(-1)
    if path_costs.size < 2 or not np.all(np.isfinite(path_costs)):
        raise ValueError('path_costs must contain at least two finite values.')
    summaries = []
    for count in counts:
        count = int(count)
        if count < 2 or count > path_costs.size:
            continue
        prefix = path_costs[:count]
        summaries.append(
            {
                'n_paths': count,
                'total': float(np.mean(prefix)),
                'standard_error': float(np.std(prefix, ddof=1) / np.sqrt(count)),
            },
        )
    if not summaries:
        raise ValueError('no requested path count is valid for the supplied paths.')
    full = summaries[-1]['total']
    for row in summaries:
        row['difference_to_largest_prefix'] = row['total'] - full
    return summaries


def _nested_em_time_grids(t_eval, breakpoints, solver_max_steps):
    """Build exactly nested EM grids that respect each requested max step.

    The generic EM grid builder only promises that every step is no larger
    than max_step.  Because it chooses a fresh ceil(width/max_step) count on
    each knot interval, grids from different max_step values need not be
    nested.  Coupled strong/time-refinement needs the stronger property.

    F3-B therefore constructs one finest segment grid and obtains every
    coarser level as an exact subset.  Successive requested max-step ratios
    must be integer refinements (the reference uses factors of two).
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    breakpoints = np.asarray([] if breakpoints is None else breakpoints, dtype=float).reshape(-1)
    steps = tuple((float(value) for value in solver_max_steps))
    if (
        t_eval.size < 2
        or not np.all(np.isfinite(t_eval))
        or np.any(np.diff(t_eval) <= 0.0)
        or not np.all(np.isfinite(breakpoints))
        or len(steps) < 2
        or not np.all(np.isfinite(steps))
        or any((value <= 0.0 for value in steps))
        or any((right >= left for left, right in zip(steps[:-1], steps[1:])))
    ):
        raise ValueError('invalid inputs for nested EM time grids.')
    factors = [1]
    for coarse, fine in zip(steps[:-1], steps[1:]):
        ratio = coarse / fine
        integer_ratio = int(round(ratio))
        if integer_ratio < 2 or not np.isclose(ratio, integer_ratio, atol=1e-12, rtol=1e-12):
            raise ValueError('successive EM max-step levels must have integer refinement ratios.')
        factors.append(factors[-1] * integer_ratio)
    knots = np.unique(np.concatenate((t_eval, breakpoints)))
    if knots[0] != t_eval[0] or knots[-1] != t_eval[-1]:
        raise ValueError('breakpoints must lie inside the output interval.')
    pieces = [[] for _ in steps]
    finest_factor = factors[-1]
    for segment_index, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
        width = right - left
        ratio = width / steps[0]
        n_coarse = max(1, int(np.ceil(np.nextafter(ratio, -np.inf))))
        n_finest = n_coarse * finest_factor
        finest_segment = np.linspace(left, right, n_finest + 1, dtype=float)
        finest_segment[0], finest_segment[-1] = (left, right)
        for level, factor in enumerate(factors):
            stride = finest_factor // factor
            segment = finest_segment[::stride]
            expected = n_coarse * factor + 1
            if segment.size != expected:
                raise RuntimeError('failed to construct an exactly nested EM segment.')
            if segment_index:
                segment = segment[1:]
            pieces[level].append(segment)
    grids = {}
    for step, level_pieces in zip(steps, pieces):
        grid = np.concatenate(level_pieces)
        if np.max(np.diff(grid)) > step + 2e-14:
            raise RuntimeError('nested EM grid violates its requested max step.')
        output_indices = np.searchsorted(grid, t_eval)
        if (
            np.any(output_indices >= grid.size)
            or not np.allclose(grid[output_indices], t_eval, atol=2e-14, rtol=0.0)
        ):
            raise RuntimeError('nested EM grid failed to contain every output time.')
        grids[step] = grid
    finest_grid = grids[steps[-1]]
    for step in steps[:-1]:
        indices = np.searchsorted(finest_grid, grids[step])
        if (
            np.any(indices >= finest_grid.size)
            or not np.allclose(finest_grid[indices], grids[step], atol=0.0, rtol=0.0)
        ):
            raise RuntimeError('constructed EM grids are not exactly nested.')
    return grids


def _em_outputs_on_integration_grid(
    model,
    evaluator,
    parameters,
    initial_states,
    integration_t,
    brownian_increments,
):
    """Run BM1 EM paths on an explicit F3-B integration grid."""
    initial_states = np.asarray(initial_states, dtype=float).reshape(-1)

    def drift(time, state):
        return model._drift(time, state, parameters)

    def diffusion(time, state):
        del time
        return model.diffusion_coefficient(state, evaluator.sigma_x)
    paths, diagnostics = model.euler_maruyama.solve_paths(
        t=integration_t,
        init_cond=initial_states,
        drift=drift,
        diffusion=diffusion,
        n_paths=initial_states.size,
        boundary='none',
        brownian_increments=brownian_increments,
        return_diagnostics=True,
    )
    output_indices = _locate_times(integration_t, evaluator.solver_t_eval)
    state = np.asarray(paths, dtype=float)[:, output_indices]
    times = evaluator.solver_t_eval
    price = np.asarray(parameters[0](times), dtype=float)
    baseline = np.asarray(model.B(times), dtype=float)
    state_response = np.asarray(parameters[1](state), dtype=float)
    price_response = np.asarray(parameters[2](price), dtype=float)
    delta = model.l(state_response, price_response, parameters[4])
    demand = model.D(baseline, parameters[5], delta)
    diagnostics = dict(diagnostics)
    diagnostics['output_indices'] = output_indices.copy()
    diagnostics['requested_output_t'] = times.copy()
    return (state, np.asarray(demand, dtype=float), diagnostics)


def _aggregate_brownian_to_grid(fine_t, fine_increments, coarse_t):
    """Sum fine Wiener increments over arbitrary nested coarse intervals."""
    fine_t = np.asarray(fine_t, dtype=float).reshape(-1)
    coarse_t = np.asarray(coarse_t, dtype=float).reshape(-1)
    increments = np.asarray(fine_increments, dtype=float)
    if (
        fine_t.size < 2
        or coarse_t.size < 2
        or increments.ndim != 2
        or increments.shape[1] != fine_t.size - 1
        or not np.all(np.isfinite(increments))
    ):
        raise ValueError('invalid fine grid/increments for Brownian aggregation.')
    indices = np.searchsorted(fine_t, coarse_t)
    if (
        np.any(indices >= fine_t.size)
        or not np.allclose(fine_t[indices], coarse_t, atol=2e-14, rtol=0.0)
        or indices[0] != 0
        or indices[-1] != fine_t.size - 1
    ):
        raise ValueError('coarse_t must be a nested subgrid of fine_t.')
    result = np.empty((increments.shape[0], coarse_t.size - 1), dtype=float)
    for index, (left, right) in enumerate(zip(indices[:-1], indices[1:])):
        if right <= left:
            raise ValueError('coarse_t must be strictly increasing.')
        result[:, index] = np.sum(increments[:, left:right], axis=1)
    return result


def _last_refinement_change(rows, candidate, finest_level, previous_level):
    finest = next(
        (row for row in rows if (
            row['candidate'] == candidate
            and np.isclose(row['level'], finest_level, atol=0.0, rtol=0.0)
        )),
    )
    previous = next(
        (row for row in rows if (
            row['candidate'] == candidate
            and np.isclose(row['level'], previous_level, atol=0.0, rtol=0.0)
        )),
    )
    return abs(finest['total'] - previous['total'])


def _decision_scale(reference_fv):
    if 'deterministic_reference' not in reference_fv or 'f3_a_stochastic' not in reference_fv:
        return float('nan')
    return abs(
        float(reference_fv['deterministic_reference'].objective.total) - float(
            reference_fv['f3_a_stochastic'].objective.total,
        ),
    )


def _decision_gap_refinement(rows, finest_level, previous_level):
    """Return the change in J_det-J_stoch between the two finest levels."""

    def value(candidate, level):
        matches = [
            row['total']
             for row in rows if (
                row['candidate'] == candidate
                and np.isclose(row['level'], level, atol=0.0, rtol=0.0)
            )
        ]
        if len(matches) != 1:
            raise RuntimeError('expected exactly one refinement row for each candidate/level.')
        return float(matches[0])
    finest_gap = value('deterministic_reference', finest_level) - value('f3_a_stochastic', finest_level)
    previous_gap = value('deterministic_reference', previous_level) - value(
        'f3_a_stochastic',
        previous_level,
    )
    change = abs(finest_gap - previous_gap)
    return {
        'finest_gap': float(finest_gap),
        'previous_gap': float(previous_gap),
        'last_gap_change': float(change),
        'last_gap_change_over_finest_gap': float(change / abs(finest_gap)) if finest_gap != 0.0 else float(
            'inf',
        ),
    }


def run_f3_b_reference(*, mc_paths=0, mc_time_paths=0, progress=print):
    """Run the fixed-candidate F3-B numerical refinement workflow."""
    problem, _ = cases.build_f3_a_problem()
    candidates = fixed_candidates(problem)

    def report(message):
        if progress is not None:
            progress(message)
    report('[1/4] FV spatial refinement')
    spatial = run_spatial_refinement(problem, candidates)
    report('[2/4] FV integration-time refinement')
    fv_time = run_fv_time_refinement(problem, candidates)
    report('[3/4] Isolated objective-quadrature refinement')
    quadrature = run_quadrature_refinement(problem, candidates)
    report('[4/4] Provisional high-resolution FV reevaluation')
    reference_fv = evaluate_reference_fv(problem, candidates)
    decision_scale = _decision_scale(reference_fv)
    studies = (
        ('space', 'spatial', spatial, SPATIAL_X_STEPS),
        ('fv_time', 'fv_time', fv_time, TIME_SOLVER_MAX_STEPS),
        ('quadrature', 'quadrature', quadrature, QUADRATURE_MAX_STEPS),
    )
    summary = {'decision_scale_reference_vs_f3_a': decision_scale}
    gaps = {}
    for axis, prefix, rows, levels in studies:
        finest, previous = float(levels[-1]), float(levels[-2])
        summary[prefix + '_last_change'] = _last_refinement_change(
            rows, 'f3_a_stochastic', finest, previous,
        )
        gaps[axis] = _decision_gap_refinement(rows, finest, previous)
    summary['decision_gap_refinement'] = gaps
    if decision_scale > 0.0:
        for _, prefix, _, _ in studies:
            key = prefix + '_last_change'
            summary[key + '_over_decision_scale'] = summary[key] / decision_scale
    mc = None
    if int(mc_paths) > 0:
        report(f'[MC] FV versus MC with {int(mc_paths)} common paths')
        mc = run_final_mc_comparison(problem, candidates, n_paths=int(mc_paths))
    mc_time = None
    if int(mc_time_paths) > 0:
        report(f'[MC time] Coupled EM refinement with {int(mc_time_paths)} paths')
        mc_time = run_coupled_mc_time_refinement(problem, candidates, n_paths=int(mc_time_paths))
    return {
        'problem': problem,
        'candidates': candidates,
        'spatial': spatial,
        'fv_time': fv_time,
        'quadrature': quadrature,
        'reference_fv': reference_fv,
        'summary': summary,
        'mc': mc,
        'mc_time': mc_time,
    }


def run_spatial_refinement(
    problem,
    candidates,
    *,
    x_steps=SPATIAL_X_STEPS,
    objective_max_step=SPATIAL_OBJECTIVE_MAX_STEP,
    solver_max_step=SPATIAL_SOLVER_MAX_STEP,
):
    """Refine FV state space with all time/objective settings held fixed."""
    candidates = _validate_candidates(problem, candidates)
    x_steps = tuple((int(value) for value in x_steps))
    if len(x_steps) < 2 or any((value < 3 for value in x_steps)):
        raise ValueError('x_steps must contain at least two cell counts >= 3.')
    if any((right <= left for left, right in zip(x_steps[:-1], x_steps[1:]))):
        raise ValueError('x_steps must be strictly increasing.')
    rows = []
    for cells in x_steps:
        numerics = BM1ControlNumerics(
            x_steps=cells,
            objective_max_step=objective_max_step,
            solver_max_step=solver_max_step,
        )
        evaluator = problem.build_evaluator(numerics, fast=True)
        for name, prices in candidates.items():
            evaluation = evaluator.evaluate(prices)
            rows.append(_objective_record(name, prices, evaluation, numerics, 'space', cells))
    return _add_distance_to_finest(rows, axis='space', finest_level=float(x_steps[-1]))


def run_fv_time_refinement(
    problem,
    candidates,
    *,
    x_steps=TIME_X_STEPS,
    objective_max_step=TIME_OBJECTIVE_MAX_STEP,
    solver_max_steps=TIME_SOLVER_MAX_STEPS,
):
    """Refine FV integration time while the objective quadrature stays fixed."""
    candidates = _validate_candidates(problem, candidates)
    steps = tuple((float(value) for value in solver_max_steps))
    if (
        len(steps) < 2
        or not np.all(np.isfinite(steps))
        or any((value <= 0.0 for value in steps))
        or any((right >= left for left, right in zip(steps[:-1], steps[1:])))
    ):
        raise ValueError(
            'solver_max_steps must contain at least two strictly decreasing positive finite values.',
        )
    rows = []
    for step in steps:
        numerics = BM1ControlNumerics(
            x_steps=x_steps,
            objective_max_step=objective_max_step,
            solver_max_step=step,
        )
        evaluator = problem.build_evaluator(numerics, fast=True)
        for name, prices in candidates.items():
            evaluation = evaluator.evaluate(prices)
            rows.append(_objective_record(name, prices, evaluation, numerics, 'fv_time', step))
    return _add_distance_to_finest(rows, axis='fv_time', finest_level=steps[-1])


def run_quadrature_refinement(
    problem,
    candidates,
    *,
    x_steps=QUADRATURE_X_STEPS,
    quadrature_max_steps=QUADRATURE_MAX_STEPS,
    solver_max_step=QUADRATURE_SOLVER_MAX_STEP,
):
    """Refine objective quadrature using one common FV propagation per candidate.

    Every requested quadrature midpoint is included in one union output grid.
    The distribution is therefore propagated once for a candidate, and each
    quadrature formula samples that same numerical state solution.  This avoids
    attributing a changed FV trajectory to the quadrature rule itself.
    """
    candidates = _validate_candidates(problem, candidates)
    levels = tuple((float(value) for value in quadrature_max_steps))
    if (
        len(levels) < 2
        or not np.all(np.isfinite(levels))
        or any((value <= 0.0 for value in levels))
        or any((right >= left for left, right in zip(levels[:-1], levels[1:])))
    ):
        raise ValueError(
            'quadrature_max_steps must contain at least two strictly decreasing positive finite values.',
        )
    breakpoints = _problem_breakpoints(problem)
    quadratures = {}
    all_midpoints = []
    for step in levels:
        _, midpoints, widths, _ = tracking.midpoint_quadrature(
            problem.horizon,
            breakpoints=breakpoints,
            max_step=step,
        )
        quadratures[step] = (midpoints, widths)
        all_midpoints.append(midpoints)
    common_t = np.unique(np.concatenate(([problem.horizon[0]], *all_midpoints, [problem.horizon[1]])))
    numerics = BM1ControlNumerics(
        x_steps=x_steps,
        objective_max_step=levels[-1],
        solver_max_step=solver_max_step,
    )
    model = problem.build_model(numerics)
    initial_mass = problem.initial_mass(model, x_steps)
    support = problem.build_evaluator(numerics, fast=False)
    rows = []
    for name, prices in candidates.items():
        parameters = support.parameters_for_prices(prices)
        result = model.fokker_planc_reference_solver(
            initial_mass,
            parameters,
            problem.parameters.sigma_x,
            t_eval=common_t,
            breakpoints=breakpoints,
            max_step=solver_max_step,
        )
        returned_t, x, _, cell_mass, _, _, _, _, _, diagnostics = result
        np.testing.assert_allclose(returned_t, common_t, atol=0.0, rtol=0.0)
        for step in levels:
            midpoints, widths = quadratures[step]
            indices = _locate_times(common_t, midpoints)
            values = tracking.demand_grid(model, parameters, midpoints, x)
            expected, second, _ = tracking.discrete_demand_moments(values, cell_mass[indices])
            target = np.asarray(model.A(midpoints), dtype=float)
            objective = tracking.stochastic_tracking_objective(widths, expected, second, target)
            rows.append(
                {
                    'candidate': name,
                    'prices': np.asarray(prices, dtype=float).tolist(),
                    'axis': 'quadrature',
                    'level': step,
                    'x_steps': int(x_steps),
                    'objective_max_step': step,
                    'solver_max_step': float(solver_max_step),
                    'total': float(objective.total),
                    'bias': float(objective.bias),
                    'variance': float(objective.variance),
                    'common_output_points': int(common_t.size),
                    'common_fv_steps': int(diagnostics.get('n_steps', -1)),
                    'maximum_abs_mass_error': float(diagnostics.get('maximum_abs_mass_error', np.nan)),
                    'minimum_cell_mass': float(diagnostics.get('minimum_cell_mass', np.nan)),
                },
            )
    return _add_distance_to_finest(rows, axis='quadrature', finest_level=levels[-1])


def evaluate_reference_fv(problem, candidates, numerics=REFERENCE_NUMERICS):
    """Evaluate all fixed candidates at the provisional high-resolution FV setup."""
    candidates = _validate_candidates(problem, candidates)
    evaluator = problem.build_evaluator(numerics, fast=True)
    return {name: evaluator.evaluate(prices) for name, prices in candidates.items()}


def run_final_mc_comparison(
    problem,
    candidates,
    *,
    n_paths,
    numerics=REFERENCE_NUMERICS,
    prefix_counts=MC_PREFIX_COUNTS,
    initial_seed=MC_INITIAL_SEED,
    brownian_seed=MC_BROWNIAN_SEED,
):
    """Compare provisional high-resolution FV against common-random-number MC."""
    n_paths = int(n_paths)
    if n_paths < 2:
        raise ValueError('n_paths must be at least two.')
    candidates = _validate_candidates(problem, candidates)
    evaluator = problem.build_evaluator(numerics, fast=True)
    fv = {name: evaluator.evaluate(prices) for name, prices in candidates.items()}
    mc = evaluate_mc_candidates_common_random_numbers(
        evaluator,
        candidates,
        n_paths=n_paths,
        initial_seed=initial_seed,
        brownian_seed=brownian_seed,
    )
    rows = []
    prefixes = {}
    for name in candidates:
        fv_total = float(fv[name].objective.total)
        objective = mc[name].objective
        mc_total = float(objective.objective.total)
        se = float(objective.standard_error)
        rows.append(
            {
                'candidate': name,
                'n_paths': n_paths,
                'fv_total': fv_total,
                'fv_bias': float(fv[name].objective.bias),
                'fv_variance': float(fv[name].objective.variance),
                'mc_total': mc_total,
                'mc_bias': float(objective.objective.bias),
                'mc_variance': float(objective.objective.variance),
                'mc_standard_error': se,
                'mc_minus_fv': mc_total - fv_total,
                'abs_mc_minus_fv_over_se': abs(mc_total - fv_total) / se if se > 0.0 else float('inf'),
                'raw_boundary_violations': int(mc[name].diagnostics['total_violations']),
            },
        )
        counts = sorted(
            set((int(value) for value in prefix_counts if 2 <= int(value) <= n_paths)) | {n_paths},
        )
        prefixes[name] = _prefix_statistics(objective.path_objectives, counts)
    paired = {}
    reference_name = 'deterministic_reference'
    if reference_name in mc:
        for name, item in mc.items():
            if name == reference_name:
                continue
            mean, se = paired_mc_improvement(mc[reference_name], item)
            paired[name] = {
                'mean_reference_minus_candidate': float(mean),
                'standard_error': float(se),
                'ratio': float(mean / se) if se > 0.0 else float('inf'),
            }
    return {'rows': rows, 'prefixes': prefixes, 'paired_improvement': paired}


def run_coupled_mc_time_refinement(
    problem,
    candidates,
    *,
    n_paths,
    solver_max_steps=TIME_SOLVER_MAX_STEPS,
    objective_max_step=TIME_OBJECTIVE_MAX_STEP,
    initial_seed=MC_TIME_INITIAL_SEED,
    brownian_seed=MC_TIME_BROWNIAN_SEED,
):
    """Refine EM time using the same paths with summed fine Wiener increments."""
    n_paths = int(n_paths)
    if n_paths < 2:
        raise ValueError('n_paths must be at least two.')
    candidates = _validate_candidates(problem, candidates)
    steps = tuple((float(value) for value in solver_max_steps))
    if (
        len(steps) < 2
        or any((value <= 0.0 for value in steps))
        or any((right >= left for left, right in zip(steps[:-1], steps[1:])))
    ):
        raise ValueError('solver_max_steps must be strictly decreasing.')
    evaluators = {}
    for step in steps:
        numerics = BM1ControlNumerics(
            x_steps=31,
            objective_max_step=objective_max_step,
            solver_max_step=step,
        )
        evaluators[step] = problem.build_evaluator(numerics, fast=True)
    finest_step = steps[-1]
    finest = evaluators[finest_step]
    for evaluator in evaluators.values():
        np.testing.assert_allclose(evaluator.solver_t_eval, finest.solver_t_eval, atol=0.0, rtol=0.0)
    model = finest.model
    initial_states = sample_beta_initial_states(n_paths, initial_seed)
    integration_grids = _nested_em_time_grids(finest.solver_t_eval, finest.breakpoints, steps)
    fine_t = integration_grids[finest_step]
    fine_dW = model.euler_maruyama.brownian_increments(fine_t, n_paths=n_paths, seed=brownian_seed)
    rows = []
    path_costs = {}
    terminal_states = {}
    integration_steps = {}
    for step in steps:
        evaluator = evaluators[step]
        integration_t = integration_grids[step]
        dW = _aggregate_brownian_to_grid(fine_t, fine_dW, integration_t)
        integration_steps[step] = int(integration_t.size - 1)
        for name, prices in candidates.items():
            parameters = evaluator.parameters_for_prices(prices)
            state, demand, diagnostics = _em_outputs_on_integration_grid(
                model,
                evaluator,
                parameters,
                initial_states,
                integration_t,
                dW,
            )
            midpoint_demand = np.asarray(demand, dtype=float)[:, 1:-1]
            objective = tracking.monte_carlo_tracking_objective(
                evaluator.widths,
                midpoint_demand,
                evaluator.target,
            )
            key = (name, step)
            path_costs[key] = objective.path_objectives
            terminal_states[key] = np.asarray(state, dtype=float)[:, -1]
            rows.append(
                {
                    'candidate': name,
                    'solver_max_step': step,
                    'n_paths': n_paths,
                    'integration_steps': integration_steps[step],
                    'total': float(objective.objective.total),
                    'standard_error': float(objective.standard_error),
                    'raw_boundary_violations': int(diagnostics['total_violations']),
                },
            )
    for row in rows:
        name = row['candidate']
        step = row['solver_max_step']
        cost_difference = path_costs[name, step] - path_costs[name, finest_step]
        state_difference = terminal_states[name, step] - terminal_states[name, finest_step]
        row['mean_path_cost_difference_to_finest'] = float(np.mean(cost_difference))
        row['se_path_cost_difference_to_finest'] = float(np.std(cost_difference, ddof=1) / np.sqrt(n_paths))
        row['rms_terminal_state_difference_to_finest'] = float(
            np.sqrt(np.mean(np.square(state_difference))),
        )
    return rows


def run_fixed_policy_validation(
    problem,
    candidates,
    *,
    reference_name,
    mc_paths,
    mc_time_paths=0,
    numerics=REFERENCE_NUMERICS,
    solver_max_steps=TIME_SOLVER_MAX_STEPS,
    objective_max_step=None,
    initial_seed=MC_INITIAL_SEED,
    brownian_seed=MC_BROWNIAN_SEED,
    mc_time_initial_seed=MC_TIME_INITIAL_SEED,
    mc_time_brownian_seed=MC_TIME_BROWNIAN_SEED,
):
    """High-resolution FV/MC validation of already-fixed price policies.

    No optimization is performed.  This is useful when candidate policies are
    so close that the cheaper F3 working resolution cannot resolve their cost
    difference reliably.
    """
    if not isinstance(numerics, BM1ControlNumerics):
        raise TypeError('numerics must be a BM1ControlNumerics instance.')
    if isinstance(mc_paths, (bool, np.bool_)) or int(mc_paths) != mc_paths or int(mc_paths) < 2:
        raise ValueError('mc_paths must be an integer of at least two.')
    if isinstance(mc_time_paths, (bool, np.bool_)) or int(mc_time_paths) != mc_time_paths or int(mc_time_paths) < 0:
        raise ValueError('mc_time_paths must be a non-negative integer.')
    candidates = _validate_candidates(problem, candidates)
    reference_name = str(reference_name)
    if reference_name not in candidates:
        raise ValueError('reference_name must identify one supplied candidate.')

    evaluator = problem.build_evaluator(numerics, fast=True)
    fv = {name: evaluator.evaluate(prices) for name, prices in candidates.items()}
    fv_differences = {
        name: float(fv[name].total - fv[reference_name].total)
        for name in candidates
        if name != reference_name
    }

    mc = evaluate_mc_candidates_common_random_numbers(
        evaluator,
        candidates,
        n_paths=int(mc_paths),
        initial_seed=initial_seed,
        brownian_seed=brownian_seed,
    )
    paired_mc = {}
    for name in candidates:
        if name == reference_name:
            continue
        mean, standard_error = paired_mc_improvement(mc[reference_name], mc[name])
        paired_mc[name] = {
            'reference_minus_candidate': float(mean),
            'standard_error': float(standard_error),
            'ratio': float(mean / standard_error) if standard_error > 0.0 else float('inf'),
        }

    mc_time = None
    paired_mc_time = None
    if int(mc_time_paths) > 0:
        if objective_max_step is None:
            objective_max_step = numerics.objective_max_step
        mc_time = run_coupled_mc_time_refinement(
            problem,
            candidates,
            n_paths=int(mc_time_paths),
            solver_max_steps=solver_max_steps,
            objective_max_step=float(objective_max_step),
            initial_seed=mc_time_initial_seed,
            brownian_seed=mc_time_brownian_seed,
        )
        paired_mc_time = []
        for step in solver_max_steps:
            totals = {
                row['candidate']: float(row['total'])
                for row in mc_time
                if np.isclose(row['solver_max_step'], step, atol=0.0, rtol=0.0)
            }
            if set(totals) != set(candidates):
                raise RuntimeError('MC time-refinement output is missing a candidate.')
            paired_mc_time.append(
                {
                    'solver_max_step': float(step),
                    'reference_minus_candidate': {
                        name: float(totals[reference_name] - totals[name])
                        for name in candidates
                        if name != reference_name
                    },
                },
            )

    return {
        'reference_name': reference_name,
        'numerics': numerics,
        'candidates': candidates,
        'fv': fv,
        'fv_differences': fv_differences,
        'mc': mc,
        'paired_mc': paired_mc,
        'mc_time': mc_time,
        'paired_mc_time': paired_mc_time,
    }



def run_f3_f_high_resolution_validation(
    problem,
    candidates,
    *,
    mc_paths,
    mc_time_paths=0,
    numerics=REFERENCE_NUMERICS,
    solver_max_steps=TIME_SOLVER_MAX_STEPS,
    progress=None,
):
    """Re-optimize only the FV reference finely, then validate fixed F3-F policies.

    The expensive mixture policy is not optimized again.  The high-resolution
    FV search uses the F3-B reference numerics so tiny policy regrets are not
    interpreted relative to the cheaper 201-cell working optimum.
    """
    candidates = _validate_candidates(problem, candidates)
    evaluator = problem.build_evaluator(numerics, fast=True)
    starts = np.vstack(
        (
            cases._reference_starts(problem),
            *[prices for prices in candidates.values()],
        ),
    )
    optimization = multistart_powell(
        evaluator,
        starts,
        progress=progress,
        options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
    )
    all_candidates = {'highres_fv_policy': np.asarray(optimization.best.x, dtype=float).copy()}
    all_candidates.update(candidates)
    result = run_fixed_policy_validation(
        problem,
        all_candidates,
        reference_name='highres_fv_policy',
        mc_paths=mc_paths,
        mc_time_paths=mc_time_paths,
        numerics=numerics,
        solver_max_steps=solver_max_steps,
        objective_max_step=numerics.objective_max_step,
    )
    result['highres_fv_optimization'] = optimization
    return result

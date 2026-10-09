"""Isolated positive-CN joint experiment; production additive defaults unchanged."""

import numpy as np

from src.solver.finite_volume import finite_volume_1d_fitted


# Fixed historical comparison fixture; its completed campaign drivers live in the archive.
CANDIDATES = {
    'linear': ([0.], [0.]),
    'two_saved': ([.3], [-.25]),
    'two_near': ([.32], [-.23]),
}

SEARCH_SETTINGS = {
    'joint161': dict(cells=161, max_step=1/768, accumulator_spacing=1/256,
                     max_accumulator_nodes=4096, backend='fitted'),
    'joint321': dict(cells=321, max_step=1/1536, accumulator_spacing=1/512,
                     max_accumulator_nodes=4096, backend='fitted'),
}

def _record(development, cn_reference, name):
    from dataclasses import replace
    from src.validation.bm_1.interval_curve_control import declared_control
    from src.params.bm_1_response import BM1PiecewiseResponse
    record, _ = declared_control(development, cn_reference, 'two-narrow')
    f, g = [BM1PiecewiseResponse.from_inner_values(values, pieces=2)
            for values in CANDIDATES[name]]
    return replace(record, record_id=name,
                   parameters=(record.parameters[0], f, g, *record.parameters[3:]))


class PositiveCNJointSolver(finite_volume_1d_fitted):
    """Reuse the existing implicit LAPACK solve with an explicit positive half.

    Column generator Q has nonnegative off-diagonals and column sum zero.
    If 1+h*diag(Q)/2 >= 0, both (I+hQ/2) and (I-hQ/2)^-1 are positive
    and preserve each column's mass. Their product is Crank--Nicolson.
    This also works for a matrix of accumulator columns; no repair is needed.
    """

    def _stepper(self, dx, edges, centers, drift, diffusion, coefficient_key):
        implicit = super()._stepper(dx, edges, centers, drift, diffusion, coefficient_key)

        def advance(mass, time, step):
            lower, diagonal, upper, _ = self.generator_tridiagonal(dx,
                np.broadcast_to(drift(time, edges[1:-1]), edges[1:-1].shape),
                np.broadcast_to(diffusion(time, centers), centers.shape))
            explicit = 1.+.5*step*diagonal
            if np.any(explicit < 0.):
                raise ValueError('Joint CN positivity condition fails; do not clip or adapt the step.')
            tail = (None,)*(mass.ndim-1)
            rhs = explicit[(slice(None),)+tail]*mass
            rhs[1:] += (.5*step*lower)[(slice(None),)+tail]*mass[:-1]
            rhs[:-1] += (.5*step*upper)[(slice(None),)+tail]*mass[1:]
            result, diagnostics = implicit(rhs, time, .5*step)
            diagnostics = dict(diagnostics, minimum_cn_explicit_diagonal=float(explicit.min()))
            return result, diagnostics

        return advance


def run_joint_cn(development, cn_reference, candidate, *, intervals=6, accumulator_spacing=1/512,
                 saved_prefix=()):
    """Fresh whole prefix from the declared prior; no cross-candidate states."""
    from time import perf_counter
    from src.method.bm_1_interval_likelihood import run_interval_filter
    from src.model.bm_1 import bm_1
    from src.params.bm_1 import BM1IntervalObservationSeries
    record = _record(development, cn_reference, candidate)
    settings = dict(SEARCH_SETTINGS['joint321'], accumulator_spacing=accumulator_spacing)
    model = bm_1(record.baseline, record.baseline, record.horizon, 2)
    solver = PositiveCNJointSolver(vectorized_inputs=True)
    model.fokker_planc_reference = solver
    edges, _, _ = solver.grid(settings['cells'])
    observations = record.observations.prefix(intervals/24.)
    offset = len(saved_prefix)
    if offset:
        if (offset >= intervals or any(r['candidate'] != candidate or r['interval_index'] != i
                or r['x_edges'] != edges.tolist() for i, r in enumerate(saved_prefix))):
            raise ValueError('Continuation must use the matching candidate/grid/prefix.')
        initial = np.array(saved_prefix[-1]['posterior_mass'])
        initial_time = observations.ends[offset-1]
        observations = BM1IntervalObservationSeries(observations.starts[offset:], observations.ends[offset:],
            observations.values[offset:], observations.time_unit, observations.observed[offset:])
    else:
        initial, initial_time = record.initial_mass_builder(edges), record.horizon[0]
    begun = perf_counter()
    result = run_interval_filter(model, record.parameters, record.sigma_x, record.sigma_y,
        initial, initial_time, observations,
        breakpoints=record.breakpoints,
        **{k: v for k, v in settings.items() if k not in ('cells', 'backend')})
    elapsed = perf_counter()-begun
    rows = [dict(r, reused=True) for r in saved_prefix]
    for index, (score, mass, mean) in enumerate(zip(result.log_evidence,
            result.posterior_mass, result.posterior_mean)):
        rows.append(dict(candidate=candidate, method='joint321', interval_index=index+offset,
            observed=bool(observations.observed[index]),
            log_evidence=float(score) if observations.observed[index] else None,
            x_edges=edges.tolist(), posterior_mass=mass.tolist(),
            posterior_state_mean=float(mean), elapsed_seconds=elapsed/len(observations.times),
            diagnostics={k: result.prediction_diagnostics[index][k] for k in (
                'maximum_abs_mass_error', 'minimum_cell_mass', 'accumulator_deposition_variance_bound')},
            reused=False))
    return dict(candidate=candidate, elapsed_seconds=elapsed, rows=rows,
        reused_intervals=offset,
        note='New per-hour runtime is fresh replay time divided by fresh hours; preserved prefix timing unchanged.')

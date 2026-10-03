"""Independent BM1 filters sharing a process operator across sigma_y values.

Only the right-hand-side solve is shared. Each hypothesis retains a distinct
conditional posterior and its own likelihood/evidence, including missing data.
The scalar filter remains the independent reference for this opt-in path.
"""

import numpy as np

from src.method.bm_1_estimation import (
    BM1FVFilterResult, _cell_log_likelihood, _filter_interval, _positive_filter_step,
)
from src.params.bm_1 import BM1ObservationSeries, bayes_update, mass_to_log
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation import estimation as checks
from src.validation.model.bm_1 import bm_1_validation


def run_fv_sigma_y_batch(model, parameters, sigma_x, sigma_y_values,
                         initial_mass, initial_time, observations, *,
                         breakpoints=None, max_step=np.inf,
                         likelihood_quadrature_order=1):
    """Return one ordinary filter result per sigma_y, in caller order.

    All physical process parameters, inputs, grid and observation times are
    shared. ``initial_mass`` is a common vector or one probability row per
    hypothesis. Batch width controls memory, not scientific approximation.
    """
    if not isinstance(observations, BM1ObservationSeries):
        raise TypeError('observations must be a BM1ObservationSeries.')
    if not isinstance(model.fokker_planc_reference, finite_volume_1d_cached):
        raise TypeError('Batched filtering requires finite_volume_1d_cached.')
    _, parameters, sigma_x = bm_1_validation.stochastic_inputs([.5], parameters, sigma_x)
    noises = checks.finite_vector(sigma_y_values, 'sigma_y_values')
    if np.any(noises <= 0.):
        raise ValueError('Every sigma_y must be strictly positive.')
    time = checks.finite_scalar(initial_time, 'initial_time')
    if observations.times[0] <= time:
        raise ValueError('All observation times must be strictly after initial_time.')
    max_step = _positive_filter_step(max_step, 'max_step')
    order = checks.positive_integer(likelihood_quadrature_order, 'likelihood_quadrature_order')
    all_breaks = (np.empty(0) if breakpoints is None else
                  checks.finite_vector(breakpoints, 'breakpoints', minimum_size=0))
    if np.iscomplexobj(initial_mass):
        raise ValueError('initial_mass must be real.')
    initial = np.asarray(initial_mass, dtype=float)
    if initial.ndim == 1:
        initial = np.tile(checks.probability_mass(initial), (len(noises), 1))
    else:
        initial = checks.probability_rows(initial, 'initial_mass')
    if len(initial) != len(noises):
        raise ValueError('One initial probability row is required per sigma_y.')
    current = initial.copy()
    shape = (len(noises), len(observations.times), initial.shape[1])
    predicted, posterior = np.empty(shape), np.empty(shape)
    scores = np.full(shape[:2], np.nan)
    diagnostics = [[] for _ in noises]
    grids = []
    for index, (stop, value, observed) in enumerate(zip(
            observations.times, observations.values, observations.observed)):
        start, stop = _filter_interval(model, time, float(stop))
        local_breaks = all_breaks[(all_breaks > start) & (all_breaks < stop)]
        t_eval = np.array([start, stop])
        drift, diffusion, options = model._fv_evolution_fields(
            parameters, sigma_x, t_eval, local_breaks, max_step)
        evolution = model.fokker_planc_reference.solve_many(
            t_eval, current, drift, diffusion, breakpoints=local_breaks,
            max_step=max_step, **options)
        predicted[:, index] = evolution['cell_mass'][-1]
        grid = evolution['prediction_diagnostics'][0]['integration_t']
        grids.append(grid if not grids else grid[1:])
        for column, noise in enumerate(noises):
            diagnostic = dict(evolution['prediction_diagnostics'][column],
                actual_max_step=float(np.max(np.diff(grid))), likelihood_quadrature_order=order)
            diagnostics[column].append(diagnostic)
            if observed:
                likelihood = _cell_log_likelihood(model, parameters, stop, start,
                    evolution['x_edges'], float(value), float(noise), observations.price_side, order)
                update = bayes_update(mass_to_log(predicted[column, index]), likelihood)
                posterior[column, index] = update.mass
                scores[column, index] = update.log_evidence
            else:
                posterior[column, index] = predicted[column, index]
        current = posterior[:, index]
        time = stop
    integration_t = np.concatenate(grids)
    return tuple(BM1FVFilterResult(initial_time=initial_time, initial_mass=initial[column],
        times=observations.times, x_edges=evolution['x_edges'], x_grid=evolution['x_grid'],
        predicted_mass=predicted[column], posterior_mass=posterior[column],
        observed=observations.observed, log_evidence=scores[column], integration_t=integration_t,
        prediction_diagnostics=tuple(diagnostics[column])) for column in range(len(noises)))

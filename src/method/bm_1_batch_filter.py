"""Independent BM1 filters sharing a process operator across sigma_y values.

Only the right-hand-side solve is shared. Each hypothesis retains a distinct
conditional posterior and its own likelihood/evidence, including missing data.
The scalar filter remains the independent reference for this opt-in path.
"""

import numpy as np

from src.method.bm_1_estimation import (
    BM1FVFilterResult, _cell_log_likelihood, _filter_interval, _positive_filter_step,
)
from src.params.bm_1 import BM1ObservationSeries
from src.method.bm_1_observation import bayes_update, mass_to_log
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


def collapse_sigma_y_for_process_forecast(hypotheses, log_weights, conditional_mass):
    """Exactly retain the represented node-bank state--sigma_x law.

    The hypothesis columns must be (sigma_x, sigma_y), with all other process
    parameters and future inputs fixed. Common linear propagation at each
    sigma_x then preserves the full state/demand distribution and expected F3
    objective under one shared policy. No sigma_x values are averaged.

    This reduction is for a frozen origin with no further assimilation. The
    resulting law does not retain sigma_y--state dependence, so it cannot
    replace the full bank for future Y densities or observation updates.
    A zero-weight process group retains a valid arbitrary representative state.
    """
    from scipy.special import logsumexp
    if np.iscomplexobj(hypotheses):
        raise ValueError('hypotheses must be real.')
    hypotheses = np.asarray(hypotheses, dtype=float)
    if (hypotheses.ndim != 2 or hypotheses.shape[1] != 2 or len(hypotheses) == 0
            or not np.all(np.isfinite(hypotheses)) or np.any(hypotheses[:, 0] < 0.)
            or np.any(hypotheses[:, 1] <= 0.)):
        raise ValueError('hypotheses must contain finite (sigma_x >= 0, sigma_y > 0) rows.')
    logs = checks.log_probability_mass(log_weights, 'log_weights')
    states = checks.probability_rows(conditional_mass, 'conditional_mass')
    if len(hypotheses) != len(logs) or len(states) != len(logs):
        raise ValueError('Hypotheses, weights and conditional states must have equal row counts.')
    processes = np.unique(hypotheses[:, 0])
    process_logs, initial = [], []
    for process in processes:
        selected = hypotheses[:, 0] == process
        local_logs = logs[selected]
        weight = float(logsumexp(local_logs))
        state = (np.exp(local_logs-weight) @ states[selected] if np.isfinite(weight) else
                 states[selected][0])
        initial.append(checks.probability_mass(state, 'grouped initial state'))
        process_logs.append(weight)
    return dict(sigma_x=processes, log_weights=np.array(process_logs), initial_mass=np.stack(initial),
                scope='Frozen-origin state/demand/F3 objective only; not Y likelihood or continued assimilation')

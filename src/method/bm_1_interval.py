"""Recursive interval-average BM1 likelihood, separate from point observations."""

import numpy as np

from src.method.bm_1_estimation import (BM1FVFilterStep, BM1FVFilterResult,
    BM1FVLikelihoodResult, fv_filter_step, _filter_interval)
from src.params.bm_1 import bayes_update, mass_to_log, gaussian_log_likelihood
from src.params.bm_1_interval import BM1IntervalObservationSeries
from src.solver.finite_volume_additive import propagate_additive
from src.validation import estimation as checks
from src.validation.model.bm import bm_validation


def interval_filter_step(model, parameters, sigma_x, sigma_y, prior_mass,
                         start, end, observation, *, observed=True, breakpoints=None,
                         max_step=1/768, accumulator_spacing=1/256,
                         max_accumulator_nodes=4096):
    """Assimilate the joint interval law and carry its end-state posterior."""
    start, end = _filter_interval(model, start, end)
    parameters = bm_validation.parameters(parameters)
    sigma_x = checks.finite_scalar(sigma_x, 'sigma_x')
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    if sigma_x < 0.:
        raise ValueError('sigma_x must be nonnegative.')
    if not isinstance(observed, (bool, np.bool_)):
        raise TypeError('observed must be boolean.')
    breaks = checks.finite_vector([] if breakpoints is None else breakpoints,
                                  'breakpoints', minimum_size=0)
    local = breaks[(breaks > start) & (breaks < end)]
    drift, diffusion, options = model._fv_evolution_fields(
        parameters, sigma_x, np.array([start, end]), local, max_step)

    def reward(time, states):
        return model.eval(time, states, parameters[0], parameters[1],
                          parameters[2], parameters[4], parameters[5])[-1]

    prediction = propagate_additive(model.fokker_planc_reference, (start, end),
        prior_mass, drift, diffusion, reward, breakpoints=local, max_step=max_step,
        accumulator_spacing=accumulator_spacing, max_accumulator_nodes=max_accumulator_nodes,
        **options)
    joint = prediction['joint_mass']
    predicted = joint.sum(axis=1)
    if observed:
        logs = gaussian_log_likelihood(observation, prediction['accumulator'], sigma_y)
        # Bayes on the JOINT law preserves the dependence of hour consumption
        # on the terminal state. A Gaussian of E[D] is not this likelihood.
        update = bayes_update(mass_to_log(joint.ravel()), np.broadcast_to(logs, joint.shape).ravel())
        posterior = update.mass.reshape(joint.shape).sum(axis=1)
        evidence = update.log_evidence
    else:
        posterior = predicted.copy()
        evidence = np.nan
    return BM1FVFilterStep(end, prediction['x_edges'], prediction['x_grid'],
        predicted, posterior, bool(observed), evidence, prediction['diagnostics'])


def run_interval_filter(model, parameters, sigma_x, sigma_y, initial_mass,
                        initial_time, observations, *, breakpoints=None,
                        max_step=1/768, accumulator_spacing=1/256,
                        max_accumulator_nodes=4096, keep_history=True):
    """Replay all complete intervals from one prior; never reset at midnight.

    Gaps before/between observations propagate the state without likelihood.
    Missing intervals propagate the joint law without scoring or imputation.
    Only one interval's joint law is kept, even with full marginal history.
    """
    if not isinstance(observations, BM1IntervalObservationSeries):
        raise TypeError('observations must be BM1IntervalObservationSeries.')
    if not isinstance(keep_history, (bool, np.bool_)):
        raise TypeError('keep_history must be boolean.')
    initial_time = checks.finite_scalar(initial_time, 'initial_time')
    if initial_time < model.T[0] or observations.starts[0] < initial_time:
        raise ValueError('Prior cannot follow the first interval or precede the model.')
    current = checks.probability_mass(initial_mass, 'initial_mass')
    time = initial_time
    steps, scores, grids = [], [], []
    maximum_error, minimum_mass, actual_max_step, n_steps = 0., np.inf, 0., 0
    for start, end, value, seen in zip(observations.starts, observations.ends,
                                     observations.values, observations.observed):
        gap = None
        if start > time:
            gap = fv_filter_step(model, parameters, sigma_x, sigma_y, current,
                time, start, np.nan, observed=False, breakpoints=breakpoints, max_step=max_step)
            current = gap.posterior_mass
        step = interval_filter_step(model, parameters, sigma_x, sigma_y, current,
            start, end, float(value) if seen else np.nan, observed=bool(seen),
            breakpoints=breakpoints, max_step=max_step, accumulator_spacing=accumulator_spacing,
            max_accumulator_nodes=max_accumulator_nodes)
        d = step.diagnostics
        if gap is not None:
            gd = gap.diagnostics
            d['integration_t'] = np.r_[gd['integration_t'], d['integration_t'][1:]]
            d['n_steps'] += gd['n_steps']
            for name in ('maximum_abs_mass_error', 'actual_max_step'):
                d[name] = max(d[name], gd[name])
            d['minimum_cell_mass'] = min(d['minimum_cell_mass'], gd['minimum_cell_mass'])
        scores.append(step.log_evidence)
        maximum_error = max(maximum_error, d['maximum_abs_mass_error'])
        minimum_mass = min(minimum_mass, d['minimum_cell_mass'])
        actual_max_step = max(actual_max_step, d['actual_max_step'])
        n_steps += d['n_steps']
        if keep_history:
            steps.append(step)
            grids.append(d['integration_t'] if not grids else d['integration_t'][1:])
        current, time = step.posterior_mass, step.time
    if not keep_history:
        return BM1FVLikelihoodResult(step, scores, observations.observed,
            maximum_error, minimum_mass, actual_max_step, n_steps)
    return BM1FVFilterResult(initial_time, initial_mass, observations.ends,
        steps[0].x_edges, steps[0].x_grid,
        np.stack([s.predicted_mass for s in steps]), np.stack([s.posterior_mass for s in steps]),
        observations.observed, scores, np.concatenate(grids), tuple(s.diagnostics for s in steps))

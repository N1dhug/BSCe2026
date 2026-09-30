"""Recursive BM1 state estimation with known parameters.

This module composes the existing conservative FV/Fokker--Planck solver with
the generic F4 observation/Bayes primitives. It contains estimator mechanics,
not synthetic experiments, plotting, control, or parameter fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from src.method.beta_mixture import PreparedBetaProjector, evolve_projected_density

from src.params.bm_1 import (
    BM1ObservationSeries, bayes_update, gaussian_log_likelihood, mass_to_log,
)
from src.validation import estimation as checks

@dataclass(frozen=True)
class BM1FVFilterStep:
    """One known-parameter FV prediction/update at one observation time."""

    time: float
    x_edges: ArrayLike
    x_grid: ArrayLike
    predicted_mass: ArrayLike
    posterior_mass: ArrayLike
    observed: bool
    log_evidence: float
    diagnostics: dict

    def __post_init__(self):
        time = checks.finite_scalar(self.time, 'time')
        x_edges = checks.increasing_grid(self.x_edges, 'x_edges')
        x_grid = checks.finite_vector(self.x_grid, 'x_grid')
        predicted = checks.probability_mass(self.predicted_mass, 'predicted_mass')
        posterior = checks.probability_mass(self.posterior_mass, 'posterior_mass')
        if x_grid.size != predicted.size or posterior.size != predicted.size or x_edges.size != predicted.size + 1:
            raise ValueError('Filter grids and masses must have consistent cell counts.')
        if not isinstance(self.observed, (bool, np.bool_)):
            raise TypeError('observed must be boolean.')
        observed = bool(self.observed)
        log_evidence = float(self.log_evidence)
        if observed and not np.isfinite(log_evidence):
            raise ValueError('Observed filter steps require finite log evidence.')
        if not observed and not np.isnan(log_evidence):
            raise ValueError('Missing-observation filter steps use NaN log evidence.')
        diagnostics = dict(self.diagnostics)
        for array in (x_edges, x_grid, predicted, posterior):
            array.setflags(write=False)
        object.__setattr__(self, 'time', time)
        object.__setattr__(self, 'x_edges', x_edges)
        object.__setattr__(self, 'x_grid', x_grid)
        object.__setattr__(self, 'predicted_mass', predicted)
        object.__setattr__(self, 'posterior_mass', posterior)
        object.__setattr__(self, 'observed', observed)
        object.__setattr__(self, 'log_evidence', log_evidence)
        object.__setattr__(self, 'diagnostics', diagnostics)

    @property
    def predicted_mean(self):
        return float(self.predicted_mass @ self.x_grid)

    @property
    def posterior_mean(self):
        return float(self.posterior_mass @ self.x_grid)


@dataclass(frozen=True)
class BM1FVFilterResult:
    """Replay result for the known-parameter finite-volume state filter."""

    initial_time: float
    initial_mass: ArrayLike
    times: ArrayLike
    x_edges: ArrayLike
    x_grid: ArrayLike
    predicted_mass: ArrayLike
    posterior_mass: ArrayLike
    observed: ArrayLike
    log_evidence: ArrayLike
    integration_t: ArrayLike
    prediction_diagnostics: tuple

    def __post_init__(self):
        initial_time = checks.finite_scalar(self.initial_time, 'initial_time')
        initial_mass = checks.probability_mass(self.initial_mass, 'initial_mass')
        times = checks.increasing_grid(self.times, 'filter times', minimum_size=1)
        x_edges = checks.increasing_grid(self.x_edges, 'x_edges')
        x_grid = checks.finite_vector(self.x_grid, 'x_grid')
        predicted = np.asarray(self.predicted_mass, dtype=float).copy()
        posterior = np.asarray(self.posterior_mass, dtype=float).copy()
        observed = np.asarray(self.observed).copy()
        log_evidence = np.asarray(self.log_evidence, dtype=float).copy()
        integration_t = checks.increasing_grid(self.integration_t, 'integration_t', minimum_size=2)
        shape = (times.size, x_grid.size)
        if predicted.shape != shape or posterior.shape != shape:
            raise ValueError('Filter mass histories must have shape (times, cells).')
        if x_edges.size != x_grid.size + 1 or initial_mass.size != x_grid.size:
            raise ValueError('Filter grid and initial mass must share the same cell count.')
        if observed.shape != times.shape or observed.dtype.kind != 'b':
            raise ValueError('observed must be a boolean vector matching filter times.')
        if log_evidence.shape != times.shape:
            raise ValueError('log_evidence must match filter times.')
        if np.any(~np.isfinite(log_evidence[observed])) or np.any(~np.isnan(log_evidence[~observed])):
            raise ValueError('Log evidence must be finite exactly where observations are present.')
        for row in predicted:
            checks.probability_mass(row, 'predicted mass row')
        for row in posterior:
            checks.probability_mass(row, 'posterior mass row')
        diagnostics = tuple(dict(item) for item in self.prediction_diagnostics)
        if len(diagnostics) != times.size:
            raise ValueError('One prediction diagnostic record is required per filter time.')
        for array in (initial_mass, times, x_edges, x_grid, predicted, posterior, observed, log_evidence, integration_t):
            array.setflags(write=False)
        object.__setattr__(self, 'initial_time', initial_time)
        object.__setattr__(self, 'initial_mass', initial_mass)
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'x_edges', x_edges)
        object.__setattr__(self, 'x_grid', x_grid)
        object.__setattr__(self, 'predicted_mass', predicted)
        object.__setattr__(self, 'posterior_mass', posterior)
        object.__setattr__(self, 'observed', observed)
        object.__setattr__(self, 'log_evidence', log_evidence)
        object.__setattr__(self, 'integration_t', integration_t)
        object.__setattr__(self, 'prediction_diagnostics', diagnostics)

    @property
    def predicted_mean(self):
        return self.predicted_mass @ self.x_grid

    @property
    def posterior_mean(self):
        return self.posterior_mass @ self.x_grid

    @property
    def total_log_likelihood(self):
        return float(np.sum(self.log_evidence[self.observed]))

    @property
    def maximum_abs_mass_error(self):
        return float(max(item['maximum_abs_mass_error'] for item in self.prediction_diagnostics))

    @property
    def minimum_cell_mass(self):
        return float(min(item['minimum_cell_mass'] for item in self.prediction_diagnostics))



@dataclass(frozen=True)
class BM1MixtureFilterResult:
    """Replay result for the known-parameter recursive fixed-basis mixture filter."""

    initial_time: float
    initial_mass: ArrayLike
    initial_coefficients: ArrayLike
    times: ArrayLike
    x_edges: ArrayLike
    x_grid: ArrayLike
    predicted_mass: ArrayLike
    posterior_mass: ArrayLike
    predicted_coefficients: ArrayLike
    posterior_coefficients: ArrayLike
    observed: ArrayLike
    log_evidence: ArrayLike
    prediction_diagnostics: tuple
    posterior_projection_diagnostics: tuple
    initial_projection_diagnostics: dict

    def __post_init__(self):
        initial_time = checks.finite_scalar(self.initial_time, 'initial_time')
        initial_mass = checks.probability_mass(self.initial_mass, 'initial_mass')
        initial_coefficients = checks.finite_vector(
            self.initial_coefficients, 'initial_coefficients',
        )
        if np.any(initial_coefficients < 0.0) or abs(float(np.sum(initial_coefficients)) - 1.0) > 1e-9:
            raise ValueError('initial_coefficients must satisfy the probability simplex.')
        times = checks.increasing_grid(self.times, 'filter times', minimum_size=1)
        x_edges = checks.increasing_grid(self.x_edges, 'x_edges')
        x_grid = checks.finite_vector(self.x_grid, 'x_grid')
        predicted = np.asarray(self.predicted_mass, dtype=float).copy()
        posterior = np.asarray(self.posterior_mass, dtype=float).copy()
        pred_coeff = np.asarray(self.predicted_coefficients, dtype=float).copy()
        post_coeff = np.asarray(self.posterior_coefficients, dtype=float).copy()
        observed = np.asarray(self.observed).copy()
        log_evidence = np.asarray(self.log_evidence, dtype=float).copy()
        mass_shape = (times.size, x_grid.size)
        coeff_shape = (times.size, initial_coefficients.size)
        if predicted.shape != mass_shape or posterior.shape != mass_shape:
            raise ValueError('Mixture filter mass histories must have shape (times, cells).')
        if pred_coeff.shape != coeff_shape or post_coeff.shape != coeff_shape:
            raise ValueError('Mixture coefficient histories must have shape (times, components).')
        if x_edges.size != x_grid.size + 1 or initial_mass.size != x_grid.size:
            raise ValueError('Mixture filter grid and initial mass must share cell count.')
        if observed.shape != times.shape or observed.dtype.kind != 'b':
            raise ValueError('observed must be boolean and match times.')
        if log_evidence.shape != times.shape:
            raise ValueError('log_evidence must match filter times.')
        if np.any(~np.isfinite(log_evidence[observed])) or np.any(~np.isnan(log_evidence[~observed])):
            raise ValueError('Log evidence must be finite exactly for observed steps.')
        for row in predicted:
            checks.probability_mass(row, 'mixture predicted mass row')
        for row in posterior:
            checks.probability_mass(row, 'mixture posterior mass row')
        for row in pred_coeff:
            if np.any(row < 0.0) or abs(float(np.sum(row)) - 1.0) > 1e-9:
                raise ValueError('Predicted mixture coefficients must satisfy the simplex.')
        for row in post_coeff:
            if np.any(row < 0.0) or abs(float(np.sum(row)) - 1.0) > 1e-9:
                raise ValueError('Posterior mixture coefficients must satisfy the simplex.')
        prediction_diagnostics = tuple(dict(item) for item in self.prediction_diagnostics)
        projection_diagnostics = tuple(dict(item) for item in self.posterior_projection_diagnostics)
        if len(prediction_diagnostics) != times.size or len(projection_diagnostics) != times.size:
            raise ValueError('One diagnostic record per mixture filter time is required.')
        initial_projection_diagnostics = dict(self.initial_projection_diagnostics)
        for array in (
            initial_mass, initial_coefficients, times, x_edges, x_grid,
            predicted, posterior, pred_coeff, post_coeff, observed, log_evidence,
        ):
            array.setflags(write=False)
        object.__setattr__(self, 'initial_time', initial_time)
        object.__setattr__(self, 'initial_mass', initial_mass)
        object.__setattr__(self, 'initial_coefficients', initial_coefficients)
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'x_edges', x_edges)
        object.__setattr__(self, 'x_grid', x_grid)
        object.__setattr__(self, 'predicted_mass', predicted)
        object.__setattr__(self, 'posterior_mass', posterior)
        object.__setattr__(self, 'predicted_coefficients', pred_coeff)
        object.__setattr__(self, 'posterior_coefficients', post_coeff)
        object.__setattr__(self, 'observed', observed)
        object.__setattr__(self, 'log_evidence', log_evidence)
        object.__setattr__(self, 'prediction_diagnostics', prediction_diagnostics)
        object.__setattr__(self, 'posterior_projection_diagnostics', projection_diagnostics)
        object.__setattr__(self, 'initial_projection_diagnostics', initial_projection_diagnostics)

    @property
    def predicted_mean(self):
        return self.predicted_mass @ self.x_grid

    @property
    def posterior_mean(self):
        return self.posterior_mass @ self.x_grid

    @property
    def total_log_likelihood(self):
        return float(np.sum(self.log_evidence[self.observed]))


def _simplex_coefficients(values, size, name):
    coefficients = checks.finite_vector(values, name)
    if coefficients.size != size:
        raise ValueError(f'{name} must have one value per mixture component.')
    if np.any(coefficients < 0.0) or abs(float(np.sum(coefficients)) - 1.0) > 1e-9:
        raise ValueError(f'{name} must satisfy the probability simplex.')
    return coefficients


def mixture_filter_step(
    model,
    parameters,
    sigma_x,
    sigma_y,
    prior_coefficients,
    previous_time,
    observation_time,
    observation_value,
    *,
    projector,
    observed=True,
    price_side='right',
    breakpoints=None,
    max_step=np.inf,
    cfl=0.4,
    likelihood_quadrature_order=1,
):
    """One recursive fixed-basis mixture prediction/update with no FV restart."""
    if not isinstance(projector, PreparedBetaProjector):
        raise TypeError('projector must be a PreparedBetaProjector.')
    previous_time = checks.finite_scalar(previous_time, 'previous_time')
    observation_time = checks.finite_scalar(observation_time, 'observation_time')
    if observation_time <= previous_time:
        raise ValueError('observation_time must be strictly after previous_time.')
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    max_step = _positive_filter_step(max_step, 'max_step')
    cfl = checks.finite_scalar(cfl, 'cfl', positive=True)
    if cfl >= 1.0:
        raise ValueError('cfl must lie strictly below one.')
    if not isinstance(observed, (bool, np.bool_)):
        raise TypeError('observed must be boolean.')
    if price_side not in ('left', 'right'):
        raise ValueError('price_side must be left or right.')
    order = checks.positive_integer(likelihood_quadrature_order, 'likelihood_quadrature_order')
    coefficients = _simplex_coefficients(
        prior_coefficients, projector.basis.shape[0], 'prior_coefficients',
    )

    edges, x_grid, dx = model.fokker_planc_reference.grid(
        projector.basis.shape[1], bounds=(0.0, 1.0),
    )
    if not np.isclose(dx, projector.dx, atol=1e-14, rtol=1e-10):
        raise ValueError('projector spacing does not match the BM1 state grid.')
    if breakpoints is None:
        local_breakpoints = np.empty(0, dtype=float)
    else:
        all_breakpoints = checks.finite_vector(breakpoints, 'breakpoints', minimum_size=0)
        local_breakpoints = all_breakpoints[
            (all_breakpoints > previous_time) & (all_breakpoints < observation_time)
        ]

    def drift(time, state):
        return model._drift(time, state, parameters)

    def diffusion(time, state):
        del time
        return model.diffusion_coefficient(state, sigma_x)

    evolved = evolve_projected_density(
        method=model.fokker_planc,
        t_eval=np.array([previous_time, observation_time], dtype=float),
        breakpoints=local_breakpoints,
        x=x_grid,
        dx=dx,
        basis=projector.basis,
        initial_coefficients=coefficients,
        drift=drift,
        diffusion=diffusion,
        max_step=max_step,
        cfl=cfl,
        projection_data=projector.projection_data,
        projection_solver=projector.projection_solver,
    )
    predicted_density = evolved['density'][-1]
    predicted_coefficients = evolved['coefficients'][-1]
    predicted_mass = predicted_density * dx
    checks.probability_mass(predicted_mass, 'mixture predicted mass')

    if bool(observed):
        update = bm1_observation_update(
            model,
            parameters,
            sigma_y,
            predicted_mass,
            previous_time,
            observation_time,
            observation_value,
            price_side=price_side,
            likelihood_quadrature_order=order,
        )
        raw_posterior = update.mass
        projected = projector.project(
            raw_posterior / dx,
            initial_coefficients=predicted_coefficients,
        )
        posterior_mass = projected['density'] * dx
        posterior_coefficients = projected['coefficients']
        log_evidence = update.log_evidence
        projection_diagnostics = dict(projected['diagnostics'])
        projection_diagnostics['raw_to_projected_cell_mass_l1'] = float(
            np.sum(np.abs(raw_posterior - posterior_mass))
        )
    else:
        posterior_mass = predicted_mass.copy()
        posterior_coefficients = predicted_coefficients.copy()
        log_evidence = np.nan
        projection_diagnostics = {
            'optimality_gap': 0.0,
            'density_l2_error': 0.0,
            'cell_mass_l1_error': 0.0,
            'boundary_cell_mass': float(posterior_mass[0] + posterior_mass[-1]),
            'mass_error': abs(float(np.sum(posterior_mass)) - 1.0),
            'used_fallback': False,
            'hessian_condition_number': projector.projection_data[2],
            'projection_matrix_condition_number': projector.projection_solver.matrix_condition_number,
            'used_clipping': False,
            'used_renormalization': False,
            'raw_to_projected_cell_mass_l1': 0.0,
        }
    checks.probability_mass(posterior_mass, 'mixture posterior mass')
    return {
        'time': observation_time,
        'x_edges': edges,
        'x_grid': x_grid,
        'predicted_mass': predicted_mass,
        'posterior_mass': posterior_mass,
        'predicted_coefficients': predicted_coefficients,
        'posterior_coefficients': posterior_coefficients,
        'observed': bool(observed),
        'log_evidence': log_evidence,
        'prediction_diagnostics': dict(evolved['diagnostics']),
        'posterior_projection_diagnostics': projection_diagnostics,
    }


def run_mixture_filter(
    model,
    parameters,
    sigma_x,
    sigma_y,
    initial_mass,
    initial_time,
    observations,
    *,
    projector,
    breakpoints=None,
    max_step=np.inf,
    cfl=0.4,
    likelihood_quadrature_order=1,
):
    """Replay observations through the self-contained recursive mixture filter."""
    if not isinstance(observations, BM1ObservationSeries):
        raise TypeError('observations must be a BM1ObservationSeries.')
    if not isinstance(projector, PreparedBetaProjector):
        raise TypeError('projector must be a PreparedBetaProjector.')
    initial_time = checks.finite_scalar(initial_time, 'initial_time')
    if observations.times[0] <= initial_time:
        raise ValueError('All observation times must be strictly after initial_time.')
    initial_mass = checks.probability_mass(initial_mass, 'initial_mass')
    if initial_mass.size != projector.basis.shape[1]:
        raise ValueError('initial_mass must use the projector cell grid.')
    initial_projection = projector.project(initial_mass / projector.dx)
    current_coefficients = initial_projection['coefficients']
    current_time = initial_time
    steps = []
    for time, value, is_observed in zip(
        observations.times, observations.values, observations.observed,
    ):
        step = mixture_filter_step(
            model,
            parameters,
            sigma_x,
            sigma_y,
            current_coefficients,
            current_time,
            float(time),
            float(value) if is_observed else np.nan,
            projector=projector,
            observed=bool(is_observed),
            price_side=observations.price_side,
            breakpoints=breakpoints,
            max_step=max_step,
            cfl=cfl,
            likelihood_quadrature_order=likelihood_quadrature_order,
        )
        steps.append(step)
        current_coefficients = step['posterior_coefficients']
        current_time = step['time']

    return BM1MixtureFilterResult(
        initial_time=initial_time,
        initial_mass=initial_projection['density'] * projector.dx,
        initial_coefficients=initial_projection['coefficients'],
        times=observations.times,
        x_edges=steps[0]['x_edges'],
        x_grid=steps[0]['x_grid'],
        predicted_mass=np.stack([step['predicted_mass'] for step in steps]),
        posterior_mass=np.stack([step['posterior_mass'] for step in steps]),
        predicted_coefficients=np.stack([step['predicted_coefficients'] for step in steps]),
        posterior_coefficients=np.stack([step['posterior_coefficients'] for step in steps]),
        observed=observations.observed,
        log_evidence=np.array([step['log_evidence'] for step in steps], dtype=float),
        prediction_diagnostics=tuple(step['prediction_diagnostics'] for step in steps),
        posterior_projection_diagnostics=tuple(
            step['posterior_projection_diagnostics'] for step in steps
        ),
        initial_projection_diagnostics=initial_projection['diagnostics'],
    )


def _observation_demand(model, parameters, time, states, price_side, previous_time):
    """Evaluate demand with explicit left/right price semantics at a measurement."""
    if price_side not in ('left', 'right'):
        raise ValueError('price_side must be left or right.')
    U, f, g, _, k, Xi = parameters
    price_time = time if price_side == 'right' else np.nextafter(time, previous_time)
    Ut = U(price_time)
    Bt = model.B(time)
    delta = model.l(f(states), g(Ut), k)
    return np.asarray(model.D(Bt, Xi, delta), dtype=float)


def _cell_log_likelihood(model, parameters, time, previous_time, edges, observation,
                         sigma_y, price_side, quadrature_order):
    """Approximate the mean likelihood in each piecewise-constant FV cell."""
    order = checks.positive_integer(quadrature_order, 'quadrature_order')
    centers = 0.5 * (edges[:-1] + edges[1:])
    if order == 1:
        demand = _observation_demand(model, parameters, time, centers, price_side, previous_time)
        return gaussian_log_likelihood(observation, demand, sigma_y)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points = centers[:, None] + 0.5 * np.diff(edges)[:, None] * nodes[None, :]
    demand = _observation_demand(
        model, parameters, time, points.reshape(-1), price_side, previous_time,
    ).reshape(points.shape)
    logs = gaussian_log_likelihood(observation, demand.reshape(-1), sigma_y).reshape(points.shape)
    return logsumexp(logs + np.log(0.5 * weights)[None, :], axis=1)


def bm1_observation_update(model, parameters, sigma_y, state_mass, previous_time,
                           observation_time, observation_value, *, price_side='right',
                           likelihood_quadrature_order=1):
    """Apply one BM1 point-observation update to an already predicted state mass.

    This helper exposes the same predictive evidence used by fv_filter_step.
    It is useful when a fixed prediction (for example open propagation) must be
    scored against an observation without carrying the resulting posterior
    forward. No state propagation occurs here.
    """
    previous_time = checks.finite_scalar(previous_time, 'previous_time')
    observation_time = checks.finite_scalar(observation_time, 'observation_time')
    if observation_time <= previous_time:
        raise ValueError('observation_time must be strictly after previous_time.')
    state_mass = checks.probability_mass(state_mass, 'state_mass')
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    if price_side not in ('left', 'right'):
        raise ValueError('price_side must be left or right.')
    order = checks.positive_integer(likelihood_quadrature_order, 'likelihood_quadrature_order')
    edges, _, _ = model.fokker_planc_reference.grid(state_mass.size, bounds=(0.0, 1.0))
    log_likelihood = _cell_log_likelihood(
        model, parameters, observation_time, previous_time, edges, observation_value,
        sigma_y, price_side, order,
    )
    return bayes_update(mass_to_log(state_mass), log_likelihood)

def _positive_filter_step(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind in ('b', 'c'):
        raise TypeError(f'{name} must be a positive real scalar.')
    value = float(array)
    if np.isnan(value) or value <= 0.0:
        raise ValueError(f'{name} must be positive or np.inf.')
    return value


def fv_filter_step(model, parameters, sigma_x, sigma_y, prior_mass, previous_time,
                   observation_time, observation_value, *, observed=True,
                   price_side='right', breakpoints=None, max_step=np.inf,
                   likelihood_quadrature_order=1):
    """Propagate one posterior mass and optionally assimilate one point observation.

    Prediction uses the existing conservative finite-volume solver. Missing
    observations perform prediction only. No prediction mass is clipped or
    renormalized; Bayes normalization is the only intentional normalization.
    """
    previous_time = checks.finite_scalar(previous_time, 'previous_time')
    observation_time = checks.finite_scalar(observation_time, 'observation_time')
    if observation_time <= previous_time:
        raise ValueError('observation_time must be strictly after previous_time.')
    prior_mass = checks.probability_mass(prior_mass, 'prior_mass')
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    max_step = _positive_filter_step(max_step, 'max_step')
    if not isinstance(observed, (bool, np.bool_)):
        raise TypeError('observed must be boolean.')
    if price_side not in ('left', 'right'):
        raise ValueError('price_side must be left or right.')
    order = checks.positive_integer(likelihood_quadrature_order, 'likelihood_quadrature_order')
    if breakpoints is None:
        local_breakpoints = np.empty(0, dtype=float)
    else:
        all_breakpoints = checks.finite_vector(breakpoints, 'breakpoints', minimum_size=0)
        local_breakpoints = all_breakpoints[
            (all_breakpoints > previous_time) & (all_breakpoints < observation_time)
        ]
    result = model.fokker_planc_reference_solver(
        prior_mass, parameters, sigma_x,
        t_eval=np.array([previous_time, observation_time], dtype=float),
        breakpoints=local_breakpoints, max_step=max_step,
    )
    x_grid = result[1]
    predicted = result[3][-1]
    diagnostics = result[-1].copy()
    integration_t = diagnostics['integration_t']
    diagnostics['actual_max_step'] = float(np.max(np.diff(integration_t)))
    diagnostics['likelihood_quadrature_order'] = order
    edges, centers, _ = model.fokker_planc_reference.grid(predicted.size, bounds=(0.0, 1.0))
    if not np.array_equal(centers, x_grid):
        raise RuntimeError('FV prediction grid changed unexpectedly during filtering.')
    if bool(observed):
        update = bm1_observation_update(
            model, parameters, sigma_y, predicted, previous_time, observation_time,
            observation_value, price_side=price_side,
            likelihood_quadrature_order=order,
        )
        posterior = update.mass
        log_evidence = update.log_evidence
    else:
        posterior = predicted.copy()
        log_evidence = np.nan
    return BM1FVFilterStep(
        time=observation_time, x_edges=edges, x_grid=x_grid,
        predicted_mass=predicted, posterior_mass=posterior,
        observed=bool(observed), log_evidence=log_evidence, diagnostics=diagnostics,
    )


def run_fv_filter(model, parameters, sigma_x, sigma_y, initial_mass, initial_time,
                  observations, *, breakpoints=None, max_step=np.inf,
                  likelihood_quadrature_order=1):
    """Replay point observations through the same pure one-step FV filter kernel."""
    if not isinstance(observations, BM1ObservationSeries):
        raise TypeError('observations must be a BM1ObservationSeries.')
    initial_time = checks.finite_scalar(initial_time, 'initial_time')
    if observations.times[0] <= initial_time:
        raise ValueError('All observation times must be strictly after initial_time.')
    current_mass = checks.probability_mass(initial_mass, 'initial_mass')
    current_time = initial_time
    steps = []
    full_grid = None
    for time, value, is_observed in zip(observations.times, observations.values, observations.observed):
        step = fv_filter_step(
            model, parameters, sigma_x, sigma_y, current_mass, current_time, float(time),
            float(value) if is_observed else np.nan, observed=bool(is_observed),
            price_side=observations.price_side, breakpoints=breakpoints, max_step=max_step,
            likelihood_quadrature_order=likelihood_quadrature_order,
        )
        steps.append(step)
        grid = np.asarray(step.diagnostics['integration_t'], dtype=float)
        full_grid = grid.copy() if full_grid is None else np.concatenate((full_grid, grid[1:]))
        current_mass = step.posterior_mass
        current_time = step.time
    return BM1FVFilterResult(
        initial_time=initial_time, initial_mass=initial_mass, times=observations.times,
        x_edges=steps[0].x_edges, x_grid=steps[0].x_grid,
        predicted_mass=np.stack([step.predicted_mass for step in steps]),
        posterior_mass=np.stack([step.posterior_mass for step in steps]),
        observed=observations.observed,
        log_evidence=np.array([step.log_evidence for step in steps], dtype=float),
        integration_t=full_grid,
        prediction_diagnostics=tuple(step.diagnostics for step in steps),
    )

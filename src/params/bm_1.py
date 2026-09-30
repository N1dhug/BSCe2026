"""F4 estimation primitives: observation contract, Bayes update and conservative remap.

These are not a complete filter, parameter fitter or forecasting application.
FP prediction remains in the existing method/solver modules. The module does
not import control, execute experiments, read data, or change BM1 equations.
See root/F4.md and root/f4/main.tex for the staged implementation contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from src.validation import estimation as checks


@dataclass(frozen=True)
class BM1ObservationSeries:
    """Owned point observations; times use the explicitly declared time unit.

    price_side records whether an observation at an input jump sees its left
    or right limit. A future filter must honor it; this container samples no
    signals. Interval averages/energies require a different observation model.
    Missing measurements require an explicit False mask. No truth or control
    target is stored, and no unobserved value is inferred from a future sample.
    """
    times: ArrayLike
    values: ArrayLike
    time_unit: str
    observed: ArrayLike | None = None
    observation_kind: str = 'point'
    price_side: str = 'right'

    def __post_init__(self):
        if self.observation_kind != 'point':
            raise NotImplementedError('Only point observations are prepared; interval data need an explicit model.')
        if self.time_unit not in ('second', 'hour', 'day'):
            raise ValueError('time_unit must be second, hour or day; no implicit conversion is performed.')
        if self.price_side not in ('left', 'right'):
            raise ValueError('price_side must be left or right.')
        times, values, observed = checks.point_observations(self.times, self.values, self.observed)
        for array in (times, values, observed):
            array.setflags(write=False)
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'observed', observed)

    def prefix(self, cutoff: float) -> 'BM1ObservationSeries':
        """Copy observations at or before cutoff; never include future values."""
        cutoff = checks.finite_scalar(cutoff, 'cutoff')
        keep = self.times <= cutoff
        if not np.any(keep):
            raise ValueError('The requested prefix contains no observation times.')
        return BM1ObservationSeries(
            self.times[keep], self.values[keep], self.time_unit, self.observed[keep],
            self.observation_kind, self.price_side,
        )


@dataclass(frozen=True)
class BayesUpdate:
    """Normalized posterior and predictive log density/evidence.

    Keep log_mass for future parameter-weight updates: a tiny positive
    posterior can underflow to zero when converted to ordinary mass.
    log_evidence is a log DENSITY for continuous observations; it can be > 0.
    """
    log_mass: NDArray[np.float64]
    log_evidence: float
    mass: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        logs = checks.log_probability_mass(self.log_mass)
        log_evidence = checks.finite_scalar(self.log_evidence, 'log_evidence')
        mass = np.exp(logs)
        logs.setflags(write=False)
        mass.setflags(write=False)
        object.__setattr__(self, 'log_mass', logs)
        object.__setattr__(self, 'mass', mass)
        object.__setattr__(self, 'log_evidence', log_evidence)


def mass_to_log(mass: ArrayLike) -> NDArray[np.float64]:
    """Convert normalized cell/hypothesis masses; genuine zeros become -inf."""
    values = checks.probability_mass(mass)
    logs = np.full(values.shape, -np.inf)
    positive = values > 0.0
    logs[positive] = np.log(values[positive])
    return logs


def gaussian_log_likelihood(observation: float, predicted_values: ArrayLike,
                            sigma_y: float) -> NDArray[np.float64]:
    """Point-observation Gaussian log density on supplied state/hypothesis nodes.

    predicted_values should be D_theta(x_i, U(t_k), B(t_k)), not D_theta(E[X]).
    No dt factor is applied to sigma_y. Overflow in a remote tail becomes -inf;
    a later all-zero-support Bayes update fails explicitly rather than repairing it.
    """
    observation = checks.finite_scalar(observation, 'observation')
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    means = checks.finite_vector(predicted_values, 'predicted_values')
    with np.errstate(over='ignore', invalid='raise', divide='raise'):
        residual = (observation - means) / sigma_y
        logs = -0.5 * residual * residual - np.log(sigma_y) - 0.5 * np.log(2.0 * np.pi)
    return logs


def bayes_update(log_prior: ArrayLike, log_likelihood: ArrayLike) -> BayesUpdate:
    """One Bayes update for normalized log masses (no extra cell-width factor).

    Applicable to state cell masses or discrete parameter weights. This is
    ONLY the algebraic update, not a parameter bank: each future hypothesis
    must supply its own predictive evidence from its own conditional filter.
    """
    prior = checks.log_probability_mass(log_prior, 'log_prior')
    if np.iscomplexobj(log_likelihood):
        raise ValueError('log_likelihood must be real-valued.')
    likelihood = np.asarray(log_likelihood, dtype=float)
    if (likelihood.shape != prior.shape or np.any(np.isnan(likelihood))
            or np.any(np.isposinf(likelihood))):
        raise ValueError('log_likelihood must match log_prior and contain finite values or -inf.')
    with np.errstate(over='ignore', invalid='raise'):
        combined = prior + likelihood
    pivot = float(np.max(combined))
    if not np.isfinite(pivot):
        raise FloatingPointError('No representable prior/likelihood overlap; inspect support, grid and noise scale.')
    # Center before subtracting the normalizer; large common offsets then
    # do not corrupt normalization of the returned posterior.
    shifted = combined - pivot
    local_log_z = float(logsumexp(shifted))
    log_z = pivot + local_log_z
    if not np.isfinite(log_z):
        raise FloatingPointError('Predictive log evidence is not representable in float64.')
    return BayesUpdate(shifted - local_log_z, log_z)


def conservative_mass_transfer(source_edges: ArrayLike, source_mass: ArrayLike,
                               target_edges: ArrayLike) -> NDArray[np.float64]:
    """Rebin a piecewise-constant density by exact interval overlaps.

    Nonuniform grids are supported. Supports must coincide exactly: no
    truncation, extrapolation or tail renormalization is inferred. Conservation
    is exact in real arithmetic, checked in float arithmetic. Refinement cannot
    recover subcell information that was absent on the source grid.
    """
    source = checks.increasing_grid(source_edges, 'source_edges')
    target = checks.increasing_grid(target_edges, 'target_edges')
    mass = checks.probability_mass(source_mass, 'source_mass')
    if mass.size != source.size - 1:
        raise ValueError('source_mass must have one value per source cell.')
    if source[0] != target[0] or source[-1] != target[-1]:
        raise ValueError('Source and target supports must coincide; truncation is not a remap.')
    if np.array_equal(source, target):
        return mass.copy()
    result = np.zeros(target.size - 1)
    i = j = 0
    while i < mass.size and j < result.size:
        left = max(source[i], target[j])
        right = min(source[i + 1], target[j + 1])
        if right > left:
            result[j] += mass[i] * ((right - left) / (source[i + 1] - source[i]))
        source_right, target_right = source[i + 1], target[j + 1]
        if source_right <= target_right:
            i += 1
        if target_right <= source_right:
            j += 1
    if (not np.all(np.isfinite(result)) or np.any(result < 0.0)
            or abs(float(np.sum(result)) - float(np.sum(mass))) > checks.MASS_ATOL):
        raise FloatingPointError('Conservative mass transfer failed; no normalization repair is applied.')
    return result


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
        log_likelihood = _cell_log_likelihood(
            model, parameters, observation_time, previous_time, edges, observation_value,
            sigma_y, price_side, order,
        )
        update = bayes_update(mass_to_log(predicted), log_likelihood)
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

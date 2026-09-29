"""F4 preparation: observation contract, Bayes primitives and conservative remap.

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

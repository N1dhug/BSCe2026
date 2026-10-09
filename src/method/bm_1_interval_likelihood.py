"""Shared Gaussian interval likelihood numerics for native and reference F.

The latent hourly reward is bounded in [0,1], but additive Gaussian observation
noise gives Y support on the real line. The contour bank resolves the same
likelihood for every reachable terminal state; it changes no model parameter.
There are no pilot drivers, data loaders or optional research imports here.
"""

import numpy as np

ORIGINAL_CENTERS = (.5, .25, .75, 0., 1.)
EXTENDED_CENTERS = ORIGINAL_CENTERS + (.125, .375, .625, .875)


class CharacteristicNumericalError(ValueError):
    """Expected unresolved numerical control, never a substituted fit cost."""
    def __init__(self, reason, **diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def _positive_implicit_factors(lower, diagonal, upper):
    """Positive Thomas factors for one exact tridiagonal matrix."""
    pivots = diagonal.copy()
    multipliers = np.empty_like(lower)
    for j in range(1, len(diagonal)):
        multipliers[j-1] = -lower[j-1]/pivots[j-1]
        pivots[j] += multipliers[j-1]*upper[j-1]
    if (np.any(pivots <= 0.) or not np.isfinite(pivots).all()
            or np.any(lower > 0.) or np.any(upper > 0.)):
        raise CharacteristicNumericalError('Positive implicit factorization failed.')
    return pivots, multipliers


def _terminal_candidate(prediction, value, sigma_y, relative_guard):
    """Conditional Fourier error is checked before small row amplitudes enter."""
    span = prediction['span']
    omega = prediction['frequencies']
    tilt, center = prediction['tilt'], prediction['center']
    z = value - sigma_y**2 * tilt
    coefficients = np.exp(-.5 * (sigma_y * omega)**2 - 1j * omega * (z - center))
    density = (prediction['cf'][:, 0].real
               + 2. * np.real(prediction['cf'][:, 1:] @ coefficients[1:])) / span
    distance = max(abs(z), abs(z - 1.))
    if span <= distance:
        raise ValueError('Periodic span must separate the physical reward range.')
    gap = span - distance
    alias = (2. * np.exp(-.5 * (gap / sigma_y)**2) / (np.sqrt(2. * np.pi) * sigma_y)
             / (-np.expm1(-(2. * gap * span + span**2) / (2. * sigma_y**2))))
    alpha = .5 * (2. * np.pi * sigma_y / span)**2
    k = len(omega) - 1
    tail = 2. / span * np.exp(-alpha * (k + 1)**2) / (-np.expm1(-alpha * (2 * k + 3)))
    roundoff = 64. * np.finfo(float).eps * (1. + 2. * np.sum(np.abs(coefficients[1:]))) / span
    absolute_error = alias + tail + roundoff
    active = np.isfinite(prediction['log_amplitude'])
    positive = active & np.isfinite(density) & (density > absolute_error)
    relative = np.full(density.shape, np.inf)
    relative[positive] = absolute_error / (density[positive] - absolute_error)
    eligible = positive & (relative <= relative_guard)
    log_joint = np.full(density.shape, -np.inf)
    log_prefactor = .5 * sigma_y**2 * tilt**2 - tilt * (value - center)
    log_joint[eligible] = (prediction['log_amplitude'][eligible]
                           + np.log(density[eligible]) + log_prefactor)
    return dict(log_joint=log_joint, eligible=eligible, relative=relative, active=active,
                density=density, alias_bound=float(alias), truncation_bound=float(tail),
                roundoff_allowance=float(roundoff))


def native_terminal_update_extended(operator, logs, value, sigma_y, settings, *, mapper=None):
    """Use the declared nine-center bank with every existing guard intact.

This is a finite representation change, not a guarantee that every possible
candidate will resolve. Original failures and source/job identities must be
preserved by any future provenance-aware adoption adapter.
    """
    from src.method.bm_1_interval_native import native_terminal_update
    if tuple(settings['centers']) != ORIGINAL_CENTERS:
        raise ValueError('Require the exact original ordered contour prefix.')
    extended = dict(settings, centers=list(EXTENDED_CENTERS))
    return native_terminal_update(operator, logs, value, sigma_y, extended, mapper=mapper)

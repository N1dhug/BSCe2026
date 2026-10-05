"""Positive joint propagation of an FV Markov state and its additive reward.

This development kernel reuses the one-dimensional FV generator and implicit
Euler solver. Frozen-left reward transport / state propagation / transport is
first-order overall. Linear nodal deposition preserves mass and first moments
but adds accumulator dispersion, which must be checked by grid refinement.
The accumulator support grows rather than truncating or reflecting its tails.
"""

import numpy as np

from src.solver.finite_volume import finite_volume_1d_cached
from src.validation import estimation as checks
from src.validation.solver.finite_volume import finite_volume_1d_validation as fv_checks


def _transport(joint, increments, spacing, node_budget):
    """Translate each state's nodal law without discarding probability."""
    common = float(np.min(increments))
    shifts = (increments-common)/spacing
    lower = np.floor(shifts).astype(np.int64)
    fraction = shifts-lower
    added = int(np.max(lower+(fraction > 0.)))
    size = joint.shape[1]+added
    if size > node_budget:
        raise ValueError('Accumulator node budget exceeded; change declared numerics, never truncate tails.')
    result = np.zeros((joint.shape[0], size))
    rows = np.arange(joint.shape[0])[:, None]
    columns = np.arange(joint.shape[1])[None, :]+lower[:, None]
    result[rows, columns] = joint*(1.-fraction[:, None])
    moving = np.flatnonzero(fraction > 0.)
    if moving.size:
        result[moving[:, None], columns[moving]+1] += joint[moving]*fraction[moving, None]
    return result, common


def propagate_additive(solver, interval, initial_mass, drift, diffusion, reward, *,
                       max_step, accumulator_spacing=1/256, max_accumulator_nodes=4096,
                       breakpoints=None, coefficient_key=None):
    """Return joint law of X_end and interval-average reward in [0,1].

    The continuous reward is bounded; the numerical nodal law may extend past
    that range through deposition error. Such mass is retained and scored.
    Each call resets the accumulator to zero, but never resets X. Its finite
    node budget is a memory guard, not a boundary condition or acceptance gate.
    """
    if not isinstance(solver, finite_volume_1d_cached):
        raise TypeError('Additive propagation requires the existing LAPACK/cached FV backend.')
    if not all(callable(f) for f in (drift, diffusion, reward)):
        raise TypeError('Drift, diffusion and reward must be callable.')
    interval = checks.increasing_grid(interval, 'interval')
    if interval.size != 2:
        raise ValueError('Supply exactly the interval start and end.')
    prior = checks.probability_mass(initial_mass, 'initial_mass')
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    spacing = checks.finite_scalar(accumulator_spacing, 'accumulator_spacing', positive=True)
    budget = checks.positive_integer(max_accumulator_nodes, 'max_accumulator_nodes', minimum=2)
    edges, centers, dx = solver.grid(prior.size)
    grid, _ = solver.build_time_grid(interval, breakpoints, max_step)
    advance = solver._stepper(dx, edges, centers, drift, diffusion, coefficient_key)
    joint = prior[:, None].copy()
    offset = 0.
    maximum_error = abs(float(joint.sum())-1.)
    minimum_mass = float(joint.min())
    maximum_column_sum = 0.
    minimum_rate = np.inf
    duration = float(interval[1]-interval[0])
    dispersion_bound = 0.
    for left, right in zip(grid[:-1], grid[1:]):
        step = float(right-left)
        rates = fv_checks.field(reward(float(left), centers), centers.shape, 'reward')
        if np.any(rates < 0.) or np.any(rates > 1.):
            raise ValueError('Normalized demand reward must lie in [0,1].')
        increments = rates*(step/(2.*duration))
        joint, common = _transport(joint, increments, spacing, budget)
        offset += common
        joint, generator = advance(joint, float(left), step)
        joint, common = _transport(joint, increments, spacing, budget)
        offset += common
        # Evolved joint mass is checked directly; individual columns are not
        # normalized probability vectors. No clipping or prediction repair.
        if (not np.isfinite(joint).all() or np.any(joint < 0.)
                or abs(float(joint.sum())-1.) > checks.MASS_ATOL):
            raise ValueError('Joint probability guard failed; no repair is applied.')
        maximum_error = max(maximum_error, abs(float(joint.sum())-1.))
        minimum_mass = min(minimum_mass, float(joint.min()))
        maximum_column_sum = max(maximum_column_sum, generator['maximum_abs_column_sum'])
        minimum_rate = min(minimum_rate, generator['minimum_off_diagonal_rate'])
        # Each linear deposition adds <= dz * increment variance. This bound
        # concerns deposition only, not state-grid or time-discretization error.
        dispersion_bound += spacing*float(np.ptp(rates))*step/duration
    accumulator = offset+spacing*np.arange(joint.shape[1])
    diagnostics = dict(integration_t=grid, n_steps=len(grid)-1,
        actual_max_step=float(np.max(np.diff(grid))),
        maximum_abs_mass_error=maximum_error, minimum_cell_mass=minimum_mass,
        maximum_abs_generator_column_sum=maximum_column_sum,
        minimum_off_diagonal_rate=minimum_rate,
        accumulator_spacing=spacing, accumulator_nodes=joint.shape[1],
        accumulator_deposition_variance_bound=dispersion_bound,
        predicted_interval_mean=float(joint.sum(axis=0) @ accumulator),
        predicted_interval_second_moment=float(joint.sum(axis=0) @ accumulator**2),
        accumulator_mass_outside_physical_range=float(joint[:, accumulator > 1.].sum()),
        used_clipping=False, used_renormalization=False,
        observation_kind='interval_average', method='frozen-left split; implicit state; linear nodal reward')
    return dict(joint_mass=joint, accumulator=accumulator, x_edges=edges,
                x_grid=centers, diagnostics=diagnostics)

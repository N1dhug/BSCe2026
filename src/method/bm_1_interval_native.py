"""Experimental exact-discretization H2 kernel; no fitting or data reads.

Positive Thomas addition in log-amplitude/conditional-CF coordinates. The
original terminal inversion and its error policy are reused unchanged.
Numba is an explicit optional dependency; there is no silent Python fallback.
"""

import math
from time import perf_counter

import numpy as np
from numba import njit
from scipy.special import logsumexp

from src.validation.bm_1.interval_characteristic import (
    CharacteristicNumericalError, _positive_implicit_factors)
from src.validation.bm_1.interval_terminal_tilt import _terminal_candidate


@njit(cache=True, fastmath=False, inline='always')
def _add(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a-b)))


@njit(cache=True, fastmath=False, inline='always')
def _weight(a, total):
    return 0. if a == -np.inf else math.exp(a-total)


@njit(cache=True, fastmath=False, nogil=True)
def _advance(logs, cf, neutral, step_ids, ed, el, eu, lp, lbeta, lu,
             pivots, beta, upper, phases, shifts):
    """Frozen-segment substeps, including the independent neutral CN solve."""
    logs, cf, neutral = logs.copy(), cf.copy(), neutral.copy()
    n, k = cf.shape
    explicit_logs, forward_logs = np.empty(n), np.empty(n)
    explicit_cf, forward_cf = np.empty_like(cf), np.empty_like(cf)
    neutral_rhs = np.empty(n)
    maximum_error = 0.
    for profile in step_ids:
        for j in range(n):
            logs[j] += shifts[profile, j]
            for f in range(k):
                cf[j, f] *= phases[profile, j, f]
        for j in range(n):
            a = logs[j]+ed[profile, j]
            b = logs[j-1]+el[profile, j-1] if j else -np.inf
            c = logs[j+1]+eu[profile, j] if j+1 < n else -np.inf
            total = _add(_add(a, b), c)
            explicit_logs[j] = total
            w0, w1, w2 = _weight(a, total), _weight(b, total), _weight(c, total)
            for f in range(k):
                value = w0*cf[j, f]
                if j:
                    value += w1*cf[j-1, f]
                if j+1 < n:
                    value += w2*cf[j+1, f]
                explicit_cf[j, f] = value
            value = math.exp(ed[profile, j])*neutral[j]
            if j:
                value += math.exp(el[profile, j-1])*neutral[j-1]
            if j+1 < n:
                value += math.exp(eu[profile, j])*neutral[j+1]
            neutral_rhs[j] = value
        forward_logs[0] = explicit_logs[0]
        for f in range(k):
            forward_cf[0, f] = explicit_cf[0, f]
        for j in range(1, n):
            a, b = explicit_logs[j], lbeta[profile, j-1]+forward_logs[j-1]
            total = _add(a, b)
            forward_logs[j] = total
            w0, w1 = _weight(a, total), _weight(b, total)
            for f in range(k):
                forward_cf[j, f] = w0*explicit_cf[j, f]+w1*forward_cf[j-1, f]
            neutral_rhs[j] += beta[profile, j-1]*neutral_rhs[j-1]
        logs[-1] = forward_logs[-1]-lp[profile, -1]
        neutral[-1] = neutral_rhs[-1]/pivots[profile, -1]
        for f in range(k):
            cf[-1, f] = forward_cf[-1, f]
        for j in range(n-2, -1, -1):
            a, b = forward_logs[j], lu[profile, j]+logs[j+1]
            total = _add(a, b)
            logs[j] = total-lp[profile, j]
            w0, w1 = _weight(a, total), _weight(b, total)
            for f in range(k):
                cf[j, f] = w0*forward_cf[j, f]+w1*cf[j+1, f]
            neutral[j] = (neutral_rhs[j]+upper[profile, j]*neutral[j+1])/pivots[profile, j]
        mass = 0.
        for j in range(n):
            if not math.isfinite(neutral[j]) or neutral[j] < 0.:
                raise ValueError('Native neutral probability guard failed.')
            mass += neutral[j]
            if logs[j] != -np.inf:
                zero = cf[j, 0]
                if zero.real <= 0. or zero.imag != 0. or not math.isfinite(zero.real):
                    raise ValueError('Native zero-frequency amplitude guard failed.')
                logs[j] += math.log(zero.real)
                for f in range(k):
                    cf[j, f] /= zero.real
            logs[j] += shifts[profile, j]
            if math.isnan(logs[j]) or logs[j] == np.inf:
                raise ValueError('Native log-amplitude guard failed.')
            for f in range(k):
                cf[j, f] *= phases[profile, j, f]
                if not math.isfinite(cf[j, f].real) or not math.isfinite(cf[j, f].imag) or abs(cf[j, f]) > 1.+1e-10:
                    raise ValueError('Native characteristic modulus guard failed.')
        error = abs(mass-1.)
        maximum_error = max(maximum_error, error)
        if error > 1e-10:
            raise ValueError('Native neutral mass guard failed.')
    return logs, cf, neutral, maximum_error


def freeze_native_operator(spec, candidate, unit):
    """Use exactly the old FV fields/time grid and original interval duration."""
    from src.method.bm_1_response import curve_pair_from_logits
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_fitted
    from src.tools.input_generator.input_builder import input_builder
    from src.validation.solver.finite_volume import finite_volume_1d_validation as checks
    solver = finite_volume_1d_fitted(vectorized_inputs=True)
    edges, x, dx = solver.grid(spec['settings']['cells'])
    f, g = curve_pair_from_logits(candidate['logits'], candidate['pieces'])
    B = input_builder.price_intervals((spec['input_edges'], spec['baseline']))
    U = input_builder.price_intervals((spec['input_edges'], spec['price']))
    p = spec['physical']; model = bm_1(B, B, spec['horizon'], 2)
    start, end = unit['start'], unit['end']
    breaks = np.asarray(spec['input_edges'])[1:-1]
    breaks = breaks[(breaks > start) & (breaks < end)]
    grid, _ = solver.build_time_grid([start, end], breaks, spec['settings']['max_step'])
    fields, profiles, ids, keys = {}, [], [], {}
    for left, right in zip(grid[:-1], grid[1:]):
        input_index = int(np.searchsorted(spec['input_edges'], left, side='right')-1)
        if input_index not in fields:
            drift = checks.field(model._drift(float(left), edges[1:-1],
                (U, f, g, p['C'], p['k'], p['Xi'])), edges[1:-1].shape, 'drift')
            diffusion = checks.field(model.diffusion_coefficient(x, p['sigma_x']), x.shape, 'diffusion')
            lo, di, up, diagnostic = solver.generator_tridiagonal(dx, drift, diffusion)
            r = np.asarray(model.eval(float(left), x, U, f, g, p['k'], p['Xi'])[-1])
            if np.any(r < 0.) or np.any(r > 1.) or not np.isfinite(r).all():
                raise ValueError('Native reward guard failed.')
            fields[input_index] = lo, di, up, r
        h = float(right-left); key = (input_index, h.hex())
        if key not in keys:
            lo, di, up, reward = fields[input_index]; s = h/2.
            ed, el, eu = 1.+s*di, s*lo, s*up
            if np.any(ed < 0.) or np.any(el < 0.) or np.any(eu < 0.):
                raise ValueError('Native positive-CN guard failed; no step repair.')
            pivots, beta = _positive_implicit_factors(-s*lo, 1.-s*di, -s*up)
            with np.errstate(divide='ignore'):
                logs = [np.log(v) for v in (ed, el, eu, pivots, beta, s*up)]
            keys[key] = len(profiles)
            profiles.append((*logs, pivots, beta, s*up, reward*h/(2.*(end-start)), h/(2.*(end-start))))
        ids.append(keys[key])
    # Every array is ordinary float64; exact h keys never use rounding.
    names = ('ed', 'el', 'eu', 'lp', 'lbeta', 'lu', 'pivots', 'beta', 'upper', 'reward_half', 'fraction')
    result = {name: np.array([r[i] for r in profiles]) for i, name in enumerate(names)}
    result.update(step_ids=np.array(ids, dtype=np.int64), x=x)
    return result


def advance_frequency_block(job):
    """Picklable pool operation; index zero accompanies every positive block."""
    operator, logs, indices, span, tilt, center = job
    omega = 2.*np.pi*np.asarray(indices)/span
    centered = operator['reward_half']-center*operator['fraction'][:, None]
    phases = np.exp(1j*centered[:, :, None]*omega[None, None, :])
    shifts = tilt*centered
    n = len(logs)
    args = [operator[k] for k in ('step_ids', 'ed', 'el', 'eu', 'lp', 'lbeta', 'lu', 'pivots', 'beta', 'upper')]
    try:
        result = _advance(np.asarray(logs), np.ones((n, len(indices)), dtype=np.complex128),
                          np.exp(logs), *args, phases, shifts)
    except ValueError as exc:
        raise CharacteristicNumericalError(str(exc)) from exc
    return dict(indices=list(indices), log_amplitude=result[0], cf=result[1],
                predicted_mass=result[2], maximum_abs_mass_error=result[3])


def native_prediction(operator, logs, *, frequencies, span, tilt=0., center=0., mapper=None):
    """Canonical column assembly with strict duplicated-amplitude checks."""
    positive = np.arange(1, frequencies+1)
    blocks = np.array_split(positive, min(8, frequencies))
    jobs = [(operator, logs, np.r_[0, b], span, tilt, center) for b in blocks]
    results = list(map(advance_frequency_block, jobs) if mapper is None else mapper(jobs))
    first = results[0]
    cf = np.empty((len(logs), frequencies+1), dtype=np.complex128)
    cf[:, 0] = first['cf'][:, 0]
    covered = []
    for result in results:
        if any(not np.array_equal(first[k], result[k]) for k in ('log_amplitude', 'predicted_mass')) or not np.array_equal(first['cf'][:, 0], result['cf'][:, 0]):
            raise CharacteristicNumericalError('Frequency blocks disagree on common state/amplitude.')
        cols = result['indices'][1:]; covered.extend(cols)
        cf[:, cols] = result['cf'][:, 1:]
    if covered != list(range(1, frequencies+1)):
        raise ValueError('Incomplete or noncanonical frequency blocks.')
    return dict(cf=cf, log_amplitude=first['log_amplitude'], predicted_mass=first['predicted_mass'],
        maximum_abs_mass_error=max(r['maximum_abs_mass_error'] for r in results),
        frequencies=2.*np.pi*np.arange(frequencies+1)/span, span=span, tilt=tilt, center=center)


def native_terminal_update(operator, logs, value, sigma_y, settings, *, mapper=None):
    """Reuse the original terminal candidate calculation/guards, unchanged."""
    selected_logs = np.full(len(logs), -np.inf)
    selected_guard = np.full(len(logs), np.inf)
    selected = np.full(len(logs), -1, dtype=int)
    active, neutral, contours, overlap_max, mass_error = None, None, [], 0., 0.
    for index, center in enumerate(settings['centers']):
        begun = perf_counter(); tilt = (value-center)/sigma_y**2
        prediction = native_prediction(operator, logs, frequencies=settings['frequencies'],
            span=settings['span'], tilt=tilt, center=center, mapper=mapper)
        row = _terminal_candidate(prediction, value, sigma_y, settings['relative_guard'])
        if active is None:
            active, neutral = row['active'], prediction['predicted_mass']
        elif not np.array_equal(active, row['active']) or not np.array_equal(neutral, prediction['predicted_mass']):
            raise CharacteristicNumericalError('Native contours changed neutral state/support.')
        mass_error = max(mass_error, prediction['maximum_abs_mass_error'])
        overlap = row['eligible'] & (selected >= 0)
        discrepancy = float(np.max(np.abs(row['log_joint'][overlap]-selected_logs[overlap]))) if np.any(overlap) else 0.
        overlap_max = max(overlap_max, discrepancy)
        if discrepancy > settings['overlap_log_tolerance']:
            raise CharacteristicNumericalError('Native resolved contours disagree.', maximum_log_joint_difference=discrepancy)
        better = row['eligible'] & (row['relative'] < selected_guard)
        selected_logs[better], selected_guard[better], selected[better] = row['log_joint'][better], row['relative'][better], index
        contours.append(dict(center=float(center), tilt=float(tilt), eligible_cells=int(row['eligible'].sum()),
            unresolved_cells=int(np.sum(active & (selected < 0))),
            negative_candidate_cells=int(np.sum(row['density'][active] < 0.)),
            maximum_overlap_log_difference=discrepancy, elapsed_seconds=perf_counter()-begun))
        if np.all(selected[active] >= 0):
            break
    if np.any(active & (selected < 0)):
        raise CharacteristicNumericalError('Native terminal contour bank exhausted.', contours=contours)
    score = float(logsumexp(selected_logs)); posterior = selected_logs-score
    if not np.isfinite(score) or not np.isfinite(posterior[active]).all() or abs(float(logsumexp(posterior))) > 1e-10:
        raise CharacteristicNumericalError('Native posterior normalization guard failed.')
    return dict(log_posterior_mass=posterior, log_evidence=score, contours=contours,
        maximum_relative_conditional_guard=float(selected_guard[active].max()),
        maximum_overlap_log_difference=overlap_max, maximum_abs_mass_error=mass_error)

"""F4-H2 development: choose a resolved contour for EVERY terminal state.

All contours represent the same Gaussian interval likelihood. The bank is
declared before a replay and exhausted only until every reachable state has
a positive, relatively resolved representation. No probability is discarded.
The Fourier roundoff allowance is a diagnostic, not a verified-arithmetic
certificate for the FV/Thomas solves; see H2_TERMINAL_TILT_REVIEW.md.
"""

from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from src.validation import estimation as checks
from src.validation.bm_1.interval_characteristic import (
    CharacteristicNumericalError, balanced_characteristic_prediction)


DEFAULT_CENTERS = (.5, .25, .75, 0., 1.)


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


def terminal_tilt_update(solver, interval, prior, drift, diffusion, reward, value, sigma_y, *,
                         max_step, frequencies=64, span=2., centers=DEFAULT_CENTERS,
                         log_prior=None, breakpoints=None, coefficient_key=None,
                         relative_guard=1e-8, overlap_log_tolerance=1e-7):
    """Positive-CN control with terminal-wise choice among exact tilt identities.

    Stop once every reachable terminal cell is resolved. This is explicit
    contour selection, with unchanged time/state grids and no retries beyond
    the finite declared bank. An unresolved cell aborts the entire update.
    Tiny ordinary masses may underflow on export; finite log masses are kept
    and MUST be supplied to the next update.
    """
    value = checks.finite_scalar(value, 'value')
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    relative_guard = checks.finite_scalar(relative_guard, 'relative_guard', positive=True)
    overlap_log_tolerance = checks.finite_scalar(overlap_log_tolerance, 'overlap_log_tolerance', positive=True)
    centers = checks.finite_vector(centers, 'centers')
    if np.any(centers < 0.) or np.any(centers > 1.) or len(set(centers)) != len(centers):
        raise ValueError('Declare distinct contour centers inside [0,1].')
    selected_logs = np.full(len(prior), -np.inf)
    selected_guard = np.full(len(prior), np.inf)
    selected = np.full(len(prior), -1, dtype=int)
    diagnostics, active, neutral = [], None, None
    maximum_mass_error, maximum_overlap = 0., 0.
    for index, center in enumerate(centers):
        begun = perf_counter()
        tilt = (value - center) / sigma_y**2
        prediction = balanced_characteristic_prediction(solver, interval, prior, drift, diffusion, reward,
            max_step=max_step, frequencies=frequencies, span=span, tilt=tilt, center=center,
            log_prior=log_prior, breakpoints=breakpoints, coefficient_key=coefficient_key,
            state_scheme='positive_crank_nicolson')
        row = _terminal_candidate(prediction, value, sigma_y, relative_guard)
        if active is None:
            active, neutral = row['active'], prediction['predicted_mass']
        elif (not np.array_equal(active, row['active'])
              or not np.array_equal(neutral, prediction['predicted_mass'])):
            raise CharacteristicNumericalError('Contour predictions changed the process or reachable support.')
        maximum_mass_error = max(maximum_mass_error, prediction['maximum_abs_mass_error'])
        overlap = row['eligible'] & (selected >= 0)
        discrepancy = (float(np.max(np.abs(row['log_joint'][overlap] - selected_logs[overlap])))
                       if np.any(overlap) else 0.)
        maximum_overlap = max(maximum_overlap, discrepancy)
        if discrepancy > overlap_log_tolerance:
            raise CharacteristicNumericalError('Resolved contour representations disagree.',
                center=float(center), maximum_log_joint_difference=discrepancy)
        better = row['eligible'] & (row['relative'] < selected_guard)
        selected_logs[better] = row['log_joint'][better]
        selected_guard[better] = row['relative'][better]
        selected[better] = index
        diagnostics.append(dict(center=float(center), tilt=float(tilt),
            eligible_cells=int(row['eligible'].sum()), unresolved_cells=int(np.sum(active & (selected < 0))),
            negative_candidate_cells=int(np.sum(row['density'][active] < 0.)),
            minimum_candidate_density=float(row['density'][active].min()),
            maximum_overlap_log_difference=discrepancy, elapsed_seconds=perf_counter() - begun,
            alias_bound=row['alias_bound'], truncation_bound=row['truncation_bound'],
            roundoff_allowance=row['roundoff_allowance']))
        if np.all(selected[active] >= 0):
            break
    if np.any(active & (selected < 0)):
        raise CharacteristicNumericalError('Declared terminal contour bank exhausted; unresolved cells retained as failure.',
            unresolved_cells=int(np.sum(active & (selected < 0))), contours=diagnostics)
    log_evidence = float(logsumexp(selected_logs))
    log_posterior = selected_logs - log_evidence
    if not np.isfinite(log_evidence) or not np.isfinite(log_posterior[active]).all():
        raise CharacteristicNumericalError('Nonfinite terminal log likelihood.')
    posterior = checks.probability_mass(np.exp(log_posterior))
    return dict(log_evidence=log_evidence, log_joint_mass=selected_logs,
                log_posterior_mass=log_posterior, posterior_mass=posterior, predicted_mass=neutral,
                selected_contour=selected, contours=diagnostics,
                maximum_relative_conditional_guard=float(selected_guard[active].max()),
                maximum_overlap_log_difference=maximum_overlap,
                maximum_abs_mass_error=maximum_mass_error, used_clipping=False,
                rigorous_roundoff_certificate=False)


def terminal_reference_control():
    """Archive the exact Gaussian and positive one-jump references."""
    from scipy.special import roots_legendre
    from src.params.bm_1 import gaussian_log_likelihood
    from src.solver.finite_volume import finite_volume_1d_fitted
    solver = finite_volume_1d_fitted()
    zero = lambda t, x: np.zeros(x.shape)
    rewards = np.array([.05, .5, .95])
    exact, rows = np.log([.2, .5, .3]), []
    for label, initial, values in (('original_cancellation', exact, [.6]),
                                    ('sub_float_revival', np.array([0., -900., -950.]), [1.5, 1.5])):
        logs, exact = initial.copy(), initial.copy()
        for index, value in enumerate(values):
            result = terminal_tilt_update(solver, [0., 1.], np.exp(logs), zero, zero,
                lambda t, x: rewards, value, .04, log_prior=logs, max_step=1/8)
            exact += gaussian_log_likelihood(value, rewards, .04)
            score = float(logsumexp(exact)); exact -= score
            rows.append(dict(case=label, interval_index=index, value=value, sigma_y=.04,
                log_evidence=result['log_evidence'], exact_log_evidence=score,
                log_posterior_mass=result['log_posterior_mass'].tolist(),
                exact_log_posterior_mass=exact.tolist(),
                score_error=abs(result['log_evidence']-score),
                maximum_log_posterior_error=float(np.max(np.abs(result['log_posterior_mass']-exact))),
                contours=result['contours'], relative_guard=result['maximum_relative_conditional_guard']))
            logs = result['log_posterior_mass']
    rewards, rate, value, sigma = np.array([.1, .9, .9]), .8, .55, .04
    refs = []
    for order in (96, 160):
        nodes, weights = roots_legendre(order)
        times = (nodes + 1.) / 2.
        average = rewards[1] + (rewards[0]-rewards[1]) * times
        jumped = logsumexp(np.log(weights/2.*rate)-rate*times+gaussian_log_likelihood(value, average, sigma))
        refs.append(np.array([-rate+gaussian_log_likelihood(value, rewards[:1], sigma)[0], jumped]))
    jump_rows = []
    for steps in (128, 256):
        result = terminal_tilt_update(solver, [0., 1.], [1., 0., 0.],
            lambda t, x: np.where(x < .5, rate/3., 0.), zero, lambda t, x: rewards,
            value, sigma, max_step=1/steps)
        jump_rows.append(dict(steps=steps, log_joint=result['log_joint_mass'][:2].tolist(),
            exact_log_joint=refs[-1].tolist(),
            maximum_log_joint_error=float(np.max(np.abs(result['log_joint_mass'][:2]-refs[-1]))),
            disconnected_state_is_zero=bool(np.isneginf(result['log_joint_mass'][2]))))
    passed = (max(max(r['score_error'], r['maximum_log_posterior_error']) for r in rows) < 2e-8
              and float(np.max(np.abs(refs[0]-refs[1]))) < 1e-12
              and jump_rows[-1]['maximum_log_joint_error'] < 2e-4
              and jump_rows[0]['maximum_log_joint_error']/jump_rows[1]['maximum_log_joint_error'] > 3.5
              and all(r['disconnected_state_is_zero'] for r in jump_rows))
    return dict(schema='f4-h2-terminal-tilt-references-v1', gaussian_controls=rows,
                one_jump_controls=jump_rows, one_jump_reference_order_error=float(np.max(np.abs(refs[0]-refs[1]))),
                development_checks_pass=bool(passed), scientific_acceptance=False)


def declared_terminal_control(report, reference, stage):
    from src.validation.bm_1.interval_curve_control import declared_control
    if stage not in ('hour', 'day', 'cross'):
        raise ValueError('Declare hour, day or cross.')
    record, old = declared_control(report, reference, 'two-narrow')
    return record, dict(case='two-narrow-terminal-tilt', intervals=1 if stage == 'hour' else 24,
        centers=list(DEFAULT_CENTERS), relative_guard=1e-8, overlap_log_tolerance=1e-7,
        settings=([dict(cells=1281, max_step=1/24576, frequencies=64, span=2.)]
                  if stage == 'cross' else
                  [dict(cells=1281, max_step=1/12288, frequencies=64, span=2.),
                   dict(cells=1921, max_step=1/24576, frequencies=64, span=2.)]),
        f=old['f'], g=old['g'], sigma_y=old['sigma_y'],
        observations_reused=True, physical_parameters_reused=True, prior_reused=True,
        comparison=('one cross replay; reuse saved day to separate time and state'
                    if stage == 'cross' else
                    'combined time/state refinement; not separate time and state attribution'))


def run_terminal_control(report, reference, stage, *, saved_rows=(), checkpoint=None,
                         progress=None, deadline=None, terminal_reference=None):
    """Fixed resolutions, one atomic pilot unit per completed interval.

    This checkpoint stores the pilot's actual log state. It is not an adapter
    for the historical fitter, arbitrary records, or continuously varying B/U.
    """
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_fitted
    from src.validation.bm_1.interval_curve_control import _compare
    record, design = declared_terminal_control(report, reference, stage)
    if stage == 'cross':
        _, expected = declared_terminal_control(report, reference, 'day')
        if (terminal_reference is None or terminal_reference.get('design') != expected
                or terminal_reference.get('status') != 'completed'
                or terminal_reference.get('stage') != 'day'
                or len(terminal_reference.get('replays', [])) != 2):
            raise ValueError('Cross control requires the matching completed terminal day.')
    rows, count = list(saved_rows), design['intervals']
    total = len(design['settings'])*count
    if len(rows) > total or any((r['setting_index'], r['interval_index']) != divmod(i, count)
                                for i, r in enumerate(rows)):
        raise ValueError('Checkpoint does not follow the declared interval order.')
    model = bm_1(record.baseline, record.baseline, record.horizon, 2)
    solver = finite_volume_1d_fitted(vectorized_inputs=True)
    model.fokker_planc_reference = solver
    weights = [s['cells']/s['max_step']*s['frequencies'] for s in design['settings'] for _ in range(count)]
    while len(rows) < total:
        if deadline is not None and perf_counter() >= deadline:
            return dict(status='paused', rows=rows, development_checks_pass=False)
        setting_index, index = divmod(len(rows), count)
        settings = design['settings'][setting_index]
        edges, x, _ = solver.grid(settings['cells'])
        if index == 0:
            current = record.initial_mass_builder(edges)
            with np.errstate(divide='ignore'):
                logs = np.log(current)
        else:
            logs = np.array([-np.inf if v is None else v for v in rows[-1]['log_posterior_mass']])
            if logs.shape != x.shape or np.isnan(logs).any() or np.isposinf(logs).any():
                raise ValueError('Invalid checkpoint log state.')
            current = np.exp(logs)
        start, end = record.observations.starts[index], record.observations.ends[index]
        seen, value = record.observations.observed[index], record.observations.values[index]
        drift, diffusion, options = model._fv_evolution_fields(record.parameters, record.sigma_x,
            np.array([start, end]), [], settings['max_step'])
        p = record.parameters
        reward = lambda t, states: model.eval(t, states, p[0], p[1], p[2], p[4], p[5])[-1]
        begun = perf_counter()
        try:
            if seen:
                result = terminal_tilt_update(solver, [start, end], current, drift, diffusion, reward,
                    value, record.sigma_y, log_prior=logs, **{k:v for k,v in settings.items() if k != 'cells'},
                    centers=design['centers'], relative_guard=design['relative_guard'],
                    overlap_log_tolerance=design['overlap_log_tolerance'], **options)
                logs = result['log_posterior_mass']
                diagnostics = {k:result[k] for k in ('log_evidence', 'contours',
                    'maximum_relative_conditional_guard', 'maximum_overlap_log_difference', 'maximum_abs_mass_error')}
                diagnostics['selected_contour_counts'] = np.bincount(result['selected_contour'][result['selected_contour'] >= 0],
                    minlength=len(result['contours'])).tolist()
            else:
                result = balanced_characteristic_prediction(solver, [start, end], current, drift, diffusion, reward,
                    max_step=settings['max_step'], frequencies=1, log_prior=logs,
                    state_scheme='positive_crank_nicolson', **options)
                logs = result['log_amplitude']
                diagnostics = dict(log_evidence=None, contours=[], selected_contour_counts=[],
                    maximum_relative_conditional_guard=0., maximum_overlap_log_difference=0.,
                    maximum_abs_mass_error=result['maximum_abs_mass_error'])
        except CharacteristicNumericalError as exc:
            return dict(status='numerical_guard_failed', rows=rows, development_checks_pass=False,
                        failed_setting=setting_index, failed_interval=index, reason=str(exc), diagnostics=exc.diagnostics)
        row = dict(setting_index=setting_index, interval_index=index,
            log_posterior_mass=[None if np.isneginf(v) else float(v) for v in logs],
            posterior_state_mean=float(np.exp(logs) @ x), elapsed_seconds=perf_counter()-begun, **diagnostics)
        rows.append(row)
        if checkpoint is not None:
            checkpoint(rows)
        if progress is not None:
            progress(sum(weights[:len(rows)]), sum(weights))
    replays = []
    for i, settings in enumerate(design['settings']):
        units = rows[i*count:(i+1)*count]
        logs = np.array([-np.inf if v is None else v for v in units[-1]['log_posterior_mass']])
        edges, _, _ = solver.grid(settings['cells'])
        replays.append(dict(status='completed', **settings, x_edges=edges.tolist(),
            final_posterior_mass=np.exp(logs).tolist(),
            log_likelihood=float(sum(r['log_evidence'] for r in units if r['log_evidence'] is not None)),
            posterior_state_means=[r['posterior_state_mean'] for r in units],
            elapsed_seconds=sum(r['elapsed_seconds'] for r in units)))
    if stage == 'cross':
        coarse, fine = terminal_reference['replays']
        comparisons = [
            _compare('time_at_1281_cells', coarse, replays[0], report['development_tolerances']),
            _compare('space_at_h_1_24576', replays[0], fine, report['development_tolerances'])]
        return dict(status='completed', rows=rows, replays=replays, comparisons=comparisons,
                    reference_replays_reused=2, reference_intervals_recomputed=0,
                    saved_combined_comparison=terminal_reference['comparison'],
                    numerical_fit_readiness=False,
                    development_checks_pass=all(c['development_tolerances_pass'] for c in comparisons))
    comparison = _compare('combined_time_state', replays[0], replays[1], report['development_tolerances'])
    return dict(status='completed', rows=rows, replays=replays, comparison=comparison,
                development_checks_pass=comparison['development_tolerances_pass'])

"""F interval likelihood and continuous, candidate-owned replay/checkpoints.

The Gaussian observation has real support; its latent interval reward lies in
[0,1]. Native and independent reference solvers share this numerical contract.
Only replay_precise_candidate imports the independent reference implementation.
"""

import hashlib, json, numpy as np
from time import perf_counter
from scipy.special import logsumexp
from src.params.bm_1 import BM1IntervalObservationSeries
from src.solver.finite_volume import finite_volume_1d_fitted


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


PRECISE_SETTINGS = dict(cells=1921, max_step=1/24576, frequencies=64, span=2.,
    centers=[.5, .25, .75, 0., 1.], relative_guard=1e-8, overlap_log_tolerance=1e-7)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def candidate_panel(step=.5):
    """One linear reference plus +/- each of 2 and 6 independent drop logits.

    Exactly 17 fixed candidates; a single coordinate probe, not a converged fit.
    The shared linear reference is not re-evaluated for each curve family.
    """
    if not np.isfinite(step) or not 0. < step <= 4.:
        raise ValueError('Declare a positive logit step <=4.')
    rows = [dict(id='linear', pieces=2, logits=[0., 0.])]
    for pieces in (2, 4):
        for coordinate in range(2*(pieces-1)):
            for sign in (1, -1):
                logits = [0.] * (2*(pieces-1)); logits[coordinate] = sign*float(step)
                rows.append(dict(id=f'p{pieces}_c{coordinate}_{"plus" if sign == 1 else "minus"}',
                                 pieces=pieces, logits=logits))
    return rows


def freeze_hourly_record(record, settings=PRECISE_SETTINGS):
    """Freeze a record produced by the historical or saved-development adapter."""
    if not isinstance(record.observations, BM1IntervalObservationSeries) or record.observations.time_unit != 'day':
        raise ValueError('Precise H2 requires the explicit interval-average/day contract.')
    edges = np.r_[record.horizon[0], record.breakpoints, record.horizon[-1]]
    left, right = edges[:-1], edges[1:]
    b, u = np.asarray(record.baseline(left)), np.asarray(record.parameters[0](left))
    # A sanity check on the registered hourly-hold callbacks, not a proof for
    # arbitrary time-dependent functions. Callers only use the two named adapters.
    for fraction in (.25, .75):
        if (not np.array_equal(record.baseline(left+fraction*(right-left)), b)
                or not np.array_equal(record.parameters[0](left+fraction*(right-left)), u)):
            raise ValueError('Only frozen time-held B/U inputs are supported here.')
    grid, _, _ = finite_volume_1d_fitted().grid(settings['cells'])
    initial = np.asarray(record.initial_mass_builder(grid), dtype=float)
    with np.errstate(divide='ignore'):
        logs = np.log(initial)
    obs = record.observations
    return dict(record_id=record.record_id, horizon=list(record.horizon),
        input_edges=edges.tolist(), baseline=b.tolist(), price=u.tolist(),
        physical=dict(C=record.parameters[3], k=record.parameters[4], Xi=record.parameters[5],
                      sigma_x=record.sigma_x, sigma_y=record.sigma_y),
        starts=obs.starts.tolist(), ends=obs.ends.tolist(), observed=obs.observed.tolist(),
        values=[float(y) if seen else None for y, seen in zip(obs.values, obs.observed)],
        initial_log_mass=[None if np.isneginf(x) else float(x) for x in logs],
        settings=dict(settings), time_unit='day', midnight_resets=0)


def replay_units(spec):
    """Include pre-TRAIN evolution and gaps, split only their prediction units."""
    observations = BM1IntervalObservationSeries(spec['starts'], spec['ends'],
        [np.nan if y is None else y for y in spec['values']], 'day', spec['observed'])
    if spec['time_unit'] != 'day' or spec['midnight_resets'] != 0:
        raise ValueError('Never change time units or reset daily state.')
    edges = np.asarray(spec['input_edges'], dtype=float)
    if (np.any(np.diff(edges) <= 0.) or edges[0] != spec['horizon'][0]
            or edges[-1] != spec['horizon'][-1] or len(edges)-1 != len(spec['baseline'])
            or len(spec['baseline']) != len(spec['price'])
            or observations.ends[-1] != edges[-1] or observations.starts[0] < edges[0]):
        raise ValueError('Frozen inputs must cover the whole declared TRAIN and prior lead.')
    units, current = [], float(edges[0])
    for index, (start, end, seen, value) in enumerate(zip(observations.starts,
            observations.ends, observations.observed, observations.values)):
        gap_edges = np.r_[current, edges[(edges > current) & (edges < start)], start]
        for a, z in zip(gap_edges[:-1], gap_edges[1:]):
            if z > a:
                units.append(dict(start=float(a), end=float(z), observed=False,
                                  value=None, observation_index=None))
        units.append(dict(start=float(start), end=float(end), observed=bool(seen),
                          value=float(value) if seen else None, observation_index=index))
        current = float(end)
    return units


def validate_checkpoint(saved, spec, candidate):
    """Reject changed candidates/inputs and corrupt evidence before state reuse."""
    identity = digest(dict(spec=spec, candidate=candidate))
    if saved['identity'] != identity or saved['checksum'] != digest(saved['payload']):
        raise ValueError('Checkpoint candidate/input identity or checksum changed.')
    payload, units = saved['payload'], replay_units(spec)
    rows = payload['rows']
    if len(rows) > len(units):
        raise ValueError('Checkpoint exceeds TRAIN.')
    for index, row in enumerate(rows):
        if row['unit'] != index or any(row[k] != units[index][k] for k in units[index]):
            raise ValueError('Checkpoint chronology or observation mask changed.')
        if (row['observed'] != (row['log_evidence'] is not None)
                or not np.isfinite(row['state_mean']) or row['maximum_abs_mass_error'] > 1e-10):
            raise ValueError('Invalid saved evidence or probability guard.')
    score = sum(r['log_evidence'] for r in rows if r['observed'])
    if score != payload['log_likelihood'] or not np.isfinite(score):
        raise ValueError('Checkpoint cumulative evidence changed.')
    logs = np.array([-np.inf if x is None else x for x in payload['log_mass']])
    if (logs.shape != (spec['settings']['cells'],) or np.isnan(logs).any()
            or np.isposinf(logs).any() or abs(float(logsumexp(logs))) > 1e-10):
        raise ValueError('Invalid checkpoint log state; finite tails must be retained.')
    if not rows and payload['log_mass'] != spec['initial_log_mass']:
        raise ValueError('The declared initial law changed.')
    return payload


def replay_precise_candidate(spec, candidate, *, saved=None, limit_units=None,
                             should_pause=None, checkpoint=None):
    """Whole-stream candidate; hourly checkpoints own the actual log posterior."""
    from src.method.bm_1_response import curve_pair_from_logits
    from src.model.bm_1 import bm_1
    from src.tools.input_generator.input_builder import input_builder
    from src.validation.bm_1.interval_characteristic import balanced_characteristic_prediction
    from src.validation.bm_1.interval_terminal_tilt import terminal_tilt_update
    settings = dict(spec['settings']); cells = settings.pop('cells')
    f, g = curve_pair_from_logits(candidate['logits'], candidate['pieces'])
    units = replay_units(spec)
    payload = (dict(rows=[], log_mass=spec['initial_log_mass'], log_likelihood=0., status='paused')
               if saved is None else dict(validate_checkpoint(saved, spec, candidate)))
    if payload.get('status') == 'numerical_guard_failed':
        raise ValueError('Preserve the failed candidate; no automatic numerical retry.')
    rows = list(payload['rows'])
    logs = np.array([-np.inf if x is None else x for x in payload['log_mass']])
    if not rows:
        validate_checkpoint(dict(identity=digest(dict(spec=spec, candidate=candidate)),
            payload=payload, checksum=digest(payload)), spec, candidate)
    B = input_builder.price_intervals((spec['input_edges'], spec['baseline']))
    U = input_builder.price_intervals((spec['input_edges'], spec['price']))
    p = spec['physical']; parameters = (U, f, g, p['C'], p['k'], p['Xi'])
    model = bm_1(B, B, spec['horizon'], 2)
    solver = finite_volume_1d_fitted(vectorized_inputs=True)
    model.fokker_planc_reference = solver
    _, states, _ = solver.grid(cells)
    identity = digest(dict(spec=spec, candidate=candidate))
    def publish(status, **extra):
        nonlocal payload
        payload = dict(rows=rows, log_mass=[None if np.isneginf(x) else float(x) for x in logs],
            log_likelihood=float(sum(r['log_evidence'] for r in rows if r['observed'])),
            status=status, **extra)
        result = dict(identity=identity, checksum=digest(payload), payload=payload)
        if checkpoint is not None:
            checkpoint(result)
        return result
    stop = len(units) if limit_units is None else min(len(units), limit_units)
    for index in range(len(rows), stop):
        if should_pause is not None and should_pause():
            return publish('paused')
        unit = units[index]; start, end = unit['start'], unit['end']
        breaks = np.asarray(spec['input_edges'])[1:-1]
        breaks = breaks[(breaks > start) & (breaks < end)]
        drift, diffusion, options = model._fv_evolution_fields(parameters, p['sigma_x'],
            np.array([start, end]), breaks, settings['max_step'])
        reward = lambda t, x: model.eval(t, x, U, f, g, p['k'], p['Xi'])[-1]
        begun = perf_counter()
        try:
            if unit['observed']:
                result = terminal_tilt_update(solver, [start, end], np.exp(logs), drift, diffusion,
                    reward, unit['value'], p['sigma_y'], log_prior=logs, breakpoints=breaks,
                    **settings, **options)
                next_logs = result['log_posterior_mass']; evidence = float(result['log_evidence'])
                diagnostics = {k: result[k] for k in ('contours', 'maximum_relative_conditional_guard',
                    'maximum_overlap_log_difference', 'maximum_abs_mass_error')}
            else:
                result = balanced_characteristic_prediction(solver, [start, end], np.exp(logs),
                    drift, diffusion, lambda t, x: np.zeros_like(x), log_prior=logs,
                    max_step=settings['max_step'], frequencies=1, breakpoints=breaks,
                    state_scheme='positive_crank_nicolson', **options)
                next_logs = result['log_amplitude']; evidence = None
                diagnostics = dict(contours=[], maximum_relative_conditional_guard=0.,
                    maximum_overlap_log_difference=0., maximum_abs_mass_error=result['maximum_abs_mass_error'])
        except (CharacteristicNumericalError, ValueError) as exc:
            return publish('numerical_guard_failed', failed_unit=index, reason=str(exc),
                           diagnostics=getattr(exc, 'diagnostics', {}))
        logs = next_logs
        rows.append(dict(unit=index, **unit, log_evidence=evidence,
            state_mean=float(np.exp(logs) @ states), elapsed_seconds=perf_counter()-begun,
            **diagnostics))
        publish('completed' if len(rows) == len(units) else 'paused')
    return publish('completed' if len(rows) == len(units) else 'prefix_complete')

def interval_filter_step(model, parameters, sigma_x, sigma_y, prior_mass,
                         start, end, observation, *, observed=True, breakpoints=None,
                         max_step=1/768, accumulator_spacing=1/256,
                         max_accumulator_nodes=4096):
    """Assimilate the joint interval law and carry its end-state posterior."""
    from src.method.bm_1_estimation import BM1FVFilterStep, BM1FVFilterResult, BM1FVLikelihoodResult, fv_filter_step, _filter_interval
    from src.method.bm_1_observation import bayes_update, mass_to_log, gaussian_log_likelihood
    from src.solver.finite_volume_additive import propagate_additive
    from src.validation import estimation as checks
    from src.validation.model.bm import bm_validation

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
    from src.method.bm_1_estimation import BM1FVFilterStep, BM1FVFilterResult, BM1FVLikelihoodResult, fv_filter_step, _filter_interval
    from src.method.bm_1_observation import bayes_update, mass_to_log, gaussian_log_likelihood
    from src.solver.finite_volume_additive import propagate_additive
    from src.validation import estimation as checks
    from src.validation.model.bm import bm_validation

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

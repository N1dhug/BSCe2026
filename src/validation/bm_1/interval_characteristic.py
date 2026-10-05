"""Bounded characteristic-function control; not a historical fit backend.

This experiment removes nodal accumulator deposition. Complex characteristic
columns evolve with the SAME real FV implicit state solve, and exact frozen
reward phases. A truncated periodic Gaussian Fourier series supplies evidence.
Positivity and finite-series guards abort unresolved calculations; no clipping,
cost substitution or production fallback is introduced.
"""

from time import perf_counter

import numpy as np
from scipy.special import betainc, logsumexp

from src.method.bm_1_response import CurveTrainingRecord
from src.model.bm_1 import bm_1
from src.params.bm_1 import conservative_mass_transfer
from src.params.bm_1_interval import BM1IntervalObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse
from src.solver.finite_volume import finite_volume_1d_fitted
from src.tools.input_generator.input_builder import input_builder
from src.validation import estimation as checks


class CharacteristicNumericalError(ValueError):
    """Expected unresolved numerical control, never a substituted fit cost."""
    def __init__(self, reason, **diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def _log_scan(values, weights):
    """Positive triangular recurrence z[i]=v[i]+w[i-1]*z[i-1].

    Accumulated products are evaluated in log space. Zero weights split the
    recurrence into disconnected blocks; no floor is added to zero masses.
    Extended precision reduces cancellation in the cumulative products where
    supported, but the method also supports Windows float64 longdouble.
    """
    values = np.asarray(values, dtype=np.longdouble)
    result = np.empty_like(values)
    cuts = np.r_[0, np.flatnonzero(weights == 0.)+1, len(values)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        products = np.r_[np.longdouble(0.),
            np.cumsum(np.log(weights[lo:hi-1].astype(np.longdouble)))]
        result[lo:hi] = products+np.logaddexp.accumulate(values[lo:hi]-products)
    return result


def _balanced_state_step(log_amplitude, conditional_cf, lower, diagonal, upper):
    """The SAME implicit FV equation, with positive diagonal row scaling.

    A positive Thomas solve in log space supplies a scale for each row. The
    real LAPACK solver then solves D^-1 M D for the conditional transforms.
    Its zero-frequency column gives the actual amplitude correction; retaining
    that correction is an algebraic representation, not mass normalization.
    """
    pivots = diagonal.copy()
    multipliers = np.empty_like(lower)
    for j in range(1, len(diagonal)):
        multipliers[j-1] = -lower[j-1]/pivots[j-1]
        pivots[j] += multipliers[j-1]*upper[j-1]
    if (np.any(pivots <= 0.) or not np.isfinite(pivots).all()
            or np.any(lower > 0.) or np.any(upper > 0.)):
        raise CharacteristicNumericalError('Positive implicit factorization failed.')
    forward = _log_scan(log_amplitude, multipliers)
    scale = _log_scan((forward-np.log(pivots))[::-1],
        (-upper/pivots[:-1])[::-1])[::-1]
    scale = np.asarray(scale, dtype=float)
    active = np.isfinite(scale)
    row_diag = diagonal.copy()
    balanced_lower, balanced_upper = np.zeros_like(lower), np.zeros_like(upper)
    linked = active[1:] & active[:-1]
    lower_active, upper_active = linked & (lower < 0.), linked & (upper < 0.)
    balanced_lower[lower_active] = -np.exp(np.log(-lower[lower_active])
        +scale[:-1][lower_active]-scale[1:][lower_active])
    balanced_upper[upper_active] = -np.exp(np.log(-upper[upper_active])
        +scale[1:][upper_active]-scale[:-1][upper_active])
    rhs = np.zeros_like(conditional_cf)
    rhs[active] = np.exp(log_amplitude[active]-scale[active])[:, None]*conditional_cf[active]
    row_diag[~active] = 1.
    from scipy.linalg.lapack import dgtsv
    columns = rhs.shape[1]
    real_rhs = np.array(np.concatenate((rhs.real, rhs.imag), axis=1), order='F')
    _, _, _, result, info = dgtsv(balanced_lower, row_diag, balanced_upper, real_rhs,
        overwrite_dl=1, overwrite_d=1, overwrite_du=1, overwrite_b=1)
    if info != 0:
        raise CharacteristicNumericalError('Balanced LAPACK state solve failed.', info=int(info))
    result = result[:, :columns]+1j*result[:, columns:]
    zero = result[:, 0].real
    if np.any(zero[active] <= 0.) or not np.isfinite(result).all():
        raise CharacteristicNumericalError('Balanced zero-frequency amplitude failed.')
    scale[active] += np.log(zero[active])
    result[active] /= zero[active, None]
    return scale, result


def _positive_explicit_step(log_amplitude, cf, lower, diagonal, upper):
    """Positive tridiagonal product, without exponentiating small masses."""
    if np.any(diagonal < 0.) or np.any(lower < 0.) or np.any(upper < 0.):
        raise CharacteristicNumericalError('Crank-Nicolson explicit half is not positive; decrease the declared step.',
            minimum_explicit_diagonal=float(diagonal.min()))
    with np.errstate(divide='ignore'):
        terms = np.full((3,len(log_amplitude)), -np.inf)
        terms[0] = log_amplitude+np.log(diagonal)
        terms[1,1:] = log_amplitude[:-1]+np.log(lower)
        terms[2,:-1] = log_amplitude[1:]+np.log(upper)
    scale = logsumexp(terms,axis=0)
    active = np.isfinite(scale)
    weights = np.zeros_like(terms)
    weights[:,active] = np.exp(terms[:,active]-scale[active])
    result = weights[0,:,None]*cf
    result[1:] += weights[1,1:,None]*cf[:-1]
    result[:-1] += weights[2,:-1,None]*cf[1:]
    return scale,result


def balanced_characteristic_prediction(solver, interval, prior, drift, diffusion, reward, *,
        max_step, frequencies=64, span=4., breakpoints=None, coefficient_key=None,
        tilt=0., center=0., log_prior=None, state_scheme='implicit_euler'):
    """Row-scaled transform control preserving arbitrarily small amplitudes."""
    from src.validation.solver.finite_volume import finite_volume_1d_validation as fv_checks
    prior = checks.probability_mass(prior)
    frequencies = checks.positive_integer(frequencies, 'frequencies')
    span = checks.finite_scalar(span, 'span', positive=True)
    tilt = checks.finite_scalar(tilt, 'tilt')
    center = checks.finite_scalar(center, 'center')
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    interval = checks.increasing_grid(interval, 'interval')
    if state_scheme not in ('implicit_euler','positive_crank_nicolson'):
        raise ValueError('Declare implicit_euler or positive_crank_nicolson.')
    if interval.size != 2:
        raise ValueError('Supply interval start/end.')
    if log_prior is None:
        with np.errstate(divide='ignore'):
            log_prior = np.log(prior)
    log_amplitude = np.array(log_prior, dtype=float, copy=True)
    if (log_amplitude.shape != prior.shape or np.isnan(log_amplitude).any()
            or np.isposinf(log_amplitude).any() or abs(float(logsumexp(log_amplitude))) > 1e-10
            or not np.allclose(np.exp(log_amplitude), prior, rtol=1e-12, atol=0.)):
        raise ValueError('Log prior and ordinary prior must describe the same probability law.')
    omega = 2.*np.pi*np.arange(frequencies+1)/span
    edges, x, dx = solver.grid(len(prior))
    grid, _ = solver.build_time_grid(interval, breakpoints, max_step)
    advance = solver._stepper(dx, edges, x, drift, diffusion, coefficient_key)
    neutral = prior.copy()
    cf = np.ones((len(prior), frequencies+1), dtype=complex)
    maximum_error = 0.
    generators = {}
    for start, end in zip(grid[:-1], grid[1:]):
        h = float(end-start)
        r = np.broadcast_to(np.asarray(reward(float(start), x), dtype=float), x.shape)
        if not np.isfinite(r).all() or np.any(r < 0.) or np.any(r > 1.):
            raise ValueError('Reward must remain finite in [0,1].')
        phase = np.exp(1j*(r-center)[:, None]*omega[None, :]*h/(2.*(interval[1]-interval[0])))
        shift = tilt*(r-center)*h/(2.*(interval[1]-interval[0]))
        cf *= phase
        log_amplitude += shift
        key = coefficient_key(float(start)) if coefficient_key is not None else None
        if key not in generators or coefficient_key is None:
            a = fv_checks.field(drift(float(start), edges[1:-1]), edges[1:-1].shape, 'drift')
            b = fv_checks.field(diffusion(float(start), x), x.shape, 'diffusion')
            generators[key] = solver.generator_tridiagonal(dx, a, b)[:3]
        lo, di, up = generators[key]
        state_h = h/2. if state_scheme == 'positive_crank_nicolson' else h
        if state_scheme == 'positive_crank_nicolson':
            explicit_diagonal = 1.+state_h*di
            log_amplitude,cf = _positive_explicit_step(log_amplitude,cf,
                state_h*lo,explicit_diagonal,state_h*up)
            explicit_neutral = explicit_diagonal*neutral
            explicit_neutral[1:] += state_h*lo*neutral[:-1]
            explicit_neutral[:-1] += state_h*up*neutral[1:]
            neutral = explicit_neutral
        log_amplitude, cf = _balanced_state_step(log_amplitude, cf, -state_h*lo, 1.-state_h*di, -state_h*up)
        cf *= phase
        log_amplitude += shift
        neutral, _ = advance(neutral, float(start), state_h)
        error = abs(float(neutral.sum())-1.)
        if (np.any(neutral < 0.) or error > 1e-10 or not np.isfinite(cf).all()
                or np.isnan(log_amplitude).any() or np.isposinf(log_amplitude).any()
                or np.any(np.abs(cf) > 1.+1e-10)):
            raise CharacteristicNumericalError('Balanced characteristic probability/modulus guard failed.',
                mass_error=error, maximum_modulus=float(np.abs(cf).max()))
        maximum_error = max(maximum_error, error)
    return dict(cf=cf, log_amplitude=log_amplitude, frequencies=omega, span=span,
        predicted_mass=neutral, tilt=tilt, center=center, x_edges=edges, x_grid=x,
        maximum_abs_mass_error=maximum_error)


def characteristic_prediction(solver, interval, prior, drift, diffusion, reward, *,
                               max_step, frequencies=64, span=4., breakpoints=None,
                               coefficient_key=None, tilt=0., center=0.):
    """Finite-state terminal characteristic vector of the interval average."""
    prior = checks.probability_mass(prior)
    frequencies = checks.positive_integer(frequencies, 'frequencies')
    span = checks.finite_scalar(span, 'span', positive=True)
    tilt = checks.finite_scalar(tilt, 'tilt')
    center = checks.finite_scalar(center, 'center')
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    interval = checks.increasing_grid(interval, 'interval')
    if interval.size != 2:
        raise ValueError('Supply interval start/end.')
    omega = 2.*np.pi*np.arange(frequencies+1)/span
    exponents = np.r_[0.+0.j, tilt+1j*omega]
    edges, centers, dx = solver.grid(len(prior))
    grid, _ = solver.build_time_grid(interval, breakpoints, max_step)
    advance = solver._stepper(dx, edges, centers, drift, diffusion, coefficient_key)
    cf = np.broadcast_to(prior[:, None], (len(prior), len(exponents))).astype(complex).copy()
    maximum_error = 0.
    for index, (start, end) in enumerate(zip(grid[:-1], grid[1:])):
        h = float(end-start)
        r = np.broadcast_to(np.asarray(reward(float(start), centers), dtype=float), centers.shape)
        if not np.isfinite(r).all() or np.any(r < 0.) or np.any(r > 1.):
            raise ValueError('Reward must remain finite in [0,1].')
        phase = np.exp((r-center)[:, None]*exponents[None, :]*h/(2.*(interval[1]-interval[0])))
        cf *= phase
        # LAPACK backend is real. Keep real/imaginary columns explicitly;
        # passing complex RHS through its float cast would lose information.
        rhs = np.concatenate((cf.real, cf.imag), axis=1)
        evolved, _ = advance(rhs, float(start), h)
        cf = (evolved[:, :len(exponents)]+1j*evolved[:, len(exponents):])*phase
        prediction = cf[:, 0].real
        error = abs(float(prediction.sum())-1.)
        # Relative tolerances alone round back to the same number for
        # subnormal tails. Account for accumulated subnormal rounding in this diagnostic;
        # every coefficient is retained unchanged, including tiny tails.
        modulus_bound = (cf[:, 1].real*(1.+1e-10)
            +8.*(index+1)*np.nextafter(0.,1.))[:, None]
        if (not np.isfinite(cf).all() or np.any(prediction < 0.) or error > 1e-10
                or np.any(cf[:, 1].real < 0.)
                or np.any(np.abs(cf[:, 1:]) > modulus_bound)):
            raise CharacteristicNumericalError('Characteristic probability/modulus guard failed; no repair.',
                mass_error=error, minimum_neutral=float(prediction.min()),
                minimum_tilted=float(cf[:,1].real.min()),
                maximum_modulus_excess=float(np.max(np.abs(cf[:,1:])-modulus_bound)))
        maximum_error = max(maximum_error, error)
    return dict(cf=cf[:, 1:], frequencies=omega, span=span, predicted_mass=cf[:, 0].real,
        tilt=tilt, center=center,
        x_edges=edges, x_grid=centers, maximum_abs_mass_error=maximum_error)


def characteristic_gaussian_update(prediction, value, sigma_y, *, relative_guard=1e-8):
    """Score the periodic Gaussian, failing unresolved series or signs.

    Analytic alias/truncation bounds use continuous A in [0,1]. The extra
    floating-point guard is a conservative heuristic, not a proof of all
    matrix-solve roundoff. Time/state error requires separate refinements.
    """
    value = checks.finite_scalar(value, 'value')
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    relative_guard = checks.finite_scalar(relative_guard, 'relative_guard', positive=True)
    span = prediction['span']
    omega = prediction['frequencies']
    tilt, center = prediction['tilt'], prediction['center']
    shifted_value = value-center-sigma_y**2*tilt
    coefficients = np.exp(-.5*(sigma_y*omega)**2)*np.exp(-1j*omega*shifted_value)
    density = (prediction['cf'][:, 0].real+
               2.*np.real(prediction['cf'][:, 1:] @ coefficients[1:]))/span
    scaled = 'log_amplitude' in prediction
    active = np.isfinite(prediction['log_amplitude']) if scaled else np.ones(len(density), dtype=bool)
    evidence = float(density.sum())
    if (np.any(density < 0.) or not np.isfinite(density).all() or evidence <= 0.
            or (scaled and np.any(density[active] <= 0.))):
        raise CharacteristicNumericalError('Fourier likelihood positivity/cancellation guard failed; no clipping.',
            minimum_density=float(density.min()), summed_density=evidence,
            negative_cells=int(np.sum(density < 0.)))
    distance = max(abs(shifted_value+center), abs(shifted_value+center-1.))
    if span <= distance:
        raise ValueError('Periodic span does not separate the physical reward range.')
    gap = span-distance
    alias = (2.*np.exp(-.5*(gap/sigma_y)**2)/(np.sqrt(2.*np.pi)*sigma_y)
             /(-np.expm1(-(2.*gap*span+span**2)/(2.*sigma_y**2))))
    alpha = .5*(2.*np.pi*sigma_y/span)**2
    K = len(omega)-1
    tail = 2./span*np.exp(-alpha*(K+1)**2)/(-np.expm1(-alpha*(2*K+3)))
    roundoff = 64.*np.finfo(float).eps*(1.+2.*np.sum(np.abs(coefficients[1:])))/span
    if scaled:
        log_density = np.full(len(density), -np.inf)
        log_density[active] = prediction['log_amplitude'][active]+np.log(density[active])
        log_evidence = float(logsumexp(log_density))
        log_tilted_mass = float(logsumexp(prediction['log_amplitude']))
    else:
        log_evidence = float(np.log(evidence))
        log_tilted_mass = float(np.log(prediction['cf'][:, 0].real.sum()))
    error_guard = (alias+tail+roundoff)*np.exp(log_tilted_mass-log_evidence)
    if error_guard > relative_guard:
        raise CharacteristicNumericalError('Fourier alias/tail/roundoff guard unresolved; refine the declared method.',
            relative_error_guard=float(error_guard), evidence=evidence, alias=float(alias),
            tail=float(tail), roundoff=float(roundoff))
    log_posterior = log_density-log_evidence if scaled else np.log(density/evidence)
    posterior = checks.probability_mass(np.exp(log_posterior))
    log_prefactor = .5*sigma_y**2*tilt**2-tilt*(value-center)
    return dict(posterior_mass=posterior, log_posterior_mass=log_posterior,
                log_evidence=float(log_evidence+log_prefactor),
                relative_series_guard=float(error_guard), alias_bound=float(alias), truncation_bound=float(tail))


def record_from_development_report(report):
    """Reuse exactly saved synthetic inputs; no workbook or arbitrary history."""
    if report.get('schema') != 'bm1-hourly-interval-development-v1' or report.get('historical_observation_values_read') != 0:
        raise ValueError('Use the registered synthetic hourly development report.')
    design = report['declared_design']
    obs = design['observations']
    if design['rows'] != 24 or len(obs['starts']) != 24:
        raise ValueError('This control is bounded to the registered one-day panel.')
    observations = BM1IntervalObservationSeries(obs['starts'], obs['ends'],
        [np.nan if y is None else y for y in obs['values']], 'day', obs['observed'])
    edges = np.r_[observations.starts, observations.ends[-1]]
    B = input_builder.price_intervals((edges, design['baseline_values']))
    U = input_builder.price_intervals((edges, design['price_values']))
    p = design['parameters']
    linear = BM1PiecewiseResponse.linear()
    def initial(grid):
        return np.diff(betainc(2., 3., grid))
    return CurveTrainingRecord('synthetic-characteristic-control', 'TRAIN', B, (0., 1.),
        (U, linear, linear, p['C'], p['k'], p['Xi']), p['sigma_x'], p['sigma_y'],
        initial, observations, edges[1:-1])


def replay_characteristic(record, *, cells, max_step, frequencies=64, span=4., exponential_tilt=True,
                          balanced_transform=True, state_scheme='implicit_euler'):
    """Development replay only, accepting the registered 24-hour interval case."""
    if not isinstance(record.observations, BM1IntervalObservationSeries) or len(record.observations.times) != 24:
        raise ValueError('Only the registered synthetic interval case enters this experiment.')
    model = bm_1(record.baseline, record.baseline, record.horizon, 2)
    model.fokker_planc_reference = finite_volume_1d_fitted(vectorized_inputs=True)
    solver = model.fokker_planc_reference
    edges, x, _ = solver.grid(cells)
    current = record.initial_mass_builder(edges)
    with np.errstate(divide='ignore'):
        log_current = np.log(current)
    scores, posterior, guards = [], [], []
    maximum_error = 0.
    for start, end, value, seen in zip(record.observations.starts, record.observations.ends,
                                     record.observations.values, record.observations.observed):
        breaks = record.breakpoints[(record.breakpoints > start) & (record.breakpoints < end)]
        drift, diffusion, options = model._fv_evolution_fields(record.parameters, record.sigma_x,
            np.array([start, end]), breaks, max_step)
        def reward(t, states):
            p = record.parameters
            return model.eval(t, states, p[0], p[1], p[2], p[4], p[5])[-1]
        # Exact Gaussian exponential-tilting identity avoids cancellation for
        # improbable measurements. This changes representation, not the model:
        # phi(y-A)=exp(-(y-c)^2/(2s^2))*exp((y-c)(A-c)/s^2)*phi(c-A).
        # The neutral column still carries the ordinary unconditioned state.
        center = float(reward(float(start), x) @ current) if exponential_tilt else 0.
        tilt = (float(value)-center)/record.sigma_y**2 if seen and exponential_tilt else 0.
        try:
            prediction_fn = balanced_characteristic_prediction if balanced_transform else characteristic_prediction
            if not balanced_transform and state_scheme != 'implicit_euler':
                raise ValueError('The new time scheme requires the balanced transform.')
            extra = dict(log_prior=log_current,state_scheme=state_scheme) if balanced_transform else {}
            prediction = prediction_fn(solver, [start, end], current, drift, diffusion,
                reward, max_step=max_step, frequencies=frequencies, span=span, breakpoints=breaks,
                tilt=tilt, center=center, **extra, **options)
            maximum_error = max(maximum_error, prediction['maximum_abs_mass_error'])
            if seen:
                update = characteristic_gaussian_update(prediction, value, record.sigma_y)
                current = update['posterior_mass']
                log_current = update['log_posterior_mass']
                scores.append(update['log_evidence'])
                guards.append(update['relative_series_guard'])
            else:
                log_current = prediction['log_amplitude'] if balanced_transform else np.log(prediction['predicted_mass'])
                current = np.exp(log_current)
                scores.append(None)
        except CharacteristicNumericalError as exc:
            exc.diagnostics.update(interval_start=float(start), interval_end=float(end),
                completed_intervals=len(scores), partial_log_evidence=scores)
            raise
        posterior.append(current.copy())
    return dict(status='completed', cells=cells, max_step=max_step, frequencies=frequencies, span=span,
        exponential_tilt=exponential_tilt, balanced_transform=balanced_transform,state_scheme=state_scheme,
        log_likelihood=float(sum(y for y in scores if y is not None)), log_evidence=scores,
        x_edges=edges.tolist(), final_posterior_mass=current.tolist(),
        posterior_state_means=list(map(float, np.asarray(posterior) @ x)),
        maximum_abs_mass_error=maximum_error, maximum_relative_series_guard=max(guards),
        used_clipping=False, scientific_acceptance=False)


def run_characteristic_control(report, *, progress=None):
    """Cheap bounded panel; no optimizer, historical values or backend adoption."""
    begun = perf_counter()
    record = record_from_development_report(report)
    settings = [(161,1/768,64),(321,1/1536,64),(641,1/3072,64),
                (1281,1/6144,64),(1281,1/6144,96),(2561,1/12288,96)]
    outputs = []
    weights = [cells/h*freq for cells,h,freq in settings]
    for cells, h, frequencies in settings:
        clock = perf_counter()
        try:
            row = replay_characteristic(record, cells=cells, max_step=h, frequencies=frequencies)
        except CharacteristicNumericalError as exc:
            row = dict(status='numerical_guard_failed', cells=cells, max_step=h,
                frequencies=frequencies, reason=str(exc), diagnostics=exc.diagnostics,
                scientific_acceptance=False)
        row['elapsed_seconds'] = perf_counter()-clock
        outputs.append(row)
        if progress is not None:
            progress(sum(weights[:len(outputs)]), sum(weights))
    fine = outputs[-1]
    comparisons = []
    tol = report['development_tolerances']
    fine_available = fine['status'] == 'completed'
    fine_edges = np.asarray(fine['x_edges']) if fine_available else None
    fine_mass = np.asarray(fine['final_posterior_mass']) if fine_available else None
    for row in (outputs[:-1] if fine_available else ()):
        if row['status'] != 'completed':
            continue
        mass = conservative_mass_transfer(row['x_edges'], row['final_posterior_mass'], fine_edges)
        errors = dict(total_log_likelihood=abs(row['log_likelihood']-fine['log_likelihood']),
            final_state_mean=abs(row['posterior_state_means'][-1]-fine['posterior_state_means'][-1]),
            final_mass_l1=float(np.abs(mass-fine_mass).sum()))
        comparisons.append(dict(cells=row['cells'], frequencies=row['frequencies'], errors=errors,
            development_tolerances_pass=all(errors[k] <= tol[k] for k in tol)))
    try:
        unshifted = replay_characteristic(record,cells=161,max_step=1/768,exponential_tilt=False,
                                         balanced_transform=False)
    except CharacteristicNumericalError as exc:
        unshifted = dict(status='numerical_guard_failed', reason=str(exc), diagnostics=exc.diagnostics)
    return dict(schema='bm1-interval-characteristic-control-v2',
        fixed_development_design=report['declared_design'], replays=outputs,
        plain_fourier_control=unshifted, fine_reference_available=fine_available,
        comparisons_with_fine=comparisons, development_tolerances=tol,
        frequency_64_96_score_change=(abs(outputs[3]['log_likelihood']-outputs[4]['log_likelihood'])
            if all(row['status']=='completed' for row in outputs[3:5]) else None),
        elapsed_seconds=perf_counter()-begun, historical_observation_values_read=0,
        numerical_acceptance=False, scientific_acceptance=False,
        interpretation='Synthetic numerical control only. Row-balanced log-amplitude transforms preserve tiny tails. '
            'Same frozen-left FV state scheme; exact reward phases. '
            'Periodic/Fourier truncation and positivity guards are checked, with heuristic roundoff allowance. '
            'Not a historical fitter, production backend, proof of continuous-state accuracy or H2/H3 acceptance.')


def run_characteristic_time_control(report, reference, *, progress=None):
    """Two declared time refinements at fixed state/frequency resolution.

    Reuses completed spatial controls; no optimizer or repeated spatial panel.
    The second replay tests time error independently of the saved spatial grid.
    """
    if (reference.get('schema') != 'bm1-interval-characteristic-control-v2'
            or reference.get('fixed_development_design') != report.get('declared_design')
            or reference.get('historical_observation_values_read') != 0
            or not reference.get('fine_reference_available')):
        raise ValueError('Use a completed balanced control on the identical synthetic design.')
    saved = reference['replays']
    baseline, spatial = saved[3], saved[5]
    for row, expected in ((baseline,(1281,1/6144,64)),(spatial,(2561,1/12288,96))):
        if (row.get('status') != 'completed' or not row.get('balanced_transform')
                or (row['cells'],row['max_step'],row['frequencies']) != expected):
            raise ValueError('Saved reference settings differ from the declared control.')
    clock = perf_counter()
    record = record_from_development_report(report)
    outputs = []
    for index, h in enumerate((1/12288,1/24576)):
        begun = perf_counter()
        try:
            row = replay_characteristic(record,cells=1281,max_step=h,frequencies=64)
        except CharacteristicNumericalError as exc:
            row = dict(status='numerical_guard_failed',cells=1281,max_step=h,frequencies=64,
                reason=str(exc),diagnostics=exc.diagnostics,scientific_acceptance=False)
        row['elapsed_seconds'] = perf_counter()-begun
        outputs.append(row)
        if progress is not None:
            progress((1.,3.)[index],3.)
    def compare(name, a, b):
        if a['status'] != 'completed' or b['status'] != 'completed':
            return dict(name=name,available=False)
        mass = conservative_mass_transfer(a['x_edges'],a['final_posterior_mass'],b['x_edges'])
        errors = dict(total_log_likelihood=abs(a['log_likelihood']-b['log_likelihood']),
            final_state_mean=abs(a['posterior_state_means'][-1]-b['posterior_state_means'][-1]),
            final_mass_l1=float(np.abs(mass-b['final_posterior_mass']).sum()))
        return dict(name=name,available=True,errors=errors,
            development_tolerances_pass=all(errors[k] <= report['development_tolerances'][k] for k in errors))
    return dict(schema='bm1-interval-characteristic-time-control-v1',
        fixed_development_design=report['declared_design'],replays=outputs,
        saved_baseline=baseline,saved_spatial_reference=spatial,
        comparisons=[compare('fixed_state_h6144_h12288',baseline,outputs[0]),
            compare('fixed_state_h12288_h24576',outputs[0],outputs[1]),
            compare('same_time_1281_2561',outputs[0],spatial)],
        development_tolerances=report['development_tolerances'],elapsed_seconds=perf_counter()-clock,
        historical_observation_values_read=0,numerical_acceptance=False,scientific_acceptance=False,
        interpretation='Two predeclared time controls on the same synthetic day, not a fit. '
            'Spatial comparison reuses the saved 2561-cell result at the same time step; '
            'its 96 frequencies differ from 64 (the saved frequency change is reported separately). '
            'Passing one spatial comparison does not establish time convergence or historical readiness.')


def run_characteristic_cn_control(report, *, progress=None):
    """Fixed three-replay development panel for positive second-order splitting.

    The saved inputs are piecewise constant within every hourly interval. Only
    under that condition do frozen reward / CN state / frozen reward steps
    provide the declared second-order time control. General time-dependent
    fields are not claimed second order and no historical backend is changed.
    """
    record = record_from_development_report(report)
    settings = [(1281,1/12288,64),(1281,1/24576,64),(1921,1/24576,64)]
    outputs,weights = [],[n/h*k for n,h,k in settings]
    clock = perf_counter()
    for n,h,k in settings:
        begun = perf_counter()
        try:
            row = replay_characteristic(record,cells=n,max_step=h,frequencies=k,
                state_scheme='positive_crank_nicolson')
        except CharacteristicNumericalError as exc:
            row = dict(status='numerical_guard_failed',cells=n,max_step=h,frequencies=k,
                reason=str(exc),diagnostics=exc.diagnostics,scientific_acceptance=False)
        row['elapsed_seconds'] = perf_counter()-begun
        outputs.append(row)
        if progress is not None:
            progress(sum(weights[:len(outputs)]),sum(weights))
    comparisons = []
    for name,i,j in (('fixed_state_time_refinement',0,1),('fixed_time_state_refinement',1,2),
                     ('combined_candidate_reference',0,2)):
        a,b = outputs[i],outputs[j]
        if a['status'] != 'completed' or b['status'] != 'completed':
            comparisons.append(dict(name=name,available=False))
            continue
        mass = conservative_mass_transfer(a['x_edges'],a['final_posterior_mass'],b['x_edges'])
        errors = dict(total_log_likelihood=abs(a['log_likelihood']-b['log_likelihood']),
            final_state_mean=abs(a['posterior_state_means'][-1]-b['posterior_state_means'][-1]),
            final_mass_l1=float(np.abs(mass-b['final_posterior_mass']).sum()))
        comparisons.append(dict(name=name,available=True,errors=errors,
            development_tolerances_pass=all(errors[k] <= report['development_tolerances'][k] for k in errors)))
    return dict(schema='bm1-interval-characteristic-cn-control-v1',
        fixed_development_design=report['declared_design'],replays=outputs,comparisons=comparisons,
        development_tolerances=report['development_tolerances'],elapsed_seconds=perf_counter()-clock,
        historical_observation_values_read=0,
        development_numerics_pass=all(c.get('development_tolerances_pass',False) for c in comparisons),
        numerical_acceptance=False,scientific_acceptance=False,
        interpretation='Declared three-replay synthetic CN panel with explicit-half positivity guard. '
            'State reference ratio is 1921/1281, not a doubling, and the finest replay is not exact truth. '
            'Time inputs are constant inside each hour; no general variable-input order claim. '
            'Not historical readiness, production adoption, model selection or H2/H3 acceptance.')

"""Bounded characteristic-function control; not a historical fit backend.

This experiment removes nodal accumulator deposition. Complex characteristic
columns evolve with the SAME real FV implicit state solve, and exact frozen
reward phases. A truncated periodic Gaussian Fourier series supplies evidence.
Positivity and finite-series guards abort unresolved calculations; no clipping,
cost substitution or production fallback is introduced.
"""

from time import perf_counter

import numpy as np
from scipy.special import betainc

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
    span = prediction['span']
    omega = prediction['frequencies']
    tilt, center = prediction['tilt'], prediction['center']
    shifted_value = value-center-sigma_y**2*tilt
    coefficients = np.exp(-.5*(sigma_y*omega)**2)*np.exp(-1j*omega*shifted_value)
    density = (prediction['cf'][:, 0].real+
               2.*np.real(prediction['cf'][:, 1:] @ coefficients[1:]))/span
    evidence = float(density.sum())
    if np.any(density < 0.) or not np.isfinite(density).all() or evidence <= 0.:
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
    tilted_mass = float(prediction['cf'][:, 0].real.sum())
    error_guard = (alias+tail+roundoff)*tilted_mass/evidence
    if error_guard > relative_guard:
        raise CharacteristicNumericalError('Fourier alias/tail/roundoff guard unresolved; refine the declared method.',
            relative_error_guard=float(error_guard), evidence=evidence, alias=float(alias),
            tail=float(tail), roundoff=float(roundoff))
    posterior = checks.probability_mass(density/evidence)
    log_prefactor = .5*sigma_y**2*tilt**2-tilt*(value-center)
    return dict(posterior_mass=posterior, log_evidence=float(np.log(evidence)+log_prefactor),
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


def replay_characteristic(record, *, cells, max_step, frequencies=64, span=4., exponential_tilt=True):
    """Development replay only, accepting the registered 24-hour interval case."""
    if not isinstance(record.observations, BM1IntervalObservationSeries) or len(record.observations.times) != 24:
        raise ValueError('Only the registered synthetic interval case enters this experiment.')
    model = bm_1(record.baseline, record.baseline, record.horizon, 2)
    model.fokker_planc_reference = finite_volume_1d_fitted(vectorized_inputs=True)
    solver = model.fokker_planc_reference
    edges, x, _ = solver.grid(cells)
    current = record.initial_mass_builder(edges)
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
            prediction = characteristic_prediction(solver, [start, end], current, drift, diffusion,
                reward, max_step=max_step, frequencies=frequencies, span=span, breakpoints=breaks,
                tilt=tilt, center=center, **options)
            maximum_error = max(maximum_error, prediction['maximum_abs_mass_error'])
            if seen:
                update = characteristic_gaussian_update(prediction, value, record.sigma_y)
                current = update['posterior_mass']
                scores.append(update['log_evidence'])
                guards.append(update['relative_series_guard'])
            else:
                current = prediction['predicted_mass']
                scores.append(None)
        except CharacteristicNumericalError as exc:
            exc.diagnostics.update(interval_start=float(start), interval_end=float(end),
                completed_intervals=len(scores), partial_log_evidence=scores)
            raise
        posterior.append(current.copy())
    return dict(status='completed', cells=cells, max_step=max_step, frequencies=frequencies, span=span,
        exponential_tilt=exponential_tilt,
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
        unshifted = replay_characteristic(record,cells=161,max_step=1/768,exponential_tilt=False)
    except CharacteristicNumericalError as exc:
        unshifted = dict(status='numerical_guard_failed', reason=str(exc), diagnostics=exc.diagnostics)
    return dict(schema='bm1-interval-characteristic-control-v1',
        fixed_development_design=report['declared_design'], replays=outputs,
        plain_fourier_control=unshifted, fine_reference_available=fine_available,
        comparisons_with_fine=comparisons, development_tolerances=tol,
        frequency_64_96_score_change=(abs(outputs[3]['log_likelihood']-outputs[4]['log_likelihood'])
            if all(row['status']=='completed' for row in outputs[3:5]) else None),
        elapsed_seconds=perf_counter()-begun, historical_observation_values_read=0,
        numerical_acceptance=False, scientific_acceptance=False,
        interpretation='Synthetic numerical control only. Same frozen-left FV state scheme; exact reward phases. '
            'Periodic/Fourier truncation and positivity guards are checked, with heuristic roundoff allowance. '
            'Not a historical fitter, production backend, proof of continuous-state accuracy or H2/H3 acceptance.')

"""Small independent interval controls; no historical acceptance study."""

import numpy as np
from time import perf_counter
from scipy.linalg import expm
from scipy.special import roots_hermite

from src.validation import estimation as checks


def finite_state_gaussian_reference(generator, rewards, prior, duration, value,
                                    sigma_y, *, order=96):
    """Independent finite-state Feynman--Kac/Fourier Gaussian calculation.

    Q uses column-probability convention. A=integral reward/duration. The
    characteristic terminal-state vector is exp((Q+iw*diag(r)/duration)*T)p.
    Fourier integration of the Gaussian observation uses Gauss--Hermite.
    Order refinement is required; this does not certify a continuous FV grid.
    """
    prior = checks.probability_mass(prior)
    generator = np.asarray(generator, dtype=float)
    rewards = checks.finite_vector(rewards, 'rewards')
    duration = checks.finite_scalar(duration, 'duration', positive=True)
    sigma_y = checks.finite_scalar(sigma_y, 'sigma_y', positive=True)
    value = checks.finite_scalar(value, 'value')
    order = checks.positive_integer(order, 'order', minimum=2)
    n = len(prior)
    if generator.shape != (n, n) or rewards.shape != (n,):
        raise ValueError('Generator, rewards and prior must use the same finite state space.')
    if (not np.isfinite(generator).all() or np.any(generator-np.diag(np.diag(generator)) < 0.)
            or np.max(np.abs(generator.sum(axis=0))) > 1e-10):
        raise ValueError('Supply a conservative column Markov generator.')
    nodes, weights = roots_hermite(order)
    density = np.zeros(n, dtype=complex)
    R = np.diag(rewards)
    for node, weight in zip(nodes, weights):
        frequency = np.sqrt(2.)*node/sigma_y
        characteristic = expm(generator*duration+1j*frequency*R) @ prior
        density += weight*np.exp(-1j*frequency*value)*characteristic
    density *= np.sqrt(2.)/(2.*np.pi*sigma_y)
    if np.max(np.abs(density.imag)) > 1e-10 or np.any(density.real <= 0.):
        raise ValueError('Independent Fourier quadrature is unresolved; refine its order.')
    likelihood = float(density.real.sum())
    return dict(log_evidence=float(np.log(likelihood)), posterior_mass=density.real/likelihood)


def run_interval_pilot(*, refine=False, progress=None):
    """Frozen cheap development panel: one synthetic day, no raw workbook.

    The measurements are constructed test inputs, not a truth/recovery study.
    Separate time/state/accumulator comparisons and a combined fine reference
    expose numerical sensitivity. No positive criterion releases H2 or H3.
    """
    import platform
    import scipy
    from scipy.special import betainc
    from src.method.bm_1_response import (CurveTrainingRecord, replay_curve_record,
        fit_shared_curve_pair)
    from src.method.bm_1_observation import conservative_mass_transfer, bayes_update, mass_to_log, gaussian_log_likelihood
    from src.params.bm_1 import BM1IntervalObservationSeries
    from src.params.bm_1_response import BM1PiecewiseResponse
    from src.solver.finite_volume import finite_volume_1d_fitted
    from src.solver.finite_volume_additive import propagate_additive
    from src.tools.input_generator.input_builder import input_builder
    from src.validation.bm_1.response import _score_record

    begun = perf_counter()
    linear = BM1PiecewiseResponse.linear()
    edges = np.arange(25)/24.
    B = input_builder.price_intervals((edges, .5+.04*np.sin(2*np.pi*(np.arange(24)+.5)/24.)))
    U = input_builder.price_intervals((edges, np.repeat([.2,.8,.35,.65], 6)))
    values = .48+.12*np.cos(2*np.pi*(np.arange(24)+.5)/24.)
    seen = np.ones(24, dtype=bool)
    seen[[7, 19]] = False
    values[~seen] = np.nan
    observations = BM1IntervalObservationSeries(edges[:-1], edges[1:], values, 'day', seen)
    def initial(grid):
        return np.diff(betainc(2., 3., grid))
    record = CurveTrainingRecord('synthetic-hourly-development', 'TRAIN', B, (0., 1.),
        (U, linear, linear, .15, 3., .8), .4, .08, initial, observations, edges[1:-1])
    settings = dict(cells=81, max_step=1/384, backend='fitted',
                    accumulator_spacing=1/128, max_accumulator_nodes=4096)
    panel = [('base', {}), ('time', dict(max_step=1/768)),
        ('accumulator', dict(accumulator_spacing=1/256)), ('state', dict(cells=161)),
        ('combined', dict(cells=161, max_step=1/768, accumulator_spacing=1/256)),
        ('fine', dict(cells=321, max_step=1/1536, accumulator_spacing=1/512))]
    if refine:
        panel = [('fine', dict(cells=321, max_step=1/1536, accumulator_spacing=1/512)),
            ('finer', dict(cells=641, max_step=1/3072, accumulator_spacing=1/1024)),
            ('finest', dict(cells=1281, max_step=1/6144, accumulator_spacing=1/2048))]
    weights = [dict(settings, **change)['cells']/settings['cells']
        *settings['max_step']/dict(settings, **change)['max_step']
        *settings['accumulator_spacing']/dict(settings, **change)['accumulator_spacing'] for _,change in panel]
    raw, results = [], []
    for name, changes in panel:
        numerics = dict(settings, **changes)
        clock = perf_counter()
        result = replay_curve_record(record, linear, linear, **numerics, keep_history=True)
        elapsed = perf_counter()-clock
        results.append(result)
        raw.append(dict(name=name, numerics=numerics, elapsed_seconds=elapsed,
            log_likelihood=result.total_log_likelihood,
            log_evidence=[float(s) if flag else None for s, flag in zip(result.log_evidence, seen)],
            x_edges=result.x_edges.tolist(), final_posterior_mass=result.posterior_mass[-1].tolist(),
            posterior_state_means=result.posterior_mean.tolist(),
            predicted_interval_means=[d['predicted_interval_mean'] for d in result.prediction_diagnostics],
            maximum_abs_mass_error=result.maximum_abs_mass_error,
            minimum_cell_mass=result.minimum_cell_mass,
            maximum_accumulator_nodes=max(d['accumulator_nodes'] for d in result.prediction_diagnostics),
            maximum_deposition_variance_bound=max(d['accumulator_deposition_variance_bound'] for d in result.prediction_diagnostics),
            maximum_accumulator_mass_above_one=max(d['accumulator_mass_outside_physical_range'] for d in result.prediction_diagnostics)))
        if progress is not None:
            progress(sum(weights[:len(raw)]), sum(weights))
    # These development tolerances are declared here before observing scores.
    tolerances = dict(total_log_likelihood=.05, final_state_mean=.002, final_mass_l1=.05)
    reference = results[-1]
    comparisons = []
    for name, result in zip([p[0] for p in panel[:-1]], results[:-1]):
        transferred = conservative_mass_transfer(result.x_edges, result.posterior_mass[-1], reference.x_edges)
        errors = dict(total_log_likelihood=abs(result.total_log_likelihood-reference.total_log_likelihood),
            final_state_mean=abs(float(result.posterior_mean[-1]-reference.posterior_mean[-1])),
            final_mass_l1=float(np.sum(np.abs(transferred-reference.posterior_mass[-1]))))
        comparisons.append(dict(name=name, errors=errors,
            development_tolerances_pass=all(errors[k] <= tolerances[k] for k in tolerances)))

    # Independent three-state reference, resolving its Fourier quadrature.
    solver = finite_volume_1d_fitted()
    prior = np.array([.2, .5, .3])
    rewards = np.array([.15, .5, .85])
    lo, diag, up, _ = solver.generator_tridiagonal(1/3, [.03]*2, [.3]*3)
    Q = np.diag(diag)+np.diag(lo, -1)+np.diag(up, 1)
    exact = finite_state_gaussian_reference(Q, rewards, prior, 1., .62, .12, order=96)
    exact_fine = finite_state_gaussian_reference(Q, rewards, prior, 1., .62, .12, order=160)
    finite_rows = []
    for resolution in (64, 256, 512):
        joint = propagate_additive(solver, [0., 1.], prior,
            lambda t,x: np.full(x.shape, .03), lambda t,x: np.full(x.shape, .3),
            lambda t,x: rewards, max_step=1/resolution, accumulator_spacing=1/resolution)
        update = bayes_update(mass_to_log(joint['joint_mass'].ravel()),
            np.broadcast_to(gaussian_log_likelihood(.62, joint['accumulator'], .12), joint['joint_mass'].shape).ravel())
        posterior = update.mass.reshape(joint['joint_mass'].shape).sum(axis=1)
        finite_rows.append(dict(resolution=resolution, log_evidence=update.log_evidence,
            posterior_mass=posterior.tolist(),
            score_error=abs(update.log_evidence-exact_fine['log_evidence']),
            posterior_max_error=float(np.max(np.abs(posterior-exact_fine['posterior_mass'])))))

    fits = []
    for pieces, budget in (() if refine else ((2, 6), (4, 8))):
        fit = fit_shared_curve_pair((record,), pieces=pieces, starts=np.zeros((1,2*(pieces-1))),
            regularization=.1, max_evaluations=budget, **settings)
        fits.append(dict(pieces=pieces, max_evaluations=budget,
            distinct_evaluations=fit['distinct_evaluations'], budget_exhausted=fit['budget_exhausted'],
            elapsed_seconds=fit['elapsed_seconds'], trace=fit['trace'],
            f=fit['f'].to_record(), g=fit['g'].to_record(),
            selected_replay=_score_record(fit['replay']),
            scientific_acceptance=False))
    return dict(schema='bm1-hourly-interval-development-v1',
        panel='bounded-fine-refinement' if refine else 'short-kernel-and-fit',
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        declared_design=dict(horizon_days=1., rows=24, scored_rows=int(seen.sum()),
            parameters=dict(C=.15, k=3., Xi=.8, sigma_x=.4, sigma_y=.08),
            prior='Beta(2,3)', curves='linear canonical for numerical panel',
            measurement_interpretation='Constructed normalized hourly averages, not simulated truth or historical data',
            observations=dict(starts=observations.starts.tolist(), ends=observations.ends.tolist(),
                values=[float(v) if flag else None for v,flag in zip(values,seen)], observed=seen.tolist()),
            baseline_values=list(map(float, B(edges[:-1]))), price_values=list(map(float,U(edges[:-1])))),
        numerical_replays=raw, development_tolerances=tolerances, comparisons_with_fine=comparisons,
        independent_finite_state_reference=dict(generator=Q.tolist(), rewards=rewards.tolist(), prior=prior.tolist(),
            value=.62, sigma_y=.12, duration=1., order=160,
            log_evidence=exact_fine['log_evidence'], posterior_mass=exact_fine['posterior_mass'].tolist(),
            quadrature_score_change=abs(exact['log_evidence']-exact_fine['log_evidence']), comparisons=finite_rows),
        bounded_shared_fits=fits, elapsed_seconds=perf_counter()-begun,
        historical_observation_values_read=0, historical_fit_run=False,
        numerical_acceptance=False, scientific_acceptance=False,
        interpretation='Implementation/runtime development only. Numerical comparisons and budget stops are explicit; '
            'no historical model selection, recovery, blind holdout, forecast or H2/H3 acceptance.')

"""One small independent control of the new F-only interval forecast adapter."""

import math

import numpy as np
from scipy.special import ndtr


def forecast_control():
    from src.method.bm_1_interval_forecast import (
        IntervalDistribution, interval_cdf_plan, native_interval_moments,
        native_interval_distribution, compare_interval_distributions,
        compare_interval_moments, forecast_open_intervals, interval_forecast_checkpoint)
    from src.validation.bm_1.interval_native import small_operator
    from src.validation.bm_1.interval_characteristic import CharacteristicNumericalError

    lo, di, up = np.array([.8, 0., 0.]), np.array([-.8, -.2, 0., 0.]), np.array([.2, 0., 0.])
    rewards = (np.array([.1, .9, .05, .95]), np.array([.8, .2, .95, .05]))
    logs = np.array([np.log(.2), np.log(.8), -np.inf, -900.])
    halves = [small_operator(lo, di, up, r, 2) for r in rewards]
    for op in halves:
        op['step_ids'] = np.array([0], dtype=np.int64)
    keys = ('ed', 'el', 'eu', 'lp', 'lbeta', 'lu', 'pivots', 'beta', 'upper', 'reward_half', 'fraction')
    joined = {k: np.concatenate([op[k] for op in halves]) for k in keys}
    joined.update(step_ids=np.array([0, 1], dtype=np.int64), x=halves[0]['x'])
    full = native_interval_moments(joined, logs, .04)
    first = native_interval_moments(halves[0], logs, .04)
    continued = native_interval_moments(halves[1], first['log_state'], .04,
        conditional_mean=first['conditional_mean'], conditional_variance=first['conditional_variance'])

    # Independent dense matrix/RAW moments, not the log-mixture implementation.
    Q = np.diag(di)+np.diag(lo, -1)+np.diag(up, 1)
    transition = np.linalg.solve(np.eye(4)-Q/4., np.eye(4)+Q/4.)
    raw = np.column_stack((np.exp(logs), np.zeros((4, 2))))
    for reward in rewards:
        a = reward/4.  # half of a half-hour, divided by original Delta=1
        raw[:, 2] += 2*a*raw[:, 1]+a*a*raw[:, 0]
        raw[:, 1] += a*raw[:, 0]
        raw = transition@raw
        raw[:, 2] += 2*a*raw[:, 1]+a*a*raw[:, 0]
        raw[:, 1] += a*raw[:, 0]
    actual = np.exp(full['log_state'])[:, None]*np.column_stack((
        np.ones(4), full['conditional_mean'],
        full['conditional_variance']+full['conditional_mean']**2))
    raw_error = float(np.max(np.abs(raw-actual)))
    continuation_error = max(float(np.max(np.abs(full[k][[0, 1, 3]]-continued[k][[0, 1, 3]])))
        for k in ('log_state', 'conditional_mean', 'conditional_variance'))
    mean_error = abs(full['interval_mean']-float(raw[:, 1].sum()))
    variance_error = abs(full['interval_variance']-float(raw[:, 2].sum()-raw[:, 1].sum()**2))
    tail_error = max(abs(full['log_state'][3]+900.), abs(full['conditional_mean'][3]-.5),
                     abs(full['conditional_variance'][3]))
    assert max(raw_error, continuation_error, mean_error, variance_error, tail_error) <= 2e-12
    assert np.isneginf(full['log_state'][2])
    compare_interval_moments(full, continued)
    try:
        compare_interval_moments(full, dict(continued, interval_variance=continued['interval_variance']+.003))
    except CharacteristicNumericalError:
        pass
    else:
        raise AssertionError('Moment precision failure was admitted.')

    # Transform independently evaluated through dense positive matrices.
    plan = interval_cdf_plan(.04)
    observed = native_interval_distribution(joined, logs, plan)
    omega = 2*np.pi*np.arange(plan.frequencies+1)/plan.span
    dense = np.exp(logs)[:, None]*np.ones_like(omega, dtype=complex)
    for reward in rewards:
        phase = np.exp(1j*reward[:, None]*omega/4.)
        dense = phase*(transition@(phase*dense))
    exact_cf = dense.sum(axis=0)*np.exp(-.5*(plan.sigma_y*omega)**2)
    cf_error = float(np.max(np.abs(observed['distribution'].characteristic-exact_cf)))
    assert cf_error <= 2e-12

    # Narrow noise forces N>64. Gaussian CDF/quantiles are analytic controls.
    plan = interval_cdf_plan(.01)
    assert plan.frequencies > 64
    beta = 2*math.pi**2*plan.sigma_y**2/plan.span**2
    q = beta*(plan.frequencies-1)**2
    assert math.exp(-q)/(math.pi*q) > 1e-10
    zero = np.zeros(3)
    op = small_operator(zero[:-1], zero, zero[:-1], np.full(3, .35), 4)
    tiny_logs = np.array([0., -900., -np.inf])
    gaussian = native_interval_distribution(op, tiny_logs, plan)
    points = np.r_[np.linspace(.3, .4, 81), plan.left-1., plan.right+1.]
    gaussian_error = float(np.max(np.abs(gaussian['distribution'].cdf(points)-ndtr((points-.35)/.01))))
    assert gaussian_error <= 2e-10
    exact_quantiles = [.35-.01*1.6448536269514722, .35, .35+.01*1.6448536269514722]
    quantile_error = max(abs(r['value']-q) for r, q in zip(gaussian['quantiles'], exact_quantiles))
    assert quantile_error <= .0001
    assert gaussian['log_state'][1] == -900. and np.isneginf(gaussian['log_state'][2])
    comparison = compare_interval_distributions(gaussian['distribution'], gaussian['distribution'], observed_values=[.35])
    shifted = gaussian['distribution'].characteristic*np.exp(1j*.01*2*np.pi*np.arange(plan.frequencies+1)/plan.span)
    try:
        compare_interval_distributions(gaussian['distribution'], IntervalDistribution(plan, shifted))
    except CharacteristicNumericalError:
        pass
    else:
        raise AssertionError('Forecast precision failure was admitted.')

    # Two distinct hourly rewards prove reset/ownership: hour 2 is .8, not .5.
    ops = [small_operator(zero[:-1], zero, zero[:-1], np.full(3, r), 2) for r in (.2, .8)]
    checkpoints = []
    saved_prior = tiny_logs.copy()
    open_result = forecast_open_intervals(ops, tiny_logs, interval_cdf_plan(.04),
        interval_count=2, horizons=(1, 2), checkpoint=lambda row, state, full: checkpoints.append((row['hour'], state)))
    assert np.array_equal(tiny_logs, saved_prior)
    assert max(abs(r['interval_mean']-expected) for r, expected in zip(open_result['rows'], (.2, .8))) <= 2e-12
    assert open_result['assimilated_observations'] == 0 and len(checkpoints) == 2
    assert open_result['log_state'][1] == -900.
    resumed = forecast_open_intervals(ops[1:], checkpoints[0][1], interval_cdf_plan(.04),
        interval_count=2, horizons=(1, 2), completed_intervals=1)
    assert np.array_equal(resumed['log_state'], open_result['log_state'])
    assert resumed['rows'] == open_result['rows'][1:]
    # Existing array checkpoint I/O must retain true zero and sub-float logs.
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from src.tools.bm_1_reports import _atomic_report, _read_likelihood_report
    payload = interval_forecast_checkpoint(open_result['rows'][-1], open_result['log_state'],
                                          open_result['distributions'][2], 'a'*64)
    with TemporaryDirectory(prefix='h2-forecast-control-') as directory:
        path = Path(directory)/'hour.npz'
        _atomic_report(path, payload)
        recovered = _read_likelihood_report(path)[0]
    assert np.array_equal(recovered['log_state'], open_result['log_state'])
    assert np.array_equal(recovered['distribution']['characteristic_real'],
                          open_result['distributions'][2]['distribution'].characteristic.real)
    assert np.array_equal(recovered['distribution']['characteristic_imag'],
                          open_result['distributions'][2]['distribution'].characteristic.imag)
    try:
        forecast_open_intervals(ops, tiny_logs, interval_cdf_plan(.04),
            interval_count=2, horizons=(1, 2), should_stop=lambda: True)
    except TimeoutError:
        pass
    else:
        raise AssertionError('Open forecast ignored its caller deadline.')
    for bad_logs in ([0., 0., -np.inf], [np.nan, -np.inf, -np.inf]):
        try:
            native_interval_moments(op, bad_logs, .04)
        except CharacteristicNumericalError:
            pass
        else:
            raise AssertionError('Invalid prior was admitted.')
    return dict(passed=True, dense_raw_moment_error=raw_error,
        within_interval_continuation_error=continuation_error,
        marginal_mean_error=mean_error, marginal_variance_error=variance_error,
        finite_log_tail_error=tail_error, true_zero_preserved=True,
        independent_dense_characteristic_error=cf_error,
        gaussian_cdf_error=gaussian_error, gaussian_quantile_error=quantile_error,
        narrow_noise_frequencies=plan.frequencies, base_fine_comparison=comparison,
        hourly_accumulator_reset=True, prior_ownership_preserved=True,
        boundary_resume_exact=True, checkpoint_roundtrip_exact=True,
        precision_failure_rejected=True, deadline_stop_rejected=True,
        invalid_prior_rejected=True, historical_observation_values_read=0)

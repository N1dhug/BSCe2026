"""One focused independent algebra/control check for the new H2 recurrence."""

import numpy as np
from scipy.special import logsumexp, roots_legendre


def small_operator(lo, di, up, rewards, steps):
    from src.validation.bm_1.interval_characteristic import _positive_implicit_factors
    s = .5/steps
    pivots, beta = _positive_implicit_factors(-s*lo, 1.-s*di, -s*up)
    with np.errstate(divide='ignore'):
        rows = [np.log(v) for v in (1.+s*di, s*lo, s*up, pivots, beta, s*up)]
    keys = ('ed', 'el', 'eu', 'lp', 'lbeta', 'lu')
    op = {k: v[None, :] for k, v in zip(keys, rows)}
    op.update(pivots=pivots[None, :], beta=beta[None, :], upper=(s*up)[None, :],
        reward_half=(rewards/(2*steps))[None, :], fraction=np.array([1./(2*steps)]),
        step_ids=np.zeros(steps, dtype=np.int64), x=np.linspace(0., 1., len(di)))
    return op


def algebra_control():
    from src.method.bm_1_interval_native import native_prediction, native_terminal_update
    from src.method.bm_1_interval_likelihood import PRECISE_SETTINGS
    from src.params.bm_1 import gaussian_log_likelihood
    # Asymmetric generator, two disconnected cells, one true zero and one
    # finite sub-float log mass. Dense solve is independent of log-Thomas.
    lo, di, up = np.array([.8, 0., 0.]), np.array([-.8, -.2, 0., 0.]), np.array([.2, 0., 0.])
    reward = np.array([.1, .9, .05, .95])
    logs = np.array([np.log(.2), np.log(.8), -np.inf, -900.])
    op = small_operator(lo, di, up, reward, 1)
    prediction = native_prediction(op, logs, frequencies=2, span=2., tilt=5., center=.5)
    Q = np.diag(di)+np.diag(lo, -1)+np.diag(up, 1)
    omega = prediction['frequencies']
    phase = np.exp((5.+1j*omega[None, :])*(reward[:, None]-.5)/2.)
    expected = phase*np.linalg.solve(np.eye(4)-Q/2., (np.eye(4)+Q/2.)@(np.exp(logs)[:, None]*phase))
    actual = np.exp(prediction['log_amplitude'])[:, None]*prediction['cf']
    dense_error = float(np.max(np.abs(expected-actual)))
    tail_error = abs(float(prediction['log_amplitude'][3])-(-900.+5*(.95-.5)))
    zero_preserved = bool(np.isneginf(prediction['log_amplitude'][2]))
    # Analytic Gaussian law, including revival of a finite tail through Bayes.
    r = np.array([.05, .5, .95]); z = np.zeros(3)
    op = small_operator(z[:-1], z, z[:-1], r, 8)
    logs = np.array([0., -900., -950.]); gaussian = []
    exact = logs.copy()
    for value in (1.5, 1.5):
        result = native_terminal_update(op, logs, value, .04, PRECISE_SETTINGS)
        exact += gaussian_log_likelihood(value, r, .04)
        score = float(logsumexp(exact)); exact -= score
        gaussian.append(dict(score_error=abs(result['log_evidence']-score),
            log_posterior_error=float(np.max(np.abs(result['log_posterior_mass']-exact)))))
        logs = result['log_posterior_mass']
    # Positive one-jump density integrated independently, same declared 256
    # substeps as the old witness. No old FV reference is rerun.
    lo, di, up = np.array([.8, 0.]), np.array([-.8, 0., 0.]), np.zeros(2)
    r = np.array([.1, .9, .9])
    op = small_operator(lo, di, up, r, 256)
    result = native_terminal_update(op, np.array([0., -np.inf, -np.inf]), .55, .04, PRECISE_SETTINGS)
    nodes, weights = roots_legendre(160); t = (nodes+1)/2.
    reference = np.array([-.8+gaussian_log_likelihood(.55, r[:1], .04)[0],
        logsumexp(np.log(weights/2.*.8)-.8*t+gaussian_log_likelihood(.55, .9-.8*t, .04))])
    joint = result['log_posterior_mass']+result['log_evidence']
    jump_error = float(np.max(np.abs(joint[:2]-reference)))
    passed = (dense_error <= 1e-12 and tail_error <= 1e-10 and zero_preserved
        and all(max(r.values()) <= 2e-8 for r in gaussian) and jump_error <= 2e-4
        and np.isneginf(joint[2]))
    report = dict(dense_transform_error=dense_error, finite_tail_error=tail_error,
        true_zero_preserved=zero_preserved, gaussian_controls=gaussian,
        one_jump_log_joint_error=jump_error, passed=bool(passed))
    if not passed:
        raise ValueError('Native independent algebra/control check failed: '+str(report))
    return report

"""Small independent references for the F4-H2 terminal-contour prototype."""

import unittest

import numpy as np
from scipy.special import logsumexp, roots_legendre

from src.params.bm_1 import gaussian_log_likelihood
from src.solver.finite_volume import finite_volume_1d_fitted
from src.validation.bm_1.interval_characteristic import CharacteristicNumericalError
from src.validation.bm_1.interval_terminal_tilt import terminal_tilt_update


class TerminalTiltTests(unittest.TestCase):
    def test_exact_gaussian_tails_and_recursive_revival_below_float_range(self):
        solver = finite_volume_1d_fitted()
        zero = lambda t, x: np.zeros(x.shape)
        rewards = np.array([.05, .5, .95])
        for initial, values in ((np.log([.2, .5, .3]), [.6]),
                                (np.array([0., -900., -950.]), [1.5, 1.5]),
                                (np.array([0., -np.inf, -np.inf]), [.6])):
            logs, exact = initial.copy(), initial.copy()
            for value in values:
                result = terminal_tilt_update(solver, [0., 1.], np.exp(logs), zero, zero,
                    lambda t, x: rewards, value, .04, log_prior=logs, max_step=1/8)
                exact += gaussian_log_likelihood(value, rewards, .04)
                expected_score = logsumexp(exact)
                exact -= expected_score
                self.assertAlmostEqual(result['log_evidence'], expected_score, delta=2e-8)
                np.testing.assert_allclose(result['log_posterior_mass'], exact, rtol=0., atol=2e-8)
                logs = result['log_posterior_mass']
                self.assertLessEqual(result['maximum_relative_conditional_guard'], 1e-8)
            if len(values) == 2:
                self.assertGreater(result['posterior_mass'][2], .99)

    def test_absorbing_jump_matches_positive_path_integral_and_time_refines(self):
        # A single jump time has positive density rate*exp(-rate*t).
        # This quadrature has no Fourier inversion and no FV time stepping.
        solver, rate, value, sigma = finite_volume_1d_fitted(), .8, .55, .04
        rewards = np.array([.1, .9, .9])  # Third state is disconnected.
        references = []
        for order in (96, 160):
            nodes, weights = roots_legendre(order)
            times = (nodes + 1.) / 2.
            average = rewards[1] + (rewards[0] - rewards[1]) * times
            jumped = logsumexp(np.log(weights / 2. * rate) - rate * times
                              + gaussian_log_likelihood(value, average, sigma))
            references.append(np.array([-rate + gaussian_log_likelihood(value, rewards[:1], sigma)[0], jumped]))
        np.testing.assert_allclose(references[0], references[1], rtol=0., atol=1e-12)
        errors = []
        for resolution in (128, 256):
            result = terminal_tilt_update(solver, [0., 1.], [1., 0., 0.],
                lambda t, x: np.where(x < .5, rate / 3., 0.), lambda t, x: np.zeros(x.shape),
                lambda t, x: rewards, value, sigma, max_step=1/resolution)
            errors.append(float(np.max(np.abs(result['log_joint_mass'][:2] - references[-1]))))
            self.assertTrue(np.isneginf(result['log_joint_mass'][2]))
        self.assertLess(errors[-1], 2e-4)
        self.assertGreater(errors[0] / errors[1], 3.5)

    def test_insufficient_bank_and_frequency_budget_fail_without_prior_mutation(self):
        solver = finite_volume_1d_fitted()
        prior = np.array([.2, .5, .3])
        original = prior.copy()
        zero = lambda t, x: np.zeros(x.shape)
        for kwargs in (dict(centers=(.5,)), dict(frequencies=1)):
            with self.assertRaisesRegex(CharacteristicNumericalError, 'bank exhausted'):
                terminal_tilt_update(solver, [0., 1.], prior, zero, zero,
                    lambda t, x: np.array([.05, .5, .95]), .6, .04, max_step=1/8, **kwargs)
        np.testing.assert_array_equal(prior, original)


if __name__ == '__main__':
    unittest.main()

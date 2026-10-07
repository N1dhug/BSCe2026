"""One contract check against an independent affine bounded LS solution."""
from types import SimpleNamespace
from time import perf_counter
import unittest

import numpy as np

from src.method.bm_1_g7_deterministic import DeterministicResidualControl


class DeterministicControlTests(unittest.TestCase):
    def test_weighted_bounds_identity_and_true_probe_budget(self):
        class AffineEvaluator:
            terminal_weight = price_smoothing_weight = 0.
            widths = np.array([.1,.9])
            target = np.array([-.2,.7])

            def validate_prices(self, prices):
                prices = np.asarray(prices, dtype=float)
                if prices.shape != (2,) or np.any((prices<0)|(prices>1)):
                    raise ValueError('Two bounded prices required.')
                return prices

            def evaluate(self, prices):
                value = np.sum(self.widths*(prices-self.target)**2)
                return SimpleNamespace(total=value, prices=prices.copy(),
                    state=np.array([.5,.5]),
                    objective=SimpleNamespace(expected_demand=prices.copy()))

        control = DeterministicResidualControl(AffineEvaluator(), max_forwards=100,
            deadline=perf_counter()+30)
        fit = control.search([.5,.5], max_new_forwards=80)
        self.assertTrue(fit['success'])
        # TRF uses a bound-scaled gradient: at gtol=1e-7 a few e-6
        # from the active zero bound is legitimate. The analytic lower
        # bound remains .004; verify its objective gap independently.
        np.testing.assert_allclose(fit['prices'], [0.,.7], atol=1e-5)
        self.assertGreaterEqual(fit['objective'], .004-1e-12)
        self.assertLess(fit['objective']-.004, 2e-7)
        self.assertLess(fit['factor_two_error'], 1e-12)
        self.assertGreater(control.forward_calls, fit['nfev'])
        before = control.forward_calls
        control.residual(fit['prices'])
        self.assertEqual(before, control.forward_calls)
        small = DeterministicResidualControl(AffineEvaluator(), max_forwards=2,
            deadline=perf_counter()+30)
        stopped = small.search([.5,.5], max_new_forwards=2)
        self.assertFalse(stopped['success'])
        self.assertEqual(stopped['status'], 'budget_stop')
        self.assertEqual(small.forward_calls, 2)
        self.assertIsNotNone(stopped['prices'])


if __name__ == '__main__':
    unittest.main()

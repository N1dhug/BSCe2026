"""One independent affine WM contract check; no optimizer or historical fit."""
import unittest
import numpy as np

from src.method.bm_1_g7_deterministic import MatchedDeterministicControl


class MatchedControlTests(unittest.TestCase):
    def test_signed_price_day_clock_affinity_and_raw_cost(self):
        packet = dict(schema='g7-matched-fixed-target-v1',
                      baseline_time_hours=[0., 1., 2.], baseline=[.5]*3,
                      target=[.5]*3, price_edges_hours=[0., 1., 2.],
                      scales=dict(demand_min=4., demand_scale=3.,
                                  price_min=-500., price_scale=6500.),
                      models={'wm': dict(family='WM4', r=[-1., -.5, -3.],
                                         state_origin=[.2, .1])})
        control = MatchedDeterministicControl(packet)
        high, low = np.array([.7, .2]), np.array([.3, .8])
        a = control.evaluate('wm', high, 12)
        b = control.evaluate('wm', low, 12)
        c = control.evaluate('wm', (high+low)/2, 12)
        # Independently apply the first ODE derivative for one physical-hour step.
        day_step = 1/(24*12)
        expected_first = [.2+day_step*.1,
                          .1+day_step*(-1*.2-.5*.4-3*.1)]
        np.testing.assert_allclose(a['state'][1], expected_first, atol=1e-15, rtol=0)
        np.testing.assert_allclose(c['demand'], (a['demand']+b['demand'])/2,
                                   atol=1e-15, rtol=0)
        self.assertAlmostEqual(a['objective_raw'], 9*a['objective_normalized'])
        self.assertAlmostEqual(a['objective_normalized'],
                               np.sum((a['demand']-.5)**2)/12)
        np.testing.assert_array_equal(control.hourly_prices(high), high)
        with self.assertRaises(ValueError):
            control.evaluate('wm', [1.01, .5], 12)
        with self.assertRaises(ValueError):
            control.evaluate('wm', [.5], 12)


if __name__ == '__main__':
    unittest.main()

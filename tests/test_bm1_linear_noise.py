"""Focused G6a controls: analytical variance, derivatives/operator, calibration."""
import unittest
import numpy as np
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_linear_noise import unit_variance, demand_state_derivative, fit_constant_sigma, interval_metrics


class LinearNoiseTests(unittest.TestCase):
    def test_equilibrium_variance_reset_and_hour_day_scaling(self):
        inp=BM1ResponseInputs(np.arange(49.),np.full(49,.5),np.full(48,.5),steps_per_hour=12)
        p=[4.,2.,.4,.5]
        out=unit_variance(inp,p,reset_hour=24)
        elapsed=np.maximum(0,np.arange(49.)-24)
        rate=2*.5*p[1]*p[2]/p[0]
        expected=(1/16)/rate*(-np.expm1(-rate*elapsed))
        np.testing.assert_allclose(out['unit_state_variance'],expected,rtol=3e-14,atol=1e-15)
        np.testing.assert_allclose(out['unit_demand_variance'],(.5*p[1]*p[2])**2*expected[:-1],rtol=3e-14,atol=1e-15)
        sigma=.2
        C_days=p[0]/24; sigma_days=np.sqrt(24)*sigma
        day_rate=2*.5*p[1]*p[2]/C_days
        day_variance=(sigma_days**2/16)/day_rate*(-np.expm1(-day_rate*elapsed/24))
        np.testing.assert_allclose(sigma**2*expected,day_variance,rtol=2e-15,atol=1e-15)
        np.testing.assert_array_equal(out['demand'],inp.solve(p)['demand'])
        np.testing.assert_array_equal(0*out['unit_state_variance'],np.zeros(49))
        self.assertTrue(np.all(out['unit_state_variance']>=0))

    def test_derivative_and_reject_interval_operator(self):
        k=2.;xi=.4;b=.3
        for x in (.2,.8):
            def demand(v):
                d=np.tanh(.5*k*(1-2*v))
                return b+xi*d*(1-b if d>=0 else b)
            numeric=(demand(x+1e-6)-demand(x-1e-6))/2e-6
            self.assertAlmostEqual(float(demand_state_derivative(x,0,b,k,xi)),numeric,places=9)
        inp=BM1ResponseInputs([0,1],[.5,.5],[.5],observation_contract='hourly-mean')
        with self.assertRaisesRegex(ValueError,'Interval variance'):
            unit_variance(inp,[4,2,.4,.5])

    def test_exact_nonnegative_ls_and_interval_score(self):
        w=np.array([.2,.4,.8]);s=3*w
        fit=fit_constant_sigma(w,s)
        self.assertAlmostEqual(fit['sigma_per_sqrt_hour'],3)
        self.assertLess(fit['sigma_rmse'],1e-15)
        self.assertEqual(fit_constant_sigma(np.zeros(3),s)['status'],'unidentifiable')
        m=interval_metrics([0,3],[0,0],[1,1])
        self.assertEqual(m['coverage'],.5)
        self.assertGreater(m['mean_interval_score'],m['mean_width'])

"""Fundamental G solver, energy balance and independent ODE reference checks."""
import unittest
import numpy as np
from src.model.bm_1 import bm_1
from src.method.bm_1_deterministic_fit import BM1DeterministicInputs,exact_linear_interval

def inputs(hours=48, steps=12, contract='hourly-point'):
    time = np.arange(hours*60+1)/60
    baseline = .45+.12*np.sin(time/4)+.03*np.cos(time/1.3)
    price = np.random.default_rng(421).uniform(.05,.95,hours)
    return BM1DeterministicInputs(time,baseline,price,steps_per_hour=steps,observation_contract=contract)
class DeterministicFitTests(unittest.TestCase):
    def test_price_switches_and_linear_baseline_against_adaptive_ode(self):
        data = inputs(hours=8)
        p = np.array([2.,4.,.8,.2])
        B = lambda t:np.interp(t,data.time,data.baseline)
        def U(t):
            return data.prices[np.minimum(np.floor(np.asarray(t)).astype(int),7)]
        model = bm_1(B,lambda t:0.,(0.,8.),9)
        reference = model.deterministic_solver([p[3]],(U,lambda x:1-2*x,lambda u:1-2*u,*p[:3]),
                                               t_eval=np.arange(9.),breakpoints=np.arange(1.,8.),rtol=1e-10,atol=1e-12)
        actual = data.solve(p)
        np.testing.assert_allclose(actual['state'],reference[1],atol=2e-6,rtol=0)
        np.testing.assert_allclose(actual['point_demand'],reference[-1][:-1],atol=2e-6,rtol=0)

    def test_exact_constant_interval_and_energy(self):
        data = BM1DeterministicInputs(np.arange(5.),np.full(5,.4),np.full(4,.7),observation_contract='hourly-mean')
        actual = data.solve((6.,2.,.7,.1))
        x = .1
        for i in range(4):
            x, demand = exact_linear_interval(x,.4,.7,1.,6.,2.,.7)
            self.assertAlmostEqual(actual['state'][i+1],x,places=8)
            self.assertAlmostEqual(actual['demand'][i],demand,places=8)
        np.testing.assert_allclose(actual['mean_demand']-.4,6*np.diff(actual['state']),atol=1e-14)
        self.assertGreater(np.max(np.abs(actual['point_demand']-actual['mean_demand'])),1e-4)

    def test_no_flexibility_and_bounds(self):
        data = inputs()
        out = data.solve((12.,2.,0.,0.))
        np.testing.assert_array_equal(out['state'],0.)
        np.testing.assert_allclose(out['demand'],np.interp(np.arange(48),data.time,data.baseline))
        for x0 in (0.,1.):
            x = data.solve((1.,10.,1.,x0))['state']
            self.assertTrue(np.all((x>=0)&(x<=1)))
        with self.assertRaises(ValueError):
            inputs(8,steps=1).solve((1.,10.,1.,.5))

if __name__=='__main__':unittest.main()

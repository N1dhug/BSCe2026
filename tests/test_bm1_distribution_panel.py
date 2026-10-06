"""Two focused interface checks; the registered panel checks numerical fidelity."""
import unittest
import numpy as np
from src.method.bm_1_distribution_panel import build_problem, make_support, uniform_grid_indices
from src.control.bm_1 import BM1ControlNumerics
from src.method.euler_maruyama import euler_maruyama


class TestDistributionPanel(unittest.TestCase):
    def test_hours_days_drift_diffusion_and_integral(self):
        h=build_problem(time_unit='hours').build_evaluator(BM1ControlNumerics(x_steps=17),fast=False)
        d=build_problem(time_unit='days').build_evaluator(BM1ControlNumerics(x_steps=17),fast=False)
        x=np.array([.2,.5,.8]);u=np.array([.3,.7,.3,.7])
        hp=h.parameters_for_prices(u);dp=d.parameters_for_prices(u)
        for time in (0.,5.,7.,17.,23.):
            np.testing.assert_allclose(d.model._drift(time/24,x,dp),24*h.model._drift(time,x,hp),atol=2e-14)
        np.testing.assert_allclose(d.model.diffusion_coefficient(x,.2*np.sqrt(24)),
                                   np.sqrt(24)*h.model.diffusion_coefficient(x,.2))
        self.assertAlmostEqual(h.widths.sum(),24*d.widths.sum(),places=12)

    def test_nested_grid_and_summed_brownian_increments(self):
        fine=np.arange(2305)/96
        actual,index=euler_maruyama.build_time_grid(fine,[6.,12.,18.],np.inf)
        np.testing.assert_array_equal(actual,fine)
        np.testing.assert_array_equal(actual[index],fine)
        increments=euler_maruyama.brownian_increments(fine,3,seed=1)
        coarse=euler_maruyama.coarsen_brownian_increments(increments,2)
        np.testing.assert_allclose(coarse.sum(axis=1),increments.sum(axis=1),atol=1e-14)
        # Common quadrature midpoints and all price breaks lie on these grids.
        midpoint=make_support(17,1/24)[0].midpoints
        np.testing.assert_allclose(fine[uniform_grid_indices(fine,midpoint)],midpoint,atol=1e-12)
        with self.assertRaises(ValueError):
            uniform_grid_indices(fine,[.001])


if __name__=='__main__': unittest.main()

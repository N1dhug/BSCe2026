"""Independent equilibrium and conservative-generator checks for fitted flux."""
import unittest
import numpy as np
from scipy.special import ndtr

from src.solver.finite_volume import finite_volume_1d, finite_volume_1d_fitted


class FittedFluxTests(unittest.TestCase):
    def test_constant_coefficient_detailed_balance_and_dense_evolution(self):
        solver = finite_volume_1d_fitted()
        for velocity in (-3., 0., 3.):
            edges, centers, dx = solver.grid(31)
            drift = np.full(30, velocity)
            diffusion = np.full(31, .7)
            lower, diagonal, upper, diagnostics = solver.generator_tridiagonal(dx, drift, diffusion)
            Q = np.diag(diagonal)+np.diag(lower, -1)+np.diag(upper, 1)
            equilibrium = np.exp(velocity/(.5*.7**2)*(centers-centers.max()))
            equilibrium /= equilibrium.sum()
            np.testing.assert_allclose(Q @ equilibrium, 0., atol=2e-13, rtol=0)
            # Cancellation scales with the generator rates, not unit mass.
            roundoff = 4*np.finfo(float).eps*np.max(np.sum(abs(Q), axis=0))
            np.testing.assert_allclose(Q.sum(axis=0), 0., atol=roundoff, rtol=0)
            self.assertGreaterEqual(diagnostics['minimum_off_diagonal_rate'], 0.)
            initial = np.zeros(31); initial[15] = 1.
            got = solver.solve([0., .17], initial, lambda t, x: velocity+0*x,
                               lambda t, x: .7+0*x, max_step=.037, coefficient_key=lambda t: 0)
            want = initial.copy()
            times = got['diagnostics']['integration_t']
            for step in np.diff(times):
                want = np.linalg.solve(np.eye(31)-step*Q, want)
            np.testing.assert_allclose(got['cell_mass'][-1], want, atol=2e-15, rtol=0)
            self.assertFalse(got['diagnostics']['used_clipping'])
            self.assertFalse(got['diagnostics']['used_renormalization'])

    def test_variable_diffusion_equilibrium_converges_without_missing_D_derivative(self):
        solver = finite_volume_1d_fitted()
        errors = []
        for cells in (31, 61, 121):
            edges, centers, dx = solver.grid(cells)
            # p proportional exp(-2*(x-.5)^2), D=.1+.2*x^2,
            # a=D'+D*p'/p gives exactly zero continuum flux everywhere.
            initial = np.diff(ndtr(2*(edges-.5)))
            initial /= initial.sum()
            def D(x): return .1+.2*x*x
            def drift(t, x): return .4*x-4*D(x)*(x-.5)
            def diffusion(t, x): return np.sqrt(2*D(x))
            got = solver.solve([0., 2.], initial, drift, diffusion, max_step=.025,
                               coefficient_key=lambda t: 0)
            errors.append(np.sum(abs(got['cell_mass'][-1]-initial)))
        self.assertGreater(errors[0]/errors[1], 3.5)
        self.assertGreater(errors[1]/errors[2], 3.5)
        self.assertLess(errors[-1], 1e-4)

    def test_zero_diffusion_upwind_limit_and_extreme_peclet_remain_finite(self):
        solver = finite_volume_1d_fitted()
        drift = np.linspace(-2, 2, 16)
        zero = np.zeros(17)
        got = solver.generator_tridiagonal(1/17, drift, zero)
        want = finite_volume_1d.generator_tridiagonal(1/17, drift, zero)
        for a, b in zip(got[:3], want[:3]):
            np.testing.assert_array_equal(a, b)
        for value in (-1000., -50., -1e-8, 0., 1e-8, 50., 1000.):
            result = solver._bernoulli(np.array([value]))[0]
            self.assertTrue(np.isfinite(result))
            self.assertGreaterEqual(result, 0.)
        for diffusion in (np.ones(17)*1e-4, np.ones(17)*10):
            lower, diagonal, upper, _ = solver.generator_tridiagonal(1/17, drift, diffusion)
            self.assertTrue(np.all(np.isfinite(diagonal)))
            self.assertTrue(np.all(lower >= 0) and np.all(upper >= 0))


if __name__ == '__main__':
    unittest.main()

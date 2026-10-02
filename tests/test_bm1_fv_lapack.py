"""Opt-in LAPACK changes only the tridiagonal linear solver."""
import unittest
from unittest.mock import patch
import numpy as np
from src.solver.finite_volume import finite_volume_1d, finite_volume_1d_lapack


class FVLapackTests(unittest.TestCase):
    def test_m_matrix_solutions_preserve_inputs_and_match_reference(self):
        rng=np.random.default_rng(9012)
        for cells in (3,17,401):
            lower=rng.uniform(0,10,cells-1);upper=rng.uniform(0,10,cells-1)
            diagonal=np.ones(cells)+np.r_[lower,0.]+np.r_[0.,upper]
            args=(-lower,diagonal,-upper,rng.dirichlet(np.ones(cells)))
            originals=[v.copy() for v in args]
            want=finite_volume_1d._solve_tridiagonal(*args)
            got=finite_volume_1d_lapack._solve_tridiagonal(*args)
            np.testing.assert_allclose(got,want,atol=1e-14,rtol=0)
            for a,b in zip(args,originals):np.testing.assert_array_equal(a,b)

    def test_full_time_varying_solver_and_breakpoints_match_reference(self):
        cells=41;mass=np.zeros(cells);mass[20]=1.
        times=np.array([0.,.07,.18,.25])
        def drift(t,x):return (.4 if t<.1 else -.2)+.1*np.sin(7*t)-.3*x
        def diffusion(t,x):return (.5+.1*t)*x*(1-x)
        args=(times,mass,drift,diffusion)
        options=dict(breakpoints=[.1],max_step=.007)
        want=finite_volume_1d().solve(*args,**options)
        got=finite_volume_1d_lapack().solve(*args,**options)
        for k in ('cell_mass','density','mass','t','x_grid','x_edges'):
            np.testing.assert_allclose(got[k],want[k],atol=1e-12,rtol=0)
        for k in ('integration_t','output_indices'):
            np.testing.assert_array_equal(got['diagnostics'][k],want['diagnostics'][k])
        self.assertFalse(got['diagnostics']['used_clipping'])
        self.assertFalse(got['diagnostics']['used_renormalization'])

    def test_lapack_failure_is_not_repaired_or_ignored(self):
        with patch('scipy.linalg.lapack.dgtsv',return_value=(None,None,None,None,2)):
            with self.assertRaisesRegex(RuntimeError,'info=2'):
                finite_volume_1d_lapack._solve_tridiagonal(np.ones(2),np.ones(3),np.ones(2),np.ones(3))


if __name__=='__main__': unittest.main()

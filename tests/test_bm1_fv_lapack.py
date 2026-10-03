"""Opt-in LAPACK changes only the tridiagonal linear solver."""
import unittest
from unittest.mock import patch
import numpy as np
from src.solver.finite_volume import finite_volume_1d, finite_volume_1d_lapack, finite_volume_1d_cached


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


class FVCachedTests(unittest.TestCase):
    def test_piecewise_operator_reuse_matches_dense_implicit_euler(self):
        cells = 7
        solver = finite_volume_1d_cached()
        mass = np.arange(1., cells + 1); mass /= mass.sum()
        times = np.array([0., .1, .25, .5])
        def key(t): return 0 if t < .25 else 1
        def drift(t, x): return (.3 if key(t) == 0 else -.2) - .2*x
        def diffusion(t, x): return .6*x*(1-x)
        with patch.object(solver, 'generator_tridiagonal', wraps=solver.generator_tridiagonal) as build:
            got = solver.solve(times, mass, drift, diffusion, max_step=.04,
                               breakpoints=[.25], coefficient_key=key)
        self.assertEqual(build.call_count, 2)
        edges, x, dx = solver.grid(cells)
        internal = got['diagnostics']['integration_t']
        history = [mass]
        for left, right in zip(internal[:-1], internal[1:]):
            a, d, b, _ = solver.generator_tridiagonal(dx, drift(left, edges[1:-1]), diffusion(left, x))
            Q = np.diag(d) + np.diag(a, -1) + np.diag(b, 1)
            history.append(np.linalg.solve(np.eye(cells) - (right-left)*Q, history[-1]))
        want = np.asarray(history)[got['diagnostics']['output_indices']]
        np.testing.assert_allclose(got['cell_mass'], want, atol=2e-15, rtol=0)
        self.assertEqual(got['cell_mass'].shape, (len(times), cells))

    def test_generic_time_varying_callbacks_and_override_remain_live(self):
        from src.model.bm_1 import bm_1
        from src.validation.bm_1.estimation import build_f4_b_case, beta_initial_mass
        case = build_f4_b_case()
        base, parameters = case.build_model()
        class ChangedDrift(bm_1):
            def _drift(self, t, x, parameters):
                return super()._drift(t, x, parameters) + .1*np.sin(20*t)*x*(1-x)
        changed = ChangedDrift(base.B, base.A, base.T, base.t_steps)
        mass = beta_initial_mass(case, base, 17)
        settings = dict(t_eval=[0, .3, .6, 1], breakpoints=case.breakpoints, max_step=.02)
        changed.fokker_planc_reference = finite_volume_1d_lapack()
        want = changed.fokker_planc_reference_solver(mass, parameters, .5, **settings)
        changed.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=True)
        got = changed.fokker_planc_reference_solver(mass, parameters, .5, **settings)
        np.testing.assert_array_equal(got[3], want[3])
        base.fokker_planc_reference = finite_volume_1d_lapack()
        normal = base.fokker_planc_reference_solver(mass, parameters, .5, **settings)
        self.assertGreater(np.max(abs(got[3]-normal[3])), 1e-5)

    def test_exact_step_lengths_and_cache_lifetime_do_not_change_dynamics(self):
        solver = finite_volume_1d_cached()
        mass = np.full(31, 1/31)
        # Many distinct fields also exercise bounded cache eviction.
        for scale in (.2, 1.7):
            def drift(t, x): return scale*(.5-x)+.03*np.sin(t*200)
            def diffusion(t, x): return .5*x*(1-x)
            settings = dict(max_step=.0037, breakpoints=[.153])
            args = ([0., .17, .3], mass, drift, diffusion)
            want = finite_volume_1d_lapack().solve(*args, **settings)
            got = solver.solve(*args, coefficient_key=lambda t: t, **settings)
            np.testing.assert_array_equal(got['diagnostics']['integration_t'], want['diagnostics']['integration_t'])
            np.testing.assert_allclose(got['cell_mass'], want['cell_mass'], atol=2e-15, rtol=0)
            self.assertLess(got['diagnostics']['maximum_abs_mass_error'], 1e-12)

    def test_bm1_parameter_changes_and_varying_inputs_match_uncached(self):
        from src.validation.bm_1.estimation import build_f4_b_case, beta_initial_mass
        case = build_f4_b_case()
        model, parameters = case.build_model()
        model.B = lambda t: .5 + .1*np.sin(np.asarray(t)*8)
        mass = beta_initial_mass(case, model, 31)
        settings = dict(t_eval=[0., .31, .7, 1.], breakpoints=case.breakpoints, max_step=.012)
        for sigma, C in ((.5, 1.), (1.2, .6)):
            parameters[3] = C
            model.fokker_planc_reference = finite_volume_1d_lapack()
            want = model.fokker_planc_reference_solver(mass, parameters, sigma, **settings)
            for vectorized in (False, True):
                model.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=vectorized)
                got = model.fokker_planc_reference_solver(mass, parameters, sigma, **settings)
                np.testing.assert_allclose(got[3], want[3], atol=2e-14, rtol=0)

    def test_factorization_and_solve_failures_are_never_repaired(self):
        args = ([0., .1], np.ones(5)/5, lambda t, x: 0*x, lambda t, x: .5+0*x)
        settings = dict(coefficient_key=lambda t: 0)
        with patch('scipy.linalg.lapack.dgttrf', return_value=(None, None, None, None, None, 2)):
            with self.assertRaisesRegex(RuntimeError, 'factorization.*info=2'):
                finite_volume_1d_cached().solve(*args, **settings)
        with patch('scipy.linalg.lapack.dgttrs', return_value=(None, -1)):
            with self.assertRaisesRegex(RuntimeError, 'factored solve.*info=-1'):
                finite_volume_1d_cached().solve(*args, **settings)


if __name__=='__main__': unittest.main()

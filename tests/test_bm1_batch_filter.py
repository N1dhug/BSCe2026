"""Multiple-RHS filtering must preserve independent conditional trajectories."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from src.method.bm_1_batch_filter import run_fv_sigma_y_batch
from src.method.bm_1_estimation import run_fv_filter
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation.bm_1 import estimation as core


class BatchedEvolutionTests(unittest.TestCase):
    def test_distinct_states_match_independent_dense_systems(self):
        solver = finite_volume_1d_cached()
        initial = np.array([[.1, .2, .3, .2, .1, .05, .05], [0., 0., 1., 0., 0., 0., 0.]])
        def drift(t, x): return (.3 if t < .15 else -.2)-.1*x
        def diffusion(t, x): return .4*x*(1-x)
        settings = dict(breakpoints=[.15], max_step=.037, coefficient_key=lambda t: int(t >= .15))
        got = solver.solve_many([0., .11, .3], initial, drift, diffusion, **settings)
        edges, centers, dx = solver.grid(7)
        times = got['prediction_diagnostics'][0]['integration_t']
        dense = initial.T.copy()
        history = [dense.T.copy()]
        for left, right in zip(times[:-1], times[1:]):
            lower, diagonal, upper, _ = solver.generator_tridiagonal(dx, drift(left, edges[1:-1]), diffusion(left, centers))
            operator = np.diag(diagonal)+np.diag(lower, -1)+np.diag(upper, 1)
            dense = np.linalg.solve(np.eye(7)-(right-left)*operator, dense)
            history.append(dense.T.copy())
        expected = np.array(history)[got['prediction_diagnostics'][0]['output_indices']]
        np.testing.assert_allclose(got['cell_mass'], expected, atol=2e-15, rtol=0)
        for i, row in enumerate(initial):
            scalar = solver.solve([0., .11, .3], row, drift, diffusion, **settings)
            np.testing.assert_array_equal(got['cell_mass'][:, i], scalar['cell_mass'])
            for key, value in scalar['diagnostics'].items():
                np.testing.assert_equal(got['prediction_diagnostics'][i][key], value)

    def test_invalid_column_and_internal_solver_failures_are_not_hidden(self):
        solver = finite_volume_1d_cached()
        args = ([0., .1], np.ones((2, 7))/7, lambda t, x: x*0, lambda t, x: x*0+.3)
        for invalid in (np.ones(7)/7, np.ones((0, 7)), np.ones((2, 7)), np.ones((2, 7), complex)*1j):
            with self.subTest(shape=invalid.shape), self.assertRaises((ValueError, TypeError)):
                solver.solve_many(args[0], invalid, *args[2:])
        with patch('scipy.linalg.lapack.dgttrs', return_value=(None, 2)):
            with self.assertRaisesRegex(RuntimeError, 'info=2'):
                solver.solve_many(*args, coefficient_key=lambda t: 0)
        bad = np.ones((7, 2))/7; bad[0, 1] = -.01
        with patch.object(solver, '_stepper', return_value=lambda *a: (bad, {})):
            with self.assertRaisesRegex(RuntimeError, 'negative'):
                solver.solve_many(*args)


class BatchedFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = core.build_f4_b_case()
        cls.sequence = core.generate_synthetic_sequence(cls.case, sequence_id=15011)

    def setup_filter(self):
        model, parameters = self.case.build_model()
        model.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=True)
        return model, parameters, core.beta_initial_mass(self.case, model, 31)

    def test_noise_order_missing_observations_left_prices_and_distinct_initial_states(self):
        model, parameters, mass = self.setup_filter()
        noises = [.12, .0075, .03, .03]
        initial = np.stack([mass, mass[::-1], np.ones(31)/31, mass])
        observed = self.sequence.observations.observed.copy(); observed[::3] = False
        observations = replace(self.sequence.observations, observed=observed, price_side='left')
        for order in (1, 3, 5):
            settings = dict(breakpoints=self.case.breakpoints, max_step=1/96, likelihood_quadrature_order=order)
            got = run_fv_sigma_y_batch(model, parameters, .5, noises, initial, 0., observations, **settings)
            for noise, start, result in zip(noises, initial, got):
                scalar = run_fv_filter(model, parameters, .5, noise, start, 0., observations, **settings)
                for name in ('predicted_mass', 'posterior_mass', 'log_evidence', 'integration_t'):
                    np.testing.assert_equal(getattr(result, name), getattr(scalar, name))
                self.assertEqual(result.maximum_abs_mass_error, scalar.maximum_abs_mass_error)
        self.assertGreater(np.max(abs(got[0].posterior_mass-got[1].posterior_mass)), .001)

    def test_time_varying_model_override_keeps_live_fields(self):
        model, parameters, mass = self.setup_filter()
        original = model._drift
        model._drift = lambda t, x, p: original(t, x, p)+.03*np.sin(t*11)*x*(1-x)
        model.B = lambda t: .5+.1*np.sin(np.asarray(t)*7)
        settings = dict(breakpoints=self.case.breakpoints, max_step=1/96, likelihood_quadrature_order=3)
        got = run_fv_sigma_y_batch(model, parameters, .5, [.0075, .12], mass, 0., self.sequence.observations, **settings)
        for noise, result in zip([.0075, .12], got):
            scalar = run_fv_filter(model, parameters, .5, noise, mass, 0., self.sequence.observations, **settings)
            np.testing.assert_array_equal(result.posterior_mass, scalar.posterior_mass)
            np.testing.assert_array_equal(result.log_evidence, scalar.log_evidence)

    def test_invalid_noise_or_initial_state_cannot_enter_solver(self):
        model, parameters, mass = self.setup_filter()
        for noises, initial in (([0., .03], mass), ([-1.], mass), ([np.nan], mass),
                                ([.03], np.stack([mass, mass])), ([.03], mass.astype(complex)*1j)):
            with self.subTest(noises=noises), patch.object(model.fokker_planc_reference, 'solve_many') as solve:
                with self.assertRaises((ValueError, TypeError)):
                    run_fv_sigma_y_batch(model, parameters, .5, noises, initial, 0., self.sequence.observations)
                solve.assert_not_called()

    def group(self):
        model, _, _ = self.setup_filter()
        return dict(pair=('sigma_x', 'sigma_y'), sequence_id=self.sequence.sequence_id,
            case=self.case, observations=self.sequence.observations,
            hypotheses=grid._parameter_grid(('sigma_x', 'sigma_y'), 5, 5)[0], counts=(3, 5),
            backend='cached', keep_demand=True, label='middle', prior_bridge=False,
            settings=dict(x_steps=31, max_step=1/96, likelihood_quadrature_order=3))


class ForecastCollapseTests(unittest.TestCase):
    def test_process_state_dependence_and_objective_are_preserved(self):
        from src.method.bm_1_batch_filter import collapse_sigma_y_for_process_forecast
        rng = np.random.default_rng(15121)
        hypotheses = np.array([[.125, .12], [2., .0075], [.125, .0075], [2., .12]])
        states = rng.dirichlet(np.ones(17), size=4)
        original_states = states.copy()
        logs = np.log([.1, .25, .2, .45])
        collapsed = collapse_sigma_y_for_process_forecast(hypotheses, logs, states)
        np.testing.assert_allclose(np.exp(collapsed['log_weights']), [.3, .7], atol=1e-15, rtol=0)
        solver = finite_volume_1d_cached()
        future = []
        grouped = []
        for sigma, state in zip(hypotheses[:, 0], states):
            future.append(solver.solve([0., .4, 1.], state, lambda t, x: .2*(.5-x),
                                      lambda t, x: sigma*x*(1-x), max_step=1/48)['cell_mass'])
        for sigma, state in zip(collapsed['sigma_x'], collapsed['initial_mass']):
            grouped.append(solver.solve([0., .4, 1.], state, lambda t, x: .2*(.5-x),
                                       lambda t, x: sigma*x*(1-x), max_step=1/48)['cell_mass'])
        full = np.einsum('j,jtn->tn', np.exp(logs), np.asarray(future))
        reduced = np.einsum('j,jtn->tn', np.exp(collapsed['log_weights']), np.asarray(grouped))
        np.testing.assert_allclose(full, reduced, atol=2e-15, rtol=0)
        centers = solver.grid(17)[1]
        demand = .5+.5*np.tanh(2*(.5-centers))
        np.testing.assert_allclose(full @ ((demand-.6)**2), reduced @ ((demand-.6)**2), atol=2e-15, rtol=0)
        wrong = solver.solve([0., .4, 1.], np.exp(logs) @ states, lambda t, x: .2*(.5-x),
                             lambda t, x: float(np.exp(collapsed['log_weights']) @ collapsed['sigma_x'])*x*(1-x),
                             max_step=1/48)['cell_mass']
        self.assertGreater(np.max(abs(wrong-reduced)), .001)
        np.testing.assert_array_equal(states, original_states)

    def test_zero_weights_and_invalid_laws_are_handled_without_repair(self):
        from src.method.bm_1_batch_filter import collapse_sigma_y_for_process_forecast
        hypotheses = np.array([[.5, .01], [.5, .1], [2., .01]])
        states = np.ones((3, 7))/7
        got = collapse_sigma_y_for_process_forecast(hypotheses, [0., -np.inf, -np.inf], states)
        np.testing.assert_array_equal(got['log_weights'], [0., -np.inf])
        np.testing.assert_array_equal(got['initial_mass'], states[:2])
        for h, w, s in ((hypotheses[:, :1], [0., -np.inf, -np.inf], states),
                        (hypotheses, [0., 0., 0.], states),
                        (hypotheses, [0., -np.inf, -np.inf], states*2),
                        (hypotheses, [0., -np.inf], states)):
            with self.assertRaises(ValueError):
                collapse_sigma_y_for_process_forecast(h, w, s)


if __name__ == '__main__':
    unittest.main()

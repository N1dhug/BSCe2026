"""Multiple-RHS filtering must preserve independent conditional trajectories."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from src.method.bm_1_batch_filter import run_fv_sigma_y_batch
from src.method.bm_1_estimation import run_fv_filter
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation.bm_1 import estimation as core, joint_estimation as joint, joint_grid as grid


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

    def test_task_boundaries_reject_mixing_process_parameters(self):
        group = self.group()
        tasks = list(joint._batch_tasks([dict(group, batch_sigma_y=4)]))
        self.assertEqual([(a, b) for _, a, b, _ in tasks],
                         [(a, b) for start in range(0, 25, 5) for a, b in ((start, start+4), (start+4, start+5))])
        with self.assertRaisesRegex(ValueError, 'fixed sigma_x'):
            joint._candidate_batch_replay((0, 3, 7, group))
        for changes in (dict(batch_sigma_y=5), dict(batch_sigma_y=4, backend='reference'),
                        dict(batch_sigma_y=4, pair=('rho', 'Xi'))):
            with self.assertRaises(ValueError): list(joint._batch_tasks([dict(group, **changes)]))

    def test_complete_joint_bank_matches_scalar_and_spawn_with_demand(self):
        group = self.group()
        scalar = joint._run_groups_streaming([group], 1, None, grid._StreamingGridReduction)[0]
        for workers in (1, 2):
            progress = []
            batched = joint._run_groups_streaming([dict(group, batch_sigma_y=4)], workers, progress.append,
                                                  grid._StreamingGridReduction)[0]
            self.assertEqual([p['completed'] for p in progress], list(range(1, 26)))
            for a, b in zip(scalar, batched):
                for key in ('conditional_log_evidence', 'log_weights', 'bank_log_evidence',
                            'marginal_predicted_mass', 'marginal_posterior_mass',
                            'marginal_predicted_demand_mean', 'marginal_posterior_demand_mean'):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_batched_queue_is_bounded_and_failure_never_drops_hypotheses(self):
        events = []
        class Future:
            def result(self):
                events.append('consume'); raise RuntimeError('batch failed')
        class Pool:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, *args): events.append('submit'); return Future()
        with patch.object(joint, 'ProcessPoolExecutor', Pool), self.assertRaisesRegex(RuntimeError, 'batch failed'):
            joint._run_groups_streaming([dict(self.group(), batch_sigma_y=4)], 3, None, grid._StreamingGridReduction)
        self.assertEqual(events, ['submit']*3+['consume'])

    def test_benchmark_checks_every_configuration_and_retains_raw_scores(self):
        from src.validation.bm_1 import joint_pilot
        configurations = [(name, dict(x_steps=11, max_step=1/48, likelihood_quadrature_order=3))
                          for name in joint_pilot.LABELS]
        with patch.object(joint_pilot, 'configurations', return_value=configurations):
            report = joint.benchmark_sigma_y_batch(record_days=1, repeats=2, batch_size=4)
        self.assertTrue(report['passes'])
        self.assertEqual(report['tested_scalar_hypotheses'], 96)
        self.assertEqual(report['tested_batch_calls'], 24)
        self.assertEqual(len(report['rows']), 12)
        for row in report['rows']:
            self.assertEqual(max(row['maximum_errors'].values()), 0.)
            for key in row['final_repeat_raw']['scalar']:
                np.testing.assert_array_equal(row['final_repeat_raw']['scalar'][key],
                                              row['final_repeat_raw']['batch'][key])

    def test_benchmark_cli_requires_clean_source_and_preserves_existing_output(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from src.exe.run_model.bm_1 import estimate
        with TemporaryDirectory() as directory:
            output = Path(directory)/'benchmark.json.gz'
            args = estimate.build_parser().parse_args(['benchmark-fv-batch', '--output', str(output)])
            with patch.object(estimate, '_source_snapshot', return_value=dict(working_tree_dirty=True)):
                with self.assertRaisesRegex(ValueError, 'Commit source'):
                    args.handler(args)
            output.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                args.handler(args)
            self.assertEqual(output.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()

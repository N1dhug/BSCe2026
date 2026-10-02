"""Cross-cutting F4 regression checks for the post-D performance/correctness audit."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import copy
import gzip
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.method.beta_mixture import PreparedBetaProjector, beta_cell_average_basis
from src.method.bm_1_estimation import mixture_filter_step, run_mixture_filter
from src.validation import estimation as checks
from src.validation.bm_1 import estimation as synthetic
from src.validation.bm_1 import mixture_estimation as mixture_f4


class F4AuditRegressionTests(unittest.TestCase):
    """Concrete bugs and performance invariants found in the post-D audit."""

    @staticmethod
    def _projector():
        edges = np.linspace(0.0, 1.0, 32)
        basis = beta_cell_average_basis(edges, [1., 2., 4.], [1., 3., 2.])
        return PreparedBetaProjector(basis, 1/31), (edges[:-1] + edges[1:])/2

    def test_warm_snapshot_keeps_target_and_optimality(self):
        projector, centers = self._projector()
        mass = np.array([.2, .3, .5]) @ projector.basis * projector.dx
        original = mass.copy()
        cold, cold_mass, coefficients = mixture_f4._snapshot_projection(projector, mass, centers)
        warm, warm_mass, _ = mixture_f4._snapshot_projection(
            projector, mass, centers, initial_coefficients=coefficients,
        )
        np.testing.assert_array_equal(mass, original)
        np.testing.assert_allclose(warm_mass, cold_mass, atol=1e-12, rtol=0)
        self.assertLessEqual(warm['optimality_gap'], 1e-9)
        self.assertLessEqual(cold['optimality_gap'], 1e-9)

    def test_cached_simplex_geometry_is_readonly_and_target_free(self):
        from src.method.beta_mixture import _simplex_tangent
        tangent = _simplex_tangent(7)
        self.assertIs(tangent, _simplex_tangent(7))
        self.assertFalse(tangent.flags.writeable)
        np.testing.assert_allclose(tangent.T @ tangent, np.eye(6), atol=1e-14)
        np.testing.assert_allclose(tangent.sum(axis=0), 0.0, atol=1e-14)

    def test_initial_projection_warm_start_keeps_filter_results(self):
        case = synthetic.build_f4_b_case()
        sequence = synthetic.generate_synthetic_sequence(case)
        model, parameters = case.build_model()
        projector, _ = self._projector()
        initial = synthetic.beta_initial_mass(case, model, 31)
        kwargs = dict(projector=projector, max_step=1/192)
        args = (model, parameters, case.sigma_x, case.sigma_y, initial,
                0.0, sequence.observations.prefix(1/24))
        cold = run_mixture_filter(*args, **kwargs)
        warm = run_mixture_filter(*args, **kwargs, initial_coefficients=cold.initial_coefficients)
        np.testing.assert_array_equal(cold.posterior_mass, warm.posterior_mass)
        np.testing.assert_array_equal(cold.log_evidence, warm.log_evidence)

    def test_probability_rows_reject_nan_complex_signed_and_wrong_mass(self):
        for bad in ([[np.nan, np.nan]], [[.5+1j, .5]], [[-.1, 1.1]], [[.3, .3]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checks.probability_rows(bad)
        original = np.array([[.3, .7]])
        owned = checks.probability_rows(original)
        original[0, 0] = 9
        np.testing.assert_array_equal(owned, [[.3, .7]])

    def test_mixture_result_rejects_nan_coefficients(self):
        from src.method.bm_1_estimation import BM1MixtureFilterResult
        args = dict(initial_time=0, initial_mass=[.5, .5], initial_coefficients=[.5, .5],
                    times=[1], x_edges=[0, .5, 1], x_grid=[.25, .75],
                    predicted_mass=[[.5, .5]], posterior_mass=[[.5, .5]],
                    predicted_coefficients=[[np.nan, np.nan]], posterior_coefficients=[[.5, .5]],
                    observed=[True], log_evidence=[0], prediction_diagnostics=({},),
                    posterior_projection_diagnostics=({},), initial_projection_diagnostics={})
        with self.assertRaises(ValueError):
            BM1MixtureFilterResult(**args)

    def test_mixture_rejects_invalid_capacity_and_horizon_before_evolution(self):
        case = synthetic.build_f4_b_case()
        model, parameters = case.build_model()
        projector, _ = self._projector()
        for capacity in (-1, 0, np.nan):
            bad = list(parameters)
            bad[3] = capacity
            with self.subTest(capacity=capacity), patch(
                'src.method.bm_1_estimation.evolve_projected_density'
            ) as evolve, self.assertRaises(ValueError):
                mixture_filter_step(model, bad, .5, .03, [1, 0, 0], 0, 1/24,
                                    .5, projector=projector)
            evolve.assert_not_called()
        with self.assertRaises(ValueError):
            mixture_filter_step(model, parameters, .5, .03, [1, 0, 0], 1, 2,
                                .5, projector=projector)

    def test_unequal_history_lengths_are_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            mixture_f4._error_history(np.full((2, 2), .5), np.full((1, 2), .5),
                                      np.array([.25, .75]), .5)

    def test_spatial_comparison_rejects_mixed_time_settings(self):
        rows = [dict(requested_max_step=h, actual_max_step=h, metric=.01)
                for h in (1/32, 1/64, 1/128)]
        with self.assertRaises(ValueError):
            mixture_f4._d3c_spatial_summary(rows, 'metric')

    def test_invalid_convergence_design_fails_before_expensive_work(self):
        with patch.object(mixture_f4, 'run_f4_d_recursive_sanity') as run:
            with self.assertRaises(ValueError):
                mixture_f4.run_f4_d_recursive_convergence(time_steps=(1/100, 1/300, 1/600))
        run.assert_not_called()

    def test_negative_mse_is_not_a_valid_rmse_input(self):
        with self.assertRaises(ValueError):
            mixture_f4._d3d_rmse_summary([-.1, .1], [.1, .1], replicates=100, seed=1)


class F4ExecutionRegressionTests(unittest.TestCase):
    def test_batch_and_single_coupled_paths_and_diagnostics_match_exactly(self):
        ids = (0, 1, 1001)
        singles = [synthetic.generate_coupled_refinement(sequence_id=i) for i in ids]
        batch = synthetic.generate_synthetic_sequences(sequence_ids=ids, refine=True)
        for a, b in zip(singles, batch):
            for resolution in ('coarse', 'fine'):
                left, right = getattr(a, resolution), getattr(b, resolution)
                for field in ('times', 'state', 'price', 'baseline', 'demand'):
                    np.testing.assert_array_equal(getattr(left.truth, field), getattr(right.truth, field))
                np.testing.assert_array_equal(left.observations.values, right.observations.values)
                np.testing.assert_array_equal(left.measurement_noise, right.measurement_noise)
                self.assertEqual(left.sequence_id, right.sequence_id)
                for key in left.truth.diagnostics:
                    np.testing.assert_equal(left.truth.diagnostics[key], right.truth.diagnostics[key])

    def test_batch_rejects_empty_duplicate_and_invalid_ids(self):
        for ids in ((), (0, 0), (-1,), (1.5,), (True,)):
            with self.subTest(ids=ids), self.assertRaises((ValueError, TypeError)):
                synthetic.generate_synthetic_sequences(sequence_ids=ids)

    def test_population_serial_spawn_and_progress_are_identical(self):
        settings = dict(n_sequences=2, sequence_start=0, x_steps=31, max_step=1/24,
                        bootstrap_replicates=100, bootstrap_seed=3)
        progress = []
        serial = mixture_f4.run_f4_d_recursive_population(
            **settings, workers=1,
            progress=lambda k, n, row: progress.append((k, n, row['sequence_id'])),
        )
        # Explicit spawn exercises Windows' process model on every test platform.
        factory = partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context('spawn'))
        with patch.object(mixture_f4, 'ProcessPoolExecutor', factory):
            parallel = mixture_f4.run_f4_d_recursive_population(**settings, workers=2)
        self.assertEqual(progress, [(1, 2, 0), (2, 2, 1)])
        self.assertIsNone(mixture_f4._D3D_WORKER_STATE)
        self.assertEqual({k: v for k, v in serial.items() if k != 'workers'},
                         {k: v for k, v in parallel.items() if k != 'workers'})
        self.assertTrue(serial['complete'])
        self.assertFalse(serial['matches_registered_design'])

    def test_population_all_failed_retains_ids_settings_and_failure_ledger(self):
        def failed(sequence, *, state=None):
            return dict(sequence_id=sequence.sequence_id, valid_truth=True, status='failed',
                        failure_type='RuntimeError', failure_message='forced test failure')
        with patch.object(mixture_f4, '_run_d3d_sequence', side_effect=failed):
            result = mixture_f4.run_f4_d_recursive_population(
                n_sequences=2, sequence_start=0, x_steps=31, max_step=1/24,
                bootstrap_replicates=100,
            )
        self.assertFalse(result['passes_population'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['summaries_conditioned_on_success'])
        self.assertEqual(len(result['per_sequence']), 2)
        self.assertEqual(len(result['failed_sequences']), 2)
        self.assertEqual(result['x_steps'], 31)

    def test_progress_cannot_mutate_the_result_ledger(self):
        settings = dict(n_sequences=2, sequence_start=0, x_steps=31, max_step=1/24,
                        bootstrap_replicates=100, bootstrap_seed=4)
        def progress(completed, total, row):
            row['status'] = 'failed'
            row['mse_mixture_posterior'] = -1.0
        result = mixture_f4.run_f4_d_recursive_population(**settings, progress=progress)
        self.assertEqual(result['successful_sequences'], 2)
        self.assertTrue(all(row['status'] == 'ok' for row in result['per_sequence']))

    def test_cli_json_roundtrip_and_no_overwrite(self):
        from src.exe.run_model.bm_1 import estimate
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'result.json'
            args = estimate.build_parser().parse_args([
                'mixture-filter-population', '--sequences', '2', '--start', '0',
                '--cells', '31', '--max-step', str(1/24), '--bootstrap-replicates', '100',
                '--output', str(destination), '--quiet',
            ])
            from contextlib import redirect_stdout
            from io import StringIO
            with redirect_stdout(StringIO()):
                report = estimate.run_mixture_filter_population(args)
            stored = json.loads(destination.read_text())
            self.assertEqual(stored['per_sequence'], report['per_sequence'])
            self.assertIn('source_commit', stored)
            with patch.object(mixture_f4, 'run_f4_d_recursive_population') as run:
                with self.assertRaises(FileExistsError), redirect_stdout(StringIO()):
                    estimate.run_mixture_filter_population(args)
            run.assert_not_called()

    def test_cli_blas_control_is_explicit_and_import_is_non_mutating(self):
        from src.exe.run_model.bm_1 import estimate
        with patch.dict(os.environ, {'OPENBLAS_NUM_THREADS': '7'}, clear=False):
            estimate._configure_blas_startup(['filter'])
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '7')
            estimate._configure_blas_startup(['filter', '--blas-threads', '1'])
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '1')
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '1')
            self.assertEqual(os.environ['OPENBLAS_DEFAULT_NUM_THREADS'], '1')
        # A subprocess importing the CLI must not rewrite the host's environment.
        script = "import os; before=dict(os.environ); from src.exe.run_model.bm_1 import estimate; assert before==dict(os.environ)"
        subprocess.run([sys.executable, '-B', '-c', script], check=True)

    def test_preparation_keeps_independent_formula_and_negative_variance_example(self):
        from src.exe.run_model.bm_1 import estimate
        self.assertEqual(estimate.build_parser().parse_args(['preparation']).handler,
                         estimate.run_preparation)
        report = synthetic.run_f4_preparation_checks()
        self.assertLess(report['nonmonotone_variance_example']['initial_variance_derivative'], 0)
        self.assertAlmostEqual(report['single_measurement']['continuous_posterior_mean'], .5)
        self.assertEqual(len(report['static_projection']), 6)
        for result in report['static_projection']:
            self.assertLessEqual(result['mass_error'], 1e-9)

    def test_latex_sources_have_no_hidden_control_characters(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'root/f4').glob('*.tex'):
            data = path.read_bytes()
            # Accept conventional CRLF files, but not isolated CR from a broken \rm escape.
            data = data.replace(b'\r\n', b'\n')
            self.assertFalse(any(value < 32 and value not in (9, 10) for value in data), str(path))


class F4MidDevelopmentRegressionTests(unittest.TestCase):
    """Behavioral regressions for the current A--F implementation."""

    def test_quantile_near_one_handles_cumulative_roundoff_and_zero_tail(self):
        probability = np.nextafter(1., 0.)
        for empty_tail in (0, 3):
            mass = np.r_[np.full(7, 1/7), np.zeros(empty_tail)]
            original = mass.copy()
            edges = np.linspace(0., 1., mass.size + 1)
            # Uniform density over the first seven cells has this exact quantile.
            expected = probability * edges[7]
            actual = synthetic._piecewise_constant_quantile(edges, mass, probability)
            self.assertAlmostEqual(actual, expected, places=14)
            np.testing.assert_array_equal(mass, original)

    def test_fv_and_observation_updates_reject_out_of_horizon_times(self):
        from src.method.bm_1_estimation import fv_filter_step, bm1_observation_update
        case = synthetic.build_f4_b_case()
        model, parameters = case.build_model()
        initial = synthetic.beta_initial_mass(case, model, 21)
        for start, end in ((-.01, .01), (.99, 1.01)):
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, 'horizon'):
                    fv_filter_step(model, parameters, .5, .03, initial, start, end, .5)
                with self.assertRaisesRegex(ValueError, 'horizon'):
                    bm1_observation_update(model, parameters, .03, initial, start, end, .5)
        # The closed model endpoints are valid, including an observation at T[1].
        result = fv_filter_step(model, parameters, .5, .03, initial, .99, 1., .5)
        self.assertAlmostEqual(result.posterior_mass.sum(), 1.)

    def test_synthetic_likelihood_rejects_wrong_declared_time_unit(self):
        from src.params.bm_1 import BM1ObservationSeries
        case = synthetic.build_f4_b_case()
        for unit in ('hour', 'second'):
            observations = BM1ObservationSeries([1/24], [.5], unit)
            with self.subTest(unit=unit), self.assertRaisesRegex(ValueError, 'day'):
                synthetic.evaluate_f4_e_candidate(case, observations,
                    synthetic.f4_e_truth_candidate(case), x_steps=21)

    def test_parameter_scores_reject_complex_arrays_without_discarding_imaginary_part(self):
        from src.method.bm_1_estimation import parameter_bank_weights
        with self.assertRaises(ValueError):
            parameter_bank_weights(np.array([[1.+2j, 0.]]), np.log([.5, .5]), [True])

    def test_f_bank_probability_gate_rejects_impossible_diagnostics(self):
        good = dict(maximum_abs_mass_error=0., minimum_cell_mass=0., fallbacks=0)
        self.assertTrue(synthetic._f4_f_probability_gate([good]))
        for key, value in (('maximum_abs_mass_error', -1.),
                           ('maximum_abs_mass_error', -np.inf),
                           ('maximum_abs_mass_error', np.nan),
                           ('minimum_cell_mass', np.inf),
                           ('minimum_cell_mass', -.01), ('fallbacks', -1)):
            with self.subTest(key=key, value=value):
                self.assertFalse(synthetic._f4_f_probability_gate([dict(good, **{key: value})]))

    def test_f1_f3_reject_broadcast_summaries_and_stale_parameter_means(self):
        from tests import test_bm1_estimation as fixtures
        reports = ((fixtures.OneParameterBankTests.refinement_fixture(),
                    synthetic.f4_f_validated_refinement),
                   (fixtures.OneParameterBankTests.refined_mixture_fixture(),
                    synthetic.f4_f_validated_mixture_check))
        for original, validate in reports:
            validate(original)
            for field in ('initial_log_weights', 'log_weights', 'cumulative_log_evidence',
                          'parameter_mean'):
                report = copy.deepcopy(original)
                if field == 'parameter_mean':
                    report['rows'][0][field][0] += .5
                else:
                    report['rows'][0][field] = -np.log(41) if 'weights' in field else 0.
                with self.subTest(stage=report['stage'], field=field), self.assertRaises(ValueError):
                    validate(report)

    def test_e_progress_callbacks_cannot_change_scientific_results(self):
        from tests import test_bm1_estimation as fixtures
        options = dict(e1_refinement=fixtures.F4EParameterScreeningTests.passing_e1(),
                       n_sequences=1, sequence_start=0, x_steps=21, max_step=1/24,
                       likelihood_quadrature_order=1,
                       parameter_grids=fixtures.F4EParameterScreeningTests.small_grids())
        reference = synthetic.run_f4_e_parameter_screening(**options)

        def corrupt_screen(completed, total, key, row):
            row['slices']['rho']['total_log_likelihood'] = (999., 999., 999.)

        result = synthetic.run_f4_e_parameter_screening(**options, progress=corrupt_screen)
        self.assertTrue(result == reference, 'Progress callback changed the screening result.')

    def test_e_surface_progress_callbacks_cannot_change_scientific_results(self):
        from tests import test_bm1_estimation as fixtures
        source = fixtures.F4EPairSurfaceTests.registered_e2()
        # Reuse an existing coherent E2 fixture, but replace the expensive
        # surface computation with a real, deterministic surface summary.
        def surface(task):
            row, pair, a, b, *_ = task
            total = np.zeros((len(a), len(b)))
            item = synthetic._f4_e_surface_summary(pair, a, b, total)
            item.update(maximum_abs_mass_error=0., minimum_cell_mass=0.,
                        e2_truth_cut_maximum_abs_differences=(0., 0.))
            return (row['sequence_id'], pair), item

        def corrupt_surface(completed, total, key, row):
            row['total_log_likelihood'] = np.ones_like(row['total_log_likelihood']).tolist()

        with patch.object(synthetic, '_f4_e_pair_surface_task', side_effect=surface):
            reference = synthetic.run_f4_e_pair_surfaces(e2_screening=source, n_sequences=1)
            result = synthetic.run_f4_e_pair_surfaces(e2_screening=source, n_sequences=1,
                                                     progress=corrupt_surface)
        self.assertTrue(result == reference, 'Progress callback changed the surface result.')

    def test_compressed_e_dependencies_reach_the_next_study(self):
        from tests import test_bm1_estimation as fixtures
        from src.exe.run_model.bm_1 import estimate
        from time import perf_counter
        reports = ((fixtures.F4EParameterScreeningTests.passing_e1(),
                    'parameter-screen', '--e1-report', 'run_f4_e_parameter_screening'),
                   (fixtures.F4EPairSurfaceTests.registered_e2(),
                    'pair-surfaces', '--e2-report', 'run_f4_e_pair_surfaces'))
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            for report, command, flag, runner in reports:
                path = Path(directory) / (report['stage'] + '.json.gz')
                output_args = estimate.build_parser().parse_args(['bank-refine', '--output', str(path)])
                estimate._write_likelihood_report(output_args, report, perf_counter())
                args = estimate.build_parser().parse_args([command, flag, str(path), '--quiet'])
                with self.subTest(command=command), patch.object(synthetic, runner,
                        side_effect=RuntimeError('replay reached')) as run:
                    with self.assertRaisesRegex(RuntimeError, 'replay reached'):
                        args.handler(args)
                    self.assertEqual(run.call_count, 1)

    def test_synthetic_save_refuses_existing_file_before_generation(self):
        from src.exe.run_model.bm_1 import estimate
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / 'f4_b_sequence_0000.json'
            path.write_text('existing evidence', encoding='utf-8')
            args = estimate.build_parser().parse_args(['synthetic', '--save'])
            with patch.object(estimate, 'OUTPUT_DIR', output), \
                    patch.object(synthetic, 'generate_synthetic_sequence',
                                 wraps=synthetic.generate_synthetic_sequence) as generate:
                with self.assertRaises(FileExistsError):
                    estimate.run_synthetic(args)
                generate.assert_not_called()
            self.assertEqual(path.read_text(), 'existing evidence')

    def test_d3d_compressed_output_is_actually_gzip(self):
        from src.exe.run_model.bm_1 import estimate
        report = dict(successful_sequences=0, invalid_truth_sequences=[],
                      failed_sequences=[dict(sequence_id=0)], passes_population=False)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            path = Path(directory) / 'failed.json.gz'
            args = estimate.build_parser().parse_args(['mixture-filter-population', '--output', str(path)])
            with patch.object(mixture_f4, 'run_f4_d_recursive_population', return_value=report):
                estimate.run_mixture_filter_population(args)
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                stored = json.load(stream)
            self.assertFalse(stored['passes_population'])
            self.assertEqual(stored['failed_sequences'], report['failed_sequences'])


if __name__ == '__main__':
    unittest.main()

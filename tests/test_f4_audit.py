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


if __name__ == '__main__':
    unittest.main()

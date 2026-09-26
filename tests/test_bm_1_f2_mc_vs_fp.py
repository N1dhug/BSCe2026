"""F2-D tests for the shared BM reference case and MC-vs-FV comparison."""

import unittest

import numpy as np

from src.diagnostics.bm_1 import (
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    empirical_cell_mass,
)
from src.exe.run_model.bm_1.f2_reference_case import (
    BASELINE,
    INITIAL_ALPHA,
    INITIAL_BETA,
    PRICE_DROP_TIME,
    PRICE_VALUES,
    SIGMA_X,
    T_EVAL,
    beta_initial_cell_mass,
    build_reference_case,
    sample_beta_initial_states,
)
from src.exe.run_model.bm_1.mc_vs_fp_reference import run_comparison


class ReferenceCaseTests(unittest.TestCase):
    def test_shared_case_builds_expected_bm_inputs(self):
        model, parameters = build_reference_case()
        self.assertEqual(model.T, (0.0, 1.0))
        np.testing.assert_array_equal(model.t, T_EVAL)
        self.assertAlmostEqual(float(model.B(0.25)), BASELINE)
        self.assertAlmostEqual(float(parameters[0](0.25)), PRICE_VALUES[0])
        self.assertAlmostEqual(float(parameters[0](PRICE_DROP_TIME)), PRICE_VALUES[1])

    def test_beta_cell_mass_is_normalized_and_centered(self):
        model, _ = build_reference_case()
        edges, centers, _ = model.fokker_planc_reference.grid(201)
        mass = beta_initial_cell_mass(edges)
        self.assertAlmostEqual(np.sum(mass), 1.0, places=14)
        self.assertGreaterEqual(np.min(mass), 0.0)
        self.assertAlmostEqual(np.sum(mass * centers), 0.5, places=12)

    def test_beta_sampling_is_reproducible_and_matches_initial_law(self):
        left = sample_beta_initial_states(20_000, seed=42)
        right = sample_beta_initial_states(20_000, seed=42)
        np.testing.assert_array_equal(left, right)
        self.assertTrue(np.all((left > 0.0) & (left < 1.0)))
        self.assertLess(abs(np.mean(left) - 0.5), 0.005)
        expected_variance = (
            INITIAL_ALPHA
            * INITIAL_BETA
            / (
                (INITIAL_ALPHA + INITIAL_BETA) ** 2
                * (INITIAL_ALPHA + INITIAL_BETA + 1.0)
            )
        )
        self.assertLess(abs(np.var(left, ddof=1) - expected_variance), 0.001)


class DistributionMetricTests(unittest.TestCase):
    def test_empirical_mass_and_distances_match_simple_case(self):
        edges = np.array([0.0, 0.5, 1.0])
        empirical = empirical_cell_mass(
            np.array([0.1, 0.2, 0.7, 0.8]),
            edges,
        )
        np.testing.assert_array_equal(empirical, [0.5, 0.5])

        other = np.array([0.75, 0.25])
        self.assertAlmostEqual(cell_mass_l1(empirical, other), 0.5)
        self.assertAlmostEqual(cell_mass_cdf_sup(empirical, other), 0.25)
        self.assertAlmostEqual(
            cell_mass_wasserstein_1(empirical, other, 0.5),
            0.125,
        )


class MCvsFVIntegrationTests(unittest.TestCase):
    def test_small_controlled_comparison_is_consistent(self):
        result = run_comparison(
            n_paths=5_000,
            x_steps=101,
            max_step=1.0 / 96.0,
            initial_seed=77,
            brownian_seed=78,
        )

        self.assertEqual(result["em_diagnostics"]["total_violations"], 0)
        self.assertLess(
            result["fp_diagnostics"]["maximum_abs_mass_error"],
            1.0e-10,
        )
        self.assertLess(
            np.max(np.abs(result["mc_mean"] - result["fp_mean"])),
            0.02,
        )
        self.assertLess(
            np.max(
                np.abs(
                    result["mc_expected_demand"]
                    - result["fp_expected_demand"]
                )
            ),
            0.04,
        )
        self.assertLess(result["cdf_sup"][-1], 0.10)
        self.assertLess(result["l1"][-1], 0.40)
        np.testing.assert_allclose(
            np.sum(result["mc_cell_mass"], axis=1),
            1.0,
            atol=1.0e-14,
        )


if __name__ == "__main__":
    unittest.main()

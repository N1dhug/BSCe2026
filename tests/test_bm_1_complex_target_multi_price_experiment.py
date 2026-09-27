"""Structure tests for the complex-target multi-price BM1 experiment."""

import unittest

import numpy as np

from src.exe.run_model.bm_1.complex_target_multi_price_experiment import (
    HOURLY_EDGES,
    INTERVAL_COUNTS,
    REFERENCE_PRICES_24,
    _coarsen_reference_prices,
    _edges,
    _sample_piecewise_prices,
)


class ComplexTargetMultiPriceStructureTests(unittest.TestCase):
    def test_reference_signal_is_valid_and_genuinely_time_varying(self):
        self.assertEqual(REFERENCE_PRICES_24.shape, (24,))
        self.assertTrue(np.all(REFERENCE_PRICES_24 >= 0.0))
        self.assertTrue(np.all(REFERENCE_PRICES_24 <= 1.0))
        self.assertGreater(np.count_nonzero(np.diff(REFERENCE_PRICES_24)), 12)
        self.assertTrue(np.any(np.diff(REFERENCE_PRICES_24) > 0.0))
        self.assertTrue(np.any(np.diff(REFERENCE_PRICES_24) < 0.0))

    def test_candidate_grids_are_nested_in_hourly_breakpoints(self):
        hourly = set(np.round(HOURLY_EDGES, 15))
        for count in INTERVAL_COUNTS:
            candidate = set(np.round(_edges(count), 15))
            self.assertTrue(candidate.issubset(hourly))

    def test_coarsening_preserves_reference_mean(self):
        reference_mean = float(np.mean(REFERENCE_PRICES_24))
        for count in INTERVAL_COUNTS:
            coarse = _coarsen_reference_prices(count)
            self.assertEqual(coarse.size, count)
            self.assertAlmostEqual(float(np.mean(coarse)), reference_mean, places=15)

    def test_24_interval_coarsening_is_exact_reference(self):
        np.testing.assert_array_equal(
            _coarsen_reference_prices(24),
            REFERENCE_PRICES_24,
        )

    def test_resampling_preserves_constant_signal(self):
        old_edges = _edges(2)
        old_prices = np.array([0.4, 0.4], dtype=float)
        for count in INTERVAL_COUNTS:
            sampled = _sample_piecewise_prices(
                old_edges,
                old_prices,
                _edges(count),
            )
            np.testing.assert_array_equal(
                sampled,
                np.full(count, 0.4, dtype=float),
            )


if __name__ == "__main__":
    unittest.main()

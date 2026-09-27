"""Tests for the exploratory multi-price BM1 experiment."""

import unittest

import numpy as np

from src.exe.run_model.bm_1.multi_price_optimization_experiment import (
    DETERMINISTIC_REFERENCE_PRICES,
    INTERVAL_COUNTS,
    _edges,
    _lift_half_day_prices,
    _sample_piecewise_prices,
    _union_breakpoints,
)


class MultiPriceExperimentStructureTests(unittest.TestCase):
    def test_reference_policy_is_embedded_exactly_on_even_grids(self):
        for count in INTERVAL_COUNTS:
            prices = _lift_half_day_prices(
                DETERMINISTIC_REFERENCE_PRICES,
                count,
            )
            self.assertEqual(prices.size, count)
            np.testing.assert_array_equal(
                prices[: count // 2],
                DETERMINISTIC_REFERENCE_PRICES[0],
            )
            np.testing.assert_array_equal(
                prices[count // 2 :],
                DETERMINISTIC_REFERENCE_PRICES[1],
            )

    def test_common_breakpoints_contains_every_candidate_edge(self):
        common = _union_breakpoints()
        for count in INTERVAL_COUNTS:
            for breakpoint in _edges(count)[1:-1]:
                self.assertTrue(np.any(common == breakpoint))

    def test_piecewise_resampling_preserves_two_price_policy(self):
        old_edges = _edges(2)
        old_prices = DETERMINISTIC_REFERENCE_PRICES

        for count in (4, 6):
            resampled = _sample_piecewise_prices(
                old_edges,
                old_prices,
                _edges(count),
            )
            expected = _lift_half_day_prices(old_prices, count)
            np.testing.assert_array_equal(resampled, expected)

    def test_odd_interval_count_is_rejected_for_half_day_lift(self):
        with self.assertRaises(ValueError):
            _lift_half_day_prices(
                DETERMINISTIC_REFERENCE_PRICES,
                5,
            )


if __name__ == "__main__":
    unittest.main()

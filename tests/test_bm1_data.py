"""BM1 regression tests grouped by responsibility."""

import numpy as np
import pandas as pd
import unittest
from src.method.bm_1_control import MonteCarloCandidateResult
from src.method.bm_1_objective import monte_carlo_tracking_objective
from src.tools.bm_1_data import (_complete_day_values, _hourly_hold, _mc_summary, build_day_signals,
    select_representative_days)

def _constant_day(series, day, value):
    index = pd.date_range(pd.Timestamp(day).normalize(), pd.Timestamp(day).normalize() + pd.Timedelta(days=1),
        freq='h', inclusive='left')
    series.loc[index] = float(value)

class RealDataDailyExperimentTests(unittest.TestCase):

    def test_history_baseline_is_hourwise_trimmed_mean(self):
        index = pd.date_range('2022-01-01', '2022-03-08', freq='h', inclusive='left')
        consumption = pd.Series(np.nan, index=index, dtype=float)
        price = pd.Series(np.nan, index=index, dtype=float)
        target_day = pd.Timestamp('2022-03-07')
        history_levels = [0.1, 0.2, 0.3, 0.4, 0.9]
        for week, level in enumerate(history_levels, start=1):
            _constant_day(consumption, target_day - pd.Timedelta(weeks=week), level)
        _constant_day(consumption, target_day, 0.55)
        _constant_day(price, target_day, 0.6)
        signals = build_day_signals(consumption, price, consumption, price, target_day, baseline_weeks=5)
        np.testing.assert_allclose(signals.baseline, 0.3)
        np.testing.assert_allclose(signals.target, 0.55)
        np.testing.assert_allclose(signals.observed_price, 0.6)

    def test_mc_summary_handles_candidate_wrapper(self):
        tracking = monte_carlo_tracking_objective(np.array([0.5, 0.5], dtype=float),
            np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]], dtype=float), np.array([0.15, 0.35], dtype=float))
        candidate = MonteCarloCandidateResult(prices=np.array([0.4, 0.6], dtype=float), objective=tracking,
            diagnostics={})
        summary = _mc_summary(candidate)
        self.assertAlmostEqual(summary['total'], tracking.objective.total)
        self.assertAlmostEqual(summary['standard_error'], tracking.standard_error)

    def test_incomplete_day_is_rejected(self):
        index = pd.date_range('2022-04-01', periods=24, freq='h')
        values = pd.Series(np.arange(24, dtype=float), index=index)
        values.iloc[5] = np.nan
        with self.assertRaises(ValueError):
            _complete_day_values(values, '2022-04-01', 'test signal')

    def test_hourly_hold_is_right_continuous(self):
        values = np.linspace(0.0, 1.0, 24)
        signal = _hourly_hold(values)
        breakpoint = 6.0 / 24.0
        query = np.array([np.nextafter(breakpoint, 0.0), breakpoint, np.nextafter(breakpoint, 1.0)])
        result = np.asarray(signal(query), dtype=float)
        self.assertAlmostEqual(result[0], values[5])
        self.assertAlmostEqual(result[1], values[6])
        self.assertAlmostEqual(result[2], values[6])

    def test_auto_selection_returns_distinct_consumption_regimes(self):
        index = pd.date_range('2022-01-01', '2022-05-01', freq='h', inclusive='left')
        consumption = pd.Series(index=index, dtype=float)
        price = pd.Series(index=index, dtype=float)
        day_number = (index.normalize() - pd.Timestamp('2022-01-01')).days.to_numpy()
        hour = index.hour.to_numpy()
        consumption[:] = 0.1 + 0.002 * day_number + 0.001 * hour
        price[:] = 0.25 + 0.002 * hour
        selected = select_representative_days(consumption, price, consumption, price, n_days=3, baseline_weeks=5)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        means = []
        for day in selected:
            signals = build_day_signals(consumption, price, consumption, price, day, baseline_weeks=5)
            means.append(float(np.mean(signals.target)))
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])


if __name__ == '__main__':
    unittest.main()

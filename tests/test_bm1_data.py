"""BM1 regression tests grouped by responsibility."""

import numpy as np
import pandas as pd
import unittest
from src.control.bm_1 import MonteCarloCandidateResult
from src.control.objective import control_penalties, monte_carlo_tracking_objective
from src.exe.run_model.bm_1.real_data import build_numerics, build_problem, run_day
from src.tools.bm_1_data import (
    DaySignals,
    HOURLY_EDGES,
    _complete_day_values,
    _hourly_hold,
    _mc_summary,
    build_day_signals,
    select_representative_days,
)

def _constant_day(series, day, value):
    index = pd.date_range(pd.Timestamp(day).normalize(), pd.Timestamp(day).normalize() + pd.Timedelta(days=1),
        freq='h', inclusive='left')
    series.loc[index] = float(value)

def _synthetic_day_signals():
    target = np.linspace(0.25, 0.55, 24, dtype=float)
    baseline = np.linspace(0.30, 0.50, 24, dtype=float)
    observed_price = np.linspace(0.20, 0.80, 24, dtype=float)
    day = pd.Timestamp('2022-05-17')
    return DaySignals(
        day=day,
        target=target,
        baseline=baseline,
        observed_price=observed_price,
        raw_target=100.0 * target,
        raw_price=1000.0 * observed_price,
        baseline_days=tuple(day - pd.Timedelta(weeks=i) for i in range(1, 6)),
    )


class F3GHistoricalContractTests(unittest.TestCase):

    def test_default_application_target_is_history_only_and_observed_mode_is_explicit(self):
        signals = _synthetic_day_signals()
        midpoints = 0.5 * (HOURLY_EDGES[:-1] + HOURLY_EDGES[1:])

        application = build_problem(signals)
        retrospective = build_problem(signals, target_mode='observed')

        np.testing.assert_allclose(application.target(midpoints), signals.baseline)
        np.testing.assert_allclose(retrospective.target(midpoints), signals.target)
        self.assertGreater(
            float(np.max(np.abs(application.target(midpoints) - signals.target))),
            1e-3,
        )
        with self.assertRaisesRegex(ValueError, 'target_mode'):
            build_problem(signals, target_mode='invalid')

    def test_historical_problem_owns_observed_prices_and_uses_hourly_contract(self):
        signals = _synthetic_day_signals()
        expected = signals.observed_price.copy()
        problem = build_problem(signals)

        np.testing.assert_array_equal(problem.observed_prices, expected)
        np.testing.assert_array_equal(problem.price_edges, HOURLY_EDGES)
        np.testing.assert_array_equal(problem.breakpoints, HOURLY_EDGES[1:-1])
        self.assertEqual(problem.n_intervals, 24)
        self.assertEqual(problem.deterministic_initial_state, 0.5)

        signals.observed_price[:] = 0.0
        np.testing.assert_array_equal(problem.observed_prices, expected)
        with self.assertRaises(ValueError):
            problem.observed_prices[0] = 0.5

        template = problem.parameters.parameter_template(problem.price_edges)
        np.testing.assert_array_equal(
            template[0](np.array([0.1, 0.9], dtype=float)),
            np.array([0.5, 0.5], dtype=float),
        )

    def test_historical_observed_evaluation_matches_direct_common_evaluator(self):
        signals = _synthetic_day_signals()
        problem = build_problem(signals)
        numerics = build_numerics(cells=21)

        attached = problem.evaluate_observed(numerics, fast=True)
        direct = problem.build_evaluator(numerics, fast=True).evaluate(problem.observed_prices)

        np.testing.assert_allclose(attached.cell_mass, direct.cell_mass, atol=0.0, rtol=0.0)
        self.assertAlmostEqual(attached.total, direct.total, places=15)
        self.assertAlmostEqual(attached.objective.bias, direct.objective.bias, places=15)
        self.assertAlmostEqual(attached.objective.variance, direct.objective.variance, places=15)

    def test_fixed_input_mode_skips_retrospective_optimizer(self):
        signals = _synthetic_day_signals()
        result = run_day(
            signals,
            cells=21,
            validate_mc=False,
            optimize=False,
            save=False,
        )

        self.assertIsNone(result.optimization)
        self.assertIsNone(result.fixed_paired_mc)
        self.assertIsNone(result.paired_mc)
        self.assertEqual(set(result.evaluations), {'observed_spot', 'neutral'})
        np.testing.assert_array_equal(result.problem.observed_prices, signals.observed_price)
        self.assertTrue(np.isfinite(result.evaluations['observed_spot'].total))
        self.assertTrue(np.isfinite(result.evaluations['neutral'].total))


    def test_fixed_input_mode_mc_contains_only_supplied_fixed_candidates(self):
        result = run_day(
            _synthetic_day_signals(),
            cells=21,
            mc_paths=50,
            validate_mc=True,
            optimize=False,
            save=False,
        )

        self.assertEqual(set(result.mc), {'observed_spot', 'neutral'})
        self.assertIsNotNone(result.fixed_paired_mc)
        self.assertIsNone(result.paired_mc)
        row = result.fixed_paired_mc
        self.assertTrue(np.isfinite(row['neutral_minus_observed_spot']))
        self.assertTrue(np.isfinite(row['standard_error']))
        self.assertTrue(np.isfinite(row['ratio']))
        for item in result.mc.values():
            self.assertEqual(item.path_objectives.shape, (50,))
            self.assertTrue(np.isfinite(item.total))
            self.assertTrue(np.isfinite(item.standard_error))


class HistoricalEstimationPreparationTests(unittest.TestCase):
    def test_five_weekday_arithmetic_mean_keeps_local_hour_and_no_future_history(self):
        from src.tools.bm_1_data import previous_weekday_mean
        index = pd.date_range('2022-02-01', '2022-05-01', freq='h', tz='Europe/Copenhagen')
        history = pd.Series(99., index=index)
        target = pd.Timestamp('2022-03-28 10:00', tz=index.tz)
        expected = [target-pd.DateOffset(weeks=k) for k in range(1,6)]
        history.loc[expected] = [.1,.2,.3,.4,.9]
        baseline, used = previous_weekday_mean(history, [target], origin=target.normalize())
        self.assertAlmostEqual(baseline.iloc[0], .38)
        self.assertEqual(used[0], tuple(expected))
        self.assertTrue(all(t.hour==10 and t.weekday()==0 for t in used[0]))
        history.loc[history.index>=target.normalize()] = -1000
        again, _ = previous_weekday_mean(history, [target], origin=target.normalize())
        np.testing.assert_array_equal(again, baseline)
        history.loc[expected[2]] = np.nan
        with self.assertRaisesRegex(ValueError, 'Five complete'):
            previous_weekday_mean(history, [target], origin=target.normalize())

    def test_longer_forecast_uses_only_pre_origin_weekday_occurrences(self):
        from src.tools.bm_1_data import previous_weekday_mean
        origin = pd.Timestamp('2022-05-02', tz='Europe/Copenhagen')
        history = pd.Series(1., index=pd.date_range('2022-03-01','2022-06-01',freq='h',tz=origin.tz))
        target = origin+pd.DateOffset(weeks=2, hours=12)
        _, sources = previous_weekday_mean(history, [target], origin=origin)
        self.assertTrue(all(t<origin for t in sources[0]))
        self.assertEqual(sources[0][0], pd.Timestamp('2022-04-25 12:00',tz=origin.tz))

    def test_point_adapter_preserves_dst_elapsed_time_missing_mask_and_unclipped_values(self):
        from src.tools.bm_1_data import EstimationPowerScale, historical_point_observations
        origin = pd.Timestamp('2022-10-30',tz='Europe/Copenhagen')
        times = pd.date_range(origin+pd.Timedelta(hours=1), periods=5, freq='h')
        data = pd.Series([5.,np.nan,12.,-1.,8.], index=times)
        obs, metadata = historical_point_observations(data, origin=origin,
            scale=EstimationPowerScale(10.,'kW'), measurement_id='declared cohort', observation_kind='point')
        np.testing.assert_array_equal(obs.times, np.arange(1,6)/24)
        np.testing.assert_array_equal(obs.observed, [True,False,True,True,True])
        np.testing.assert_array_equal(obs.values[[0,2,3,4]], [.5,1.2,-.1,.8])
        self.assertNotEqual(metadata['timestamps'][1],metadata['timestamps'][2])
        self.assertEqual(times[1].hour,times[2].hour)
        for kind in ('interval_mean','energy','unknown'):
            with self.assertRaises(NotImplementedError):
                historical_point_observations(data, origin=origin, scale=EstimationPowerScale(10.,'kW'),
                    measurement_id='cohort', observation_kind=kind)

    def test_training_scale_is_frozen_before_origin_and_naive_times_are_rejected(self):
        from src.tools.bm_1_data import EstimationPowerScale, historical_point_observations
        times = pd.date_range('2022-01-01', periods=8, freq='h',tz='UTC')
        data = pd.Series([1,2,3,4,1000,2000,3000,4000],index=times)
        scale = EstimationPowerScale.from_training(data,cutoff=times[4],unit='kW')
        self.assertEqual(scale.power,4.)
        with self.assertRaisesRegex(ValueError,'after the estimation origin'):
            historical_point_observations(data.iloc[5:],origin=times[3],scale=scale,
                                         measurement_id='cohort',observation_kind='point')
        with self.assertRaises(ValueError):
            historical_point_observations(data.set_axis(times.tz_localize(None)),origin=times[0],scale=scale,
                                         measurement_id='cohort',observation_kind='point')
        with self.assertRaises(ValueError):EstimationPowerScale(0.,'kW')
        with self.assertRaises(ValueError):EstimationPowerScale(1.,'kWh')


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
        prices = np.array([0.4, 0.6], dtype=float)
        penalties = control_penalties(
            np.array([0.4, 0.5, 0.6]),
            np.full(3, 1.0 / 3.0),
            prices,
        )
        candidate = MonteCarloCandidateResult(
            prices=prices,
            objective=tracking,
            diagnostics={},
            penalties=penalties,
            path_objectives=tracking.path_objectives.copy(),
            standard_error=tracking.standard_error,
        )
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

    def test_historical_count_inputs_are_not_silently_truncated(self):
        index = pd.date_range('2022-01-01', '2022-05-01', freq='h', inclusive='left')
        consumption = pd.Series(0.4, index=index, dtype=float)
        price = pd.Series(0.5, index=index, dtype=float)
        with self.assertRaises(ValueError):
            build_day_signals(
                consumption, price, consumption, price, '2022-03-07',
                baseline_weeks=5.5,
            )
        with self.assertRaises(ValueError):
            select_representative_days(
                consumption, price, consumption, price,
                n_days=2.5,
                baseline_weeks=5,
            )

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


class AuditSerializationTests(unittest.TestCase):

    def test_strict_json_replaces_nested_nonfinite_diagnostics_with_null(self):
        import json
        from src.tools.bm_1_data import _json_safe
        payload = {'correlation': float('nan'), 'values': np.array([1., np.inf]),
                   'nested': [np.float64('-inf'), {'ok': np.int64(3)}]}
        safe = _json_safe(payload)
        self.assertEqual(json.loads(json.dumps(safe, allow_nan=False)),
                         {'correlation': None, 'values': [1., None], 'nested': [None, {'ok': 3}]})


if __name__ == '__main__':
    unittest.main()

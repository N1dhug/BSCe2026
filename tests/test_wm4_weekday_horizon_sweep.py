import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    BASELINE_NAMES, baseline_from_history, corresponding_weekday_baseline, evaluate_configuration,
    historical_weekdays, load_tables, matched_cohort, report_only, save_top_three_plots, summarise,
)
from src.tools.impute_hourly import impute_hourly_days
from src.validation.exe.run_model.wm_4.sweep_weekday_horizon import sweep_weekday_horizon_validation


class fake_joint_fitter:
    def __init__(self, model): self.model = model

    def fit_many(self, datasets):
        n = len(datasets)
        return {
            "r": np.array([-0.8, -0.15, -1.4]),
            "init_cond": np.zeros((n, 2)),
            "qp_loss": 0.01, "deterministic_loss": 0.02,
            "design_rank": n+3, "beta": np.zeros(n+3),
        }


class test_wm4_weekday_horizon_sweep(unittest.TestCase):
    @staticmethod
    def sample_frame():
        idx = pd.date_range("2022-01-01", periods=24*95, freq="h")
        hours = idx.hour.to_numpy(dtype=float)
        days = np.arange(len(idx), dtype=float)/24
        demand = 0.5+0.08*np.cos(2*np.pi*hours/24)+0.03*np.cos(days/3)
        price = 0.25*np.sin(2*np.pi*hours/24+days/2)
        return pd.DataFrame({"I": demand, "U": price, "Temp": 5+5*np.cos(days/7)}, index=idx)

    def test_corresponding_weekday_baseline_is_exact_historical_mean(self):
        a = np.linspace(0.2, 0.6, 24)
        b = np.linspace(0.6, 0.2, 24)
        baseline = corresponding_weekday_baseline(np.vstack((a, b)))
        timestamps = pd.to_datetime(["2022-04-04 00:00", "2022-04-04 00:30", "2022-04-04 23:30"])
        expected = np.array([0.4, 0.4, 0.4])
        np.testing.assert_allclose(baseline.predict(timestamps, np.zeros(3)), expected)
        self.assertTrue(np.allclose(baseline.profile, 0.4))

    def test_historical_profiles_are_strictly_prior_and_matched(self):
        frame = self.sample_frame()
        day = pd.Timestamp("2022-03-26")
        dates = historical_weekdays(day, 10)
        validation = sweep_weekday_horizon_validation()
        self.assertEqual(len(dates), 10)
        self.assertTrue(all(past < day and past.dayofweek == day.dayofweek for past in dates))
        values = validation.training_data(frame, dates)
        self.assertEqual(values.shape, (10, 24))
        self.assertAlmostEqual(values[-1, 7], frame.loc[dates[-1]+pd.Timedelta(hours=7), "I"])

    def test_single_joint_fit_reused_for_multiple_oracle_horizons(self):
        frame = self.sample_frame()
        day = pd.Timestamp("2022-03-12")
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params", fake_joint_fitter):
            fit, score, component, states = evaluate_configuration(frame, day, 3, 4, 6, 24)
            changed = frame.copy()
            changed.loc[day+pd.Timedelta(hours=20), "I"] += 0.3
            _, changed_scores, _, _ = evaluate_configuration(changed, day, 3, 4, 6, 24)
        self.assertEqual(len(score), 19)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(component), 24)
        self.assertAlmostEqual(fit["theta_U"], -0.15)
        self.assertAlmostEqual(fit["theta_R"], -1.4)
        self.assertAlmostEqual(score[0]["oracle_F0"], changed_scores[0]["oracle_F0"])
        self.assertAlmostEqual(score[0]["rmse"], changed_scores[0]["rmse"])
        self.assertNotAlmostEqual(score[-1]["rmse"], changed_scores[-1]["rmse"])
        first = score[0]
        self.assertIn("oracle_R0",first)
        self.assertAlmostEqual(score[0]["oracle_R0"],changed_scores[0]["oracle_R0"])
        forecast = np.array([row["prediction_at_initial_zero"]+
                             first["oracle_F0"]*row["sensitivity_F0"]+
                             first["oracle_R0"]*row["sensitivity_R0"] for row in component[:6]])
        truth = np.array([row["observed"] for row in component[:6]])
        self.assertAlmostEqual(np.sqrt(np.mean((forecast-truth)**2)), first["rmse"])

    def test_pooled_ranking_uses_mse_and_requires_matched_days(self):
        days = pd.to_datetime(["2022-03-12", "2022-03-13", "2022-03-14"])
        fits = pd.DataFrame([{"day": day, "training_weekdays": n}
                             for day in days[:2] for n in (2, 3)])
        scores = pd.DataFrame([{"day": day, "training_weekdays": n, "horizon_hours": h,
                                "rmse": float(i+1), "mse": float((i+1)**2),
                                "baseline_rmse": 3.0, "baseline_mse": 9.0, "bias": 0.0}
                               for i, day in enumerate(days[:2]) for n in (2, 3) for h in (6, 7)])
        tables = {"fit": fits, "score": scores}
        cohort = matched_cohort(tables, (2, 3), 2)
        self.assertEqual(list(cohort), list(days[:2]))
        ranking = summarise(scores)
        self.assertAlmostEqual(ranking.iloc[0]["aggregate_rmse"], np.sqrt(2.5))
        missing = scores.drop(scores[(scores["day"] == days[0]) &
                                     (scores["training_weekdays"] == 3) &
                                     (scores["horizon_hours"] == 7)].index)
        self.assertEqual(list(matched_cohort({"fit": fits, "score": missing}, (2, 3), 2)), [days[1]])

    def test_top_three_boxplot_does_not_pass_version_specific_label_keyword(self):
        day = pd.Timestamp("2022-03-12")
        score = pd.DataFrame([{"day": day+pd.Timedelta(days=i), "baseline": "weekly_average", "training_weekdays": 2,
                               "horizon_hours": 6, "rmse": .1+i*.01} for i in range(3)])
        top = pd.DataFrame([{"rank": 1, "baseline": "weekly_average", "training_weekdays": 2, "horizon_hours": 6}])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("matplotlib.axes.Axes.boxplot") as boxplot:
                save_top_three_plots(score, top, pd.DatetimeIndex(score["day"]), Path(tmp))
                self.assertTrue(boxplot.called)
                self.assertNotIn("labels", boxplot.call_args.kwargs)
                self.assertNotIn("tick_labels", boxplot.call_args.kwargs)
            self.assertTrue((Path(tmp)/"top_three_rmse_distribution.png").is_file())

    def test_absolute_aggregate_error_wins_over_relative_baseline_improvement(self):
        # N=2 predicts better in absolute terms, even though N=3 has more baseline skill.
        data = pd.DataFrame([
            {"day": pd.Timestamp("2022-03-12"), "training_weekdays": 2, "horizon_hours": 6,
             "rmse": .10, "mse": .01, "baseline_rmse": .11,
             "baseline_mse": .11**2, "bias": 0.0},
            {"day": pd.Timestamp("2022-03-12"), "training_weekdays": 3, "horizon_hours": 6,
             "rmse": .15, "mse": .15**2, "baseline_rmse": .30,
             "baseline_mse": .30**2, "bias": 0.0},
        ])
        ranking = summarise(data)
        self.assertEqual(int(ranking.iloc[0]["training_weekdays"]), 2)
        self.assertLess(float(ranking.iloc[0]["aggregate_rmse"]), float(ranking.iloc[1]["aggregate_rmse"]))
        self.assertLess(float(ranking.iloc[0]["baseline_improvement_pct"]),
                        float(ranking.iloc[1]["baseline_improvement_pct"]))

    def test_hourly_imputer_fills_short_gaps_without_making_full_days(self):
        dates = pd.date_range("2022-03-01", periods=4*24, freq="h")
        data = pd.DataFrame({"I": np.r_[np.arange(24)*.01, np.arange(24)*0+.9,
                                       np.arange(24)*0+.7, np.arange(24)*.01+.2],
                             "U": np.r_[np.arange(24)*.02, np.arange(24)*0+.4,
                                       np.arange(24)*0+.6, np.arange(24)*.02],
                             "Temp": np.arange(96, dtype=float)}, index=dates)
        missing_day = pd.Timestamp("2022-03-02")
        data = data.loc[~(data.index.normalize() == missing_day)].copy()
        data = data.drop(pd.Timestamp("2022-03-01 06:00"))
        data.loc[pd.Timestamp("2022-03-03 07:00"), "I"] = np.nan
        data.loc[pd.Timestamp("2022-03-03 09:00"), "U"] = np.nan
        data.loc[pd.Timestamp("2022-03-03 10:00"), "Temp"] = np.nan
        filled, report = impute_hourly_days(
            data, days=pd.date_range("2022-03-01", "2022-03-04", freq="D"),
            columns=("I", "U", "Temp"))
        self.assertEqual(len(filled), 72)
        self.assertNotIn(missing_day, filled.index.normalize())
        self.assertEqual(report["missing_days"], ["2022-03-02"])
        self.assertEqual(report["imputed_hourly_slots"], 4)
        self.assertAlmostEqual(filled.at[pd.Timestamp("2022-03-01 06:00"), "I"], .06)
        self.assertAlmostEqual(filled.at[pd.Timestamp("2022-03-03 09:00"), "U"], .6)
        self.assertAlmostEqual(filled.at[pd.Timestamp("2022-03-03 10:00"), "Temp"], 58.0)
        with self.assertRaisesRegex(ValueError, "Missing full day"):
            impute_hourly_days(data, days=pd.date_range("2022-03-01", "2022-03-04", freq="D"),
                               columns=("I", "U", "Temp"), require_days=True)

    def test_baseline_fits_impute_hours_and_never_fill_whole_days(self):
        frame = self.sample_frame()
        origin = pd.Timestamp("2022-03-12")
        history = historical_weekdays(origin, 2)
        fragment = pd.concat([frame.reindex(pd.date_range(day, periods=24, freq="h"))
                              for day in history]).sort_index()
        fragment = fragment.drop(history[0]+pd.Timedelta(hours=6))
        fragment.loc[history[1]+pd.Timedelta(hours=10), "Temp"] = np.nan
        for name in BASELINE_NAMES:
            model = baseline_from_history(fragment, history, name)
            self.assertEqual(model.metadata["samples"], 48)
            self.assertEqual(model.metadata["imputation"]["imputed_hourly_slots"], 2)
        incomplete = fragment.loc[fragment.index.normalize() != history[0]]
        with self.assertRaisesRegex(ValueError, "Missing full day"):
            impute_hourly_days(incomplete, days=history, columns=("I", "U", "Temp"),
                               require_days=True)

    def test_sweep_imputes_historical_hours_but_never_target_labels(self):
        original = self.sample_frame()
        day = pd.Timestamp("2022-03-12")
        history = historical_weekdays(day, 2)
        altered = original.drop(history[0]+pd.Timedelta(hours=8)).copy()
        altered.loc[history[1]+pd.Timedelta(hours=11), "I"] = np.nan
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params", fake_joint_fitter):
            fit, scores, components, states = evaluate_configuration(
                altered, day, 2, 2, 6, 8, "weekly_average")
            self.assertEqual(fit["imputed_training_hours"], 2)
            self.assertEqual(len(scores), 3)
            self.assertEqual(len(states), 2)
            # Never fabricate forecast labels to create optimistic oracle scores.
            altered.loc[day+pd.Timedelta(hours=7), "I"] = np.nan
            with self.assertRaisesRegex(ValueError, "Missing consumption"):
                evaluate_configuration(altered, day, 2, 2, 6, 8, "weekly_average")

    def test_all_three_baselines_use_exactly_corresponding_weekday_data(self):
        frame = self.sample_frame()
        day = pd.Timestamp("2022-03-12")
        history = historical_weekdays(day, 2)
        self.assertEqual(set(BASELINE_NAMES),
                         {"statistical_output", "weekly_average", "weekly_temperature"})
        selected = pd.concat([frame.reindex(pd.date_range(past, periods=24, freq="h"))
                              for past in history])
        for name in BASELINE_NAMES:
            baseline = baseline_from_history(frame, history, name)
            self.assertEqual(baseline.metadata["samples"], 48)
            self.assertEqual(baseline.metadata["fit_end"], selected.index.max())
            reference = baseline.predict(pd.date_range(day, periods=24, freq="h"),
                                         frame.loc[day:day+pd.Timedelta(hours=23), "Temp"].to_numpy())
            self.assertEqual(reference.shape, (24,))
            self.assertTrue(np.isfinite(reference).all())
            corrupted = frame.copy()
            corrupted.loc[day:day+pd.Timedelta(hours=23), "I"] += 0.2
            other = baseline_from_history(corrupted, history, name)
            np.testing.assert_allclose(other.coef, baseline.coef)
        self.assertEqual(len(history), 2)

    def test_three_baseline_rank_uses_absolute_error_on_matched_days(self):
        day = pd.Timestamp("2022-03-12")
        baselines = BASELINE_NAMES
        ns = (2, 3)
        fit = pd.DataFrame([{"day": day, "baseline": b, "training_weekdays": n}
                            for b in baselines for n in ns])
        rows = []
        for b in baselines:
            for n in ns:
                for h in (6, 7):
                    error = .1 if b == "weekly_average" else .2 if b == "weekly_temperature" else .3
                    rows.append({"day": day, "baseline": b, "training_weekdays": n,
                                 "horizon_hours": h, "rmse": error, "mse": error**2,
                                 "baseline_rmse": error+.1, "baseline_mse": (error+.1)**2,
                                 "bias": 0.0})
        scores = pd.DataFrame(rows)
        self.assertEqual(list(matched_cohort({"fit": fit, "score": scores}, ns, 2, baselines)), [day])
        ranking = summarise(scores)
        self.assertEqual(ranking.iloc[0]["baseline"], "weekly_average")
        self.assertEqual(ranking.iloc[0]["aggregate_rmse"], .1)
        missing = scores.loc[~((scores["baseline"] == "statistical_output") &
                               (scores["training_weekdays"] == 3) &
                               (scores["horizon_hours"] == 7))]
        self.assertTrue(matched_cohort({"fit": fit, "score": missing}, ns, 2, baselines).empty)

    def test_all_three_baselines_expose_own_parameters_in_configurations(self):
        frame = self.sample_frame()
        day = pd.Timestamp("2022-03-12")
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params", fake_joint_fitter):
            for name in BASELINE_NAMES:
                fit, scores, components, states = evaluate_configuration(frame, day, 2, 2, 6, 8, name)
                self.assertEqual(fit["baseline"], name)
                self.assertEqual(fit["baseline_samples"], 48)
                self.assertEqual(len(scores), 3)
                self.assertTrue(all(item["baseline"] == name for item in scores))
                self.assertTrue(all(item["baseline"] == name and "baseline_value" in item
                                    for item in components))
                self.assertEqual(len(states), 2)

    def test_saved_reporting_produces_top_three_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            days = pd.date_range("2022-03-12", periods=3, freq="D")
            fits = [{"day": day, "training_weekdays": n,
                     "theta_F": -0.7, "theta_R": -1.5, "theta_U": -0.2,
                     "fit_qp_loss": .01, "fit_seconds": .02} for day in days for n in (2, 3)]
            scores = [{"day": day, "training_weekdays": n, "horizon_hours": h,
                       "rmse": .05+.01*n+.001*h, "mse": (.05+.01*n+.001*h)**2,
                       "baseline_rmse": .12, "baseline_mse": .12**2,
                       "bias": 0.0} for day in days for n in (2, 3) for h in (6, 7)]
            pd.DataFrame(fits).to_csv(output/"parameter_estimates.csv", index=False)
            pd.DataFrame(scores).to_csv(output/"oracle_horizon_scores.csv", index=False)
            ranking, top = report_only(output, (2, 3), 2)
            self.assertEqual(len(top), 3)
            self.assertEqual(len(ranking), 4)
            self.assertEqual(top.iloc[0]["training_weekdays"], 2)
            self.assertEqual(top.iloc[0]["horizon_hours"], 6)
            self.assertTrue((output/"top_three.csv").is_file())
            self.assertTrue((output/"rmse_heatmap.png").is_file())
            self.assertTrue((output/"top_three_rmse_distribution.png").is_file())
            tables = load_tables(output)
            self.assertEqual(len(tables["fit"]), 6)


if __name__ == "__main__": unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    BASELINE_NAMES, baseline_from_history, evaluate_configuration,
    historical_weekdays, historical_calendar_days, hourly_origins, load_tables, matched_cohort,
    report_only, save_top_three_plots, summarise,
)
from src.tools.impute_hourly import impute_hourly_days
from src.validation.exe.run_model.wm_4.sweep_weekday_horizon import sweep_weekday_horizon_validation


class fake_joint_fitter:
    seen = []
    def __init__(self, model): self.model = model

    def fit_many(self, datasets):
        n = len(datasets)
        self.seen.append((n, self.model.t_steps, tuple(len(d["I"]) for d in datasets)))
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

    def test_hourly_origin_grid_and_last_complete_horizon(self):
        frame = self.sample_frame()
        starts = hourly_origins(frame, max_weekdays=3, max_horizon=6,
                                start_day="2022-03-12", end_day="2022-03-13",
                                max_origins=6)
        self.assertEqual(len(starts), 6)
        self.assertEqual(starts[0], pd.Timestamp("2022-03-12 00:00"))
        self.assertEqual(starts[-1], pd.Timestamp("2022-03-12 05:00"))
        self.assertTrue(all((starts[j+1]-starts[j]) == pd.Timedelta(hours=1)
                            for j in range(len(starts)-1)))
        # The final six-hour origin requires the observation at hour +6.
        last_day = frame.index.max().normalize()
        last = hourly_origins(frame, max_weekdays=3, max_horizon=6,
                              start_day=last_day, end_day=last_day)
        self.assertEqual(last[-1], frame.index.max()-pd.Timedelta(hours=6))
        # For 24 hours only samples at offsets 0..23 are required.
        last_24 = hourly_origins(frame, max_weekdays=3, max_horizon=24,
                                 start_day=last_day, end_day=last_day)
        self.assertEqual(last_24.tolist(), [last_day])

    def test_hourly_origin_historical_intervals_cross_midnight(self):
        day = pd.Timestamp("2022-03-12 21:00")
        history = historical_weekdays(day, 3)
        dates = historical_calendar_days(history, 6)
        self.assertEqual(len(dates), 6)
        self.assertEqual(history[-1], pd.Timestamp("2022-03-05 21:00"))
        self.assertEqual(dates[-1], pd.Timestamp("2022-03-06"))
        frame = self.sample_frame()
        fake_joint_fitter.seen = []
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params", fake_joint_fitter):
            fits, scores, components, states = evaluate_configuration(
                frame, day, 3, 4, 6, 6, "weekly_average")
        self.assertEqual(len(fits), 1)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(components), 6)
        self.assertEqual(fake_joint_fitter.seen, [(3, 25, (25, 25, 25))])
        self.assertEqual(fits[0]["baseline_samples"], 3*6)
        self.assertEqual(fits[0]["origin"], day)
        self.assertEqual(scores[0]["origin"], day)
        self.assertEqual(components[3]["time"], pd.Timestamp("2022-03-13 00:00"))
        self.assertTrue(all(row["historical_day"].hour == 21 for row in states))
        self.assertTrue(all(pd.Timestamp(value).hour == 21 for value in
                            fits[0]["training_days"].split(";")))

    def test_joint_drift_and_initial_states_are_specific_to_both_N_and_k(self):
        frame = self.sample_frame()
        day = pd.Timestamp("2022-03-12")
        fake_joint_fitter.seen = []
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params", fake_joint_fitter):
            fits, scores, components, states = evaluate_configuration(frame, day, 6, 4, 6, 8)
            changed = frame.copy()
            changed.loc[day+pd.Timedelta(hours=20), "I"] += 0.3
            _, changed_scores, _, _ = evaluate_configuration(changed, day, 6, 4, 6, 8)
        self.assertEqual(len(fits), 3)
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(states), 6*3)
        self.assertEqual(len(components), 6+7+8)
        self.assertEqual({fit["horizon_hours"] for fit in fits}, {6,7,8})
        self.assertTrue(all(fit["training_weekdays"] == 6 for fit in fits))
        self.assertEqual(fake_joint_fitter.seen[:3], [
            (6, 6*4+1, (6*4+1,)*6),
            (6, 7*4+1, (7*4+1,)*6),
            (6, 8*4+1, (8*4+1,)*6),
        ])
        self.assertTrue(all(len([r for r in states if r["horizon_hours"] == k]) == 6
                            for k in (6,7,8)))
        self.assertEqual([fit["baseline_samples"] for fit in fits], [6*6,6*7,6*8])
        self.assertAlmostEqual(fits[0]["theta_U"], -0.15)
        self.assertAlmostEqual(fits[0]["theta_R"], -1.4)
        self.assertAlmostEqual(scores[0]["rmse"], changed_scores[0]["rmse"])
        self.assertAlmostEqual(scores[-1]["rmse"], changed_scores[-1]["rmse"])
        first = scores[0]
        forecast = np.array([row["prediction_at_initial_zero"]+
                             first["oracle_F0"]*row["sensitivity_F0"]+
                             first["oracle_R0"]*row["sensitivity_R0"] for row in components[:6]])
        truth = np.array([row["observed"] for row in components[:6]])
        self.assertAlmostEqual(np.sqrt(np.mean((forecast-truth)**2)), first["rmse"])

    def test_expanded_fourteen_weekdays_and_one_to_twentyfour_hours(self):
        frame=self.sample_frame()
        # Need 14 complete weekly lookbacks, equivalent to 98 days.
        extended=pd.date_range("2022-01-01",periods=24*150,freq="h")
        hours=extended.hour.to_numpy(dtype=float)
        days=np.arange(len(extended),dtype=float)/24
        frame=pd.DataFrame({
            "I":.5+.08*np.cos(2*np.pi*hours/24)+.03*np.cos(days/3),
            "U":.25*np.sin(2*np.pi*hours/24+days/2),
            "Temp":5+5*np.cos(days/7),
        },index=extended)
        day=pd.Timestamp("2022-05-21")
        history=historical_weekdays(day,14)
        self.assertEqual(len(history),14)
        self.assertEqual(history[0],day-pd.Timedelta(weeks=14))
        validation=sweep_weekday_horizon_validation()
        self.assertEqual(validation.training_data(frame,history).shape,(14,24))
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params",fake_joint_fitter):
            fit,scores,components,states=evaluate_configuration(
                frame,day,14,4,1,24,"weekly_average")
        self.assertEqual(len(scores),24)
        self.assertEqual(len(components),sum(range(1,25)))
        self.assertEqual(len(states),14*24)
        self.assertEqual(len(fit),24)
        self.assertEqual(scores[0]["horizon_hours"],1)
        self.assertEqual(scores[-1]["horizon_hours"],24)
        self.assertEqual(scores[0]["oracle_state_design_rank"],1)
        self.assertEqual(scores[0]["oracle_state_observations"],1)
        self.assertAlmostEqual(scores[0]["rmse"],0.0,delta=2e-5)
        self.assertTrue(all(-1<=row["oracle_F0"]<=1 and -1<=row["oracle_R0"]<=1
                            for row in scores))
        self.assertTrue(all(np.isfinite(row["rmse"]) for row in scores))
        self.assertEqual([r["baseline_samples"] for r in fit], [14*k for k in range(1,25)])

    def test_expanded_range_validation_rejects_out_of_bounds(self):
        from types import SimpleNamespace
        good=SimpleNamespace(group="MED",baselines=list(BASELINE_NAMES),
                             min_weekdays=2,max_weekdays=14,
                             min_horizon=1,max_horizon=24,
                             steps_per_hour=4,max_days=0,progress_every=1,
                             start_day=None,end_day=None)
        sweep_weekday_horizon_validation.arguments(good)
        for name,value in (("min_weekdays",1),("max_weekdays",15),
                           ("min_horizon",0),("max_horizon",25)):
            data=vars(good).copy()
            data[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):
                sweep_weekday_horizon_validation.arguments(SimpleNamespace(**data))

    def test_pooled_ranking_uses_mse_and_requires_matched_days(self):
        days = pd.to_datetime(["2022-03-12", "2022-03-13", "2022-03-14"])
        fits = pd.DataFrame([{"day": day, "training_weekdays": n, "horizon_hours": h}
                             for day in days[:2] for n in (2, 3) for h in (6, 7)])
        scores = pd.DataFrame([{"day": day, "training_weekdays": n, "horizon_hours": h,
                                "rmse": float(i+1), "mse": float((i+1)**2),
                                "baseline_rmse": 3.0, "baseline_mse": 9.0, "bias": 0.0}
                               for i, day in enumerate(days[:2]) for n in (2, 3) for h in (6, 7)])
        tables = {"fit": fits, "score": scores}
        cohort = matched_cohort(tables, (2, 3), 2)
        self.assertEqual(list(cohort), list(days[:2]))
        incomplete_fit = fits.loc[~((fits["day"] == days[0]) &
                                    (fits["training_weekdays"] == 3) &
                                    (fits["horizon_hours"] == 7))]
        self.assertEqual(list(matched_cohort({"fit": incomplete_fit, "score": scores},
                                              (2, 3), 2)), [days[1]])
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

    def test_baseline_requires_filled_selected_historical_hours_only(self):
        frame = self.sample_frame()
        origin = pd.Timestamp("2022-03-12")
        history = historical_weekdays(origin, 2)
        fragment = pd.concat([frame.reindex(pd.date_range(day, periods=24, freq="h"))
                              for day in history]).sort_index()
        fragment = fragment.drop(history[0]+pd.Timedelta(hours=6))
        fragment.loc[history[1]+pd.Timedelta(hours=10), "Temp"] = np.nan
        for name in BASELINE_NAMES:
            with self.assertRaisesRegex(ValueError, "complete"):
                baseline_from_history(fragment, history, name, 12)
        filled, report = impute_hourly_days(fragment, days=history, columns=("I","U","Temp"), require_days=True)
        self.assertEqual(report["imputed_hourly_slots"], 2)
        for name in BASELINE_NAMES:
            baseline = baseline_from_history(filled, history, name, 12)
            self.assertEqual(baseline.metadata["samples"], 24)
            self.assertEqual(baseline.metadata["imputation"]["method"], "exact_selected_historical_hours")
        incomplete = fragment.loc[fragment.index.normalize() != history[0]]
        with self.assertRaisesRegex(ValueError, "Missing full day"):
            impute_hourly_days(incomplete, days=history, columns=("I","U","Temp"), require_days=True)

    def test_sweep_imputes_historical_hours_but_never_target_labels(self):
        original = self.sample_frame()
        day = pd.Timestamp("2022-03-12")
        history = historical_weekdays(day, 2)
        altered = original.drop(history[0]+pd.Timedelta(hours=8)).copy()
        altered.loc[history[1]+pd.Timedelta(hours=11), "I"] = np.nan
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params", fake_joint_fitter):
            fit, scores, components, states = evaluate_configuration(
                altered, day, 2, 2, 6, 8, "weekly_average")
            self.assertTrue(all(r["imputed_training_hours"] == 2 for r in fit))
            self.assertEqual(len(scores), 3)
            self.assertEqual(len(states), 2*3)
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
        fit = pd.DataFrame([{"day": day, "baseline": b, "training_weekdays": n, "horizon_hours": h}
                            for b in baselines for n in ns for h in (6, 7)])
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
                self.assertTrue(all(item["baseline"] == name for item in fit))
                self.assertEqual([item["baseline_samples"] for item in fit], [12,14,16])
                self.assertEqual(len(scores), 3)
                self.assertTrue(all(item["baseline"] == name for item in scores))
                self.assertTrue(all(item["baseline"] == name and "baseline_value" in item
                                    for item in components))
                self.assertEqual(len(states), 2*3)

    def test_mixed_midnight_and_hourly_csv_timestamps_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            pd.DataFrame([
                {"day":"2022-04-09", "origin":"2022-04-09", "baseline":"weekly_average",
                 "training_weekdays":2, "horizon_hours":6},
                {"day":"2022-04-09 01:00:00", "origin":"2022-04-09 01:00:00",
                 "baseline":"weekly_average", "training_weekdays":2, "horizon_hours":6},
            ]).to_csv(output/"parameter_estimates.csv", index=False)
            frame = load_tables(output)["fit"]
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame["day"].tolist(),
                             [pd.Timestamp("2022-04-09"), pd.Timestamp("2022-04-09 01:00")])
            self.assertEqual(frame["origin"].tolist(), frame["day"].tolist())

    def test_saved_reporting_produces_top_three_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            days = pd.date_range("2022-03-12", periods=3, freq="D")
            fits = [{"day": day, "training_weekdays": n, "horizon_hours": h,
                     "theta_F": -0.7, "theta_R": -1.5, "theta_U": -0.2,
                     "fit_qp_loss": .01, "fit_seconds": .02}
                    for day in days for n in (2,3) for h in (6,7)]
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
            self.assertTrue((output/"top_three_h6plus.csv").is_file())
            self.assertTrue((pd.read_csv(output/"top_three_h6plus.csv")["horizon_hours"]>=6).all())
            self.assertTrue((output/"rmse_heatmap.png").is_file())
            self.assertTrue((output/"top_three_rmse_distribution.png").is_file())
            tables = load_tables(output)
            self.assertEqual(len(tables["fit"]), 12)


if __name__ == "__main__": unittest.main()

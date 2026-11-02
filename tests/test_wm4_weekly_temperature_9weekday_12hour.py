import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour import (
    BASELINE, H, N, ORIGINS, evaluate_day, fit_origin, load_sweep_params,
    plot_origins, report, summarize,
)
from src.model.wm_4 import wm_4


class constant_baseline:
    def predict(self, times, temperature): return np.full(len(times),.5,dtype=float)


class test_wm4_weekly_temperature_9weekday_12hour(unittest.TestCase):
    @staticmethod
    def example_frame():
        day=pd.Timestamp("2022-05-07")
        days=[day-pd.Timedelta(weeks=w) for w in range(N,0,-1)]+[day]
        values=[]
        for d in days:
            for hour in range(24):
                price=.25*np.sin(2*np.pi*hour/24)
                demand=.5+.055*np.cos(2*np.pi*hour/24)+.004*d.weekday()
                values.append((d+pd.Timedelta(hours=hour),demand,price,12.))
        frame=pd.DataFrame(values,columns=["time","I","U","Temp"]).set_index("time").sort_index()
        return frame,day

    @staticmethod
    def artificial_errors(days=3):
        daily,hourly=[],[]
        for k in range(days):
            day=pd.Timestamp("2022-04-01")+pd.Timedelta(days=k)
            for origin in ORIGINS:
                error=(k+1)*.02+(origin/100)  # Different origin, no hidden shared forecast.
                errors=np.array([error]+[0.]*(H-1))
                baseline=np.full(H,.10+origin/200)
                daily.append({
                    "day":day,"origin_hour":origin,"oracle_F0":.10+origin/100,
                    "oracle_R0":-.02-origin/100,
                    "rmse":float(np.sqrt(np.mean(errors**2))),
                    "mse":float(np.mean(errors**2)),
                    "baseline_mse":float(np.mean(baseline**2)),
                })
                for lead in range(H):
                    hourly.append({
                        "day":day,"origin_hour":origin,"lead_hour":lead,
                        "absolute_hour":origin+lead,"hour_end":origin+lead+1,
                        "error":errors[lead],"baseline_error":baseline[lead],
                    })
        return pd.DataFrame(daily),pd.DataFrame(hourly)

    def test_twelve_distinct_origins_and_absolute_clock_hours(self):
        self.assertEqual(len(ORIGINS),12)
        self.assertEqual(tuple(ORIGINS),tuple(range(12)))
        daily,hourly=self.artificial_errors()
        self.assertEqual(len(daily),3*12)
        self.assertEqual(len(hourly),3*12*H)
        last=hourly.loc[hourly["origin_hour"]==11]
        self.assertEqual(sorted(last["absolute_hour"].unique().tolist()),list(range(11,23)))
        self.assertEqual(sorted(last["hour_end"].unique().tolist()),list(range(12,24)))

    def test_pooled_cumulative_rmse_per_origin(self):
        daily,hourly=self.artificial_errors()
        cumulative,final=summarize(daily,hourly)
        self.assertEqual(len(cumulative),12*H)
        self.assertEqual(len(final),12)
        self.assertTrue((final["days"]==3).all())
        self.assertEqual(final["samples"].tolist(),[3*H]*12)
        for origin in ORIGINS:
            prefix=cumulative.loc[cumulative["origin_hour"]==origin]
            values=daily.loc[daily["origin_hour"]==origin]
            total=np.sqrt(np.mean(values["mse"]))
            self.assertAlmostEqual(prefix.iloc[-1]["cumulative_rmse"],total)
            self.assertEqual(prefix["elapsed_hours"].tolist(),list(range(1,H+1)))
            self.assertEqual(prefix["hour_end"].tolist(),list(range(origin+1,origin+H+1)))
            # Only lead zero has an error: subsequent averages decrease as 1/sqrt(h).
            np.testing.assert_allclose(prefix["cumulative_rmse"].to_numpy(),
                                       prefix.iloc[0]["cumulative_rmse"] /
                                       np.sqrt(np.arange(1,H+1)))
            self.assertAlmostEqual(prefix.iloc[-1]["baseline_cumulative_rmse"],
                                   float(np.sqrt(np.mean(values["baseline_mse"]))))
            self.assertTrue((prefix["cumulative_squared_error"].diff().dropna()>=0).all())
        self.assertNotEqual(cumulative.loc[cumulative["origin_hour"]==0,
                                          "cumulative_rmse"].iloc[-1],
                            cumulative.loc[cumulative["origin_hour"]==11,
                                           "cumulative_rmse"].iloc[-1])

    def test_rejects_missing_origins_and_missing_lead_hour(self):
        daily,hourly=self.artificial_errors()
        with self.assertRaises(ValueError):
            summarize(daily.loc[daily["origin_hour"]!=11],hourly)
        with self.assertRaises(ValueError):
            summarize(daily,hourly.iloc[:-1])
        with self.assertRaises(ValueError):
            summarize(daily,hourly.iloc[1:])
        with self.assertRaises(ValueError):
            summarize(daily.drop(columns="oracle_R0"),hourly)

    def test_full_state_oracle_can_recover_both_initial_states(self):
        sph=4
        model=wm_4((0,H/24),H*sph+1,lambda t:0.)
        t=np.arange(H*sph+1)/sph
        U=np.full(len(t),.10)
        O=np.full(len(t),.50)
        theta=np.array([-3.,-.20,-5.])
        true=np.array([.15,-.04])
        state=model._solve_deterministic(true,U,U,t,O,theta)
        actual=np.asarray(state["I"])[np.arange(H)*sph]
        data={"I":state["I"],"U":U,"PE":U,"Time":t,"O":O}
        estimate,predicted,baseline=fit_origin(model,data,actual,theta,sph)
        np.testing.assert_allclose(estimate,true,atol=1e-6)
        np.testing.assert_allclose(predicted,actual,atol=1e-8)
        np.testing.assert_allclose(baseline,.50)

    def test_one_historical_parameter_fit_and_twelve_independent_oracles(self):
        frame,day=self.example_frame()
        theta=np.array([-3.,-.2,-5.])
        with patch("src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour.baseline_from_history",
                   return_value=constant_baseline()),patch(
                "src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour.joint_qp_params") as fitter:
            fitter.return_value.fit_many.return_value={"r":theta,"qp_loss":.001}
            daily,hourly=evaluate_day(frame,day)
            fitter.return_value.fit_many.assert_called_once()
            self.assertEqual(len(daily),12)
            self.assertEqual(len(hourly),12*H)
            self.assertTrue(all("oracle_R0" in row and "oracle_F0" in row for row in daily))
            self.assertEqual(set(row["origin_hour"] for row in daily),set(ORIGINS))
            self.assertEqual({row["hour_end"] for row in hourly if row["origin_hour"]==11},
                             set(range(12,24)))
            # With saved theta, DO NOT refit historical parameters.
            reference={"training_weekdays":N,"baseline":BASELINE,
                       "theta_F":theta[0],"theta_U":theta[1],"theta_R":theta[2]}
            again,points=evaluate_day(frame,day,reference=reference)
            fitter.return_value.fit_many.assert_called_once()
        self.assertEqual(len(again),12)
        self.assertEqual(len(points),12*H)
        self.assertAlmostEqual(again[11]["rmse"],daily[11]["rmse"])
        self.assertTrue(all(row["horizon_hours"]==H for row in daily))

    def test_saved_theta_reuse_and_twelve_panel_report(self):
        daily,hourly=self.artificial_errors()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            root.joinpath("origin_daily_metrics.csv").write_text(daily.to_csv(index=False))
            root.joinpath("origin_hourly_predictions.csv").write_text(hourly.to_csv(index=False))
            curve,final=report(root)
            self.assertEqual(len(curve),144)
            self.assertEqual(len(final),12)
            for name in ("origin_cumulative_rmse.csv","origin_aggregate_rmse.csv",
                         "origin_cumulative_rmse_panels.png"):
                self.assertTrue((root/name).is_file(),name)
            source=root/"sweep";source.mkdir()
            pd.DataFrame([
                {"day":"2022-05-07","baseline":BASELINE,"training_weekdays":N,
                 "theta_F":-3.,"theta_R":-5.,"theta_U":-.2}
            ]).to_csv(source/"parameter_estimates.csv",index=False)
            (source/"configuration.json").write_text('{"group":"MED"}')
            fitted=load_sweep_params(source,"MED")
            self.assertEqual(len(fitted),1)
            with self.assertRaises(ValueError):
                load_sweep_params(source,"ALL")


if __name__=="__main__":unittest.main()

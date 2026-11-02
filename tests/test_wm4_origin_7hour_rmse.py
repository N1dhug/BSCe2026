import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_origin_7hour_rmse import (
    HORIZON, ORIGINS, check_scores, cumulative_rmse_by_origin, evaluate_day,
    oracle_prediction, report, save_rows, summarize,
)
from src.model.wm_4 import wm_4


class constant_baseline:
    def predict(self, index, temperature): return np.full(len(index), 0.5, dtype=float)


class test_wm4_origin_7hour_rmse(unittest.TestCase):
    @staticmethod
    def frame_and_day():
        day = pd.Timestamp("2022-05-07")
        days = [day-pd.Timedelta(weeks=w) for w in range(8,0,-1)]+[day]
        points=[]
        for previous in days:
            for h in range(24):
                u=.4*np.sin(2*np.pi*h/24)
                i=.5+.05*np.cos(2*np.pi*h/24)+.005*(previous.weekday()+1)
                points.append((previous+pd.Timedelta(hours=h),i,u,12.0))
        frame=pd.DataFrame(points,columns=["time","I","U","Temp"]).set_index("time").sort_index()
        return frame,day

    def test_exact_oracle_f0_projection_and_no_origin_zero_sensitivity(self):
        sph=4
        model=wm_4((0.0,7/24),7*sph+1,lambda t:0.)
        times=np.arange(7*sph+1)/sph
        U=np.full(7*sph+1,0.13)
        O=np.full(7*sph+1,0.5)
        theta=np.array([-3.,-.2,-5.])
        r0=-.02
        f0_true=.18
        actual=model._solve_deterministic([f0_true,r0],U,U,times,O,theta)
        observed=actual["I"][::sph][:HORIZON]
        guess,pred,baseline,sensitivity=oracle_prediction(
            model,{"U":U,"PE":U,"Time":times,"O":O},observed,theta,r0,sph)
        self.assertGreater(sensitivity,0)
        self.assertAlmostEqual(guess,f0_true,places=8)
        np.testing.assert_allclose(pred,observed,atol=1e-12)
        np.testing.assert_allclose(baseline,np.full(HORIZON,.5))

    def test_eighteen_independent_origin_oracles_with_one_daily_fit(self):
        frame,day=self.frame_and_day()
        theta=np.array([-3.,-.2,-5.])
        with patch("src.exe.run_model.wm_4.analyze_origin_7hour_rmse.baseline_from_history",
                   return_value=constant_baseline()), patch(
                "src.exe.run_model.wm_4.analyze_origin_7hour_rmse.joint_qp_params") as fitted:
            fitted.return_value.fit_many.return_value={"r":theta,"qp_loss":0.001}
            daily,hourly=evaluate_day(frame,day)
            fitted.return_value.fit_many.assert_called_once()
            origin0=daily[0]
            self.assertAlmostEqual(origin0["historical_R0"],
                                   float(np.median([frame.at[day-pd.Timedelta(weeks=w),"I"]-.5
                                                    for w in range(1,9)])))
            reference={**origin0, "rmse":origin0["rmse"]}
            second,second_hour=evaluate_day(frame,day,reference=reference)
            fitted.return_value.fit_many.assert_called_once()
        self.assertEqual(len(daily),18)
        self.assertEqual(len(hourly),18*HORIZON)
        self.assertEqual(set(row["origin_hour"] for row in daily),set(ORIGINS))
        self.assertEqual(set(row["absolute_hour"] for row in hourly if row["origin_hour"]==17),
                         set(range(17,24)))
        self.assertEqual(len({row["origin_hour"] for row in hourly}),18)
        self.assertAlmostEqual(second[0]["rmse"],origin0["rmse"])
        self.assertEqual(len(second_hour),126)
        self.assertGreater(len(set(round(row["historical_R0"],6) for row in daily)),1)

    def test_cumulative_rmse_exact_prefixes_for_all_origins(self):
        # Only the first hour has a nonzero error, so cumulative RMSE
        # decreases as more zero-error observations enter the denominator.
        # The same seven-hour fit is used across all seven prefixes.
        hourly=[]
        for origin in ORIGINS:
            for day,initial in ((pd.Timestamp("2022-04-12"),1.0),
                                (pd.Timestamp("2022-04-19"),3.0)):
                for lead in range(HORIZON):
                    hourly.append({"day":day,"origin_hour":origin,"lead_hour":lead,
                                   "error":initial if lead==0 else 0.0,
                                   "baseline_error":2.0})
        cumulative=cumulative_rmse_by_origin(pd.DataFrame(hourly))
        self.assertEqual(len(cumulative),len(ORIGINS)*HORIZON)
        for origin in ORIGINS:
            part=cumulative[cumulative["origin_hour"]==origin].sort_values("elapsed_hours")
            expected=np.sqrt(5.0/np.arange(1,HORIZON+1))
            np.testing.assert_allclose(part["cumulative_rmse"],expected,atol=1e-12)
            np.testing.assert_allclose(part["baseline_cumulative_rmse"],2.0,atol=1e-12)
            np.testing.assert_array_equal(part["hour_end"],
                                          origin+np.arange(1,HORIZON+1))
            self.assertEqual(int(part["interval_start_hour"].iloc[0]),origin)
            self.assertEqual(int(part["interval_end_hour"].iloc[0]),origin+HORIZON)
            self.assertEqual(int(part["samples"].iloc[-1]),14)
        self.assertEqual(cumulative.loc[
            cumulative["origin_hour"]==17,"hour_end"].iloc[-1],24)
        # Strictly reject incomplete origins, rather than biasing the
        # comparison by averaging over different numbers of forecast days.
        with self.assertRaises(ValueError):
            cumulative_rmse_by_origin(pd.DataFrame(hourly).iloc[:-1])

    def test_aggregate_rmse_pools_errors_and_saves_figures(self):
        frame,day=self.frame_and_day()
        with patch("src.exe.run_model.wm_4.analyze_origin_7hour_rmse.baseline_from_history",
                   return_value=constant_baseline()), patch(
                "src.exe.run_model.wm_4.analyze_origin_7hour_rmse.joint_qp_params") as fitted:
            fitted.return_value.fit_many.return_value={"r":np.array([-3.,-.2,-5.]),"qp_loss":0.01}
            daily,hourly=evaluate_day(frame,day)
        d,h=check_scores(pd.DataFrame(daily),pd.DataFrame(hourly))
        summary,leads=summarize(d,h)
        self.assertEqual(len(summary),18)
        self.assertEqual(len(leads),18*HORIZON)
        self.assertTrue((summary["days"]==1).all())
        self.assertTrue((summary["samples"]==HORIZON).all())
        self.assertEqual(summary["origin_hour"].tolist(),list(ORIGINS))
        self.assertAlmostEqual(summary.iloc[0]["aggregate_rmse"],daily[0]["rmse"])
        self.assertAlmostEqual(summary.iloc[-1]["aggregate_rmse"],daily[-1]["rmse"])
        with tempfile.TemporaryDirectory() as dirname:
            output=Path(dirname)
            save_rows(output,daily,hourly,[])
            reported=report(output)
            self.assertEqual(len(reported),18)
            for name in ("origin_aggregate_rmse.csv","origin_lead_rmse.csv",
                         "origin_cumulative_rmse.csv","origin_aggregate_rmse.png",
                         "origin_rmse_quantiles.png","origin_lead_rmse_heatmap.png",
                         "origin_cumulative_rmse_panels.png"):
                self.assertTrue((output/name).is_file(),name)
            prefix=pd.read_csv(output/"origin_cumulative_rmse.csv")
            self.assertEqual(len(prefix),18*HORIZON)
            for row in reported.itertuples(index=False):
                final=prefix.loc[(prefix["origin_hour"]==row.origin_hour) &
                                 (prefix["elapsed_hours"]==HORIZON)].iloc[0]
                self.assertAlmostEqual(float(final["cumulative_rmse"]),
                                       float(row.aggregate_rmse),places=12)
                self.assertAlmostEqual(float(final["baseline_cumulative_rmse"]),
                                       float(row.baseline_rmse),places=12)
        # Missing an origin or lead hour is an error, not an unmatched average.
        with self.assertRaises(ValueError): check_scores(d[d["origin_hour"]!=17],h)
        with self.assertRaises(ValueError): check_scores(d,h.iloc[:-1])


if __name__=="__main__": unittest.main()

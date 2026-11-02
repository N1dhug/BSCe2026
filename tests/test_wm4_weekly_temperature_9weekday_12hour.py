import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour import (
    BASELINE, H, N, convert_day, from_sweep, report, summarize,
)


class test_wm4_weekly_temperature_9weekday_12hour(unittest.TestCase):
    @staticmethod
    def example(days=3):
        daily,hourly,score_rows,components=[],[],[],[]
        for j in range(days):
            day=pd.Timestamp("2022-05-01")+pd.Timedelta(days=j)
            f0,r0=.10,-.02
            # Construct consistent zero-state trajectories with two linearly
            # independent state sensitivity columns.
            hour=np.arange(24)
            sF=np.sin(hour/7)
            sR=1+.1*np.cos(hour/6)
            zeros=.5+.025*np.sin(hour/3)
            pred=zeros+f0*sF+r0*sR
            actual=pred+.012*(j+1)*np.cos(hour/5)
            baseline=np.full(24,.50)
            score={
                "day":day,"baseline":BASELINE,"training_weekdays":N,"horizon_hours":H,
                "oracle_F0":f0,"oracle_R0":r0,"theta_F":-4.0,
                "theta_R":-4.2,"theta_U":-.15,
                "historical_R0":-.015,
                "mse":float(np.mean((pred[:H]-actual[:H])**2)),
                "baseline_mse":float(np.mean((baseline[:H]-actual[:H])**2)),
            }
            parts=[{
                "day":day,"baseline":BASELINE,"training_weekdays":N,
                "hour":int(h),"observed":float(actual[h]),
                "baseline_value":float(baseline[h]),
                "prediction_at_initial_zero":float(zeros[h]),
                "sensitivity_F0":float(sF[h]),"sensitivity_R0":float(sR[h]),
            } for h in range(24)]
            summary,prefix=convert_day(score,parts)
            daily.append(summary);hourly.extend(prefix)
            score_rows.append(score);components.extend(parts)
        return pd.DataFrame(daily),pd.DataFrame(hourly),pd.DataFrame(score_rows),pd.DataFrame(components)

    def test_reconstruction_has_twelve_hours_and_both_oracle_states(self):
        daily,hourly,_,_=self.example()
        self.assertEqual(len(daily),3)
        self.assertEqual(len(hourly),3*H)
        self.assertTrue((daily["oracle_F0"]==.10).all())
        self.assertTrue((daily["oracle_R0"]==-.02).all())
        self.assertEqual(set(hourly["hour"]),set(range(H)))

    def test_pooled_prefix_matches_full_horizon_and_does_not_reset_state(self):
        daily,hourly,_,_=self.example()
        prefix=summarize(daily,hourly)
        self.assertEqual(prefix["hour_end"].tolist(),list(range(1,H+1)))
        self.assertEqual(prefix["samples"].tolist(),[3*h for h in range(1,H+1)])
        self.assertAlmostEqual(prefix.iloc[-1]["cumulative_rmse"],np.sqrt(np.mean(daily["mse"])))
        self.assertAlmostEqual(prefix.iloc[-1]["baseline_cumulative_rmse"],
                               np.sqrt(np.mean(daily["baseline_mse"])))
        expected_first=np.sqrt(np.mean(hourly.loc[hourly["hour"]==0,"error"]**2))
        self.assertAlmostEqual(prefix.iloc[0]["cumulative_rmse"],expected_first)
        # Cumulative squared error cannot decrease, but the mean RMSE can.
        self.assertTrue((prefix["cumulative_squared_error"].diff().dropna()>=0).all())
        self.assertEqual(int(prefix.iloc[-1]["days"]),3)

    def test_rejects_incomplete_days_and_wrong_oracle_model(self):
        daily,hourly,scores,parts=self.example()
        with self.assertRaises(ValueError):
            summarize(daily,hourly.iloc[:-1])
        with self.assertRaises(ValueError):
            summarize(daily,hourly.iloc[1:])
        broken=scores.iloc[0].to_dict()
        broken["horizon_hours"]=11
        with self.assertRaises(ValueError):
            convert_day(broken,parts.loc[parts["day"]==broken["day"]].to_dict("records"))
        missing=parts.loc[parts["day"]==scores.iloc[0]["day"]].copy()
        missing=missing.drop(columns="sensitivity_R0")
        with self.assertRaises(ValueError):
            convert_day(scores.iloc[0].to_dict(),missing.to_dict("records"))

    def test_can_use_previous_sweep_and_regenerate_plots(self):
        daily,hourly,scores,components=self.example()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"sweep"
            output=root/"report"
            source.mkdir();output.mkdir()
            scores.to_csv(source/"oracle_horizon_scores.csv",index=False)
            components.to_csv(source/"forecast_components.csv",index=False)
            rows,points=from_sweep(source,max_days=2)
            self.assertEqual(len(rows),2)
            self.assertEqual(len(points),2*H)
            pd.DataFrame(rows).to_csv(output/"daily_metrics.csv",index=False)
            pd.DataFrame(points).to_csv(output/"hourly_predictions.csv",index=False)
            curve=report(output)
            self.assertEqual(len(curve),H)
            for name in ("cumulative_rmse.csv","cumulative_rmse_vs_baseline.png",
                         "hourly_rmse_vs_baseline.png","cumulative_improvement_pct.png"):
                self.assertTrue((output/name).is_file(),name)
            stale=scores.drop(columns="oracle_R0")
            stale.to_csv(source/"oracle_horizon_scores.csv",index=False)
            with self.assertRaisesRegex(ValueError,"obsolete F0-only"):
                from_sweep(source)


if __name__=="__main__": unittest.main()

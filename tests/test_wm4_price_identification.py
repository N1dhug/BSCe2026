import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_price_identification import (
    PRICE_FEATURES, analyze, association_tests, benjamini_hochberg,
    block_bootstrap_spearman, historical_price_features, price_summary,
)


class test_wm4_price_identification(unittest.TestCase):
    @staticmethod
    def example(days=16):
        index = pd.date_range("2022-01-15", "2022-05-30 23:00", freq="h")
        prices = np.clip(.2+.25*np.sin(np.arange(len(index))*2*np.pi/24) +
                         .15*np.sin(np.arange(len(index))*2*np.pi/(24*17)), -1,1)
        frame = pd.DataFrame({"U": prices}, index=index)
        dates = pd.date_range("2022-04-01", periods=days, freq="D")
        d, h = [], []
        for k, day in enumerate(dates):
            history = [day-pd.Timedelta(weeks=w) for w in range(8,0,-1)]
            strength = .03+.07*(k%4)
            row = {
                "day": day, "baseline": "weekly_temperature",
                "training_weekdays": 8, "horizon_hours": 7,
                "training_days": ";".join(f"{date:%Y-%m-%d}" for date in history),
                "oracle_F0": .1, "historical_R0": -.01,
                "rmse": .04+k*.002, "mse": (.04+k*.002)**2, "baseline_mse": .01,
                "theta_F": -7.0, "theta_R": -7.0,
                "theta_U": -strength if k%3 else 1e-9,
                "price_rms": strength, "price_mean": -.02,
            }
            d.append(row)
            for hr in range(7):
                h.append({
                    "day":day, "time":day+pd.Timedelta(hours=hr), "hour":hr,
                    "baseline":"weekly_temperature","training_weekdays":8,
                    "horizon_hours":7, "observed":.5, "predicted":.55,
                    "baseline_prediction":.6,"error":.05,"absolute_error":.05,
                    "restoring":.1,"damping":-.1,"price":-.02,
                    "U":float(frame.at[day+pd.Timedelta(hours=hr),"U"])
                })
        return frame,pd.DataFrame(d),pd.DataFrame(h)

    def test_historical_volatility_within_days_and_imputation(self):
        frame, daily, hourly = self.example()
        historical = pd.to_datetime(daily.iloc[0]["training_days"].split(";"))
        gap = historical[0]+pd.Timedelta(hours=9)
        frame.loc[gap,"U"] = np.nan
        features = historical_price_features(frame,historical)
        self.assertEqual(features["historical_price_imputed_hours"],1)
        self.assertTrue(all(np.isfinite(features[feature]) for feature in PRICE_FEATURES))
        self.assertGreater(features["historical_price_std"],0)
        frame.loc[historical[0]:historical[0]+pd.Timedelta(hours=23),"U"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing full day"):
            historical_price_features(frame,historical)

    def test_price_response_flag_depends_on_fitted_theta_not_target_prices(self):
        frame,daily,hourly = self.example()
        table = price_summary(daily,hourly,frame)
        self.assertEqual(len(table),len(daily))
        self.assertEqual(table["price_active"].sum(),10)
        self.assertEqual(table["price_boundary"].sum(),6)
        self.assertEqual(table["price_positive_above_tolerance"].sum(),0)
        changed = frame.copy()
        changed.loc[pd.Timestamp("2022-04-01"):,"U"] = np.nan
        # Training history of earlier dates is unchanged; target price is
        # sourced from SAVED hourly predictions, not from the original frame.
        subset = price_summary(daily.iloc[:1],hourly.loc[hourly["day"]==daily.iloc[0]["day"]],changed)
        self.assertAlmostEqual(subset.iloc[0]["historical_price_std"],table.iloc[0]["historical_price_std"])
        self.assertEqual(subset.iloc[0]["price_active"],table.iloc[0]["price_active"])

    def test_bootstrap_correlations_and_bh_adjustment(self):
        n=91
        rng=np.random.default_rng(4)
        x=rng.normal(size=n)
        y=(x>0).astype(float)
        result=block_bootstrap_spearman(x,y,block_size=7,draws=200,seed=42)
        self.assertGreater(result["spearman_rho"],.7)
        self.assertLess(result["p_value_block_approx"],.05)
        self.assertGreater(result["bootstrap_draws_used"],150)
        q=benjamini_hochberg([.01,.04,.2,np.nan])
        self.assertTrue(np.all(np.diff(q[:3])>=0))
        self.assertTrue(np.isnan(q[-1]))
        self.assertAlmostEqual(q[0],.03)

    def test_saved_data_produces_price_diagnostic_tables_and_plots(self):
        frame,daily,hourly=self.example()
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"input"
            output=Path(temp)/"output"
            source.mkdir()
            daily.to_csv(source/"daily_metrics.csv",index=False)
            hourly.to_csv(source/"hourly_predictions.csv",index=False)
            with patch("src.exe.run_model.wm_4.analyze_price_identification.group_frame",return_value=frame):
                result,associations=analyze(source,output,block_size=3,draws=100)
            self.assertEqual(len(result),16)
            self.assertEqual(len(associations),2*len(PRICE_FEATURES))
            self.assertEqual(associations["period"].nunique(),1)
            for name in ("daily_price_identification.csv","historical_price_associations.csv",
                         "price_volatility_quartiles.csv","error_quartiles_price_activity.csv",
                         "historical_price_vs_coefficient.png","price_activation_by_quantiles.png",
                         "historical_vs_forecast_price_volatility.png",
                         "price_identification_and_error.png","price_activity_association_intervals.png"):
                self.assertTrue((output/name).is_file(),name)


if __name__=="__main__": unittest.main()

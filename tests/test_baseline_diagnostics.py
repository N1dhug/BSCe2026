import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.data import baseline_diagnostics as mod


class test_baseline_diagnostics(unittest.TestCase):
    @staticmethod
    def frame(days=140):
        index=pd.date_range("2022-01-01",periods=days*24,freq="h")
        hour=index.hour.to_numpy(dtype=float)
        weekday=index.dayofweek.to_numpy(dtype=float)
        day=np.arange(len(index),dtype=float)/24.0
        temperature=10+5*np.sin(2*np.pi*day/41)+3*np.cos(2*np.pi*hour/24)
        chill=np.maximum(0.0,18-temperature)
        price=.3+.04*np.cos(2*np.pi*day/9)+.1*np.sin(2*np.pi*hour/24)
        noise=.004*np.sin(2*np.pi*day/13)
        demand=(.40+.08*np.sin(2*np.pi*hour/24)
                +.02*(weekday==5)+.01*(weekday==6)
                +.013*chill-.045*price+noise)
        return pd.DataFrame({"I":demand,"U":price,"Temp":temperature},index=index)

    def test_weekday_profile_and_exact_hour_autocorrelation(self):
        frame=self.frame()
        profile=mod.weekday_profile(frame)
        self.assertEqual(len(profile),168)
        self.assertEqual(set(profile["weekday"]),set(range(7)))
        series=frame["I"].copy()
        series.iloc[100]=np.nan
        result=mod.acf(series,168)
        self.assertEqual(result.iloc[0]["autocorrelation"],1)
        self.assertGreater(result.loc[result["lag_hours"]==168,"autocorrelation"].iloc[0],.6)
        expected=(series.notna().to_numpy()&
                  np.isfinite(series.reindex(series.index-pd.Timedelta(hours=24)).to_numpy())).sum()
        self.assertEqual(result.loc[result["lag_hours"]==24,"pair_count"].iloc[0],expected)

    def test_weekly_similarity_uses_equal_comparison_samples(self):
        idx=pd.date_range("2022-01-01",periods=75*24,freq="h")
        h=idx.hour.to_numpy(dtype=float)
        weekday=idx.dayofweek.to_numpy(dtype=float)
        I=.5+.02*weekday+.06*np.sin(2*np.pi*h/24)
        frame=pd.DataFrame({"I":I,"U":.2,"Temp":12.0},index=idx)
        frame.loc[idx[45*24],"I"]=np.nan
        result=mod.corresponding_similarity(frame,max_weeks=4)
        self.assertEqual(len(result),4)
        self.assertTrue((result["weekly_mse"]<1e-13).all())
        self.assertTrue((result["gain_vs_neighbor_mean_mse"]>0).all())
        self.assertTrue((result["matched_pairs"]>0).all())

    def test_fixed_effects_regression_estimates_temperature_and_price(self):
        report,resid,daily=mod.dependence(self.frame(),hac_lags=24)
        self.assertTrue(report["identified_joint"])
        self.assertAlmostEqual(report["temperature_only"]["coefficient"],.013,delta=.005)
        self.assertAlmostEqual(report["temperature_conditional"]["coefficient"],.013,delta=.004)
        self.assertAlmostEqual(report["price_conditional"]["coefficient"],-.045,delta=.025)
        self.assertGreaterEqual(report["temperature_mse_gain"],0.0)
        self.assertGreaterEqual(report["price_conditional_mse_gain"],-1e-12)
        self.assertEqual(len(resid),140*24)
        self.assertTrue(len(daily)>100)
        self.assertTrue(np.isfinite(resid.to_numpy()).all())

    def test_rolling_baselines_never_fit_target_observations(self):
        frame=self.frame()
        period=frame.copy()
        actual_fit=mod._fit_sparse_baseline
        training_sets=[]

        def checked_fit(training,B):
            training_sets.append((training.copy(),B))
            return actual_fit(training,B)

        with patch.object(mod,"_fit_sparse_baseline",side_effect=checked_fit):
            scores,rows,coverage=mod.rolling_baselines(
                frame,period,backlogs=(2,),horizons=(6,),
                stride=24,max_origins=4)

        self.assertEqual(coverage["candidate_origins"],4)
        self.assertEqual(len(scores),4*len(mod.BASELINES))
        self.assertEqual(len(rows),4*len(mod.BASELINES)*6)
        self.assertEqual(len(training_sets),4*len(mod.BASELINES))
        self.assertTrue((scores["train_samples"]==2*6).all())
        for training,_ in training_sets:
            self.assertEqual(len(training),12)
            self.assertEqual(len(training.index.unique()),12)
            self.assertEqual(training.index.hour.nunique(),6)
            self.assertTrue(np.isfinite(training.to_numpy(dtype=float)).all())
        for i,origin in enumerate(pd.DatetimeIndex(scores["origin"].unique())):
            expected=pd.DatetimeIndex(np.concatenate([
                pd.date_range(origin-pd.Timedelta(weeks=n),periods=6,freq="h").to_numpy()
                for n in (2,1)]))
            for training,_ in training_sets[i*len(mod.BASELINES):(i+1)*len(mod.BASELINES)]:
                self.assertTrue(training.index.equals(expected))
                self.assertLess(training.index.max(),origin)
        aggregates=mod.summarize_baselines(scores,rows)
        self.assertEqual(set(aggregates["baseline"]),set(mod.BASELINES))
        self.assertTrue((aggregates["origins"]==4).all())

    def test_missing_targets_are_skipped_and_exogenous_is_retrospective(self):
        frame=self.frame()
        period=frame.copy()
        # Only target-score missing samples, never fill labels.
        start=period.index.min()+pd.Timedelta(weeks=2)
        period.loc[start,"I"]=np.nan
        scores,rows,coverage=mod.rolling_baselines(
            frame,period,backlogs=(2,),horizons=(6,),
            stride=24,max_origins=4)
        self.assertGreaterEqual(coverage["skipped_windows"]["missing_target"],1)
        self.assertLess(len(scores),4*len(mod.BASELINES))
        self.assertTrue((rows["timestamp"]>=rows["origin"]).all())

    def test_full_executable_saves_csv_json_and_png_without_real_data(self):
        frame=self.frame()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(mod,"group_frame",return_value=frame):
                results=mod.run(groups=("MED",),backlogs=(2,),horizons=(6,),
                                max_weeks=3,max_acf_lag=168,hac_lags=12,
                                stride_hours=24,max_origins=4,output=tmp)
            self.assertEqual(set(results),{"MED"})
            directory=Path(tmp)/"med"
            for name in ("weekday_hour_profile.csv","consumption_acf.csv",
                         "corresponding_similarity.csv","adjusted_residuals.csv",
                         "dependence.json","baseline_origin_scores.csv",
                         "baseline_hourly_predictions.csv","baseline_summary.csv",
                         "summary.json","weekday_profiles.png","consumption_acf.png",
                         "corresponding_similarity.png","daily_temperature.png",
                         "temperature_partial.png","price_partial.png",
                         "adjusted_residual_acf.png","baseline_rmse.png",
                         "baseline_bias.png"):
                with self.subTest(file=name):
                    self.assertTrue((directory/name).is_file(),name)
            self.assertEqual(len(results["MED"]["baseline_summary"]),3)
            self.assertEqual(len(results["MED"]["origin_scores"]),12)

    def test_invalid_settings(self):
        frame=self.frame()
        with self.assertRaises(ValueError):
            mod.rolling_baselines(frame,frame,backlogs=(0,),horizons=(12,))
        with self.assertRaises(ValueError):
            mod.rolling_baselines(frame,frame,backlogs=(4,),horizons=(25,))
        with self.assertRaises(ValueError):
            mod.acf(frame["I"],max_lag=0)
        with self.assertRaises(ValueError):
            mod.corresponding_similarity(frame,max_weeks=0)


if __name__=="__main__": unittest.main()

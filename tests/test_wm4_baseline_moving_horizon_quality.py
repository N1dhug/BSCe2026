import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4 import plot_baseline_moving_horizon_quality as mod
from src.exe.run_model.wm_4.sweep_weekday_horizon import EXPERIMENT_VERSION


class test_wm4_baseline_moving_horizon_quality(unittest.TestCase):
    @staticmethod
    def frame():
        index = pd.date_range("2022-05-01", "2022-06-25 23:00", freq="h")
        data = pd.DataFrame({"I":np.full(len(index),.5), "U":np.zeros(len(index)),
                             "Temp":np.full(len(index),10.)}, index=index)
        history = [pd.Timestamp("2022-06-06 06:00"),
                   pd.Timestamp("2022-06-13 06:00")]
        for start in history:
            data.loc[pd.date_range(start,periods=3,freq="h"),"I"] = [.56,.57,.58]
        data.loc[pd.date_range("2022-06-20 06:00",periods=3,freq="h"),"I"] = [.65,.67,.69]
        return data

    @staticmethod
    def rows():
        history = ["2022-06-06T06:00:00", "2022-06-13T06:00:00"]
        common = {"day":"2022-06-20T06:00:00","training_weekdays":2,
                  "horizon_hours":3,"training_days":";".join(history),
                  "baseline_beta_C":0.,"baseline_chill_center":0.,
                  "baseline_feature_mean":"null","baseline_feature_scale":"null"}
        return [
            {**common,"baseline":"weekly_average",
             "baseline_coefficients":json.dumps([.5]*168)},
            {**common,"baseline":"weekly_temperature",
             "baseline_coefficients":json.dumps([.55]*168+[0.])},
        ]

    def test_historical_index_exact_Nk_and_weekday_matching(self):
        row = self.rows()[0]
        index = mod.historical_index(row)
        self.assertEqual(len(index),6)
        self.assertEqual(index[0],pd.Timestamp("2022-06-06 06:00"))
        self.assertEqual(index[2],pd.Timestamp("2022-06-06 08:00"))
        self.assertEqual(index[3],pd.Timestamp("2022-06-13 06:00"))
        invalid = dict(row, training_days="2022-06-05T06:00:00;2022-06-13T06:00:00")
        with self.assertRaisesRegex(ValueError, "corresponding weekdays"):
            mod.historical_index(invalid)

    def test_pooled_errors_and_shared_scale_plots(self):
        frame = self.frame()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source,output = root/"sweep",root/"plots"
            source.mkdir()
            (source/"configuration.json").write_text(json.dumps({
                "version":EXPERIMENT_VERSION,"group":"ALL",
                "baselines":["weekly_average","weekly_temperature"],
                "min_weekdays":2,"max_weekdays":2,
                "min_horizon":3,"max_horizon":3,
                "steps_per_hour":4,
            }),encoding="utf-8")
            pd.DataFrame(self.rows()).to_csv(source/"parameter_estimates.csv",index=False)
            args = SimpleNamespace(group="ALL",source=str(source),output=str(output),
                                   baselines=list(mod.BASELINES),
                                   min_weekdays=None,max_weekdays=None,
                                   min_horizon=None,max_horizon=None,
                                   start_day=None,end_day=None,cohort="all",
                                   progress_every=1)
            with patch.object(mod,"group_frame",return_value=frame):
                result=mod.run(args)
            self.assertTrue((output/"fit_cumulative_quality.png").exists())
            self.assertTrue((output/"prediction_cumulative_quality.png").exists())
            self.assertTrue((output/"fit_cumulative_quality.csv").exists())
            self.assertTrue((output/"prediction_cumulative_quality.csv").exists())
            self.assertEqual(len(result["fit"]),2)
            self.assertEqual(len(result["prediction"]),2)
            for name,base in (("weekly_average",.5),("weekly_temperature",.55)):
                fit=result["fit"].set_index("baseline").loc[name]
                pred=result["prediction"].set_index("baseline").loc[name]
                hf=np.tile(np.array([.56,.57,.58])-base,2)
                pf=np.array([.65,.67,.69])-base
                self.assertEqual(int(fit["observations"]),6)
                self.assertEqual(int(pred["observations"]),3)
                self.assertEqual(int(fit["origins"]),1)
                self.assertEqual(int(pred["origins"]),1)
                self.assertAlmostEqual(float(fit["rmse"]),float(np.sqrt(np.mean(hf**2))))
                self.assertAlmostEqual(float(fit["mae"]),float(np.mean(np.abs(hf))))
                self.assertAlmostEqual(float(fit["bias"]),float(-np.mean(hf)))
                self.assertAlmostEqual(float(pred["rmse"]),float(np.sqrt(np.mean(pf**2))))
                self.assertAlmostEqual(float(pred["mae"]),float(np.mean(np.abs(pf))))
                self.assertAlmostEqual(float(pred["bias"]),float(-np.mean(pf)))
            self.assertEqual(result["configuration"]["cohort"],"all")

    def test_pooling_weights_observations_not_window_mean_rmse(self):
        pool=mod.defaultdict(mod.pool_record)
        mod.accumulate(pool,("weekly_average",2,3),[.1,.3])
        mod.accumulate(pool,("weekly_average",2,3),[.4])
        result=mod.summarize(pool).iloc[0]
        self.assertAlmostEqual(result["rmse"],np.sqrt((.01+.09+.16)/3))
        self.assertAlmostEqual(result["mae"],(.1+.3+.4)/3)
        self.assertAlmostEqual(result["bias"],(.1+.3+.4)/3)
        self.assertEqual(int(result["origins"]),2)
        self.assertEqual(int(result["observations"]),3)


if __name__=="__main__":unittest.main()

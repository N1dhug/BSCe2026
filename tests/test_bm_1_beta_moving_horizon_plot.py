"""Synthetic, input-workbook-free checks for the single-panel BM1 beta-mixture plotter."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.plot_beta_moving_horizon import plot_saved
from src.method.bm_1_beta import bm_1_beta


class BM1BetaPlotTests(unittest.TestCase):
    @staticmethod
    def fixture(root, *, legacy=False, names=("weekly_average", "weekly_temperature")):
        origin = pd.Timestamp("2022-08-22 06:00")
        starts = [origin - pd.Timedelta(weeks=i) for i in (2, 1)]
        n, k = len(starts), 4
        p = {"alpha":2.0, "beta":2.0, "k":1.0, "Xi":0.3,
             "C":1.2, "sigma_x":0.11, "sigma_y":0.03}
        config = {"components":4, "quadrature":4, "reconstruction_quadrature":8, "max_step":0.25}
        solver = bm_1_beta(**config)
        for name in names:
            directory = root / name
            directory.mkdir(parents=True)
            history_rows = []
            states = [0.44, 0.56]
            for i, timestamp in enumerate(starts):
                index = pd.date_range(timestamp, periods=k, freq="h")
                B = np.full(k, 0.30 + 0.01*(name == "weekly_temperature"))
                U = np.array([0.4,0.5,0.6,0.55])
                result = solver.simulate({"B":B,"U":U}, p, states[i])
                bounds = solver.intervals(result, p["sigma_y"])
                for j, time in enumerate(index):
                    row = {"historical_origin":timestamp, "timestamp":time,
                           "lead_hour":j, "observed":0.31 + 0.003*j,
                           "baseline":B[j], "price":U[j], "beta_mean":result["mean"][j]}
                    if not legacy: row.update({key:bounds[key][j] for key in bounds})
                    history_rows.append(row)
            target_index = pd.date_range(origin, periods=k, freq="h")
            B = np.full(k, 0.30 + 0.01*(name == "weekly_temperature"))
            U = np.array([0.4,0.5,0.6,0.55])
            future = solver.simulate({"B":B,"U":U},p,0.51)
            future_bounds = solver.intervals(future,p["sigma_y"])
            rows = []
            for j, time in enumerate(target_index):
                rows.append({"timestamp":time, "lead_hour":j,"observed":0.31 + 0.003*j,
                             "baseline":B[j],"beta_mean":future["mean"][j],
                             **{key:future_bounds[key][j] for key in future_bounds}})
            pd.DataFrame(history_rows).to_csv(directory/"historical_fit.csv",index=False)
            pd.DataFrame(rows).to_csv(directory/"oracle_prediction.csv",index=False)
            with (directory/"result.json").open("w",encoding="utf-8") as file:
                json.dump({"N":n, "k":k, "origin":str(origin),
                           "historical_origins":[str(date) for date in starts],
                           "historical_stochastic_states":states,
                           "stochastic_parameters":p,
                           "resolutions":{"prediction":config}},file)
        with (root/"run_summary.json").open("w",encoding="utf-8") as file:
            json.dump({"tag":"ALL"},file)
        return starts, origin

    def test_both_baselines_single_axes_and_all_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            original = plt.subplots
            count = []
            def instrument(*args, **kwargs):
                fig, axes = original(*args, **kwargs)
                count.append(len(fig.axes))
                return fig, axes
            with patch("src.exe.run_model.bm_1.plot_beta_moving_horizon.plt.subplots",
                       side_effect=instrument):
                result = plot_saved(run_dir=root,baseline="both")
            self.assertEqual(result["historical_windows"],2)
            self.assertEqual(result["prediction_windows"],1)
            self.assertEqual(result["axes"],1)
            self.assertEqual(count,[1])
            self.assertTrue(result["plot"].exists())
            self.assertGreater(result["plot"].stat().st_size,10000)

    def test_old_historical_csv_reconstructs_and_caches_bands_without_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, legacy=True, names=("weekly_average",))
            result = plot_saved(run_dir=root,baseline="weekly_average")
            self.assertTrue(result["plot"].is_file())
            cached = pd.read_csv(root/"weekly_average"/"historical_fit.csv")
            for column in ("lower_68","upper_68","lower_95","upper_95"):
                self.assertIn(column,cached.columns)
                self.assertTrue(np.isfinite(cached[column].to_numpy()).all())
            self.assertTrue(np.all(cached["lower_95"] <= cached["lower_68"]))
            self.assertTrue(np.all(cached["upper_68"] <= cached["upper_95"]))
            # Next call reads cached bands; it must not need reconstruction.
            with patch("src.exe.run_model.bm_1.plot_beta_moving_horizon._restore_historical_bands",
                       side_effect=AssertionError("Unexpected beta reconstruction")):
                plot_saved(run_dir=root,baseline="weekly_average")

    def test_no_reconstruct_reports_missing_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, legacy=True, names=("weekly_average",))
            with self.assertRaisesRegex(ValueError,"lacks uncertainty bands"):
                plot_saved(run_dir=root,baseline="weekly_average",reconstruct=False)

    def test_rejects_unmatched_observed_demand_between_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            file = root/"weekly_temperature"/"historical_fit.csv"
            frame = pd.read_csv(file)
            frame.loc[0,"observed"] += 0.05
            frame.to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,"different observed"):
                plot_saved(run_dir=root,baseline="both")


if __name__ == "__main__":
    unittest.main()

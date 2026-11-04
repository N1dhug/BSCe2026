import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_weekday_backlog import BASELINES, evaluate, run


def synthetic(days=3):
    rows = []
    for day in pd.date_range("2022-04-01", periods=days):
        for baseline in BASELINES:
            for n in (2, 3, 4):
                for h in (1, 2, 3):
                    mse = (1 + .01*h)**2 if n == 3 else (2 + .01*h)**2
                    rows.append({"day": day, "baseline": baseline, "training_weekdays": n,
                                 "horizon_hours": h, "mse": mse})
    return pd.DataFrame(rows)


class test_wm4_weekday_backlog_analysis(unittest.TestCase):
    def test_pooled_weights_every_forecast_hour(self):
        result, detail, winners, days = evaluate(synthetic(), n_values=(2, 3, 4), horizons=(1, 2, 3))
        row = result.loc[(result.baseline == BASELINES[0]) &
                         (result.training_weekdays == 3)].iloc[0]
        mse = np.array([(1 + .01*h)**2 for h in (1, 2, 3)])
        self.assertAlmostEqual(row.pooled_rmse, np.sqrt(np.dot(np.arange(1, 4), mse)/6))
        self.assertAlmostEqual(row.equal_horizon_rmse, np.sqrt(np.mean(mse)))
        self.assertEqual(row.hourly_errors, 18)
        self.assertEqual(len(days), 3)
        self.assertEqual(set(winners.best_backlog_N), {3})
        self.assertEqual(len(detail), 4*3*3)

    def test_strict_date_matching(self):
        df = synthetic()
        df = df.loc[~((df.day == pd.Timestamp("2022-04-03")) &
                      (df.baseline == BASELINES[0]) &
                      (df.training_weekdays == 4) & (df.horizon_hours == 2))]
        result, detail, _, days = evaluate(df, n_values=(2, 3, 4), horizons=(1, 2, 3))
        self.assertEqual(len(days), 2)
        self.assertTrue((result.origins == 2).all())
        self.assertTrue((detail.origins == 2).all())

    def test_incomplete_configuration_raises(self):
        df = synthetic()
        df = df.loc[~((df.baseline == BASELINES[0]) & (df.training_weekdays == 4) &
                      (df.horizon_hours == 2))]
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate(df, n_values=(2, 3, 4), horizons=(1, 2, 3))

    def test_run_writes_figure_and_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            synthetic().to_csv(Path(folder)/"oracle_horizon_scores.csv", index=False)
            summary, detail, winners = run(input_dir=folder, min_weekdays=2,
                                           max_weekdays=4, min_horizon=1, max_horizon=3)
            out = Path(folder)/"backlog_analysis"
            for name in ("backlog_comparison.png", "backlog_summary.csv",
                         "backlog_by_horizon.csv", "best_backlog_by_horizon.csv"):
                self.assertTrue((out/name).is_file())
            self.assertEqual(int(summary.loc[summary.baseline == "all_baselines"].iloc[0].training_weekdays), 3)


if __name__ == "__main__": unittest.main()

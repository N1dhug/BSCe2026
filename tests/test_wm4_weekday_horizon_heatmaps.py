import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_weekday_horizon_heatmaps import (
    BASELINES, aggregate_rmse, draw_heatmaps, plot_heatmaps,
)


def sample_tables():
    days = pd.to_datetime(["2022-05-01", "2022-05-02"])
    scores, components = [], []
    for b_idx, baseline in enumerate(BASELINES, 1):
        for n in (2, 3):
            for day_idx, day in enumerate(days):
                for hour in range(24):
                    components.append({
                        "day": day, "baseline": baseline, "training_weekdays": n,
                        "hour": hour, "observed": 0.0,
                        "prediction_at_initial_zero": float(b_idx*hour+day_idx),
                        "sensitivity_F0": 1.0, "sensitivity_R0": -2.0,
                    })
                for k in (2, 3):
                    scores.append({
                        "day": day, "baseline": baseline, "training_weekdays": n,
                        "horizon_hours": k, "oracle_F0": .5 if k == 2 else 0.0,
                        "oracle_R0": .25 if k == 2 else 0.0,
                    })
    return pd.DataFrame(scores), pd.DataFrame(components)


class test_wm4_weekday_horizon_heatmaps(unittest.TestCase):
    def test_sliced_rmse_uses_each_horizons_oracle_state(self):
        scores, components = sample_tables()
        table = aggregate_rmse(scores, components, hour_start=1, hour_end=3)

        def row(b, k):
            return table.loc[(table.baseline == b) & (table.training_weekdays == 2) &
                             (table.horizon_hours == k)].iloc[0]

        # The k=2 state has a zero net sensitivity shift, and only hour 1 is scored.
        self.assertAlmostEqual(row(BASELINES[0], 2).aggregate_rmse, np.sqrt((1**2+2**2)/2))
        self.assertAlmostEqual(row(BASELINES[0], 3).aggregate_rmse,
                               np.sqrt(np.mean(np.array([1., 2., 2., 3.])**2)))
        self.assertEqual(int(row(BASELINES[0], 3).days), 2)
        self.assertEqual(int(row(BASELINES[0], 3).observations), 4)
        self.assertGreater(row(BASELINES[1], 3).aggregate_rmse, row(BASELINES[0], 3).aggregate_rmse)

    def test_require_complete_scoring_interval(self):
        scores, components = sample_tables()
        table = aggregate_rmse(scores, components, hour_start=1, hour_end=3,
                               require_full_window=True)
        self.assertTrue(table.loc[table.horizon_hours == 2].aggregate_rmse.isna().all())
        self.assertTrue(table.loc[table.horizon_hours == 3].aggregate_rmse.notna().all())

    def test_last_hour_of_each_horizon(self):
        scores, components = sample_tables()
        table = aggregate_rmse(scores, components, last_hours=1)
        row = table.loc[(table.baseline == BASELINES[0]) & (table.training_weekdays == 2) &
                        (table.horizon_hours == 3)].iloc[0]
        self.assertAlmostEqual(row.aggregate_rmse, np.sqrt((2**2+3**2)/2))
        self.assertEqual(row.hour_start, 2)
        self.assertEqual(row.hour_end, 3)

    def test_cohort_matches_across_baselines_and_configurations(self):
        scores, components = sample_tables()
        missing = ((scores.day == pd.Timestamp("2022-05-02")) &
                   (scores.baseline == BASELINES[0]) &
                   (scores.training_weekdays == 3) & (scores.horizon_hours == 3))
        scores = scores.loc[~missing]
        global_table = aggregate_rmse(scores, components)
        self.assertTrue((global_table.loc[global_table.days > 0].days == 1).all())
        cell_table = aggregate_rmse(scores, components, cohort="cell")
        self.assertTrue((cell_table.loc[(cell_table.training_weekdays == 2) &
                                        (cell_table.horizon_hours == 3)].days == 2).all())
        self.assertTrue((cell_table.loc[(cell_table.training_weekdays == 3) &
                                        (cell_table.horizon_hours == 3)].days == 1).all())

    def test_plot_has_three_panels_one_colorbar_and_exports_csv(self):
        scores, components = sample_tables()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            scores.to_csv(folder/"oracle_horizon_scores.csv", index=False)
            components.to_csv(folder/"forecast_components.csv", index=False)
            result = plot_heatmaps(input_dir=folder, hour_start=0, hour_end=3)
            output = folder/"aggregate_rmse_three_baselines_hours_00_3_global.png"
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".csv").is_file())
            self.assertEqual(len(result), 3*2*2)
            figure = draw_heatmaps(result)
            self.assertEqual(len(figure.axes), 4)
            plt.close(figure)

    def test_invalid_slices_fail(self):
        scores, components = sample_tables()
        for kwargs in ({"hour_start": -1}, {"hour_end": 25}, {"hour_start": 6, "hour_end": 3},
                       {"last_hours": 4, "hour_start": 1}, {"cohort": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                aggregate_rmse(scores, components, **kwargs)


if __name__ == "__main__": unittest.main()

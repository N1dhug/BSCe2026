"""Short synthetic integration checks for the transferred Thomas forecast recipe."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.predict_daily import predict_day, save_result
from src.exe.run_model.bm_1.predict_mpc import predict_origin
from src.exe.run_model.bm_1.plot_prediction import reconstructed_demand, plot_saved
from src.tools.baseline import statistical_output
from src.params.bm_1 import lsq_params
from src.params.bm_1_response import BM1LogisticResponse, response_from_record
from src.validation.exe.run_model.bm_1.plot_prediction import bm_1_plot_validation


def frame_fixture():
    index = pd.date_range("2022-01-01", periods=60*24, freq="h")
    t = np.arange(len(index), dtype=float)
    return pd.DataFrame({
        "I": .45+.08*np.sin(2*np.pi*index.hour/24)+.025*np.sin(t/17),
        "U": .5+.25*np.cos(t/5),
        "Temp": 10+3*np.sin(t/200),
    }, index=index)


class ThomasPipelineTests(unittest.TestCase):
    def test_logistic_geometry_matches_source_and_saved_roundtrip(self):
        grid = np.linspace(0, 1, 201)
        for slope in (.05, 2., 20.):
            curve = BM1LogisticResponse(slope)
            np.testing.assert_allclose(curve(grid), lsq_params.logistic(grid, slope), atol=1e-15)
            self.assertEqual(tuple(curve(np.array([0., .5, 1.]))), (1., 0., -1.))
            self.assertTrue(np.all(np.diff(curve(grid)) < 0))
            np.testing.assert_allclose([curve.scalar_evaluator()(x) for x in grid], curve(grid), atol=1e-15)
            np.testing.assert_array_equal(response_from_record(curve.to_record())(grid), curve(grid))
            self.assertAlmostEqual(curve.max_abs_slope, slope/(2*np.tanh(slope/4)))
        for slope in (0., 21., np.nan):
            with self.assertRaises(ValueError):
                BM1LogisticResponse(slope)

    def test_day_fit_does_not_read_target_demand_and_saved_plot_replays(self):
        frame = frame_fixture()
        day = pd.Timestamp("2022-02-20")
        result = predict_day(day, frame=frame, score=True, max_iterations=1, max_nfev=3)
        changed = frame.copy()
        changed.loc[day:day+pd.Timedelta(hours=23), "I"] = .95
        other = predict_day(day, frame=changed, score=False, max_iterations=1, max_nfev=3)
        np.testing.assert_array_equal(result["hourly"]["I_predicted"], other["hourly"]["I_predicted"])
        self.assertEqual(len(result["training_days"]), 5)
        self.assertEqual(len(result["hourly"]), 24)
        with tempfile.TemporaryDirectory() as directory:
            output = save_result(result, output_root=Path(directory), plot=True)
            hourly, metadata = bm_1_plot_validation.saved(output, day)
            _, _, mean, _ = reconstructed_demand(hourly, metadata)
            np.testing.assert_allclose(mean, result["hourly"]["I_predicted"], atol=1e-14)
            plot_saved(day, output_root=Path(directory), import_actual=False, save=True)
            self.assertTrue((output/"forecast.png").is_file())
            self.assertTrue((output/"demand_heatmap.png").is_file())

    def test_mpc_uses_origin_measurement_but_not_future_demand(self):
        frame = frame_fixture()
        origin = pd.Timestamp("2022-02-20")
        baseline = statistical_output().fit(frame, origin)
        options = dict(max_iterations=1, max_nfev=3, x_steps=41, horizon=8)
        rows, diagnostics = predict_origin(frame, origin, baseline, **options)
        changed = frame.copy()
        changed.loc[origin+pd.Timedelta(hours=1):origin+pd.Timedelta(hours=8), "I"] = .95
        other, _ = predict_origin(changed, origin, baseline, **options)
        np.testing.assert_array_equal([r["I_BM1"] for r in rows], [r["I_BM1"] for r in other])
        self.assertEqual([r["lead_hours"] for r in rows], list(range(1, 9)))
        self.assertEqual(diagnostics["origin_observed_I"], frame.loc[origin, "I"])
        self.assertAlmostEqual(float(diagnostics["posterior"].sum()), 1.)
        for row in rows:
            self.assertEqual(row["target_time"], origin+pd.Timedelta(hours=row["lead_hours"]))
            self.assertTrue(0 <= row["I_BM1"] <= 1)
            self.assertLessEqual(row["lower_95"], row["upper_95"])


if __name__ == "__main__":
    unittest.main()

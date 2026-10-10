"""Regression tests for Windows-safe origin checkpointing and restart recovery."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.validation.exe.run_model.compare_mpc_predictions import compare_mpc_validation as checks


class WindowsMpcCheckpointTests(TestCase):
    OPTIONS = {"horizon": 8, "fit_status_schema": 4, "tag": "MED"}

    @staticmethod
    def rows(origin, prediction):
        return [{
            "origin": origin, "origin_day": origin.normalize(),
            "origin_hour": origin.hour, "target_time": origin + pd.Timedelta(hours=lead),
            "lead_hours": lead, "I_observed": 0.55, "I_BM1": prediction, "I_WM4": 0.53,
            "BM1_lower_68": 0.40, "BM1_upper_68": 0.60,
            "BM1_lower_95": 0.35, "BM1_upper_95": 0.65,
            "WM4_lower_68": 0.42, "WM4_upper_68": 0.62,
            "WM4_lower_95": 0.37, "WM4_upper_95": 0.67,
        } for lead in range(1, 9)]

    def test_immutable_origin_save_does_not_replace_open_aggregate(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2022-03-03"
            checks.write_settings(folder, self.OPTIONS)
            origin = pd.Timestamp("2022-03-03 16:00")
            aggregate = folder / "matched_forecasts.csv"
            aggregate.write_text("opened elsewhere", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("WinError 5")):
                saved = checks.write_origin(folder, self.OPTIONS, origin, self.rows(origin, 0.50))
            self.assertTrue(saved.is_file())
            self.assertEqual(aggregate.read_text(encoding="utf-8"), "opened elsewhere")
            recovered = checks.saved_day(folder, self.OPTIONS)
            self.assertEqual(len(recovered), 8)
            self.assertEqual(recovered["origin"].nunique(), 1)

    def test_restart_recovers_old_pending_plus_new_origin_files(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2022-03-03"
            checks.write_settings(folder, self.OPTIONS)
            first = pd.Timestamp("2022-03-03 00:00")
            second = first + pd.Timedelta(hours=1)
            third = first + pd.Timedelta(hours=2)
            pd.DataFrame(self.rows(first, 0.51)).to_csv(folder / "matched_forecasts.csv", index=False)
            pd.DataFrame(self.rows(first, 0.51) + self.rows(second, 0.52)).to_csv(
                folder / "matched_forecasts.pending.csv", index=False)
            checks.write_origin(folder, self.OPTIONS, third, self.rows(third, 0.53))
            recovered = checks.saved_day(folder, self.OPTIONS)
            self.assertEqual(recovered["origin"].nunique(), 3)
            self.assertEqual(len(recovered), 24)
            self.assertEqual(
                recovered.groupby("origin")["I_BM1"].first().tolist(), [0.51, 0.52, 0.53])

    def test_new_settings_dont_load_previous_configuration_or_incomplete_file(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2022-03-03"
            old = dict(self.OPTIONS)
            new = {**old, "horizon": 7}
            first = pd.Timestamp("2022-03-03 00:00")
            checks.write_settings(folder, old)
            pd.DataFrame(self.rows(first, 0.50)).to_csv(folder / "matched_forecasts.csv", index=False)
            checks.write_settings(folder, new)
            self.assertIsNone(checks.saved_day(folder, new))
            good = self.rows(first, 0.49)[:7]
            checks.write_origin(folder, new, first, good)
            checkpoint_dir = checks.checkpoint_directory(folder, new)
            (checkpoint_dir / "01_incomplete.csv").write_text("origin,lead_hours\n", encoding="utf-8")
            recovered = checks.saved_day(folder, new)
            self.assertEqual(len(recovered), 7)
            self.assertTrue(np.allclose(recovered["I_BM1"], 0.49))


if __name__ == "__main__":
    import unittest
    unittest.main()

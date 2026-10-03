"""Regression for real-data BM preview eligibility; all series are explicit fixtures."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

import bm_real_wm4_preview as bm


class PreviewWeekSelectionTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2022-01-01", "2022-05-31 23:00", freq="h")
        hours = index.hour.to_numpy()
        self.I = pd.Series(10. + 2.*np.cos(2.*np.pi*hours/24.), index=index)
        self.U = pd.Series(50. + 50.*np.cos(2.*np.pi*hours/24.), index=index)
        self.args = SimpleNamespace(start="auto", fit_days=7, warmup_hours=24,
                                    search_until=bm.DEFAULT_AUTO_SEARCH_END, scales_json=None)

    def select(self, I=None, U=None):
        output = io.StringIO()
        with redirect_stdout(output):
            window, skipped = bm.choose_week(self.I if I is None else I,
                                             self.U if U is None else U, self.args)
        return window, skipped, output.getvalue()

    def test_complete_data_keeps_original_first_week(self):
        window, skipped, output = self.select()
        self.assertEqual(window["start"], pd.Timestamp("2022-02-14"))
        self.assertEqual(skipped, [])
        self.assertEqual(output, "")

    def test_reported_january_gap_can_find_a_march_week(self):
        self.I.loc["2022-01-16 00:00"] = np.nan
        window, skipped, output = self.select()
        self.assertEqual(window["start"], pd.Timestamp("2022-03-07"))
        self.assertEqual(len(skipped), 3)
        self.assertIn("2022-01-16", output)
        self.assertTrue(np.isnan(self.I.loc["2022-01-16 00:00"]))
        self.assertTrue(all(pd.Timestamp(t) < window["start"]
                            for sources in window["future_baseline_sources"] for t in sources))

    def test_explicit_short_boundary_still_stops(self):
        self.I.loc["2022-01-16 00:00"] = np.nan
        self.args.search_until = "2022-03-01"
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "checked=2"):
            bm.choose_week(self.I, self.U, self.args)

    def test_domains_and_dst_are_checked_before_returning_a_week(self):
        self.I.loc["2022-01-16 00:00"] = np.nan
        self.U.loc["2022-03-07 12:00"] = 200.
        self.U.loc["2022-03-14 12:00"] = -25.
        window, skipped, output = self.select()
        self.assertEqual(window["start"], pd.Timestamp("2022-04-11"))
        self.assertEqual(len(skipped), 8)
        self.assertIn("forecast price exceeds", output)
        self.assertIn("2022-03-27", output)
        self.assertEqual(window["scales"]["price_min"], -25.)
        self.assertEqual(window["scales"]["price_max"], 200.)
        for name in ("train_b", "future_b", "train_u", "future_u"):
            self.assertTrue(((window[name]>=0.) & (window[name]<=1.)).all())

    def test_future_consumption_does_not_change_selection_or_prepared_model_inputs(self):
        self.I.loc["2022-01-16 00:00"] = np.nan
        window, _, _ = self.select()
        changed = self.I.copy()
        changed.loc[changed.index >= window["start"]] = 999999.
        other, _, _ = self.select(I=changed)
        self.assertEqual(window["start"], other["start"])
        self.assertEqual(window["scales"], other["scales"])
        for key in ("train_y", "train_b", "future_b", "train_u", "future_u"):
            np.testing.assert_array_equal(window[key], other[key])

    def test_explicit_week_is_not_silently_moved_when_domain_fails(self):
        self.args.start = "2022-03-07"
        self.U.loc["2022-03-07 12:00"] = 200.
        with self.assertRaisesRegex(ValueError, "forecast price exceeds"):
            bm.choose_week(self.I, self.U, self.args)

    def test_missing_dst_history_is_not_filled_or_skipped_inside_baseline(self):
        self.I.loc["2022-01-16 00:00"] = np.nan
        self.I.loc["2022-03-27 02:00"] = np.nan
        self.U.loc["2022-03-07 12:00"] = 200.
        self.U.loc["2022-03-14 12:00"] = -25.
        window, skipped, output = self.select()
        self.assertEqual(window["start"], pd.Timestamp("2022-05-16"))
        self.assertIn("Missing original baseline history", output)
        self.assertTrue(np.isnan(self.I.loc["2022-03-27 02:00"]))


if __name__ == "__main__":
    unittest.main()

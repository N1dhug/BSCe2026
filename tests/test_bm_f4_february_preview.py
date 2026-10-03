"""Small fixture checks for the disposable BM preview; not scientific validation."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bm_f4_february_preview as preview


class BMPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        shutil.copytree(ROOT / "src", cls.root / "src")
        data = cls.root / "files/input/data"
        data.mkdir(parents=True)
        index = pd.date_range("2022-01-01", "2022-02-14", freq="h", inclusive="left")
        values = 100 + 15*np.sin(2*np.pi*(index.hour.to_numpy()-6)/24)
        values += 4*(index.dayofweek.to_numpy() > 4)
        book = Workbook()
        sheet = book.active
        sheet.title = "2022 med dato (3)"
        sheet.append(["Unnamed: meta", "Unnamed: time", "fixture_meter", "partial_meter", "empty_meter"])
        sheet.append(["Dato", "Time", "MED", "MED", "MED"])
        for stamp, value in zip(index, values):
            partial = None if stamp in (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-02-09 03:00")) else float(value+20)
            sheet.append([stamp.date(), int(stamp.hour), float(value), partial, None])
        book.save(data / "input.xlsx")
        # A genuine full-year time structure includes October's repeated local
        # hour; that unrelated ambiguity must not reject the February run.
        utc = pd.date_range("2021-12-31 23:00", "2022-12-31 23:00", freq="h",
                            inclusive="left", tz="UTC")
        local = utc.tz_convert("Europe/Copenhagen")
        prices = 400 + 100*np.sin(2*np.pi*(local.hour.to_numpy()-9)/24)
        pd.DataFrame(dict(HourUTC=utc.tz_localize(None), PriceArea="DK1",
                          SpotPriceDKK=prices)).to_excel(data / "prices.xlsx", index=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def args(self):
        return SimpleNamespace(power_scale=None, price_low=None, price_high=None,
            curve_mode="auto", curve_train_start="2022-01-01", curve_rho=1.,
            curve_xi=.75, k=2., sigma_x=.5, sigma_y=.03)

    def record(self):
        preview.repository(self.root)
        start = pd.Timestamp("2022-02-07")
        frame, _ = preview.load_raw(self.root, "MED", start+pd.Timedelta(days=7))
        selected, scaling, dates = preview.prepare(frame, start, self.args())
        data, n_train, _ = preview.curve_dataset(frame, start, scaling, self.args())
        return frame, selected, scaling, dates, data, n_train

    def test_partial_meters_match_wm_raw_mean_without_imputation(self):
        frame, selected, _, _, _, _ = self.record()
        from src.tools.import_data.import_input import input_data
        reader = input_data(self.root / "files/input/data/input.xlsx")
        columns = [c for c in reader.measurement_columns if reader.measurement_tag(c) == "MED"]
        expected = reader.observed_data[columns].mean(axis=1).dropna().sort_index()
        pd.testing.assert_series_equal(frame.I_raw.dropna(), expected, check_names=False)
        pd.testing.assert_series_equal(frame.observed_meter_count,
                                      reader.observed_data[columns].notna().sum(axis=1), check_names=False)
        self.assertEqual(frame.loc["2022-01-03", "observed_meter_count"].iloc[0], 1)
        self.assertEqual(frame.loc["2022-01-03 00:00", "I_raw"], 85.)
        self.assertEqual(selected.loc["2022-02-09 03:00", "observed_meter_count"], 1)
        self.assertTrue(frame.selected_meter_count.eq(3).all())
        _, metadata = preview.load_raw(self.root, "MED", pd.Timestamp("2022-02-14"))
        self.assertFalse(metadata["consumption_imputation"])
        self.assertEqual(metadata["population_policy"], "wm-observed-mean")
        strict, _ = preview.load_raw(self.root, "MED", pd.Timestamp("2022-02-14"), "strict")
        args = self.args()
        args.population_policy = "strict"
        # All-NaN meters are not silently filled or dropped in strict mode.
        self.assertTrue(strict.I_raw.isna().all())
        partial = frame.copy()
        partial.loc["2022-01-03 00:00", "I_raw"] = np.nan
        with self.assertRaisesRegex(ValueError, "2022-01-03T00:00:00.*1/3 observed meters"):
            preview.prepare(partial, pd.Timestamp("2022-02-07"), args)

    def test_all_missing_source_hour_is_rejected_with_its_timestamp(self):
        frame, *_ = self.record()
        frame.loc["2022-01-04 00:00", "I_raw"] = np.nan
        frame.loc["2022-01-04 00:00", "observed_meter_count"] = 0
        with self.assertRaisesRegex(ValueError, "2022-01-04T00:00:00.*0/3 observed meters"):
            preview.prepare(frame, pd.Timestamp("2022-02-07"), self.args())

    def test_holdout_and_selection_do_not_leak_into_training(self):
        frame, selected, scaling, dates, data, n_train = self.record()
        start = pd.Timestamp("2022-02-07")
        changed = frame.copy()
        changed.loc[changed.index >= start, "I_raw"] += 500
        s2, scale2, dates2 = preview.prepare(changed, start, self.args())
        d2, n2, _ = preview.curve_dataset(changed, start, scale2, self.args())
        self.assertEqual(scaling, scale2)
        self.assertEqual(dates, dates2)
        self.assertEqual(n_train, n2)
        pd.testing.assert_frame_equal(data, d2)
        np.testing.assert_array_equal(selected.B, s2.B)
        changed = frame.copy()
        mask = (changed.index >= start-pd.Timedelta(days=7)) & (changed.index < start)
        changed.loc[mask, "I_raw"] *= 1.7
        _, scale3, _ = preview.prepare(changed, start, self.args())
        d3, n3, _ = preview.curve_dataset(changed, start, scale3, self.args())
        self.assertEqual(scaling, scale3)
        self.assertEqual(n_train, n3)
        pd.testing.assert_frame_equal(data.iloc[:n_train], d3.iloc[:n_train])
        np.testing.assert_array_equal(data.B, d3.B)

    def test_filter_continuation_has_no_midnight_reset(self):
        _, _, _, _, data, n_train = self.record()
        args = self.args()
        curve = preview.shape_spec(np.array([.8, -.3, .2, -.7, .5, .1]), 4)
        whole = preview.curve_filter(data, curve, args, cells=11, max_step=1/24)
        train = preview.curve_filter(data.iloc[:n_train], curve, args, cells=11, max_step=1/24)
        continued = preview.curve_filter(data, curve, args, cells=11, max_step=1/24,
            initial_mass=train.posterior_mass[-1], initial_hour=n_train-1,
            observation_hours=np.arange(n_train, len(data)))
        np.testing.assert_array_equal(whole.posterior_mass[n_train-1:], continued.posterior_mass)
        np.testing.assert_array_equal(whole.log_evidence[n_train-1:], continued.log_evidence)

    def test_cli_fits_curves_and_controls_with_wm_and_network_blocked(self):
        guard = self.root / "import_guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text('''
import importlib.abc
import importlib.machinery
import socket
import sys
class BlockWM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("src.exe.run_model.wm_", "src.tools.import_data.import_weather")):
            raise RuntimeError("WM/weather imports forbidden in BM-only preview")
        if fullname.startswith(("src.model.wm_", "src.params.wm_")):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            original = spec.loader
            class GuardLoader(importlib.abc.Loader):
                def create_module(self, spec):
                    return original.create_module(spec)
                def exec_module(self, module):
                    original.exec_module(module)
                    def forbidden(*args, **kwargs):
                        raise RuntimeError("WM model/parameter construction forbidden")
                    for value in vars(module).values():
                        if isinstance(value, type) and value.__module__ == module.__name__:
                            value.__init__ = forbidden
            spec.loader = GuardLoader()
            return spec
sys.meta_path.insert(0, BlockWM())
_connect = socket.socket.connect
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("Internet connection forbidden in BM-only preview")
    return _connect(self, address)
socket.socket.connect = connect
''', encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(guard) + os.pathsep + env.get("PYTHONPATH", "")
        output = self.root / "output"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "bm_f4_february_preview.py"),
            "--repo", str(self.root), "--nodes", "2", "--cells", "41",
            "--max-step", str(1/96), "--workers", "2", "--curve-evaluations", "6",
            "--curve-cells", "21", "--curve-max-step", str(1/48),
            "--control-evaluations", "8", "--observation-contract", "hourly-point-approximation",
            "--output-dir", str(output)], capture_output=True, text=True, env=env,
            timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["models_executed"], ["BM1"])
        self.assertEqual(summary["data"]["population_policy"], "wm-observed-mean")
        self.assertFalse(summary["data"]["consumption_imputation"])
        self.assertEqual(set(summary["metrics"]), {"BM", "B"})
        self.assertEqual(set(summary["raw_unit_metrics"]), {"BM", "B"})
        self.assertNotIn("wm4", summary)
        self.assertLess(pd.Timestamp(summary["normalization"]["fitted_before"]), pd.Timestamp(summary["origin"]))
        curves = json.loads((output / "response_curves.json").read_text())
        self.assertEqual(curves["selected"], summary["response_curves"])
        self.assertEqual(len(curves["metadata"]["candidates"]), 3)
        self.assertFalse(curves["metadata"]["completed_H_fg_claim"])
        self.assertFalse(summary["control"]["curves_refitted"])
        self.assertEqual(summary["control"]["curve_family"], curves["selected"]["family"])
        hourly = pd.read_csv(output / "hourly_results.csv")
        self.assertEqual(len(hourly), 168)
        self.assertEqual(hourly.observed_meter_count.min(), 1)
        self.assertEqual(hourly.observed_meter_count.max(), 2)
        self.assertTrue(hourly.selected_meter_count.eq(3).all())
        self.assertFalse(any("wm" in c.lower() or c == "Temp" for c in hourly.columns))
        self.assertEqual(len(pd.read_csv(output / "control_hourly.csv")), 168)
        self.assertEqual(len(pd.read_csv(output / "curve_validation.csv")), 3)
        for name in ("bm_weekly_prediction.png", "response_curves.png", "control.png"):
            self.assertGreater((output / name).stat().st_size, 10000)
        self.assertIn("remaining~", result.stdout)
        self.assertIn("total~", result.stdout)
        self.assertTrue((output / "origin_bank.npz").is_file())
        for folder in ('forecast_distribution', 'control_distribution'):
            info = json.loads((output / folder / 'summary.json').read_text())
            for name in info['files']:
                self.assertGreater((output / folder / name).stat().st_size, 0)
        exported = pd.read_csv(output / 'forecast_distribution/moments.csv')
        for field, original in [('D_mean', 'I_bm'), ('X_mean', 'X_bm'),
                                ('D_p05', 'D_p05'), ('D_p95', 'D_p95')]:
            np.testing.assert_allclose(exported[field], hourly[original], rtol=0, atol=1e-14)
        counterfactual = json.loads((output / 'control_distribution/summary.json').read_text())
        self.assertEqual(counterfactual['mode'], 'counterfactual_control')
        self.assertEqual(counterfactual['metrics'], {})
        with np.load(output / 'forecast_distribution/distribution.npz', allow_pickle=False) as raw:
            np.testing.assert_allclose(raw['joint_mass'].sum(axis=(0, 2)), 1., atol=1e-10)
            self.assertEqual(raw['joint_mass'].shape, raw['demand_values'].shape)
        existing = subprocess.run([sys.executable, "-B", str(ROOT / "bm_f4_february_preview.py"),
            "--repo", str(self.root), "--output-dir", str(output)],
            capture_output=True, text=True, env=env, timeout=30)
        self.assertNotEqual(existing.returncode, 0)
        self.assertIn("Output directory already exists", existing.stderr)
        self.assertNotIn("Target 2022-", existing.stdout)


if __name__ == "__main__":
    unittest.main()

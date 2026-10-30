import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.exe.run_model.wm_5.single_step import plot_single_step
from src.exe.run_model.wm_5.single_step_input import build
from src.model.wm_5 import response_library
from src.validation.exe.run_model.wm_5.single_step import single_step_validation
from src.validation.model.wm_4 import wm_4_validation
from src.validation.model.wm_5 import wm_5_validation


class test_single_step(unittest.TestCase):
    def test_price_drop_increases_demand_above_baseline(self):
        for variant in ("wm_4", "wm_5"):
            with self.subTest(variant=variant):
                example = build(variant, duration=24.0, dt=0.02, step_at=2.0)
                state = example["state"]
                t = np.asarray(state["t"])
                after = t > example["step_at"]
                demand_shift = state["I"] - state["O"]
                self.assertGreater(float(np.max(demand_shift[after])), 0.0)
                self.assertAlmostEqual(float(demand_shift[0]), 0.0, places=12)
                self.assertLessEqual(float(example["coefficients"][1 if variant == "wm_4" else -1]), 0.0)
                self.assertLess(example["responses"][2](-1.0), example["responses"][2](1.0))
                np.testing.assert_allclose(example["contributions"].sum(axis=1), state["R_dot"], atol=1e-12)

    def test_negative_price_coefficient_required(self):
        with self.assertRaises(ValueError):
            wm_4_validation.r_coefficients([-0.25, 0.01, -0.8])
        library = response_library()
        with self.assertRaises(ValueError):
            wm_5_validation.coefficients([-0.25, -0.8, 0.01],
                                         [library["linear"]], [library["linear"]], 0.02)
        with self.assertRaises(ValueError):
            single_step_validation.price_response(0.01, lambda x: x)
        with self.assertRaises(ValueError):
            single_step_validation.price_response(-0.2, lambda x: -x)

    def test_one_figure_for_all_panels(self):
        example = build("wm_5", duration=8.0, dt=0.04, step_at=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_single_step(example, Path(tmp))
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())
            self.assertGreater(paths[0].stat().st_size, 1000)
            self.assertFalse((Path(tmp) / "wm_5_response_functions.png").exists())


if __name__ == "__main__": unittest.main()

"""Smoke tests for demand-error-based WM4 diffusion sweep."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_weekday_horizon_diffusion import (
    _prepare, _unit_process_variance, save_plot, sweep,
)


class weekday_horizon_diffusion_tests(unittest.TestCase):
    def test_kernel_starts_positive_and_accumulates(self):
        values = [_unit_process_variance(-1.0, -0.3, k, 4) for k in (1, 2, 3)]
        self.assertTrue(all(np.isfinite(values)))
        self.assertGreater(values[0], 0.0)
        self.assertGreater(values[1], values[0])

    def test_grid_estimation_and_plot(self):
        rng = np.random.default_rng(13)
        rows = []
        for baseline in ("Weekly average", "Weekly temperature"):
            for day in range(24):
                for lead in range(1, 7):
                    rf, rr = -1.0, -0.3
                    u = _unit_process_variance(rf, rr, lead, 4)
                    e = rng.normal(scale=np.sqrt(0.01 + 0.03 * u))
                    rows.append({
                        "N": 3, "lead_hours": lead, "I_observed": 0.4 + e,
                        "I_mpc": 0.4, "r_F": rf, "r_R": rr,
                        "variant": baseline, "tag": "MED",
                        "origin": f"2022-01-{day + 1:02d}T00:00:00",
                    })
        data = _prepare(pd.DataFrame(rows), 4)
        result = sweep(data, (3,), (1, 2, 3, 4, 5, 6), min_samples=6)
        self.assertFalse(result.empty)
        self.assertTrue((result["observation_q"] >= 0.0).all())
        self.assertTrue((result["process_q"] >= 0.0).all())
        self.assertTrue(result["distribution_similarity"].dropna().between(0.0, 1.0).all())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sweep.png"
            save_plot(result, path, (3,), (1, 2, 3, 4, 5, 6))
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()

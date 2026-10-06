"""Focused check of independent trajectories and warm-up masks."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1.deterministic_weekday import run
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_deterministic_weekday import BM1WeekdayInputs


class WeekdayInputsTests(unittest.TestCase):
    def trajectories(self):
        return [BM1ResponseInputs(np.arange(49.), np.full(49, .3+j*.05),
            np.full(48, .1+j*.15), np.full(48, .4), warmup_hours=24,
            steps_per_hour=2, observation_mask=np.r_[np.ones(24, bool),
                np.full(24, j != 2, dtype=bool)]) for j in range(5)]

    def test_independent_replays_and_measured_target_mask(self):
        histories = self.trajectories()
        pooled = BM1WeekdayInputs(histories)
        self.assertEqual(pooled.observation_mask.sum(), 96)
        self.assertFalse(pooled.observation_mask.reshape(5, 48)[:, :24].any())
        p = [10., .2, .3, .5]
        separate = [t.solve(p) for t in histories]
        combined = pooled.solve(p)
        np.testing.assert_array_equal(combined['demand'],
            np.concatenate([s['demand'] for s in separate]))
        np.testing.assert_array_equal(combined['historical_start_states'],
            [s['state'][24] for s in separate])
        self.assertEqual(combined['rhs_evaluations'], 5*48*2*4)
        self.assertEqual(pooled.refined().steps, 4)

        with self.assertRaises(ValueError):
            BM1WeekdayInputs(self.trajectories()[:4])

    def test_target_demand_cannot_change_frozen_forecast(self):
        index = pd.date_range('2022-01-01', periods=40*24, freq='h')
        raw = pd.DataFrame({'Demand': .4+.1*np.sin(np.arange(len(index))/7),
                            'Price': 1+np.cos(np.arange(len(index))/11),
                            'Temp': 8+np.sin(np.arange(len(index))/24)}, index=index)
        day = pd.Timestamp('2022-02-07')
        changed = raw.copy()
        changed.loc[changed.index >= day, 'Demand'] = 1e9
        p = [10., .2, .3, .5]
        def fake_fit(inputs, reference, **kwargs):
            return {'status': 'complete', 'counts': {}, 'seconds': 0., 'candidates': [
                {'regularization': strength, 'best_converged': {'parameters': p,
                 'g_mid': middle, 'mse': 0., 'active_bounds': [0, 0, 0]}}
                for strength, middle in [(0., 0.), (.001, .1)]]}
        with TemporaryDirectory() as folder, patch(
                'src.exe.run_model.bm_1.deterministic_weekday.fit_response', fake_fit):
            for name, data in [('original', raw), ('changed', changed)]:
                run(data, p, day, 1, Path(folder)/name, steps=2)
            relative = Path('2022-02-07/forecast_normalized.csv')
            self.assertEqual((Path(folder)/'original'/relative).read_bytes(),
                             (Path(folder)/'changed'/relative).read_bytes())


if __name__ == '__main__':
    unittest.main()

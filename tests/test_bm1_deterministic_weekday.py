"""Independent historical day trajectories and exclusion of warmup targets."""
import unittest
import numpy as np
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_deterministic_weekday import BM1PooledInputs

class WeekdayInputsTests(unittest.TestCase):
    def trajectories(self):
        return [BM1ResponseInputs(np.arange(49.), np.full(49, .3+j*.05),
            np.full(48, .1+j*.15), np.full(48, .4), warmup_hours=24,
            steps_per_hour=2, observation_mask=np.r_[np.ones(24, bool),
                np.full(24, j != 2, dtype=bool)]) for j in range(5)]

    def test_independent_replays_and_measured_target_mask(self):
        histories = self.trajectories()
        pooled = BM1PooledInputs(histories)
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


if __name__=='__main__':unittest.main()

"""G development: five independent matching weekdays, pooled G4 loss.

This adapter reuses the G4 optimizer and forward kernel. It never concatenates
separated calendar days into a fictitious continuous state trajectory.
"""
import numpy as np
from src.method.bm_1_deterministic_response import BM1ResponseInputs


class BM1PooledInputs(BM1ResponseInputs):
    def __init__(self, trajectories):
        self.trajectories = tuple(trajectories)
        if not self.trajectories or not all(isinstance(t, BM1ResponseInputs) for t in self.trajectories):
            raise ValueError('Require independent G4 trajectories.')
        first = self.trajectories[0]
        if any(t.n_hours not in (47,48) or t.warmup != 24 or t.observations is None for t in self.trajectories):
            raise ValueError('Each history needs 24 h warm-up and a 23/24 h target.')
        super().__init__(first.time, first.baseline, first.prices,
                         state_response=first.state_response, steps_per_hour=first.steps)
        self.observations = np.concatenate([t.observations for t in self.trajectories])
        masks = []
        for t in self.trajectories:
            mask = t.observation_mask.copy()
            mask[:24] = False
            masks.append(mask)
        self.observation_mask = np.concatenate(masks)
        self.warmup = 0  # Each constituent mask already excludes its warm-up.

    def solve(self, parameters, *, deadline=None, keep_fine=False, price_response=None):
        self.last_rhs_evaluations = 0
        outputs = []
        for trajectory in self.trajectories:
            try:
                outputs.append(trajectory.solve(parameters, deadline=deadline,
                    keep_fine=keep_fine, price_response=price_response))
            finally:
                self.last_rhs_evaluations += trajectory.last_rhs_evaluations
        return {'demand': np.concatenate([o['demand'] for o in outputs]),
                'historical_start_states': np.array([o['state'][24] for o in outputs]),
                'rhs_evaluations': self.last_rhs_evaluations}

    def refined(self):
        return type(self)([t.refined() for t in self.trajectories])


class BM1WeekdayInputs(BM1PooledInputs):
    def __init__(self, trajectories):
        trajectories = tuple(trajectories)
        if len(trajectories) != 5:
            raise ValueError('Require exactly five independent G4 trajectories.')
        super().__init__(trajectories)

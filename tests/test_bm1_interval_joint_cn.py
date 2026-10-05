"""Independent semigroup accuracy and positivity gate for the isolated joint CN."""

import unittest
import numpy as np
from scipy.linalg import expm

from src.solver.finite_volume_additive import propagate_additive
from src.validation.bm_1.interval_joint_cn import PositiveCNJointSolver


class JointCNTests(unittest.TestCase):
    def test_constant_reward_markov_semigroup_refines_and_negative_step_is_rejected(self):
        solver = PositiveCNJointSolver()
        prior = np.array([.2, .5, .3])
        drift = lambda t, x: np.full(x.shape, .03)
        diffusion = lambda t, x: np.full(x.shape, .3)
        reward = lambda t, x: np.full(x.shape, .5)
        lo, di, up, _ = solver.generator_tridiagonal(1/3, [.03]*2, [.3]*3)
        exact = expm(np.diag(di)+np.diag(lo, -1)+np.diag(up, 1)) @ prior
        errors = []
        for steps in (16, 32):
            result = propagate_additive(solver, [0., 1.], prior, drift, diffusion, reward,
                max_step=1/steps, accumulator_spacing=1/11)
            self.assertEqual(result['joint_mass'].shape, (3, 1))
            self.assertAlmostEqual(result['accumulator'][0], .5, places=14)
            self.assertTrue(np.all(result['joint_mass'] >= 0.))
            self.assertAlmostEqual(result['joint_mass'].sum(), 1., places=13)
            errors.append(np.abs(result['joint_mass'][:, 0]-exact).sum())
        self.assertGreater(errors[0]/errors[1], 3.9)
        with self.assertRaisesRegex(ValueError, 'CN positivity'):
            propagate_additive(solver, [0., 1.], prior, drift,
                lambda t, x: np.full(x.shape, 5.), reward,
                max_step=1., accumulator_spacing=1/11)

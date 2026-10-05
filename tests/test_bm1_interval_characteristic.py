"""Independent checks for the optional synthetic characteristic control."""

import unittest
import numpy as np

from src.params.bm_1 import gaussian_log_likelihood
from src.solver.finite_volume import finite_volume_1d_fitted
from src.validation.bm_1.interval import finite_state_gaussian_reference
from src.validation.bm_1.interval_characteristic import (
    characteristic_prediction, characteristic_gaussian_update, CharacteristicNumericalError)


class CharacteristicIntervalTests(unittest.TestCase):
    def fields(self):
        solver = finite_volume_1d_fitted()
        prior = np.array([.2,.5,.3])
        drift = lambda t,x: np.full(x.shape,.03)
        diffusion = lambda t,x: np.full(x.shape,.3)
        reward = lambda t,x: np.array([.15,.5,.85])
        lo,di,up,_ = solver.generator_tridiagonal(1/3,[.03]*2,[.3]*3)
        Q = np.diag(di)+np.diag(lo,-1)+np.diag(up,1)
        return solver,prior,drift,diffusion,reward,Q

    def test_tilted_and_plain_laws_match_independent_finite_state_reference(self):
        solver,prior,drift,diffusion,reward,Q = self.fields()
        ref = finite_state_gaussian_reference(Q,reward(0.,None),prior,1.,.62,.12,order=160)
        scores = []
        for tilt,center in ((0.,0.),((.62-.55)/.12**2,.55)):
            pred = characteristic_prediction(solver,[0.,1.],prior,drift,diffusion,reward,
                max_step=1/512,frequencies=64,span=4.,tilt=tilt,center=center)
            result = characteristic_gaussian_update(pred,.62,.12)
            scores.append(result['log_evidence'])
            self.assertLess(abs(result['log_evidence']-ref['log_evidence']),.002)
            np.testing.assert_allclose(result['posterior_mass'],ref['posterior_mass'],atol=.002)
            self.assertLess(result['relative_series_guard'],1e-8)
            np.testing.assert_allclose(pred['predicted_mass'],
                solver.solve([0.,1.],prior,drift,diffusion,max_step=1/512)['cell_mass'][-1],atol=1e-14)
        self.assertAlmostEqual(scores[0],scores[1],places=10)

    def test_constant_reward_exponential_identity_handles_rare_gaussian_without_cancellation(self):
        solver,prior,drift,diffusion,_,_ = self.fields()
        center,value,sigma = .5,1.5,.08
        prediction = characteristic_prediction(solver,[0.,1.],prior,drift,diffusion,
            lambda t,x: np.full(x.shape,center),max_step=1/32,frequencies=128,
            tilt=(value-center)/sigma**2,center=center)
        result = characteristic_gaussian_update(prediction,value,sigma)
        expected = gaussian_log_likelihood(value,[center],sigma)[0]
        self.assertAlmostEqual(result['log_evidence'],expected,places=11)
        np.testing.assert_allclose(result['posterior_mass'],prediction['predicted_mass'],atol=3e-15)

    def test_low_frequency_and_negative_density_abort_without_clipping(self):
        solver,prior,drift,diffusion,reward,_ = self.fields()
        prediction = characteristic_prediction(solver,[0.,1.],prior,drift,diffusion,reward,
                                               max_step=.1,frequencies=1,span=4.)
        with self.assertRaisesRegex(ValueError,'guard unresolved'):
            characteristic_gaussian_update(prediction,.62,.12)
        bad = dict(prediction,cf=np.array([[-1.+0.j,-1.+0.j]]*3))
        with self.assertRaisesRegex(ValueError,'positivity'):
            characteristic_gaussian_update(bad,.62,.12)

    def test_unresolved_subnormal_transform_aborts_instead_of_repairing_tails(self):
        solver = finite_volume_1d_fitted()
        prior = np.array([1.,1e-316,1e-316])
        rewards = np.array([.15,.5,.85])
        zero = lambda t,x: np.zeros(x.shape)
        tilt,center = (.62-.4)/.12**2,.4
        original = prior.copy()
        with self.assertRaises(CharacteristicNumericalError):
            characteristic_prediction(solver,[0.,1.],prior,zero,zero,
                lambda t,x:rewards,max_step=1/128,frequencies=128,tilt=tilt,center=center)
        np.testing.assert_array_equal(prior,original)


if __name__ == '__main__':
    unittest.main()

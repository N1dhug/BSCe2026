"""Independent checks for the optional synthetic characteristic control."""

import unittest
import numpy as np

from src.params.bm_1 import gaussian_log_likelihood
from src.solver.finite_volume import finite_volume_1d_fitted
from src.validation.bm_1.interval import finite_state_gaussian_reference
from src.validation.bm_1.interval_characteristic import (
    characteristic_prediction, balanced_characteristic_prediction, characteristic_gaussian_update,
    _balanced_state_step, CharacteristicNumericalError)


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

    def test_balanced_step_matches_dense_matrix_solve_for_complex_columns(self):
        _,prior,_,_,_,Q = self.fields()
        h = .08
        matrix = np.eye(3)-h*Q
        conditional = np.array([[1., .2+.7j], [1., -.4+.2j], [1., .8-.1j]])
        expected = np.linalg.solve(matrix, prior[:,None]*conditional)
        logs, result = _balanced_state_step(np.log(prior), conditional,
            np.diag(matrix,-1),np.diag(matrix),np.diag(matrix,1))
        np.testing.assert_allclose(np.exp(logs)[:,None]*result,expected,rtol=3e-15,atol=1e-16)

    def test_balanced_law_matches_independent_reference_and_unscaled_transform(self):
        solver,prior,drift,diffusion,reward,Q = self.fields()
        tilt,center = (.62-.55)/.12**2,.55
        kwargs = dict(max_step=1/512,frequencies=64,span=4.,tilt=tilt,center=center)
        ordinary = characteristic_prediction(solver,[0.,1.],prior,drift,diffusion,reward,**kwargs)
        balanced = balanced_characteristic_prediction(solver,[0.,1.],prior,drift,diffusion,reward,**kwargs)
        np.testing.assert_allclose(np.exp(balanced['log_amplitude'])[:,None]*balanced['cf'],
            ordinary['cf'],rtol=2e-12,atol=1e-14)
        actual = characteristic_gaussian_update(balanced,.62,.12)
        reference = finite_state_gaussian_reference(Q,reward(0.,None),prior,1.,.62,.12,order=160)
        self.assertLess(abs(actual['log_evidence']-reference['log_evidence']),.002)
        np.testing.assert_allclose(actual['posterior_mass'],reference['posterior_mass'],atol=.002)

    def test_balanced_transform_preserves_below_float_range_and_disconnected_zeros(self):
        solver = finite_volume_1d_fitted()
        logs = np.array([0.,-900.,-950.])
        prior = np.exp(logs)
        zero = lambda t,x: np.zeros(x.shape)
        rewards = np.array([.15,.5,.85])
        pred = balanced_characteristic_prediction(solver,[0.,1.],prior,zero,zero,
            lambda t,x: rewards,max_step=1/128,frequencies=128,tilt=4.,center=.4,log_prior=logs)
        update = characteristic_gaussian_update(pred,.62,.12)
        expected = logs+gaussian_log_likelihood(.62,rewards,.12)
        from scipy.special import logsumexp
        expected -= logsumexp(expected)
        np.testing.assert_allclose(update['log_posterior_mass'],expected,atol=2e-12)
        self.assertTrue(np.isfinite(update['log_posterior_mass']).all())
        # A genuinely zero disconnected state stays zero, without a probability floor.
        pred = balanced_characteristic_prediction(solver,[0.,1.],[1.,0.,0.],zero,zero,
            lambda t,x: rewards,max_step=1/32,frequencies=128)
        update = characteristic_gaussian_update(pred,.62,.12)
        np.testing.assert_array_equal(update['posterior_mass'],[1.,0.,0.])
        self.assertTrue(np.isneginf(update['log_posterior_mass'][1:]).all())

    def test_cn_state_step_matches_dense_product_and_enforces_positivity(self):
        solver,prior,drift,diffusion,_,Q = self.fields()
        h = .1
        pred = balanced_characteristic_prediction(solver,[0.,h],prior,drift,diffusion,
            lambda t,x: np.zeros(x.shape),max_step=h,state_scheme='positive_crank_nicolson')
        expected = np.linalg.solve(np.eye(3)-h/2*Q,(np.eye(3)+h/2*Q) @ prior)
        np.testing.assert_allclose(pred['predicted_mass'],expected,rtol=3e-15,atol=1e-16)
        np.testing.assert_allclose(np.exp(pred['log_amplitude']),expected,rtol=3e-15,atol=1e-16)
        with self.assertRaisesRegex(CharacteristicNumericalError,'explicit half is not positive'):
            balanced_characteristic_prediction(solver,[0.,10.],prior,drift,diffusion,
                lambda t,x: np.zeros(x.shape),max_step=10.,state_scheme='positive_crank_nicolson')

    def test_cn_interval_score_converges_quadratically_to_independent_reference(self):
        solver,prior,drift,diffusion,reward,Q = self.fields()
        ref = finite_state_gaussian_reference(Q,reward(0.,None),prior,1.,.62,.12,order=160)
        errors=[]
        for h in (1/16,1/32,1/64):
            pred=balanced_characteristic_prediction(solver,[0.,1.],prior,drift,diffusion,reward,
                max_step=h,state_scheme='positive_crank_nicolson',tilt=(.62-.55)/.12**2,center=.55)
            result=characteristic_gaussian_update(pred,.62,.12)
            errors.append(abs(result['log_evidence']-ref['log_evidence']))
        self.assertGreater(errors[0]/errors[1],3.5)
        self.assertGreater(errors[1]/errors[2],3.5)
        self.assertLess(errors[-1],1e-4)

    def test_exact_operator_reuse_preserves_changing_fields_and_reward(self):
        solver,prior,drift,diffusion,_,_ = self.fields()
        for varying_state in (False,True):
            calls = {'drift':0,'reward':0}
            def a(t,x):
                calls['drift'] += 1
                return drift(t,x)+(t*.02 if varying_state else 0.)
            def reward(t,x):
                calls['reward'] += 1
                # More than sixteen distinct phases exercises cache eviction.
                return np.array([.15,.5,.85])+.02*np.sin(19.*t)
            key = None if varying_state else lambda t:0
            kwargs = dict(max_step=1/64,frequencies=12,coefficient_key=key,
                          state_scheme='positive_crank_nicolson',tilt=3.,center=.55)
            cached = balanced_characteristic_prediction(solver,[0.,1.],prior,a,diffusion,reward,**kwargs)
            cached_calls = calls.copy()
            calls.update(drift=0,reward=0)
            plain = balanced_characteristic_prediction(solver,[0.,1.],prior,a,diffusion,reward,
                reuse_frozen_operators=False,**kwargs)
            for name in ('cf','log_amplitude','predicted_mass'):
                np.testing.assert_array_equal(cached[name],plain[name])
            self.assertEqual(calls,cached_calls)
            self.assertEqual(calls['reward'],64)
            self.assertEqual(cached['operator_reuse']['phase_builds'],64)
            if varying_state:
                self.assertEqual(cached['operator_reuse']['factor_builds'],64)
            else:
                self.assertEqual(cached['operator_reuse']['factor_builds'],1)
                self.assertEqual(cached['operator_reuse']['factor_hits'],63)


if __name__ == '__main__':
    unittest.main()

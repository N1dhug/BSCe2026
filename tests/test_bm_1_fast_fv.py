"""Numerical equivalence and input-safety tests for the opt-in fast FV path."""
import unittest
from unittest.mock import patch

import numpy as np

from src.exe.run_model.bm_1.f2_reference_case import (
    build_reference_case, beta_initial_cell_mass, PRICE_EDGES,
)
from src.method.bm_1_fast_fv import FastFVStochasticPriceEvaluator
from src.method.bm_1_stochastic_optimization import FVStochasticPriceEvaluator, multistart_powell
from src.model.bm_1 import bm_1


class FastFiniteVolumeTests(unittest.TestCase):
    def pair(self, n=51, sigma=0.5, counts=2, **fast_options):
        model, params = build_reference_case()
        edges = np.linspace(0, 1, counts + 1)
        grid, _, _ = model.fokker_planc_reference.grid(n)
        initial = beta_initial_cell_mass(grid)
        args = (model, params, edges, sigma, initial)
        kwargs = dict(objective_max_step=1/24, solver_max_step=1/96,
                      breakpoints=np.array([0.37, 0.71]))
        return (FVStochasticPriceEvaluator(*args, **kwargs),
                FastFVStochasticPriceEvaluator(*args, **kwargs, **fast_options))

    def assert_equivalent(self, slow, fast, prices):
        left, right = slow.evaluate(prices), fast.evaluate(prices)
        np.testing.assert_allclose(left.cell_mass, right.cell_mass, atol=2e-12, rtol=2e-11)
        for name in ('total', 'bias', 'variance', 'expected_demand', 'second_moment'):
            np.testing.assert_allclose(getattr(left.objective,name),getattr(right.objective,name),
                                       atol=2e-12,rtol=2e-11)
        np.testing.assert_array_equal(left.diagnostics['integration_t'], right.diagnostics['integration_t'])
        self.assertLess(right.diagnostics['maximum_abs_mass_error'],1e-9)
        self.assertGreaterEqual(right.diagnostics['minimum_cell_mass'],-1e-12)
        self.assertFalse(right.diagnostics['used_clipping'])
        self.assertFalse(right.diagnostics['used_renormalization'])

    def test_grid_sizes_noise_and_extreme_prices_match_reference(self):
        for n,sigma in ((3,0.0),(31,0.5),(201,0.5),(1001,1.0)):
            with self.subTest(n=n,sigma=sigma):
                slow,fast=self.pair(n=n,sigma=sigma)
                for prices in ([0,1],[1,0],[0.73,0.26]):
                    self.assert_equivalent(slow,fast,prices)

    def test_random_24_price_schedules_match_reference(self):
        slow,fast=self.pair(counts=24)
        for seed in range(3):
            self.assert_equivalent(slow,fast,np.random.default_rng(seed).uniform(0,1,24))

    def test_time_varying_baseline_and_target_jumps(self):
        slow,fast=self.pair()
        def baseline(t):
            t=np.asarray(t)
            return 0.2+0.1*np.sin(7*t)**2+0.2*(t>=0.37)
        slow.model.B=baseline
        fast.model.B=baseline
        self.assert_equivalent(slow,fast,[0.64,0.33])

    def test_zero_flexibility_and_small_capacity(self):
        for c,xi in ((0.01,1.0),(2.0,0.0)):
            slow,fast=self.pair()
            slow.parameter_template[3]=c; fast.parameter_template[3]=c
            slow.parameter_template[5]=xi; fast.parameter_template[5]=xi
            self.assert_equivalent(slow,fast,[0.01,0.98])

    def test_batch_size_does_not_change_results(self):
        for size in (1,7,128,1000):
            slow,fast=self.pair(batch_size=size)
            self.assert_equivalent(slow,fast,[0.72,0.28])

    def test_cost_only_mode_matches_full_history_mode(self):
        slow,fast=self.pair()
        full=fast.evaluate([0.72,0.28]); cost=fast.evaluate_cost([0.72,0.28])
        self.assertEqual(full.objective.total,cost.objective.total)
        self.assertFalse(hasattr(cost,'cell_mass'))
        self.assertFalse(cost.diagnostics['saved_density_history'])
        self.assertTrue(full.diagnostics['saved_density_history'])

    def test_scalar_time_callbacks_remain_supported_explicitly(self):
        slow,fast=self.pair(vectorized=False)
        def baseline(t):
            t=np.asarray(t)
            # Existing FV model wrapper also requests 1D B(t) for outputs.
            if t.ndim>1:
                raise ValueError('No 2D inputs')
            return 0.3+0.05*t
        slow.model.B=baseline; fast.model.B=baseline
        self.assert_equivalent(slow,fast,[0.7,0.3])

    def test_initial_mass_not_mutated_by_lapack(self):
        slow,fast=self.pair()
        before=fast.initial_mass.copy()
        fast([0.8,0.2]); fast([0.1,0.9])
        np.testing.assert_array_equal(before,fast.initial_mass)

    def test_bad_controls_and_prices_are_rejected(self):
        for size in (0,1.5,True):
            with self.assertRaises((ValueError,TypeError)):
                self.pair(batch_size=size)
        _,fast=self.pair()
        for prices in ([0.3],[0.1,np.nan],[1.1,0.3]):
            with self.assertRaises(ValueError): fast(prices)

    def test_invalid_physical_response_is_not_hidden(self):
        _,fast=self.pair()
        fast.model.B=lambda t: np.zeros_like(t)+1.1
        with self.assertRaises(ValueError): fast([0.5,0.5])

    def test_lapack_failure_is_reported(self):
        _,fast=self.pair()
        with patch('src.method.bm_1_fast_fv.dgtsv',return_value=(None,None,None,None,2)):
            with self.assertRaisesRegex(RuntimeError,'LAPACK'): fast([0.5,0.5])

    def test_raw_mass_loss_is_not_normalized_away(self):
        from src.method.bm_1_fast_fv import dgtsv
        _,fast=self.pair()
        def lose_mass(*args,**kwargs):
            out=list(dgtsv(*args,**kwargs)); out[3]=0.99*out[3]; return tuple(out)
        with patch('src.method.bm_1_fast_fv.dgtsv',side_effect=lose_mass):
            with self.assertRaisesRegex(RuntimeError,'probability'): fast([0.5,0.5])

    def test_small_powell_solution_matches_reference(self):
        slow,fast=self.pair(n=21)
        starts=[[0.7,0.3]]
        options=dict(xtol=1e-5,ftol=1e-9,maxiter=30)
        a=multistart_powell(slow,starts,options=options)
        b=multistart_powell(fast,starts,options=options)
        np.testing.assert_allclose(a.best.x,b.best.x,atol=2e-6,rtol=0)
        self.assertLess(abs(a.best.fun-b.best.fun),1e-10)

if __name__=='__main__':
    unittest.main()

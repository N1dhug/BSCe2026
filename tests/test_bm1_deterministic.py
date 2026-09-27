"""BM1 regression tests grouped by responsibility."""

import numpy as np
import unittest
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from src.diagnostics import (constant_demand_capacity_certificate, constant_demand_implied_state,
    linear_response_demand_envelope, linear_response_required_price)
from src.method import tracking_error
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder

T = (0.0, 1.0)
X0 = 0.5
BASELINE = 0.3
C = 1.0
K = 2.0
XI = 1.0
PRICE_VALUES = np.array([0.75, 0.25], dtype=float)

def linear_state_response(state):
    """Benchmark response f(X) = 1 - 2X."""
    return 1.0 - 2.0 * np.asarray(state, dtype=float)

def linear_price_response(price):
    """Benchmark response g(U) = 1 - 2U."""
    return 1.0 - 2.0 * np.asarray(price, dtype=float)

def build_baseline():
    """Return the constant benchmark baseline B(t) = 0.3."""
    return input_builder.energy((np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float)))

def exact_constant_price_state(t, initial_state, initial_time, price):
    """Independent exact state for a constant-price benchmark interval."""
    query = np.asarray(t, dtype=float)
    equilibrium = 1.0 - price
    demand_scale = 1.0 - BASELINE if initial_state < equilibrium else BASELINE
    decay_rate = K * XI * demand_scale / C
    initial_offset = K * (initial_state - equilibrium)
    return equilibrium + np.arcsinh(np.sinh(initial_offset) * np.exp(-decay_rate * (query - initial_time))) / K

def exact_step_state(t):
    """Independent exact state for the two F1 price intervals."""
    query = np.asarray(t, dtype=float)
    state = np.empty_like(query)
    before_step = query <= PRICE_EDGES[1]
    state[before_step] = exact_constant_price_state(query[before_step], X0, T[0], PRICE_VALUES[0])
    state_at_step = float(exact_constant_price_state(np.array([PRICE_EDGES[1]]), X0, T[0], PRICE_VALUES[0])[0])
    state[~before_step] = exact_constant_price_state(query[~before_step], state_at_step, PRICE_EDGES[1],
        PRICE_VALUES[1])
    return state

class DeterministicSolverRegressionTests(unittest.TestCase):
    """F1-A/F1-B: exact benchmark, Euler, and breakpoint-aware solve_ivp."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = build_baseline()
        cls.parameters = build_parameters(PRICE_EDGES, PRICE_VALUES)
        cls.model = bm_1(cls.baseline, cls.baseline, T, 193)

    def test_breakpoint_solver_matches_exact_solution(self):
        output_grid = np.linspace(T[0], T[1], 200)
        result = self.model.deterministic_solver([X0], self.parameters, t_eval=output_grid,
            breakpoints=PRICE_EDGES[1:-1], rtol=1e-09, atol=1e-11)
        self.assertEqual(len(result), 8)
        np.testing.assert_array_equal(result[0], output_grid)
        for values in result:
            self.assertEqual(np.asarray(values).shape, output_grid.shape)
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertLess(np.max(np.abs(result[1] - exact_step_state(output_grid))), 1e-07)

    def test_breakpoint_restart_improves_default_solver(self):
        legacy = self.model.deterministic_solver([X0], self.parameters)
        segmented = self.model.deterministic_solver([X0], self.parameters, breakpoints=PRICE_EDGES[1:-1])
        exact = exact_step_state(self.model.t)
        legacy_error = np.max(np.abs(legacy[1] - exact))
        segmented_error = np.max(np.abs(segmented[1] - exact))
        self.assertLessEqual(segmented_error, 0.0001)
        self.assertGreaterEqual(legacy_error / segmented_error, 100.0)

    def test_zero_noise_euler_is_first_order_and_balances_energy(self):
        errors = []
        for steps_per_day in (48, 96):
            t, state, _, baseline, _, _, _, demand = solve_zero_noise_euler(steps_per_day)
            errors.append(np.max(np.abs(state - exact_step_state(t))))
            increments = np.diff(t) * (demand[:-1] - baseline[:-1])
            residual = C * (state - state[0]) - np.concatenate(([0.0], np.cumsum(increments)))
            self.assertGreaterEqual(np.min(state), -1e-12)
            self.assertLessEqual(np.max(state), 1.0 + 1e-12)
            self.assertLessEqual(np.max(np.abs(residual)), 1e-12)
        observed_order = np.log(errors[0] / errors[1]) / np.log(2.0)
        self.assertGreater(observed_order, 0.95)
        self.assertLess(observed_order, 1.05)

    def test_basic_physical_sanity_cases(self):
        equilibrium = solve_zero_noise_euler(96, np.array([0.5, 0.5]))
        no_flexibility = solve_zero_noise_euler(96, xi=0.0)
        low_price = solve_zero_noise_euler(96, np.array([0.25, 0.25]))
        high_price = solve_zero_noise_euler(96, np.array([0.75, 0.75]))
        np.testing.assert_allclose(equilibrium[1], X0, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(no_flexibility[1], X0, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(no_flexibility[7], no_flexibility[3], atol=1e-12, rtol=0.0)
        self.assertTrue(np.all(np.diff(low_price[1]) >= -1e-12))
        self.assertGreater(low_price[1][-1], X0)
        self.assertLess(low_price[1][-1], 0.75)
        self.assertTrue(np.all(np.diff(high_price[1]) <= 1e-12))
        self.assertLess(high_price[1][-1], X0)
        self.assertGreater(high_price[1][-1], 0.25)

class PriceIntervalRegressionTests(unittest.TestCase):
    """F1-C: one right-continuous decision per supplied interval."""

    def test_price_intervals_are_right_continuous_and_grid_independent(self):
        price = input_builder.price_intervals((PRICE_EDGES, PRICE_VALUES))
        left = np.nextafter(PRICE_EDGES[1], T[0])
        right = np.nextafter(PRICE_EDGES[1], T[1])
        probe = np.array([T[0], left, PRICE_EDGES[1], right, T[1]])
        expected = np.array([0.75, 0.75, 0.25, 0.25, 0.25])
        np.testing.assert_array_equal(price(probe), expected)
        self.assertEqual(price(PRICE_EDGES[1]), PRICE_VALUES[1])
        arbitrary_grid = np.linspace(T[0], T[1], 37)
        self.assertEqual(price(arbitrary_grid).shape, arbitrary_grid.shape)
        self.assertNotEqual(PRICE_VALUES.size, arbitrary_grid.size)

    def test_invalid_interval_parameterizations_are_rejected(self):
        cases = ((np.array([0.0, 1.0]), np.array([0.2, 0.8])), (np.array([0.0, 0.5, 0.5]), np.array([0.2, 0.8])),
            (PRICE_EDGES, np.array([-0.1, 0.5])), (PRICE_EDGES, np.array([0.5, 1.1])))
        for edges, prices in cases:
            with self.subTest(edges=edges, prices=prices):
                with self.assertRaises(ValueError):
                    input_builder.price_intervals((edges, prices))
PRICE_EDGES = np.array([T[0], 0.5, T[1]], dtype=float)

def build_parameters(interval_edges, interval_prices, xi=XI):
    """Return the standard F1 parameter vector."""
    price = input_builder.price_intervals((interval_edges, interval_prices))
    return [price, linear_state_response, linear_price_response, C, K, xi]

def solve_zero_noise_euler(steps_per_day, price_values=PRICE_VALUES, xi=XI):
    """Run Euler-Maruyama as ordinary Euler for a deterministic benchmark."""
    baseline = build_baseline()
    parameters = build_parameters(PRICE_EDGES, price_values, xi=xi)
    model = bm_1(baseline, baseline, T, steps_per_day + 1)
    return model.euler_maruyama_solver([X0], parameters, sigma_x=0.0, n_paths=1, seed=0, boundary='none')

class OptimizationRegressionTests(unittest.TestCase):
    """F1-D: recover a known feasible two-interval price signal."""
    KNOWN_PRICES = np.array([0.72, 0.28])
    INITIAL_PRICES = np.array([0.55, 0.45])
    OBJECTIVE_GRID = np.linspace(T[0], T[1], 193)

    @classmethod
    def setUpClass(cls):
        baseline = build_baseline()
        generator = bm_1(baseline, baseline, T, cls.OBJECTIVE_GRID.size)
        target_parameters = build_parameters(PRICE_EDGES, cls.KNOWN_PRICES)
        target_result = generator.deterministic_solver([X0], target_parameters, t_eval=cls.OBJECTIVE_GRID,
            breakpoints=PRICE_EDGES[1:-1], rtol=1e-09, atol=1e-11)
        cls.target_demand = target_result[7]
        target = interp1d(cls.OBJECTIVE_GRID, cls.target_demand, kind='linear', bounds_error=True)
        cls.model = bm_1(baseline, target, T, cls.OBJECTIVE_GRID.size)
        cls.result = cls.model.optimize(cls.INITIAL_PRICES, [X0], build_parameters(PRICE_EDGES, cls.INITIAL_PRICES),
            interval_edges=PRICE_EDGES, t_eval=cls.OBJECTIVE_GRID, rtol=1e-09, atol=1e-11,
            options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 200})
        optimized_parameters = build_parameters(PRICE_EDGES, cls.result.x)
        cls.optimized = cls.model.deterministic_solver([X0], optimized_parameters, t_eval=cls.OBJECTIVE_GRID,
            breakpoints=PRICE_EDGES[1:-1], rtol=1e-09, atol=1e-11)
        cls.recomputed_objective = float(simpson(np.square(cls.optimized[7] - cls.target_demand),
            x=cls.OBJECTIVE_GRID))

    def test_optimizer_recovers_known_feasible_prices(self):
        self.assertTrue(self.result.success, msg=self.result.message)
        np.testing.assert_allclose(self.result.x, self.KNOWN_PRICES, atol=1e-05, rtol=0.0)
        self.assertTrue(np.all(self.result.x >= 0.0))
        self.assertTrue(np.all(self.result.x <= 1.0))

    def test_optimized_objective_is_reproducibly_near_zero(self):
        self.assertLessEqual(self.recomputed_objective, 1e-12)
        public_objective = self.model.evaluate_tracking_objective(self.result.x, [X0],
            build_parameters(PRICE_EDGES, self.result.x), interval_edges=PRICE_EDGES, t_eval=self.OBJECTIVE_GRID,
            rtol=1e-09, atol=1e-11)
        shared_objective = tracking_error(self.OBJECTIVE_GRID, self.optimized[7], self.target_demand)
        self.assertTrue(np.isclose(public_objective, self.recomputed_objective, rtol=1e-12, atol=1e-16))
        self.assertTrue(np.isclose(shared_objective, self.recomputed_objective, rtol=1e-12, atol=1e-16))
        self.assertTrue(np.isclose(self.result.fun, self.recomputed_objective, rtol=1e-06, atol=1e-16))

    def test_optimizer_rejects_inconsistent_grids(self):
        parameters = build_parameters(PRICE_EDGES, self.INITIAL_PRICES)
        invalid_calls = ({'interval_edges': np.array(T, dtype=float)}, {'interval_edges': np.array([0.1, 0.5, 1.0])},
            {'interval_edges': PRICE_EDGES, 't_eval': self.OBJECTIVE_GRID[1:]})
        for keywords in invalid_calls:
            with self.subTest(keywords=keywords):
                with self.assertRaises(ValueError):
                    self.model.optimize(self.INITIAL_PRICES, [X0], parameters, **keywords)

class FeasibilityDiagnosticRegressionTests(unittest.TestCase):
    """F1-E: optimizer-independent price and capacity certificates."""
    REQUIRED_PRICE = -0.05
    OBJECTIVE_GRID = np.linspace(T[0], T[1], 193)

    @classmethod
    def price_bound_state(cls, t):
        query = np.asarray(t, dtype=float)
        equilibrium = 1.0 - cls.REQUIRED_PRICE
        rate = K * XI * (1.0 - BASELINE) / C
        return equilibrium + np.arcsinh(np.sinh(K * (X0 - equilibrium)) * np.exp(-rate * (query - T[0]))) / K

    @classmethod
    def price_bound_target(cls, t):
        query = np.asarray(t, dtype=float)
        state = cls.price_bound_state(query)
        demand = BASELINE + XI * (1.0 - BASELINE) * np.tanh(K * (1.0 - state - cls.REQUIRED_PRICE))
        return float(demand) if query.ndim == 0 else demand

    def test_price_bound_certificate_recovers_forbidden_price(self):
        state = self.price_bound_state(self.OBJECTIVE_GRID)
        target = self.price_bound_target(self.OBJECTIVE_GRID)
        required_price = linear_response_required_price(state, target, BASELINE, XI, K)
        _, admissible_maximum = linear_response_demand_envelope(state, BASELINE, XI, K)
        self.assertGreaterEqual(np.min(state), 0.0)
        self.assertLessEqual(np.max(state), 1.0)
        self.assertGreaterEqual(np.min(target), 0.0)
        self.assertLessEqual(np.max(target), 1.0)
        np.testing.assert_allclose(required_price, self.REQUIRED_PRICE, atol=1e-10, rtol=0.0)
        self.assertLess(np.max(required_price), 0.0)
        self.assertGreater(np.min(target - admissible_maximum), 0.0)

    def test_bounded_optimizer_stops_at_price_limit_with_residual(self):
        baseline = build_baseline()
        model = bm_1(baseline, self.price_bound_target, T, self.OBJECTIVE_GRID.size)
        held_edges = np.array(T, dtype=float)
        placeholder = build_parameters(held_edges, np.array([0.5]))
        result = model.optimize(np.array([0.5]), [X0], placeholder, interval_edges=held_edges,
            t_eval=self.OBJECTIVE_GRID, rtol=1e-09, atol=1e-11, options={'ftol': 1e-14, 'gtol': 1e-08, 'maxiter': 300})
        self.assertTrue(result.success, msg=result.message)
        self.assertLessEqual(result.x[0], 1e-08)
        self.assertGreater(result.fun, 0.0005)
        self.assertLess(result.fun, 0.002)

    def test_capacity_certificate_proves_nonzero_minimum_error(self):
        capacity_target = 0.82
        required_state = constant_demand_implied_state(self.OBJECTIVE_GRID, capacity_target, BASELINE, C, X0,
            initial_time=T[0])
        certificate = constant_demand_capacity_certificate(capacity_target, BASELINE, C, X0, T)
        self.assertFalse(certificate.feasible)
        self.assertEqual(certificate.violated_boundary, 'upper')
        self.assertAlmostEqual(required_state[-1], 1.02, places=12)
        self.assertAlmostEqual(certificate.required_final_state, 1.02, places=12)
        self.assertGreater(certificate.violation_time, T[0])
        self.assertLess(certificate.violation_time, T[1])
        self.assertAlmostEqual(certificate.excess_energy, 0.02, places=12)
        self.assertAlmostEqual(certificate.objective_lower_bound, 0.0004, places=12)
        feasible = constant_demand_capacity_certificate(BASELINE, BASELINE, C, X0, T)
        self.assertTrue(feasible.feasible)
        self.assertIsNone(feasible.violation_time)
        self.assertEqual(feasible.objective_lower_bound, 0.0)
        exact_upper_boundary = constant_demand_capacity_certificate(0.8, BASELINE, C, X0, T)
        self.assertTrue(exact_upper_boundary.feasible)
        self.assertAlmostEqual(exact_upper_boundary.required_final_state, 1.0, places=12)
        lower = constant_demand_capacity_certificate(0.0, 0.7, C, X0, T)
        self.assertFalse(lower.feasible)
        self.assertEqual(lower.violated_boundary, 'lower')
        self.assertAlmostEqual(lower.required_final_state, -0.2, places=12)
        self.assertAlmostEqual(lower.excess_energy, 0.2, places=12)
        self.assertAlmostEqual(lower.objective_lower_bound, 0.04, places=12)

    def test_diagnostics_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            linear_response_required_price(X0, 1.0, BASELINE, XI, K)
        with self.assertRaises(ValueError):
            linear_response_demand_envelope(np.array([]), BASELINE, XI, K)
        with self.assertRaises(ValueError):
            constant_demand_capacity_certificate(0.5, BASELINE, 0.0, X0, T)
        with self.assertRaises(ValueError):
            constant_demand_implied_state(np.array([]), 0.5, BASELINE, C, X0)
        with self.assertRaises(ValueError):
            tracking_error(np.array([0.0, 1.0]), np.array([0.2]), np.array([0.2, 0.2]))


if __name__ == '__main__':
    unittest.main()

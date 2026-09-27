"""BM1 regression tests grouped by responsibility."""

import numpy as np
import unittest
from src.control.objective import deterministic_midpoint_objective
from src.control.bm_1 import (
    BM1ControlNumerics,
    BM1ControlProblem,
    BM1Parameters,
    DynamicMixturePriceEvaluator,
    best_surface_starts,
    beta_initial_cell_mass,
    coarse_price_surface,
    evaluate_mc_candidates_common_random_numbers,
    multistart_powell,
    paired_mc_improvement,
    point_initial_cell_mass,
)
from src.validation.bm_1.cases import (
    INITIAL_ALPHA,
    INITIAL_BETA,
    SIGMA_X,
    linear_price_response,
    linear_state_response,
)
from src.validation.bm_1.control import (
    C,
    DETERMINISTIC_REFERENCE_PRICES,
    F3_C_SMOOTHING_WEIGHTS,
    F3_C_TERMINAL_WEIGHTS,
    F3_E_SCENARIO_ORDER,
    HOURLY_EDGES,
    INITIAL_PRICES,
    MULTI_INTERVAL_COUNTS as INTERVAL_COUNTS,
    INTERVAL_EDGES,
    K,
    KNOWN_DETERMINISTIC_PRICES,
    REFERENCE_PRICES_24,
    T,
    XI,
    _build_fv_evaluator,
    _coarsen_reference_prices,
    _f3_c_weight_pairs,
    multi_edges as _edges,
    _lift_half_day_prices,
    _parameter_template,
    _sample_piecewise_prices as _sample_piecewise_prices,
    _union_breakpoints,
    build_deterministic_target,
    build_f3_a_problem,
    build_f3_e_scenarios,
    build_stochastic_model,
    run_f3_d,
    INTERVAL_COUNTS as control_INTERVAL_COUNTS,
    _edges as control__edges,
    _sample_piecewise_prices as control__sample_piecewise_prices,
)

class PowellOptimizationTests(unittest.TestCase):

    def test_multistart_powell_recovers_bounded_quadratic_minimum(self):
        optimum = np.array([0.2, 0.8])

        def objective(prices):
            prices = np.asarray(prices, dtype=float)
            return float(np.sum(np.square(prices - optimum)))
        result = multistart_powell(objective, np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]),
            options={'xtol': 1e-08, 'ftol': 1e-12, 'maxiter': 100})
        self.assertTrue(result.best.success)
        np.testing.assert_allclose(result.best.x, optimum, atol=1e-06, rtol=0.0)
        self.assertLess(result.best.fun, 1e-12)

    def test_coarse_surface_returns_best_price_starts(self):

        class Toy:
            n_intervals = 2

            def __call__(self, prices):
                p = np.asarray(prices, dtype=float)
                return float((p[0] - 0.25) ** 2 + (p[1] - 0.75) ** 2)
        values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        surface = coarse_price_surface(Toy(), values)
        starts = best_surface_starts(values, surface, 2)
        np.testing.assert_array_equal(starts[0], [0.25, 0.75])

class StochasticPriceEvaluatorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        target, _, _ = build_deterministic_target()
        cls.model = build_stochastic_model(target)

    def test_fv_evaluator_is_deterministic_and_rejects_invalid_prices(self):
        evaluator = _build_fv_evaluator(self.model, x_steps=41, objective_max_step=1.0 / 24.0,
            solver_max_step=1.0 / 96.0)
        left = evaluator.evaluate(KNOWN_DETERMINISTIC_PRICES)
        right = evaluator.evaluate(KNOWN_DETERMINISTIC_PRICES)
        self.assertAlmostEqual(left.objective.total, right.objective.total, places=14)
        self.assertGreater(left.objective.total, 0.0)
        self.assertAlmostEqual(left.objective.total, left.objective.bias + left.objective.variance, places=12)
        self.assertLess(left.diagnostics['maximum_abs_mass_error'], 1e-09)
        with self.assertRaises(ValueError):
            evaluator.evaluate([0.5])
        with self.assertRaises(ValueError):
            evaluator.evaluate([-0.1, 0.5])

    def test_small_fv_optimization_does_not_worsen_reference_prices(self):
        evaluator = _build_fv_evaluator(self.model, x_steps=41, objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 96.0)
        known = evaluator(KNOWN_DETERMINISTIC_PRICES)
        surface = coarse_price_surface(evaluator, np.array([0.25, 0.5, 0.75]))
        starts = best_surface_starts(np.array([0.25, 0.5, 0.75]), surface, 2)
        starts = np.vstack((starts, KNOWN_DETERMINISTIC_PRICES, INITIAL_PRICES))
        result = multistart_powell(evaluator, starts, options={'xtol': 0.002, 'ftol': 1e-05, 'maxiter': 25})
        self.assertTrue(result.best.success)
        self.assertLessEqual(result.best.fun, known + 1e-08)
        self.assertTrue(np.all(result.best.x >= 0.0))
        self.assertTrue(np.all(result.best.x <= 1.0))

    def test_small_dynamic_mixture_evaluator_returns_valid_objective(self):
        support = _build_fv_evaluator(self.model, x_steps=31, objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 48.0)
        evaluator = DynamicMixturePriceEvaluator(support, degree=10, cfl=0.35)
        result = evaluator.evaluate(KNOWN_DETERMINISTIC_PRICES)
        self.assertTrue(np.isfinite(result.objective.total))
        self.assertGreater(result.objective.total, 0.0)
        self.assertAlmostEqual(result.objective.total, result.objective.bias + result.objective.variance, places=10)
        self.assertFalse(result.diagnostics['used_clipping'])
        self.assertFalse(result.diagnostics['used_renormalization'])

    def test_common_random_numbers_support_paired_objective_difference(self):
        evaluator = _build_fv_evaluator(self.model, x_steps=31, objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 48.0)
        candidates = {'reference': KNOWN_DETERMINISTIC_PRICES, 'initial': INITIAL_PRICES}
        result = evaluate_mc_candidates_common_random_numbers(evaluator, candidates, n_paths=500, initial_seed=111,
            brownian_seed=112)
        self.assertEqual(result['reference'].diagnostics['total_violations'], 0)
        self.assertEqual(result['initial'].diagnostics['total_violations'], 0)
        self.assertEqual(result['reference'].path_objectives.shape,
            result['initial'].path_objectives.shape)
        np.testing.assert_array_equal(
            result['reference'].path_objectives,
            result['reference'].objective.path_objectives,
        )
        improvement, standard_error = paired_mc_improvement(result['reference'], result['initial'])
        self.assertTrue(np.isfinite(improvement))
        self.assertTrue(np.isfinite(standard_error))
        self.assertGreater(standard_error, 0.0)

class InitialLawTests(unittest.TestCase):

    def test_point_initial_mass_places_all_probability_in_one_cell(self):
        edges = np.linspace(0.0, 1.0, 202)
        mass = point_initial_cell_mass(edges, 0.5)
        self.assertEqual(np.count_nonzero(mass), 1)
        self.assertAlmostEqual(float(np.sum(mass)), 1.0, places=15)
        cell = int(np.argmax(mass))
        midpoint = 0.5 * (edges[cell] + edges[cell + 1])
        self.assertAlmostEqual(midpoint, 0.5, places=15)

    def test_point_initial_mass_handles_endpoints_and_rejects_invalid_state(self):
        edges = np.linspace(0.0, 1.0, 6)
        self.assertEqual(int(np.argmax(point_initial_cell_mass(edges, 0.0))), 0)
        self.assertEqual(int(np.argmax(point_initial_cell_mass(edges, 1.0))), 4)
        with self.assertRaises(ValueError):
            point_initial_cell_mass(edges, -0.1)
        with self.assertRaises(ValueError):
            point_initial_cell_mass(edges, 1.1)
        with self.assertRaises(ValueError):
            point_initial_cell_mass(edges, [0.4, 0.6])


def _initial_mass_builder(edges):
    return beta_initial_cell_mass(edges, alpha=INITIAL_ALPHA, beta=INITIAL_BETA)

class BM1ControlProblemTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        target, _, _ = build_deterministic_target()
        baseline_model = build_stochastic_model(target)
        cls.target = staticmethod(target)
        cls.baseline = baseline_model.B
        cls.parameters = BM1Parameters(state_response=linear_state_response, price_response=linear_price_response, C=C,
            k=K, Xi=XI, sigma_x=SIGMA_X)

    def problem(self, **kwargs):
        settings = dict(baseline=self.baseline, target=self.target, horizon=T, price_edges=INTERVAL_EDGES,
            parameters=self.parameters, initial_mass_builder=_initial_mass_builder,
            deterministic_initial_state=0.5, observed_prices=KNOWN_DETERMINISTIC_PRICES)
        settings.update(kwargs)
        return BM1ControlProblem(**settings)

    def test_reference_fv_cost_matches_existing_builder(self):
        numerics = BM1ControlNumerics(x_steps=41, objective_max_step=1.0 / 24.0, solver_max_step=1.0 / 96.0)
        problem = self.problem()
        new = problem.evaluate_observed(numerics, fast=False)
        old_model = build_stochastic_model(self.target)
        old = _build_fv_evaluator(old_model, x_steps=41, objective_max_step=1.0 / 24.0,
            solver_max_step=1.0 / 96.0).evaluate(KNOWN_DETERMINISTIC_PRICES)
        self.assertAlmostEqual(new.objective.total, old.objective.total, places=14)
        self.assertAlmostEqual(new.objective.bias, old.objective.bias, places=14)
        self.assertAlmostEqual(new.objective.variance, old.objective.variance, places=14)
        np.testing.assert_array_equal(new.cell_mass, old.cell_mass)

    def test_fast_and_reference_paths_agree_for_fixed_price(self):
        numerics = BM1ControlNumerics(x_steps=31, objective_max_step=1.0 / 12.0, solver_max_step=1.0 / 48.0)
        problem = self.problem()
        reference = problem.evaluate_fixed(INITIAL_PRICES, numerics, fast=False)
        fast = problem.evaluate_fixed(INITIAL_PRICES, numerics, fast=True)
        np.testing.assert_allclose(fast.cell_mass, reference.cell_mass, atol=1e-13, rtol=0.0)
        self.assertAlmostEqual(fast.objective.total, reference.objective.total, places=13)
        self.assertAlmostEqual(fast.total, reference.total, places=13)

    def test_f3_c_penalties_add_to_cost_without_changing_forward_solution(self):
        numerics = BM1ControlNumerics(x_steps=31, objective_max_step=1.0 / 12.0, solver_max_step=1.0 / 48.0)
        legacy = self.problem().evaluate_fixed(INITIAL_PRICES, numerics, fast=False)
        problem = self.problem(
            terminal_reference=0.5,
            terminal_weight=0.4,
            price_smoothing_weight=0.2,
        )
        augmented = problem.evaluate_fixed(INITIAL_PRICES, numerics, fast=False)
        np.testing.assert_array_equal(augmented.cell_mass, legacy.cell_mass)
        self.assertAlmostEqual(augmented.objective.total, legacy.objective.total, places=14)
        self.assertAlmostEqual(
            augmented.penalties.terminal,
            0.4 * augmented.penalties.terminal_mse,
            places=15,
        )
        expected_roughness = float(np.sum(np.square(np.diff(INITIAL_PRICES))))
        self.assertAlmostEqual(augmented.penalties.price_roughness, expected_roughness, places=15)
        self.assertAlmostEqual(augmented.penalties.smoothing, 0.2 * expected_roughness, places=15)
        self.assertAlmostEqual(
            augmented.total,
            augmented.objective.total + augmented.penalties.terminal + augmented.penalties.smoothing,
            places=15,
        )
        evaluator = problem.build_evaluator(numerics, fast=True)
        fast = evaluator.evaluate(INITIAL_PRICES)
        self.assertAlmostEqual(fast.total, augmented.total, places=13)
        self.assertAlmostEqual(evaluator(INITIAL_PRICES), fast.total, places=13)

    def test_f3_c_mc_uses_the_same_augmented_path_cost(self):
        numerics = BM1ControlNumerics(x_steps=21, objective_max_step=1.0 / 12.0, solver_max_step=1.0 / 48.0)
        problem = self.problem(
            terminal_reference=0.5,
            terminal_weight=0.3,
            price_smoothing_weight=0.2,
        )
        evaluator = problem.build_evaluator(numerics, fast=True)
        results = evaluate_mc_candidates_common_random_numbers(
            evaluator,
            {'candidate': INITIAL_PRICES},
            n_paths=200,
            initial_seed=211,
            brownian_seed=212,
        )
        result = results['candidate']
        self.assertAlmostEqual(result.total, float(np.mean(result.path_objectives)), places=14)
        self.assertAlmostEqual(
            result.standard_error,
            float(np.std(result.path_objectives, ddof=1) / np.sqrt(result.path_objectives.size)),
            places=15,
        )
        self.assertGreater(result.penalties.terminal, 0.0)
        self.assertGreater(result.penalties.smoothing, 0.0)

    def test_deterministic_evaluator_uses_the_common_f3_objective(self):
        numerics = BM1ControlNumerics(
            x_steps=31,
            objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 48.0,
            deterministic_rtol=1e-09,
            deterministic_atol=1e-11,
        )
        problem = self.problem(
            terminal_reference=0.5,
            terminal_weight=0.2,
            price_smoothing_weight=0.1,
        )
        evaluator = problem.build_deterministic_evaluator(numerics)
        result = evaluator.evaluate(INITIAL_PRICES)
        reference = deterministic_midpoint_objective(
            evaluator.model,
            [problem.deterministic_initial_state],
            evaluator.parameters_for_prices(INITIAL_PRICES),
            objective_max_step=numerics.objective_max_step,
            breakpoints=evaluator.breakpoints,
            solver_max_step=numerics.solver_max_step,
            rtol=numerics.deterministic_rtol,
            atol=numerics.deterministic_atol,
        )
        self.assertAlmostEqual(result.objective.total, reference.total, places=14)
        self.assertAlmostEqual(result.objective.variance, 0.0, places=14)
        self.assertAlmostEqual(
            result.penalties.terminal_mse,
            (result.state[-1] - problem.terminal_reference) ** 2,
            places=15,
        )
        self.assertAlmostEqual(
            result.total,
            result.objective.total + result.penalties.terminal + result.penalties.smoothing,
            places=15,
        )

    def test_f3_a_problem_defines_explicit_deterministic_counterpart(self):
        problem, numerics = build_f3_a_problem()
        self.assertEqual(problem.deterministic_initial_state, 0.5)
        evaluator = problem.build_deterministic_evaluator(numerics)
        self.assertEqual(evaluator.initial_state, 0.5)

    def test_observed_prices_are_owned_and_never_used_as_template(self):
        observed = KNOWN_DETERMINISTIC_PRICES.copy()
        edges = INTERVAL_EDGES.copy()
        problem = self.problem(observed_prices=observed, price_edges=edges)
        observed[:] = 0.0
        edges[1] = 0.75
        np.testing.assert_array_equal(problem.observed_prices, KNOWN_DETERMINISTIC_PRICES)
        np.testing.assert_array_equal(problem.price_edges, INTERVAL_EDGES)
        with self.assertRaises(ValueError):
            problem.observed_prices[0] = 0.1
        with self.assertRaises(ValueError):
            problem.price_edges[1] = 0.7
        template = problem.parameters.parameter_template(problem.price_edges)
        np.testing.assert_array_equal(template[0](np.array([0.25, 0.75])), np.array([0.5, 0.5]))

    def test_observed_evaluation_requires_attached_external_prices(self):
        numerics = BM1ControlNumerics(x_steps=21, objective_max_step=1.0 / 12.0, solver_max_step=1.0 / 48.0)
        problem = self.problem(observed_prices=None)
        with self.assertRaisesRegex(ValueError, 'observed_prices'):
            problem.evaluate_observed(numerics)

    def test_initial_law_is_rebuilt_for_each_spatial_resolution(self):
        calls = []

        def builder(edges):
            calls.append(edges.copy())
            return np.full(edges.size - 1, 1.0 / (edges.size - 1))
        problem = self.problem(initial_mass_builder=builder)
        coarse = BM1ControlNumerics(x_steps=21, objective_max_step=0.1, solver_max_step=0.05)
        fine = BM1ControlNumerics(x_steps=41, objective_max_step=0.1, solver_max_step=0.05)
        coarse_model = problem.build_model(coarse)
        fine_model = problem.build_model(fine)
        coarse_mass = problem.initial_mass(coarse_model, coarse.x_steps)
        fine_mass = problem.initial_mass(fine_model, fine.x_steps)
        self.assertEqual(coarse_mass.size, 21)
        self.assertEqual(fine_mass.size, 41)
        self.assertEqual([item.size for item in calls], [22, 42])

    def test_invalid_problem_inputs_fail_before_solver(self):
        with self.assertRaises(ValueError):
            self.problem(price_edges=[0.0, 0.75])
        with self.assertRaises(ValueError):
            self.problem(observed_prices=[0.5])
        with self.assertRaises(ValueError):
            self.problem(breakpoints=[0.0])
        with self.assertRaises(ValueError):
            self.problem(terminal_weight=0.1)
        with self.assertRaises(ValueError):
            self.problem(terminal_reference=0.5, terminal_weight=-0.1)
        with self.assertRaises(ValueError):
            self.problem(terminal_reference=0.5, price_smoothing_weight=-0.1)
        with self.assertRaises(ValueError):
            self.problem(terminal_reference=1.1)
        with self.assertRaises(ValueError):
            self.problem(deterministic_initial_state=1.1)
        with self.assertRaises(ValueError):
            self.problem(deterministic_initial_state=[0.4, 0.6])
        with self.assertRaises(ValueError):
            self.problem(deterministic_initial_state=None).build_deterministic_evaluator(
                BM1ControlNumerics(x_steps=21),
            )
        with self.assertRaises(ValueError):
            BM1Parameters(state_response=linear_state_response, price_response=linear_price_response, C=0.0, k=K,
                Xi=XI, sigma_x=SIGMA_X)
        with self.assertRaises(ValueError):
            BM1ControlNumerics(x_steps=2)
        with self.assertRaises(ValueError):
            BM1ControlNumerics(deterministic_rtol=0.0)
        with self.assertRaises(ValueError):
            BM1ControlNumerics(deterministic_atol=np.inf)

    def test_existing_parameter_template_has_same_physical_parameters(self):
        problem = self.problem()
        template = problem.parameters.parameter_template(problem.price_edges)
        old = _parameter_template()
        self.assertIs(template[1], old[1])
        self.assertIs(template[2], old[2])
        self.assertEqual(template[3:], old[3:])

class F3EUncertaintyStructureTests(unittest.TestCase):

    def test_uncertainty_scenarios_isolate_initial_and_process_noise(self):
        setup = build_f3_e_scenarios()
        self.assertEqual(tuple(setup['problems']), F3_E_SCENARIO_ORDER)
        self.assertEqual(tuple(setup['evaluators']), F3_E_SCENARIO_ORDER)

        expected_sigma = {
            'deterministic': 0.0,
            'initial_only': 0.0,
            'process_only': SIGMA_X,
            'full': SIGMA_X,
        }
        for name in F3_E_SCENARIO_ORDER:
            self.assertEqual(setup['problems'][name].parameters.sigma_x, expected_sigma[name])

        process = setup['evaluators']['process_only']
        full = setup['evaluators']['full']
        self.assertEqual(np.count_nonzero(process.initial_mass), 1)
        self.assertGreater(np.count_nonzero(full.initial_mass), 1)
        self.assertAlmostEqual(process.x[int(np.argmax(process.initial_mass))], 0.5, places=15)
        self.assertEqual(setup['evaluators']['deterministic'].initial_state, 0.5)


class F3DDecisionStructureTests(unittest.TestCase):

    def test_mc_path_count_is_validated_before_running_decision_study(self):
        with self.assertRaises(ValueError):
            run_f3_d(validate_mc=True, mc_paths=1, progress=None)
        with self.assertRaises(TypeError):
            run_f3_d(validate_mc=True, mc_paths=True, progress=None)


class F3CSensitivityStructureTests(unittest.TestCase):

    def test_default_weight_sweeps_are_unique_and_grid_is_complete(self):
        axes = _f3_c_weight_pairs(F3_C_TERMINAL_WEIGHTS, F3_C_SMOOTHING_WEIGHTS)
        grid = _f3_c_weight_pairs(
            F3_C_TERMINAL_WEIGHTS,
            F3_C_SMOOTHING_WEIGHTS,
            include_grid=True,
        )
        self.assertEqual(axes[0], (0.0, 0.0))
        self.assertEqual(len(axes), len(F3_C_TERMINAL_WEIGHTS) + len(F3_C_SMOOTHING_WEIGHTS) - 1)
        self.assertEqual(len(set(axes)), len(axes))
        self.assertEqual(len(grid), len(F3_C_TERMINAL_WEIGHTS) * len(F3_C_SMOOTHING_WEIGHTS))
        self.assertEqual(len(set(grid)), len(grid))
        self.assertTrue(all((terminal == 0.0 or smoothing == 0.0) for terminal, smoothing in axes))
        self.assertTrue(
            all(
                (terminal, smoothing) in grid
                for terminal in F3_C_TERMINAL_WEIGHTS
                for smoothing in F3_C_SMOOTHING_WEIGHTS
            ),
        )

    def test_weight_sweeps_reject_negative_duplicate_or_unsorted_values(self):
        with self.assertRaises(ValueError):
            _f3_c_weight_pairs([0.0, -0.1], [0.0, 0.1])
        with self.assertRaises(ValueError):
            _f3_c_weight_pairs([0.0, 0.1, 0.1], [0.0, 0.1])
        with self.assertRaises(ValueError):
            _f3_c_weight_pairs([0.1, 0.0], [0.0, 0.1])


class MultiPriceExperimentStructureTests(unittest.TestCase):

    def test_reference_policy_is_embedded_exactly_on_even_grids(self):
        for count in INTERVAL_COUNTS:
            prices = _lift_half_day_prices(DETERMINISTIC_REFERENCE_PRICES, count)
            self.assertEqual(prices.size, count)
            np.testing.assert_array_equal(prices[:count // 2], DETERMINISTIC_REFERENCE_PRICES[0])
            np.testing.assert_array_equal(prices[count // 2:], DETERMINISTIC_REFERENCE_PRICES[1])

    def test_common_breakpoints_contains_every_candidate_edge(self):
        common = _union_breakpoints()
        for count in INTERVAL_COUNTS:
            for breakpoint in _edges(count)[1:-1]:
                self.assertTrue(np.any(common == breakpoint))

    def test_piecewise_resampling_preserves_two_price_policy(self):
        old_edges = _edges(2)
        old_prices = DETERMINISTIC_REFERENCE_PRICES
        for count in (4, 6):
            resampled = _sample_piecewise_prices(old_edges, old_prices, _edges(count))
            expected = _lift_half_day_prices(old_prices, count)
            np.testing.assert_array_equal(resampled, expected)

    def test_odd_interval_count_is_rejected_for_half_day_lift(self):
        with self.assertRaises(ValueError):
            _lift_half_day_prices(DETERMINISTIC_REFERENCE_PRICES, 5)

class ComplexTargetMultiPriceStructureTests(unittest.TestCase):

    def test_reference_signal_is_valid_and_genuinely_time_varying(self):
        self.assertEqual(REFERENCE_PRICES_24.shape, (24,))
        self.assertTrue(np.all(REFERENCE_PRICES_24 >= 0.0))
        self.assertTrue(np.all(REFERENCE_PRICES_24 <= 1.0))
        self.assertGreater(np.count_nonzero(np.diff(REFERENCE_PRICES_24)), 12)
        self.assertTrue(np.any(np.diff(REFERENCE_PRICES_24) > 0.0))
        self.assertTrue(np.any(np.diff(REFERENCE_PRICES_24) < 0.0))

    def test_candidate_grids_are_nested_in_hourly_breakpoints(self):
        hourly = set(np.round(HOURLY_EDGES, 15))
        for count in control_INTERVAL_COUNTS:
            candidate = set(np.round(control__edges(count), 15))
            self.assertTrue(candidate.issubset(hourly))

    def test_coarsening_preserves_reference_mean(self):
        reference_mean = float(np.mean(REFERENCE_PRICES_24))
        for count in control_INTERVAL_COUNTS:
            coarse = _coarsen_reference_prices(count)
            self.assertEqual(coarse.size, count)
            self.assertAlmostEqual(float(np.mean(coarse)), reference_mean, places=15)

    def test_24_interval_coarsening_is_exact_reference(self):
        np.testing.assert_array_equal(_coarsen_reference_prices(24), REFERENCE_PRICES_24)

    def test_resampling_preserves_constant_signal(self):
        old_edges = control__edges(2)
        old_prices = np.array([0.4, 0.4], dtype=float)
        for count in control_INTERVAL_COUNTS:
            sampled = control__sample_piecewise_prices(old_edges, old_prices, control__edges(count))
            np.testing.assert_array_equal(sampled, np.full(count, 0.4, dtype=float))


if __name__ == '__main__':
    unittest.main()

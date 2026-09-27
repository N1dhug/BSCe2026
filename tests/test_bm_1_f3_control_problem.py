"""F3-A tests for the common BM1 control-problem contract."""

import unittest

import numpy as np

from src.exe.run_model.bm_1.f2_reference_case import (
    INITIAL_ALPHA,
    INITIAL_BETA,
    SIGMA_X,
    beta_initial_cell_mass,
)
from src.exe.run_model.bm_1.stochastic_optimization_reference import (
    C,
    INITIAL_PRICES,
    INTERVAL_EDGES,
    K,
    KNOWN_DETERMINISTIC_PRICES,
    T,
    XI,
    _build_fv_evaluator,
    _parameter_template,
    build_deterministic_target,
    build_stochastic_model,
    linear_price_response,
    linear_state_response,
)
from src.method.bm_1_control_problem import (
    BM1ControlNumerics,
    BM1ControlProblem,
    BM1Parameters,
)


def _initial_mass_builder(edges):
    return beta_initial_cell_mass(
        edges,
        alpha=INITIAL_ALPHA,
        beta=INITIAL_BETA,
    )


class BM1ControlProblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target, _, _ = build_deterministic_target()
        baseline_model = build_stochastic_model(target)
        cls.target = staticmethod(target)
        cls.baseline = baseline_model.B
        cls.parameters = BM1Parameters(
            state_response=linear_state_response,
            price_response=linear_price_response,
            C=C,
            k=K,
            Xi=XI,
            sigma_x=SIGMA_X,
        )

    def problem(self, **kwargs):
        settings = dict(
            baseline=self.baseline,
            target=self.target,
            horizon=T,
            price_edges=INTERVAL_EDGES,
            parameters=self.parameters,
            initial_mass_builder=_initial_mass_builder,
            observed_prices=KNOWN_DETERMINISTIC_PRICES,
        )
        settings.update(kwargs)
        return BM1ControlProblem(**settings)

    def test_reference_fv_cost_matches_existing_builder(self):
        numerics = BM1ControlNumerics(
            x_steps=41,
            objective_max_step=1.0 / 24.0,
            solver_max_step=1.0 / 96.0,
        )
        problem = self.problem()
        new = problem.evaluate_observed(numerics, fast=False)

        old_model = build_stochastic_model(self.target)
        old = _build_fv_evaluator(
            old_model,
            x_steps=41,
            objective_max_step=1.0 / 24.0,
            solver_max_step=1.0 / 96.0,
        ).evaluate(KNOWN_DETERMINISTIC_PRICES)

        self.assertAlmostEqual(new.objective.total, old.objective.total, places=14)
        self.assertAlmostEqual(new.objective.bias, old.objective.bias, places=14)
        self.assertAlmostEqual(new.objective.variance, old.objective.variance, places=14)
        np.testing.assert_array_equal(new.cell_mass, old.cell_mass)

    def test_fast_and_reference_paths_agree_for_fixed_price(self):
        numerics = BM1ControlNumerics(
            x_steps=31,
            objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 48.0,
        )
        problem = self.problem()
        reference = problem.evaluate_fixed(
            INITIAL_PRICES,
            numerics,
            fast=False,
        )
        fast = problem.evaluate_fixed(
            INITIAL_PRICES,
            numerics,
            fast=True,
        )

        np.testing.assert_allclose(
            fast.cell_mass,
            reference.cell_mass,
            atol=1.0e-13,
            rtol=0.0,
        )
        self.assertAlmostEqual(
            fast.objective.total,
            reference.objective.total,
            places=13,
        )

    def test_observed_prices_are_owned_and_never_used_as_template(self):
        observed = KNOWN_DETERMINISTIC_PRICES.copy()
        edges = INTERVAL_EDGES.copy()
        problem = self.problem(observed_prices=observed, price_edges=edges)

        observed[:] = 0.0
        edges[1] = 0.75
        np.testing.assert_array_equal(
            problem.observed_prices,
            KNOWN_DETERMINISTIC_PRICES,
        )
        np.testing.assert_array_equal(problem.price_edges, INTERVAL_EDGES)
        with self.assertRaises(ValueError):
            problem.observed_prices[0] = 0.1
        with self.assertRaises(ValueError):
            problem.price_edges[1] = 0.7

        template = problem.parameters.parameter_template(problem.price_edges)
        np.testing.assert_array_equal(
            template[0](np.array([0.25, 0.75])),
            np.array([0.5, 0.5]),
        )

    def test_observed_evaluation_requires_attached_external_prices(self):
        numerics = BM1ControlNumerics(
            x_steps=21,
            objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 48.0,
        )
        problem = self.problem(observed_prices=None)
        with self.assertRaisesRegex(ValueError, "observed_prices"):
            problem.evaluate_observed(numerics)

    def test_initial_law_is_rebuilt_for_each_spatial_resolution(self):
        calls = []

        def builder(edges):
            calls.append(edges.copy())
            return np.full(edges.size - 1, 1.0 / (edges.size - 1))

        problem = self.problem(initial_mass_builder=builder)
        coarse = BM1ControlNumerics(
            x_steps=21,
            objective_max_step=0.1,
            solver_max_step=0.05,
        )
        fine = BM1ControlNumerics(
            x_steps=41,
            objective_max_step=0.1,
            solver_max_step=0.05,
        )
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
            BM1Parameters(
                state_response=linear_state_response,
                price_response=linear_price_response,
                C=0.0,
                k=K,
                Xi=XI,
                sigma_x=SIGMA_X,
            )
        with self.assertRaises(ValueError):
            BM1ControlNumerics(x_steps=2)

    def test_existing_parameter_template_has_same_physical_parameters(self):
        problem = self.problem()
        template = problem.parameters.parameter_template(problem.price_edges)
        old = _parameter_template()
        self.assertIs(template[1], old[1])
        self.assertIs(template[2], old[2])
        self.assertEqual(template[3:], old[3:])


if __name__ == "__main__":
    unittest.main()

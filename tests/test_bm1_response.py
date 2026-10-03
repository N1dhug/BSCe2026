"""H1 response geometry and existing BM1 interface checks, without fitting."""

from dataclasses import replace
import unittest

import numpy as np

from src.control.bm_1 import BM1Parameters
from src.control.bm_1_handoff import BM1ForecastOrigin, forecast_fv
from src.params.bm_1_response import BM1PiecewiseResponse


class PiecewiseResponseTests(unittest.TestCase):
    def test_linear_reference_is_nested_in_both_declared_families(self):
        x = np.linspace(0., 1., 1001).reshape(7, 143)
        for pieces in (2, 4):
            curve = BM1PiecewiseResponse.linear(pieces=pieces)
            np.testing.assert_allclose(curve(x), 1.-2.*x, atol=2e-16, rtol=0.)
            self.assertEqual(curve(x).shape, x.shape)
            self.assertEqual(float(curve(0.)), 1.)
            self.assertEqual(float(curve(1.)), -1.)

    def test_continuity_monotonicity_and_serialized_owned_geometry(self):
        values = np.array([1., .8, .2, -.7, -1.])
        curve = BM1PiecewiseResponse(np.linspace(0., 1., 5), values)
        values[1] = .9
        self.assertEqual(float(curve(.25)), .8)
        self.assertFalse(curve.values.flags.writeable)
        self.assertTrue(np.all(np.diff(curve(np.linspace(0., 1., 1000))) <= 0.))
        for knot in curve.knots[1:-1]:
            self.assertAlmostEqual(float(curve(knot-1e-9)), float(curve(knot+1e-9)), places=7)
        copy = BM1PiecewiseResponse.from_record(curve.to_record())
        np.testing.assert_array_equal(copy.values, curve.values)

    def test_invalid_curves_and_extrapolation_are_rejected_without_repair(self):
        for values in ([1., .8, .9, -.7, -1.], [1., .8, .2, -.7, -.9],
                       [1., 1.1, .2, -.7, -1.], [1., .8, np.nan, -.7, -1.]):
            with self.assertRaises(ValueError):
                BM1PiecewiseResponse(np.linspace(0., 1., 5), values)
        for inner, pieces in (([0.], 4), ([0.], True), ([0.], 3), ([1.1], 2)):
            with self.assertRaises((ValueError, TypeError)):
                BM1PiecewiseResponse.from_inner_values(inner, pieces=pieces)
        curve = BM1PiecewiseResponse.linear()
        for x in (-.01, 1.01, np.nan, 1j):
            with self.assertRaises(ValueError):
                curve(x)

    def test_same_frozen_non_linear_curves_flow_through_forecast_and_f3(self):
        f = BM1PiecewiseResponse.from_inner_values([.8, .2, -.7], pieces=4)
        g = BM1PiecewiseResponse.from_inner_values([-.3], pieces=2)
        parameters = BM1Parameters(f, g, 1., 2., 1., .5)
        origin = BM1ForecastOrigin(1., 'day', parameters, .03,
            np.linspace(0., 1., 22), np.full(21, 1/21))
        baseline = lambda t: np.full(np.asarray(t).shape, .5)
        result = forecast_fv(origin, baseline=baseline, price_edges=[1., 2.],
                             prices=[.3], times=[1.5, 2.], max_step=.125)
        self.assertTrue(np.all(np.isfinite(result['demand_mean'])))
        problem = origin.control_problem(baseline=baseline, target=baseline,
                                          horizon=(1., 2.), price_edges=[1., 2.])
        self.assertIs(problem.parameters.state_response, f)
        self.assertIs(problem.parameters.price_response, g)
        self.assertIs(replace(parameters, C=.5).state_response, f)


if __name__ == '__main__':
    unittest.main()

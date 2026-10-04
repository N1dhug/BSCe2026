"""New H1 cases preserve A--G and reveal structurally missing price support."""

import unittest
import numpy as np

from src.validation.bm_1.response_cases import (
    H1_IDS, build_h1_case, h1_candidate_panel, h1_design, h1_training_records,
    response_price_support,
)
from src.validation.bm_1.estimation import build_f4_b_case


class CurveCaseTests(unittest.TestCase):
    def test_support_detects_old_null_direction_and_new_full_rank(self):
        old = response_price_support([.35, .65])
        self.assertEqual(old['rank'], 2)
        matrix = np.asarray(old['basis_matrix'])
        np.testing.assert_allclose(matrix @ [-2/3, 1., -2/3], 0., atol=3*np.finfo(float).eps)
        np.testing.assert_allclose(matrix @ np.asarray(old['null_directions']).T, 0., atol=2e-16)
        new = response_price_support([.125, .375, .625, .875]*2)
        self.assertEqual(new['rank'], 3)
        self.assertEqual(new['null_directions'], [])
        self.assertEqual(response_price_support([0., 1.])['rank'], 0)
        for prices in ([], [-.1], [np.nan], [[.5]], [1j]):
            with self.assertRaises(ValueError): response_price_support(prices)

    def test_registered_cases_do_not_mutate_old_core_and_panel_is_monotone(self):
        case = build_f4_b_case()
        original = case.build_model()[1][1](np.linspace(0, 1, 9))
        self.assertEqual(len(h1_candidate_panel()), 16)
        for name in H1_IDS:
            for index in (0, 1):
                new = build_h1_case(name, initial_index=index)
                self.assertEqual(new.horizon, (0., 2.))
                self.assertTrue(response_price_support(new.price_values)['full_rank'])
        for _, f, g in h1_candidate_panel():
            for curve in (f, g):
                self.assertTrue(np.all(np.diff(curve.values) <= 0))
        np.testing.assert_array_equal(case.build_model()[1][1](np.linspace(0, 1, 9)), original)
        self.assertEqual(h1_design()['cases']['nonlinear']['sequence_ids'], [1764, 1765])

    def test_new_records_are_reproducible_complete_streams_with_declared_priors(self):
        records, truth = h1_training_records('no_response')
        again, _ = h1_training_records('no_response')
        self.assertEqual(len(records), 2)
        for record, repeated, pair in zip(records, again, truth):
            self.assertEqual(record.horizon, (0., 2.))
            self.assertEqual(record.observations.times.size, 48)
            np.testing.assert_array_equal(record.observations.values, repeated.observations.values)
            np.testing.assert_array_equal(pair.coarse.measurement_noise, pair.fine.measurement_noise)
            np.testing.assert_array_equal(pair.coarse.truth.demand, .5)
            self.assertNotIn('truth', record.__dict__)
            self.assertTrue(pair.coarse.truth.valid and pair.fine.truth.valid)
        low = records[0].initial_mass_builder(np.linspace(0., 1., 102))
        high = records[1].initial_mass_builder(np.linspace(0., 1., 102))
        x = (np.arange(101)+.5)/101
        self.assertAlmostEqual(float(low @ x), .2, places=4)
        self.assertAlmostEqual(float(high @ x), .8, places=4)


if __name__ == '__main__': unittest.main()

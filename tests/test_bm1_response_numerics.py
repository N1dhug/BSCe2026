"""H1 full-prefix gates, exact density distance and resumable execution guards."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from src.exe.run_model.bm_1 import curves
from src.params.bm_1_response import BM1PiecewiseResponse
from src.validation.bm_1.response_numerics import density_l1, compare_numerical_unit
from src.validation.bm_1.response_cases import h1_training_records


class CurveNumericsTests(unittest.TestCase):
    def test_density_l1_integrates_common_refinement_without_cancellation(self):
        np.testing.assert_array_equal(density_l1([0, 1], [[1]], [0, .5, 1], [[.5, .5]]), [0.])
        # Coarsening to a single cell would incorrectly report zero distance.
        np.testing.assert_allclose(density_l1([0, 1], [[1]], [0, .5, 1], [[1., 0.]]), [1.])
        np.testing.assert_allclose(density_l1([0, .5, 1], [[1., 0.]], [0, .5, 1], [[0., 1.]]), [2.])
        with self.assertRaises(ValueError): density_l1([0, 1], [[1]], [0, 2], [[1]])

    def test_all_prefix_gate_detects_error_that_cancels_at_final_time(self):
        row = dict(x_edges=np.array([0., .5, 1.]), posterior_mass=np.full((2, 2), .5),
                   log_evidence=np.zeros(2))
        base = {name: dict(row) for name in curves.H1_NUMERICS}
        changed = {name: dict(row) for name in curves.H1_NUMERICS}
        changed['time']['log_evidence'] = np.array([.2, -.2])
        checks = compare_numerical_unit(changed, base)
        self.assertFalse(checks['time']['passes'])
        self.assertEqual(checks['time']['prefix_log_evidence'], .2)
        self.assertEqual(checks['time']['prefix_evidence_difference'][-1], 0.)

    def test_resume_skips_replays_and_detects_tampering_source_and_missing_raw(self):
        records, truths = h1_training_records('no_response')
        record = replace(records[0], observations=records[0].observations.prefix(2/24))
        linear = BM1PiecewiseResponse.linear()
        curved = BM1PiecewiseResponse.from_inner_values([.7, .2, -.8], pieces=4)
        panel = [('linear', linear, linear), ('other', curved, linear)]
        numerical = {name: dict(cells=11, max_step=1/24, quadrature_order=1) for name in curves.H1_NUMERICS}
        design = curves.h1_design()
        source = dict(commit='pinned', dirty=False, sha256={'source': 'hash'})
        with TemporaryDirectory() as folder, \
             patch.dict(curves.H1_IDS, {'no_response': (1766,)}, clear=True), \
             patch.dict(curves.H1_NUMERICS, numerical, clear=True), \
             patch.object(curves, 'h1_candidate_panel', return_value=panel), \
             patch.object(curves, 'h1_design', return_value=design), \
             patch.object(curves, 'h1_training_records', return_value=((record,), (truths[0],))), \
             patch.object(curves, 'source_snapshot', return_value=source):
            args = SimpleNamespace(run_dir=Path(folder)/'run', resume=False,
                                   limit_units=1, max_hours=1., quiet=True)
            first = curves.run_numerics(args)
            self.assertFalse(first['completed'])
            self.assertFalse(first['numerical_panel_passes'])
            self.assertEqual(first['new_units'], 1)
            args.resume, args.limit_units = True, None
            with patch.object(curves, 'run_numerical_unit', wraps=curves.run_numerical_unit) as run:
                second = curves.run_numerics(args)
                self.assertEqual(run.call_count, 1)
            self.assertTrue(second['completed'])
            self.assertTrue(second['numerical_panel_passes'])
            self.assertFalse(second['scientific_acceptance'])
            with patch.object(curves, 'run_numerical_unit', side_effect=AssertionError('new replay')):
                third = curves.run_numerics(args)
                self.assertEqual(third['new_units'], 0)
            with patch.object(curves, 'source_snapshot', return_value=dict(source, commit='changed')):
                with self.assertRaisesRegex(ValueError, 'source/design changed'): curves.run_numerics(args)
            path = next((args.run_dir/'units').glob('*.npz'))
            saved = path.read_bytes()
            path.write_bytes(saved+b'changed')
            with self.assertRaisesRegex(ValueError, 'changed after'): curves.run_numerics(args)
            path.write_bytes(saved)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'): curves.run_numerics(args)

    def test_dirty_source_and_existing_run_rejected_before_numerical_work(self):
        args = SimpleNamespace(run_dir=Path('unused'), resume=False, limit_units=1, max_hours=1.)
        with patch.object(curves, 'source_snapshot', return_value={'dirty': True}), \
             patch.object(curves, 'h1_training_records', side_effect=AssertionError('truth')):
            with self.assertRaisesRegex(ValueError, 'clean pinned'): curves.run_numerics(args)


if __name__ == '__main__': unittest.main()

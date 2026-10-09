"""Crash recovery must bind saved numerical laws to the exact selected fit."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from src.exe.run_model.bm_1 import curves
from src.method.bm_1_response import CurveTrainingRecord
from src.params.bm_1 import BM1ObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse
from src.validation.bm_1.response import run_numerical_unit


class SelectedResumeTests(unittest.TestCase):
    def test_orphaned_units_are_reused_only_if_they_match_the_selected_fit(self):
        constant = lambda t: np.full(np.asarray(t).shape, .5)
        linear = BM1PiecewiseResponse.linear()
        records = tuple(CurveTrainingRecord(name, 'TRAIN', constant, (0., 1.),
            (constant, linear, linear, 1., 2., 0.), 0., .1, np.diff,
            BM1ObservationSeries([.25, .75], [.5, .6], 'day')) for name in ('a', 'b'))
        source = dict(commit='selected-resume-test', dirty=False, sha256={})
        settings = {name: dict(cells=11, max_step=.25, quadrature_order=1)
                    for name in ('base', 'time', 'spatial', 'quadrature')}
        with TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(curves, 'source_snapshot', return_value=source))
            stack.enter_context(patch.object(curves, 'read_numerical_dependency', return_value={'fixture': 'release'}))
            stack.enter_context(patch.object(curves, 'h1_training_records', return_value=(records, ())))
            stack.enter_context(patch.dict(curves.H1_V2_NUMERICS, settings, clear=True))
            root = Path(folder)
            for case in curves.H1_IDS:
                for record in records:
                    rows = run_numerical_unit(record, linear, linear, numerics=settings, backend='fitted')
                    curves._atomic_report(root/'numerics'/'units'/f'{case}_{record.record_id}_linear.npz', {'rows': rows})
            fit_args = SimpleNamespace(run_dir=root/'fit', numerics_run=root/'numerics',
                resume=False, max_hours=1., limit_candidates=None, blas_threads=1, quiet=True, workers=1)
            self.assertTrue(curves.run_fit(fit_args)['completed'])
            args = SimpleNamespace(run_dir=root/'selected', fit_run=root/'fit', numerics_run=root/'numerics',
                resume=False, max_hours=1., blas_threads=1, quiet=True)
            full = curves.run_fit_checks(args)
            self.assertTrue(full['selected_numerical_passes'])
            args.resume = True
            # Simulate a crash after atomic unit publication but before the summary.
            (args.run_dir/'summary.json').unlink()
            with patch('src.validation.bm_1.response.run_numerical_unit',
                       side_effect=AssertionError('A verified raw unit must not repeat FV work.')):
                resumed = curves.run_fit_checks(args)
                self.assertEqual(resumed['new_units'], 0)
                self.assertTrue(resumed['selected_numerical_passes'])
            path = args.run_dir/'units'/'linear_2_a.npz'
            original, _ = curves._read_likelihood_report(path)
            for kind in ('score', 'posterior', 'extra_level'):
                with self.subTest(corruption=kind):
                    bad = deepcopy(original)
                    if kind == 'score':
                        # The same offset on all levels leaves refinement differences unchanged.
                        for row in bad['rows'].values():
                            row['log_evidence'] = np.asarray(row['log_evidence']) + 1.
                    elif kind == 'posterior':
                        for row in bad['rows'].values():
                            row['posterior_mass'] = np.asarray(row['posterior_mass']).copy()
                            row['posterior_mass'][-1, 0] -= .01
                            row['posterior_mass'][-1, -1] += .01
                    else:
                        bad['rows']['unregistered'] = deepcopy(bad['rows']['base'])
                    curves._atomic_report(path, bad, replace_existing=True)
                    (args.run_dir/'summary.json').unlink(missing_ok=True)
                    with self.assertRaises(ValueError):
                        curves.run_fit_checks(args)
            curves._atomic_report(path, original, replace_existing=True)


if __name__ == '__main__':
    unittest.main()

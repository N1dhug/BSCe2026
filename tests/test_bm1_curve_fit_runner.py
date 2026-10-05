"""Small H fit execution checks; numerical-release tests do not run new FV studies."""

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


class CurveFitRunnerTests(unittest.TestCase):
    def test_fit_pause_resume_and_completed_family_reuse(self):
        const = lambda t: np.full(np.asarray(t).shape, .5)
        linear = BM1PiecewiseResponse.linear()
        record = CurveTrainingRecord('test', 'TRAIN', const, (0., 1.),
            (const, linear, linear, 1., 2., 0.), 0., .1, np.diff,
            BM1ObservationSeries([.25, .75], [.5, .6], 'day'))
        source = {'commit': 'test-source', 'dirty': False, 'sha256': {}}
        with TemporaryDirectory() as folder, \
             patch.object(curves, 'source_snapshot', return_value=source), \
             patch.object(curves, 'read_numerical_dependency', return_value={'fixture': 'synthetic release fixture'}), \
             patch.object(curves, 'h1_training_records', return_value=((record,), ())), \
             patch.dict(curves.H1_V2_NUMERICS, {'base': dict(cells=11, max_step=.25, quadrature_order=1)}, clear=True):
            args = SimpleNamespace(run_dir=Path(folder)/'fit', numerics_run=Path(folder)/'numerics',
                resume=False, max_hours=1., limit_candidates=3, blas_threads=1, quiet=True)
            part = curves.run_fit(args)
            self.assertTrue(part['paused'])
            self.assertFalse(part['completed'])
            self.assertEqual(part['new_candidates'], 3)
            args.resume, args.limit_candidates = True, None
            full = curves.run_fit(args)
            self.assertTrue(full['completed'])
            self.assertFalse(full['paused'])
            self.assertEqual(len(full['fits']), 6)
            self.assertFalse(full['scientific_acceptance'])
            with patch('src.method.bm_1_response.fit_shared_curve_pair', side_effect=AssertionError('new fit')):
                cached = curves.run_fit(args)
                self.assertTrue(cached['completed'])
                self.assertEqual(cached['new_candidates'], 0)
            with patch.object(curves, 'source_snapshot', return_value=dict(source, commit='changed')):
                with self.assertRaisesRegex(ValueError, 'changed'): curves.run_fit(args)
            path = args.run_dir/'linear_2_result.json.gz'
            payload, _ = curves._read_likelihood_report(path)
            payload['f']['values'][1] = .2
            curves._atomic_report(path, payload, replace_existing=True)
            with self.assertRaisesRegex(ValueError, 'geometry differs'): curves.run_fit(args)

    def test_parallel_cases_match_serial_fit_traces_and_use_exclusive_files(self):
        from concurrent.futures import ThreadPoolExecutor
        const = lambda t: np.full(np.asarray(t).shape, .5)
        linear = BM1PiecewiseResponse.linear()
        record = CurveTrainingRecord('test', 'TRAIN', const, (0., 1.),
            (const, linear, linear, 1., 2., 0.), 0., .1, np.diff,
            BM1ObservationSeries([.25, .75], [.5, .6], 'day'))
        source = {'commit': 'test-source', 'dirty': False, 'sha256': {}}
        with TemporaryDirectory() as folder, \
             patch.object(curves, 'source_snapshot', return_value=source), \
             patch.object(curves, 'read_numerical_dependency', return_value={'fixture': 'release'}), \
             patch.object(curves, 'h1_training_records', return_value=((record,), ())), \
             patch.dict(curves.H1_V2_NUMERICS, {'base': dict(cells=11, max_step=.25, quadrature_order=1)}, clear=True):
            args = SimpleNamespace(run_dir=Path(folder)/'serial', numerics_run=Path(folder)/'numerics',
                resume=False, max_hours=1., limit_candidates=None, blas_threads=1, quiet=True, workers=1)
            serial = curves.run_fit(args)
            args.run_dir, args.workers = Path(folder)/'parallel', 3
            # Exercise actual concurrent publication using the same tiny pure
            # fixture; the separate numerical test covers real spawn transport.
            with patch('concurrent.futures.ProcessPoolExecutor',
                       side_effect=lambda max_workers, **kw: ThreadPoolExecutor(max_workers=max_workers)):
                parallel = curves.run_fit(args)
                self.assertTrue(parallel['completed'])
                for a,b in zip(serial['fits'],parallel['fits']):
                    self.assertEqual(a['trace'],b['trace'])
                    self.assertEqual(a['f'],b['f'])
                    self.assertEqual(a['replay'],b['replay'])
                args.resume = True
                cached = curves.run_fit(args)
                self.assertEqual(cached['new_candidates'],0)
            args.limit_candidates = 3
            with self.assertRaisesRegex(ValueError,'workers=1'): curves.run_fit(args)

    def test_old_or_incomplete_numerical_report_cannot_release_fit(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            curves._atomic_report(directory/'manifest.json', {'design': curves.h1_design(1)})
            curves._atomic_report(directory/'summary.json', {'design': curves.h1_design(1), 'passes': True})
            with self.assertRaisesRegex(ValueError, 'matching v2'):
                curves.read_numerical_dependency(directory)


if __name__ == '__main__': unittest.main()

"""H and G report writers retain Windows-safe publication after extraction."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.exe.run_model.bm_1 import curves, estimate
from src.tools import checkpoint_io
from src.tools.bm_1_reports import _read_likelihood_report


def sharing_error():
    error = PermissionError(13, 'Windows sharing violation')
    error.winerror = 32
    return error


class ReportPublicationTests(unittest.TestCase):
    def test_array_checkpoint_roundtrips_more_than_four_digit_indices(self):
        from src.tools.bm_1_reports import _atomic_report
        with TemporaryDirectory() as folder:
            path = Path(folder)/'many_arrays.npz'
            arrays = [np.array([i], dtype=np.int64) for i in range(10001)]
            _atomic_report(path, {'arrays': arrays})
            report, _ = _read_likelihood_report(path)
            self.assertEqual(len(report['arrays']), 10001)
            np.testing.assert_array_equal(np.concatenate(report['arrays']), np.arange(10001))

    def test_both_entrypoints_retry_before_publishing_complete_arrays(self):
        replace = os.replace
        for writer in (estimate._atomic_report, curves._atomic_report):
            with self.subTest(writer=writer), TemporaryDirectory() as folder:
                path = Path(folder)/'checkpoint.npz'
                writer(path, {'mass': np.array([0.25, 0.75])})
                original = path.read_bytes()
                calls = []

                def temporarily_locked(source, destination):
                    calls.append(destination.read_bytes())
                    if len(calls) < 3:
                        raise sharing_error()
                    return replace(source, destination)

                with patch.object(checkpoint_io.os, 'replace', side_effect=temporarily_locked), \
                     patch.object(checkpoint_io, 'sleep'):
                    writer(path, {'mass': np.array([0.4, 0.6])}, replace_existing=True)
                self.assertEqual(calls, [original]*3)
                report, _ = _read_likelihood_report(path)
                np.testing.assert_array_equal(report['mass'], [0.4, 0.6])
                self.assertEqual(list(Path(folder).glob('*.partial')), [])

    def test_failed_publication_keeps_previous_checkpoint_and_propagates(self):
        for writer in (estimate._atomic_report, curves._atomic_report):
            with self.subTest(writer=writer), TemporaryDirectory() as folder:
                path = Path(folder)/'summary.json'
                writer(path, {'completed': False})
                original = path.read_bytes()
                with patch.object(checkpoint_io.os, 'replace', side_effect=sharing_error()), \
                     patch.object(checkpoint_io, 'sleep'):
                    with self.assertRaises(PermissionError):
                        writer(path, {'completed': True}, replace_existing=True)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(Path(folder).glob('*.partial')), [])


if __name__ == '__main__':
    unittest.main()

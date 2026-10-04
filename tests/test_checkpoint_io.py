"""Transient locks preserve published files and permit Windows cleanup retries."""
import ctypes
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

# Load this stdlib-only helper directly: src/__init__.py also imports numerical
# packages, which the cheap Windows file-handle CI job does not need to install.
_spec = importlib.util.spec_from_file_location(
    'checkpoint_io_under_test', Path(__file__).resolve().parents[1]/'src/tools/checkpoint_io.py')
io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(io)


def sharing_error(code=32):
    error = PermissionError(13, 'Windows sharing violation')
    error.winerror = code
    return error


class CheckpointIOTests(unittest.TestCase):
    def test_replace_retries_sharing_errors_without_changing_saved_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'new', Path(directory)/'checkpoint.npz'
            source.write_bytes(b'complete next block'); target.write_bytes(b'old block')
            replace = os.replace
            calls = []
            def blocked(*args):
                calls.append(target.read_bytes())
                if len(calls) < 3:
                    raise sharing_error(32 if len(calls) == 1 else 33)
                return replace(*args)
            with patch.object(io.os, 'replace', side_effect=blocked), patch.object(io, 'sleep') as pause:
                io.replace_file(source, target)
            self.assertEqual(calls, [b'old block']*3)
            self.assertEqual(pause.call_count, 2)
            self.assertEqual(target.read_bytes(), b'complete next block')

    def test_persistent_replace_error_retains_both_files_and_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'new', Path(directory)/'checkpoint.npz'
            source.write_bytes(b'next'); target.write_bytes(b'previous')
            with patch.object(io.os, 'replace', side_effect=sharing_error()), patch.object(io, 'sleep'):
                with self.assertRaises(PermissionError): io.replace_file(source, target)
            self.assertEqual(source.read_bytes(), b'next')
            self.assertEqual(target.read_bytes(), b'previous')

    def test_non_windows_permission_denial_is_not_retried_or_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'; path.write_bytes(b'saved')
            denied = PermissionError(13, 'Denied')
            with patch.object(io.os, 'replace', side_effect=denied), patch.object(io, 'sleep') as pause:
                with self.assertRaises(PermissionError): io.replace_file(path, path.with_name('new'))
                pause.assert_not_called()
            with patch.object(Path, 'unlink', side_effect=denied), patch.object(io, 'sleep') as pause:
                with self.assertRaises(PermissionError): io.discard_file(path)
                pause.assert_not_called()
            self.assertEqual(path.read_bytes(), b'saved')

    def test_windows_access_denial_has_bounded_replace_retries_but_no_cleanup_deferral(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'new', Path(directory)/'checkpoint.npz'
            source.write_bytes(b'next'); target.write_bytes(b'previous')
            denied = PermissionError(13, 'Denied'); denied.winerror = 5
            with patch.object(io.os, 'replace', side_effect=denied) as replace, \
                 patch.object(io, 'sleep') as pause:
                with self.assertRaises(PermissionError): io.replace_file(source, target)
                self.assertEqual(replace.call_count, len(io._RETRY_DELAYS) + 1)
                self.assertEqual(pause.call_count, len(io._RETRY_DELAYS))
            with patch.object(Path, 'unlink', side_effect=denied), patch.object(io, 'sleep') as pause:
                with self.assertRaises(PermissionError): io.discard_file(target)
                pause.assert_not_called()
            self.assertEqual(source.read_bytes(), b'next')
            self.assertEqual(target.read_bytes(), b'previous')

    def test_only_redundant_cleanup_can_be_deferred(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'; path.write_bytes(b'complete')
            with patch.object(Path, 'unlink', side_effect=sharing_error()), patch.object(io, 'sleep'), \
                 patch('builtins.print'):
                self.assertFalse(io.discard_file(path))
            self.assertEqual(path.read_bytes(), b'complete')
            self.assertTrue(io.discard_file(path)); self.assertFalse(path.exists())
            self.assertTrue(io.discard_file(path))


@unittest.skipUnless(os.name == 'nt', 'Requires an actual Windows file handle')
class WindowsSharingTests(unittest.TestCase):
    def lock_without_delete_sharing(self, path):
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
        return kernel, handle

    def test_atomic_replace_waits_for_real_windows_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'new', Path(directory)/'checkpoint.npz'
            source.write_bytes(b'complete'); target.write_bytes(b'previous')
            kernel, handle = self.lock_without_delete_sharing(target)
            timer = threading.Timer(.25, kernel.CloseHandle, args=(handle,))
            timer.start()
            try:
                io.replace_file(source, target)
                self.assertEqual(target.read_bytes(), b'complete')
            finally:
                timer.join()

    def test_locked_redundant_checkpoint_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'checkpoint.npz'; target.write_bytes(b'complete')
            kernel, handle = self.lock_without_delete_sharing(target)
            try:
                with patch.object(io, '_RETRY_DELAYS', ()), patch('builtins.print'):
                    self.assertFalse(io.discard_file(target))
                self.assertEqual(target.read_bytes(), b'complete')
            finally:
                kernel.CloseHandle(handle)


if __name__ == '__main__': unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.tools.run_bundle import pack, unpack, sha256


class BundleTests(unittest.TestCase):
    def test_split_roundtrip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'raw').mkdir()
            data = bytes(range(256))*8192
            (root/'raw/checkpoint.npz').write_bytes(data)
            index = root/'transfer.json'
            result = pack(root/'raw', ['checkpoint.npz'], index, part_mib=1)
            self.assertGreater(len(result['parts']), 1)
            unpack(index, root/'restored')
            self.assertEqual((root/'restored/checkpoint.npz').read_bytes(), data)
            with self.assertRaises(FileExistsError):
                unpack(index, root/'restored')
            (root/result['parts'][0]['name']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                unpack(index, root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_unsafe_archive_members_rejected_before_extraction(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'a\\b', 'x/../escape'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                part = root/'part'
                with zipfile.ZipFile(part, 'w') as archive:
                    archive.writestr('a/b' if name == 'a\\b' else name, 'data')
                if name == 'a\\b':
                    # Windows writers normalize separators. Inject the actual malformed
                    # filename into both ZIP headers so both platforms read the same bytes.
                    part.write_bytes(part.read_bytes().replace(b'a/b', b'a\\b'))
                index = dict(format='bm-data-bundle-v1', archive_sha256=sha256(part),
                    parts=[dict(name='part', bytes=part.stat().st_size, sha256=sha256(part))])
                (root/'index.json').write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    unpack(root/'index.json', root/'out')
                self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()

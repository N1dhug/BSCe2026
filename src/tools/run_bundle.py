"""Split portable data snapshots into checksum-verified upload-sized parts.

No executable code or automatic unpickling. Extraction accepts relative regular
files only, into a new directory, and verifies every archive part before writing.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _safe_name(name):
    path = PurePosixPath(name)
    if (not name or any(ord(c) < 32 for c in name) or path.is_absolute() or '..' in path.parts or '\\' in name
            or ':' in name or str(path) != name or name.endswith('/')):
        raise ValueError('Unsafe bundle member: '+name)
    devices = {'con', 'prn', 'aux', 'nul'} | {f'{p}{i}' for p in ('com', 'lpt') for i in range(1, 10)}
    if any(part.endswith(('.', ' ')) or part.split('.')[0].lower() in devices
           or any(c in part for c in '<>"|?*') for part in path.parts):
        raise ValueError('Not a portable regular filename: '+name)
    return path


def pack(root, files, index_path, *, part_mib=24):
    if not isinstance(part_mib, int) or not 1 <= part_mib <= 24:
        raise ValueError('part_mib must be between 1 and 24.')
    if index_path.exists():
        raise FileExistsError(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    with tempfile.TemporaryDirectory(prefix='bm-bundle-') as temporary:
        archive_path = Path(temporary)/'data.zip'
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED) as archive:
            for relative in sorted(set(files)):
                _safe_name(relative)
                path = root/relative
                if (path.is_symlink() or not path.is_file()
                        or not path.resolve().is_relative_to(root.resolve())
                        or any(parent.is_symlink() for parent in path.parents if parent != root.parent)):
                    raise ValueError('Bundle inputs must be regular files: '+relative)
                archive.write(path, relative)
        with archive_path.open('rb') as stream:
            while data := stream.read(part_mib*1024*1024):
                name = index_path.stem+f'.part{len(parts):04d}'
                path = index_path.parent/name
                with path.open('xb') as output:
                    output.write(data)
                parts.append(dict(name=name, bytes=len(data), sha256=sha256(path)))
        index = dict(format='bm-data-bundle-v1', archive_sha256=sha256(archive_path), parts=parts)
        # Index is the commit marker, written only after all parts exist.
        with index_path.open('x', encoding='utf-8') as output:
            json.dump(index, output, indent=2)
    return index


def unpack(index_path, destination):
    if destination.exists():
        raise FileExistsError('Extract to a new directory; never overlay a live run.')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if index.get('format') != 'bm-data-bundle-v1' or not index.get('parts'):
        raise ValueError('Invalid bundle index.')
    names = set()
    with tempfile.TemporaryDirectory(prefix='bm-unpack-') as temporary:
        archive_path = Path(temporary)/'data.zip'
        with archive_path.open('wb') as output:
            for part in index['parts']:
                name = part['name']
                if len(_safe_name(name).parts) != 1 or name in names:
                    raise ValueError('Invalid or duplicate bundle part.')
                names.add(name)
                path = index_path.parent/name
                if path.stat().st_size != part['bytes'] or sha256(path) != part['sha256']:
                    raise ValueError('Missing/changed bundle part: '+name)
                with path.open('rb') as stream:
                    shutil.copyfileobj(stream, output)
        if sha256(archive_path) != index['archive_sha256']:
            raise ValueError('Combined archive checksum mismatch.')
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len({m.filename.casefold() for m in members}) != len(members):
                raise ValueError('Duplicate or case-colliding archive members.')
            for member in members:
                # ZipInfo normalizes Windows backslashes and truncates NULs in filename.
                # Check the original archive spelling before trusting the normalized path.
                _safe_name(member.orig_filename)
                mode = member.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError('Only regular files are allowed in a bundle.')
            destination.mkdir(parents=True)
            for member in members:
                target = destination/member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as stream, target.open('xb') as output:
                    shutil.copyfileobj(stream, output)
    return destination

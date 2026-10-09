"""Verify and restore the exact archived BM source to a NEW directory.

This command never starts a model or restores private data/results. Supply
those separately under the original paths and honor original source guards.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_target(root: Path, member: str) -> Path:
    path = PurePosixPath(member)
    if path.is_absolute() or '..' in path.parts or '\\' in member or ':' in member:
        raise ValueError(f'Unsafe archive member: {member}')
    target = (root / member).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Archive member escapes destination: {member}')
    return target


def restore(destination: Path, manifest_path: Path, *, verify_only: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    archive = manifest_path.parent / manifest['archive']
    if sha256(archive.read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Original source archive checksum mismatch.')
    destination = destination.resolve()
    if not verify_only and destination.exists():
        raise ValueError('Destination must not exist; no overwrite or merge is allowed.')
    files = manifest['files']
    with ZipFile(archive) as source:
        members = source.namelist()
        if len(members) != len(set(members)) or set(members) != set(files):
            raise ValueError('Archive member set mismatch.')
        # Validate the entire archive and every resolved path before writing.
        for member in members:
            checked_target(destination, member)
            data = source.read(member)
            if len(data) != files[member]['bytes'] or sha256(data) != files[member]['sha256']:
                raise ValueError(f'Original member checksum mismatch: {member}')
        if not verify_only:
            destination.mkdir(parents=True, exist_ok=False)
            for member in members:
                target = checked_target(destination, member)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as output:
                    output.write(source.read(member))
                if sha256(target.read_bytes()) != files[member]['sha256']:
                    raise ValueError(f'Restored member checksum mismatch: {member}')
    return {'source_commit': manifest['source_commit'], 'files_verified': len(files),
            'restored': not verify_only, 'destination': str(destination), 'models_started': 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parent / 'originals/source-7249bf9.json')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(restore(args.destination, args.manifest, verify_only=args.verify_only), indent=2))


if __name__ == '__main__':
    main()

"""Build F4 notes from any working directory; no latexmk/Perl required.

Usage from the repository root: python -B root/f4/build.py
Only root/f4/build is written. No PATH changes or package-install commands.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

NOTE_DIR = Path(__file__).resolve().parent
REFERENCE_WARNINGS = (
    r'Label\(s\) may have changed', r'There were undefined references',
    r'(?:Reference|Citation) [^\n]*undefined', r'Rerun to get',
)
ANCHOR_WARNING = r'destination with the same identifier'


class BuildError(RuntimeError):
    """A missing tool, failed LaTeX run or unresolved document reference."""


def find_engine(explicit: str | None = None) -> str:
    """Find pdflatex without permanently or temporarily changing PATH."""
    if explicit:
        candidate = shutil.which(explicit)
        if candidate:
            return candidate
        raise BuildError(f'PDF engine not found: {explicit}')
    candidate = shutil.which('pdflatex')
    if candidate:
        return candidate
    for variable in ('LOCALAPPDATA', 'ProgramFiles'):
        base = os.environ.get(variable)
        if not base:
            continue
        parts = ('Programs', 'MiKTeX') if variable == 'LOCALAPPDATA' else ('MiKTeX',)
        path = Path(base).joinpath(*parts, 'miktex', 'bin', 'x64', 'pdflatex.exe')
        if path.is_file():
            return str(path)
    raise BuildError('pdflatex not found. Use the PDF artifact from the GitHub PR, '
                     'or install MiKTeX/TeX Live. Python tests do not require LaTeX.')


def reference_state(directory: Path) -> bytes:
    """Hash files whose changes require another LaTeX pass."""
    digest = hashlib.sha256()
    for suffix in ('aux', 'toc', 'out'):
        path = directory / f'main.{suffix}'
        digest.update(suffix.encode())
        digest.update(path.read_bytes() if path.exists() else b'')
    return digest.digest()


def build_notes(note_dir: Path, engine: str, *, max_runs: int = 5,
                verbose: bool = False) -> Path:
    """Run to stable references or fail; never return an old PDF after failure."""
    note_dir = note_dir.resolve()
    if max_runs < 2:
        raise BuildError('max_runs must be at least 2.')
    for name in ('main.tex', 'notes.tex'):
        if not (note_dir / name).is_file():
            raise BuildError(f'Missing source: {note_dir / name}')
    output = note_dir / 'build'
    output.mkdir(exist_ok=True)
    pdf = output / 'main.pdf'
    # An unsuccessful attempt must not be confused with a successful old artifact.
    if pdf.exists():
        try:
            pdf.unlink()
        except OSError as exc:
            raise BuildError(f'Close the PDF viewer and retry: {pdf}') from exc
    try:
        version = subprocess.run([engine, '--version'], capture_output=True,
                                 text=True, errors='replace', timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildError(f'Cannot run PDF engine {engine}: {exc}') from exc
    shell_flag = '--disable-write18' if 'MiKTeX' in version.stdout else '-no-shell-escape'
    command = [engine, shell_flag, '-interaction=nonstopmode', '-halt-on-error',
               '-file-line-error', f'-output-directory={output}', 'main.tex']
    before = reference_state(output)
    log_path = output / 'build.log'
    with log_path.open('w', encoding='utf-8') as log:
        log.write(version.stdout + version.stderr)
        for run in range(1, max_runs + 1):
            try:
                result = subprocess.run(command, cwd=note_dir, capture_output=True,
                                        text=True, errors='replace', timeout=300)
            except (OSError, subprocess.SubprocessError) as exc:
                pdf.unlink(missing_ok=True)
                raise BuildError(f'LaTeX could not finish: {exc}. See {log_path}') from exc
            text = result.stdout + result.stderr
            log.write(f'\n--- PASS {run} ---\n{text}\n')
            log.flush()
            if verbose:
                print(text)
            if result.returncode:
                pdf.unlink(missing_ok=True)
                excerpt = '\n'.join(text.splitlines()[-30:])
                raise BuildError(f'LaTeX failed on pass {run}:\n{excerpt}\nLog: {log_path}')
            if re.search(ANCHOR_WARNING, text, flags=re.IGNORECASE):
                pdf.unlink(missing_ok=True)
                raise BuildError(f'Duplicate PDF page anchors. See {log_path}')
            after = reference_state(output)
            unresolved = any(re.search(pattern, text, flags=re.IGNORECASE)
                             for pattern in REFERENCE_WARNINGS)
            if run >= 2 and before == after and not unresolved:
                if not pdf.is_file():
                    raise BuildError(f'LaTeX returned success without producing {pdf}')
                boxes = len(re.findall(r'(?:Underfull|Overfull) \\hbox', text))
                print(f'PDF ready after {run} passes: {pdf}')
                if boxes:
                    print(f'{boxes} line-layout notices; review the PDF. Details: {log_path}')
                return pdf
            before = after
    pdf.unlink(missing_ok=True)
    raise BuildError(f'References did not stabilize in {max_runs} passes. See {log_path}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', help='Explicit pdflatex executable path.')
    parser.add_argument('--max-runs', type=int, default=5)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    try:
        build_notes(NOTE_DIR, find_engine(args.engine), max_runs=args.max_runs,
                    verbose=args.verbose)
    except BuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

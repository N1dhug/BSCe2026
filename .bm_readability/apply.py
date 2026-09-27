"""One-time delivery helper. Removed from the final repository tree."""
import ast
import base64
import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

import numpy as np

EXPECTED = {
    'root/BM_GUIDE.md', 'root/BM_PERFORMANCE.md', 'root/F2.md', 'root/F3.md',
    'src/diagnostics/bm_1.py',
    'src/exe/run_model/bm_1/optimize.py', 'src/exe/run_model/bm_1/real_data.py',
    'src/exe/run_model/bm_1/refine.py', 'src/exe/run_model/bm_1/verify.py',
    'src/method/bm_1_control.py', 'src/method/bm_1_mixture.py', 'src/method/bm_1_objective.py',
    'src/model/bm_1.py', 'src/tools/bm_1_data.py',
    'src/validation/bm_1/cases.py', 'src/validation/bm_1/control.py',
    'src/validation/bm_1/deterministic.py', 'src/validation/bm_1/mixture.py',
    'src/validation/bm_1/numerics.py', 'src/validation/bm_1/stochastic.py',
    'src/validation/diagnostics/bm_1.py', 'src/validation/method/bm_1.py',
    'src/validation/model/bm_1.py', 'tests/test_bm1_control.py', 'tests/test_bm1_layout.py',
}

def apply():
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'refactor/bm-readability'
    encoded = ''.join(Path(f'.bm_readability/part{i}.b64').read_text().strip() for i in range(8))
    assert len(encoded) == 49840
    raw = base64.b64decode(encoded, validate=True)
    assert hashlib.sha256(raw).hexdigest() == '6f06cbae3dd5bcde28aae95820d445b08e74f27533bb1184df65e542bd57a690'
    delta = json.loads(lzma.decompress(raw))
    assert set(delta) == EXPECTED
    staged = {}
    for name, change in delta.items():
        source = Path(name).read_text(encoding='utf-8')
        assert hashlib.sha256(source.encode()).hexdigest() == change['before'], name
        target = ''.join(source[p[0]:p[1]] if isinstance(p, list) else p for p in change['parts'])
        assert hashlib.sha256(target.encode()).hexdigest() == change['after'], name
        if name.endswith('.py'):
            ast.parse(target, filename=name)
        staged[name] = target
    for name, target in staged.items():
        Path(name).write_text(target, encoding='utf-8')
    changed = set(subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines())
    assert changed == EXPECTED, changed ^ EXPECTED
    print(f'Applied exactly {len(changed)} checked BM/test/documentation files; no WM/shared changes.')

def compare(directory):
    directory = Path(directory)
    before = np.load(directory / 'before.npz')
    after = np.load(directory / 'after.npz')
    assert set(before.files) == set(after.files)
    timing = [k for k in before.files if k.endswith('/elapsed_seconds')]
    equal, failures = [], []
    for name in before.files:
        if name in timing:
            continue
        if np.array_equal(before[name], after[name], equal_nan=True):
            equal.append(name)
        else:
            failures.append(name)
    report = {'exact_equal_fields': len(equal), 'excluded_timing_fields': timing, 'different_fields': failures}
    print(json.dumps(report, indent=2))
    (directory / 'comparison.json').write_text(json.dumps(report, indent=2))
    assert not failures, failures

def stage():
    changed = set(subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines())
    assert changed == EXPECTED, changed ^ EXPECTED
    subprocess.run(['git', 'add', '--', *sorted(EXPECTED)], check=True)
    cached = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines())
    assert cached == EXPECTED, cached ^ EXPECTED

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'apply': apply()
    elif mode == 'compare': compare(sys.argv[2])
    elif mode == 'stage': stage()
    else: raise ValueError(mode)

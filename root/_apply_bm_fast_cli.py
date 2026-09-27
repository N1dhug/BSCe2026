"""Temporary, fixed-source integration script; removed after verification."""
from pathlib import Path
import hashlib
import subprocess


def blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def check(path, expected):
    if blob(Path(path).read_bytes()) != expected:
        raise RuntimeError('Unexpected source: ' + path)


check('src/method/bm_1_fast_fv.py', 'e0de422dac04cfca90a4687bb0f734d1946100ee')
check('tests/test_bm_1_fast_fv.py', '184b20627f92c5ebeb1559df00d3a0ff49fb50c4')
check('src/exe/run_model/bm_1/benchmark_fast_fv.py', 'e81fbfeabb34f912e97c11cba6e76643ff41a699')
check('src/exe/run_model/bm_1/multi_price_optimization_experiment.py', 'd373fafe6d92ffdca949e3374117d3b3599d2982')

# Preserve the user's latest complex experiment, not yet included in main.
source = '67766c38082ada631009a972d4301eb168219f0f'
for path, expected in (
    ('src/exe/run_model/bm_1/complex_target_multi_price_experiment.py', '8a926d0216fc27f25937eb8780121e0aee11d12d'),
    ('tests/test_bm_1_complex_target_multi_price_experiment.py', 'f0f99a46ae1280b807f4ec3f63798e6ab1f1f8bf'),
):
    data = subprocess.check_output(['git', 'show', source + ':' + path])
    if blob(data) != expected:
        raise RuntimeError('Unexpected experiment source: ' + path)
    Path(path).write_bytes(data)

base = Path('src/exe/run_model/bm_1')
for filename in ('complex_target_multi_price_experiment.py', 'multi_price_optimization_experiment.py'):
    p = base / filename
    s = p.read_text(encoding='utf-8')
    s = s.replace('from dataclasses import dataclass', 'import argparse\nfrom dataclasses import dataclass')
    s = s.replace('from src.method.bm_1_stochastic_objective import monte_carlo_tracking_objective', 'from src.method.bm_1_fast_fv import FastFVStochasticPriceEvaluator\nfrom src.method.bm_1_stochastic_objective import monte_carlo_tracking_objective')
    if filename.startswith('complex'):
        s = s.replace('def _build_evaluator(model, n_intervals):', 'def _build_evaluator(model, n_intervals, *, fast_fv=False):')
        s = s.replace('def run_complex_target_experiment(progress=print, validate_mc=True):', 'def run_complex_target_experiment(progress=print, validate_mc=True, *, fast_fv=False):')
        s = s.replace('_build_evaluator(model, count)', '_build_evaluator(model, count, fast_fv=fast_fv)')
        s = s.replace('    result = run_complex_target_experiment(\n        progress=lambda message: print(message, flush=True)\n    )', '    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--fast-fv", action="store_true", help="Use the numerically equivalent LAPACK/blocked FV evaluator.")\n    args = parser.parse_args()\n    print("FV backend: " + ("fast" if args.fast_fv else "reference"), flush=True)\n    result = run_complex_target_experiment(\n        progress=lambda message: print(message, flush=True), fast_fv=args.fast_fv\n    )')
    else:
        s = s.replace('def _build_evaluator(model, n_intervals, common_breakpoints):', 'def _build_evaluator(model, n_intervals, common_breakpoints, *, fast_fv=False):')
        s = s.replace('def run_multi_price_experiment(progress=print, validate_mc=True):', 'def run_multi_price_experiment(progress=print, validate_mc=True, *, fast_fv=False):')
        s = s.replace('_build_evaluator(model, count, common_breakpoints)', '_build_evaluator(model, count, common_breakpoints, fast_fv=fast_fv)')
        s = s.replace('    result = run_multi_price_experiment(progress=lambda message: print(message, flush=True))', '    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--fast-fv", action="store_true", help="Use the numerically equivalent LAPACK/blocked FV evaluator.")\n    args = parser.parse_args()\n    print("FV backend: " + ("fast" if args.fast_fv else "reference"), flush=True)\n    result = run_multi_price_experiment(\n        progress=lambda message: print(message, flush=True), fast_fv=args.fast_fv\n    )')
    s = s.replace('    return FVStochasticPriceEvaluator(\n', '    evaluator_class = FastFVStochasticPriceEvaluator if fast_fv else FVStochasticPriceEvaluator\n    return evaluator_class(\n')
    p.write_text(s, encoding='utf-8')

check(str(base / 'multi_price_optimization_experiment.py'), '9910287f8683c2db30ae249c8f8b4ac294198897')
check(str(base / 'complex_target_multi_price_experiment.py'), '7b6c27c1569847d17ce1ce0ccfbbc4f5b022b2d3')
print('Exact tested CLI sources installed; shared WM sources unchanged.')

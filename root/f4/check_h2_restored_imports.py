"""One registered import-only recovery control; no model or observation decode."""
from time import perf_counter
BEGUN = perf_counter()
import argparse
import importlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from root.f4 import run_h2_filesystem_recovery as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    a = p.parse_args(); reg = r.load(a.registration)
    r.require(reg['source_sha256'] == r.sha(Path(__file__)) and reg['cap_seconds'] == 5
        and len(a.source_commit) == 40, 'Unregistered import-only control.')
    decision = r.load(ROOT/reg['decision_path'])
    r.BEGUN = BEGUN
    with r.reservation(ROOT/reg['campaign_path'], decision, 'restored-import-closure-control', 5, native=True):
        closure = r.load(ROOT/reg['closure_path'])
        r.require(closure['static_complete'] is True, 'Missing static import dependency.')
        for name, expected in closure['source_sha256'].items():
            r.require(r.sha(ROOT/name) == expected, 'Restored module changed: '+name)
        modules = ('src.exe.run_model.bm_1_history', 'src.method.bm_1_interval_native',
            'src.method.bm_1_interval_forecast', 'src.control.bm_1_handoff')
        for name in modules:
            importlib.import_module(name)
        from src.exe.run_model.bm_1_history import _beta_mass
        from src.control.bm_1_handoff import BM1ForecastOrigin
        r.require(callable(_beta_mass) and callable(BM1ForecastOrigin), 'Missing required F functions.')
        report = dict(passed=True, imported_modules=list(modules), native_model_calls=0,
            final_observations_decoded=0, source_commit=a.source_commit,
            source_sha256=reg['source_sha256'], closure_sha256=r.sha(ROOT/reg['closure_path']),
            terminal_failure_preserved=True, scientific_acceptance=False)
        r.write(ROOT/reg['report_path'], report)
        print(r.json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

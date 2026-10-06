"""Focused ledger transfer/cap checks; no FV solves or worker pool."""
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def check():
    from src.exe.run_model import bm_1_h2_native as native
    from src.exe.run_model.bm_1_h2_precise import PRIOR_COMPUTE_SECONDS, _load, _write
    baseline = _load(native.OUT/'f4_h2_remaining_budget.json')
    original_seed = native.SEED
    def total(ledger):
        ledger['charged_seconds'] = PRIOR_COMPUTE_SECONDS+sum(e['charged_seconds'] for e in ledger['entries'])
        return ledger
    results = []
    with TemporaryDirectory(prefix='h2-budget-') as directory:
        directory = Path(directory)
        native.SEED = directory/'seed.json'; _write(native.SEED, baseline)
        path = directory/'ledger.json'
        try:
            older = total(dict(baseline, entries=deepcopy(baseline['entries'][:4]), validation_reserve_seconds=14400.))
            _write(path, older)
            with native.native_reservation(path, 100., 'temporary-check') as reservation:
                assert reservation['ledger']['entries'][:-1] == baseline['entries']
                assert reservation['ledger']['validation_reserve_seconds'] == 14400.
                assert reservation['ledger']['entries'][-1]['status'] == 'reserved'
            closed = _load(path)
            assert closed['entries'][-1]['status'] == 'closed'
            assert closed['charged_seconds'] >= baseline['charged_seconds']
            assert closed['charged_seconds'] < baseline['charged_seconds']+100.
            results.append('newer seed extends PC history; larger reserve and orderly refund preserved')
            newer = deepcopy(baseline)
            newer['entries'].append(dict(label='native-development|newer-PC', status='closed', charged_seconds=1.))
            _write(path, total(newer))
            with native.native_reservation(path, 100., 'temporary-check') as reservation:
                assert reservation['ledger']['entries'][:-1] == newer['entries']
            results.append('newer PC history retained; older seed cannot reset time')
            from time import perf_counter
            _write(path, baseline)
            with native.native_reservation(path, 100., 'startup-check', begun=perf_counter()-2.) as reservation:
                assert reservation['deadline'] <= __import__('time').time()+38.
            assert _load(path)['entries'][-1]['charged_seconds'] >= 2.
            results.append('startup/import time charged and removed from compute deadline')
            for kind in ('divergent', 'native-cap-exhausted'):
                invalid = deepcopy(baseline)
                if kind == 'divergent':
                    invalid['entries'][0]['label'] += '-changed'
                else:
                    invalid['entries'].append(dict(label='native-development|exhausted', status='reserved', charged_seconds=900.))
                _write(path, total(invalid)); before = path.read_bytes()
                try:
                    with native.native_reservation(path, 100., 'must-not-run'):
                        raise AssertionError('Unsafe ledger admitted.')
                except ValueError:
                    pass
                assert path.read_bytes() == before
                results.append(kind+' rejected without ledger mutation')
        finally:
            native.SEED = original_seed
    return dict(passed=True, checks=results, numerical_replays=0)


if __name__ == '__main__':
    from src.exe.run_model.bm_1_h2_precise import campaign_reservation, DEFAULT_CAMPAIGN, _write
    with campaign_reservation(DEFAULT_CAMPAIGN, 30., 'native-development|focused-bookkeeping-check'):
        result = check()
        _write(ROOT/'files/output/bm_1/runs/estimation/f4_h2_native_budget_checks_20261006.json', result)
        print(result)

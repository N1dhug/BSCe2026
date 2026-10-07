"""Authorized 18-hour controller over the unchanged frozen H2 numerical engine.

The old campaign remains immutable. A versioned extension preserves every old
entry, including the crash reservation, and debits each short invocation.
No cached model job is assigned a different identity or arithmetic trajectory.
"""
from time import perf_counter, time
BEGUN = perf_counter()-.25
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'
from src.exe.run_model import bm_1_h2_compact as c

SOURCES = ('root/f4/run_h2_same_day.py', 'root/H2_SAME_DAY_EXECUTION_20261007.md',
           'root/f4/h2_same_day_budget_decision_20261007.json')


def validate_campaign(ledger, decision):
    original = c.validate_ledger(c.load(ROOT/decision['original_campaign_path']))
    c.require(c.sha(ROOT/decision['original_campaign_path']) == decision['original_campaign_sha256'],
              'Original campaign changed; never discard or refund its reservation.')
    c.require(ledger['schema'] == 'f4-h2-authorized-campaign-v2'
              and ledger['ceiling_seconds'] == decision['ceiling_seconds'] == 64800.
              and ledger['deadline_utc_seconds'] == decision['deadline_utc_seconds']
              and ledger['decision_identity'] == decision['identity'], 'Different approved budget.')
    entries = ledger['entries']
    c.require(entries[:len(original['entries'])] == original['entries'], 'Original entries changed.')
    c.require(all(e['status'] in ('closed','reserved') and math.isfinite(e['charged_seconds'])
                  and e['charged_seconds'] >= 0. for e in entries), 'Invalid charges.')
    c.require(ledger['charged_seconds'] == c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in entries),
              'Charges do not match their full history.')
    c.require(ledger['validation_reserve_seconds'] >= original['validation_reserve_seconds'],
              'Protected validation cannot be lowered by an empirical estimate.')
    return ledger


def register_campaign(path, decision):
    c.require(decision['identity'] == c.digest({k:v for k,v in decision.items() if k!='identity'})
              and decision['user_approval'] == 'Jeg godkender, lad os fortsaette'
              and decision['invocation_cap_seconds'] == 600., 'Budget approval record differs.')
    if path.exists():
        return validate_campaign(c.load(path), decision)
    old = c.validate_ledger(c.load(ROOT/decision['original_campaign_path']))
    ledger = dict(old, schema='f4-h2-authorized-campaign-v2',
        ceiling_seconds=decision['ceiling_seconds'], deadline_utc_seconds=decision['deadline_utc_seconds'],
        decision_identity=decision['identity'], original_campaign_sha256=decision['original_campaign_sha256'])
    validate_campaign(ledger, decision)
    c.write(path, ledger)
    return ledger


@contextmanager
def reservation(path, decision, label, *, begun=None, requested=600.):
    begun = perf_counter() if begun is None else begun
    c.require(75. < requested <= decision['invocation_cap_seconds'], 'Use short approved invocations.')
    with c._RunLock(path.with_suffix('.lock')):
        ledger = validate_campaign(c.load(path), decision)
        before = ledger['charged_seconds']
        startup = perf_counter()-begun
        grant = min(requested, ledger['ceiling_seconds']-before,
                    ledger['deadline_utc_seconds']-time()+startup)
        if grant <= 75.:
            raise TimeoutError('Approved budget or same-day deadline has expired.')
        entry = dict(label='compact-production|same-day-'+label,
                     started_utc_seconds=time(), status='reserved', charged_seconds=grant)
        ledger['entries'].append(entry)
        ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        c.write(path, ledger, replace=True)
        watchdog = threading.Timer(max(.01, grant-(perf_counter()-begun)-1.), lambda:os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            yield dict(ledger=ledger, before=before, begun=begun, grant=grant,
                deadline=min(time()+max(0., grant-(perf_counter()-begun)-60.),
                             decision['deadline_utc_seconds']-60.))
        finally:
            entry.update(status='closed', charged_seconds=perf_counter()-begun)
            ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            c.write(path, ledger, replace=True)
            watchdog.cancel()


class Engine(c.Engine):
    """Replace only admission/watchdog accounting; use original model jobs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.same_day_search = False

    def check(self, *, search=None):
        c.require(not (self.directory/'FAILED.json').exists()
                  and not (self.directory/'SAME_DAY_FAILED.json').exists(), 'Preserved failure; no retry.')
        if time() >= self.allocation['deadline'] or (self.directory/'STOP').exists():
            raise TimeoutError('Paused before next job; keep cached states and ledger.')
        if self.same_day_search if search is None else search:
            used = self.allocation['before']+perf_counter()-self.allocation['begun']
            if used+self.validation_remaining+60.+1.5*max(self.rates['base'].values()) >= self.allocation['ledger']['ceiling_seconds']:
                raise TimeoutError('Search stopped to protect complete remaining validation.')

    def job(self, candidate, grid, hour, kind, prior, *, value=None, branch='prefinal', search=False):
        # The original job has a separate 36000-second search watchdog. Keep
        # it disabled here and supply the authorized equivalent around the
        # exact original JobStore/compute function. No numerical code changes.
        previous = self.same_day_search
        self.same_day_search = search
        timer = None
        try:
            self.check(search=search)
            if search:
                used = self.allocation['before']+perf_counter()-self.allocation['begun']
                allowance = self.allocation['ledger']['ceiling_seconds']-used-self.validation_remaining-60.
                if allowance <= 0.:
                    raise TimeoutError('No funded search remains.')
                timer = threading.Timer(allowance, lambda:os._exit(124))
                timer.daemon=True
                timer.start()
            return super().job(candidate, grid, hour, kind, prior, value=value, branch=branch, search=False)
        finally:
            if timer is not None: timer.cancel()
            self.same_day_search = previous


def execution_manifest(engine, decision, source_commit):
    c.require(re.fullmatch('[0-9a-f]{40}',source_commit) is not None, 'Declare immutable controller source.')
    old = c.load(engine.directory/'budgeted_execution.json')
    c.require(old['identity'] == decision['original_execution_identity']
              and old['numerical_manifest_identity'] == engine.manifest['identity'], 'Mixed execution origins.')
    for path, expected in old['source_sha256'].items():
        c.require(c.sha(ROOT/path) == expected, 'Frozen predecessor controller changed.')
    outer = dict(schema='f4-h2-authorized-execution-v2', source_commit=source_commit,
        numerical_manifest_identity=engine.manifest['identity'], predecessor_execution_identity=old['identity'],
        decision_identity=decision['identity'], source_sha256={p:c.sha(ROOT/p) for p in SOURCES},
        rates=decision['protected_rates'], ceiling_seconds=decision['ceiling_seconds'],
        deadline_utc_seconds=decision['deadline_utc_seconds'], max_invocation_seconds=600.,
        full49_scenario_admitted=False, convergence_guaranteed=False, scientific_acceptance=False)
    outer['identity'] = c.digest(outer)
    path=engine.directory/'same_day_execution.json'
    if path.exists():
        c.require(c.load(path)==outer, 'Authorized resume requires identical controller/evidence.')
    else:
        c.write(path,outer)
        with zipfile.ZipFile(engine.directory/'same_day_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in SOURCES: z.writestr(p,(ROOT/p).read_bytes())
    return outer


def prefinal(engine, outer, decision):
    from root.f4.measure_h2_compact_rates import work_graph
    c.load_probe(engine)
    engine.rates = decision['protected_rates']
    graph = work_graph(engine.rates,48,24,engine.allocation['before'])
    engine.validation_remaining = graph['protected_validation_seconds']
    families = {1:c.validated_family(engine,c.LINEAR)}
    c.require(abs(engine.validation_remaining-decision['remaining_validation_seconds']) <= 1e-7,
              'Raw-reviewed validation graph changed.')
    engine.allocation['ledger']['validation_reserve_seconds'] = max(
        engine.allocation['ledger']['validation_reserve_seconds'],engine.validation_remaining)
    for pieces in (2,4):
        fit=c.optimizer_family(engine,pieces)
        families[pieces]=c.validated_family(engine,fit['candidate'])
    selected=c.select_families(families)
    selected.update(schema='f4-h2-frozen-selection-v1',identity=engine.manifest['identity'],
        execution_identity=outer['identity'],predecessor_execution_identity=outer['predecessor_execution_identity'],
        execution_manifest_sha256=c.sha(engine.directory/'same_day_execution.json'),
        curves={str(p):families[p]['candidate'] for p in families},
        metrics={str(p):{g:c.selection_metrics(families[p]['streams'][g],engine.data['observations'])
                       for g in ('base','fine')} for p in families},
        comparisons={str(p):families[p]['comparison'] for p in families},
        cutoff=engine.data['packet']['policy']['split']['holdout_start'],
        final_observations_decoded=0,scientific_acceptance=False,blind_h3_acceptance=False)
    path=engine.directory/'selection.json'
    if path.exists(): c.require(c.load(path)==selected,'Frozen model selection changed.')
    else: c.write(path,selected)
    c.forecast_week(engine,families,selected)
    return dict(status='prefinal_completed',selected_family=selected['selected_family'],
                final_observations_decoded=0,conditional_forecast_precision_passed=True,
                scientific_acceptance=False,blind_h3_acceptance=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--stage',choices=('prefinal','final'),default='prefinal')
    p.add_argument('--max-seconds',type=float,default=600.)
    args=p.parse_args()
    decision=c.load(args.decision)
    register_campaign(args.campaign,decision)
    directory=args.run.resolve()
    c.require(not (directory/'STOP').exists(),'Review and preserve the STOP marker before explicit continuation.')
    c.require(not (directory/'SAME_DAY_FAILED.json').exists(),'Preserved failure; no retry.')
    number=len(list(directory.glob('same_day_report_*.json')))+1
    path=directory/f'same_day_report_{number:03d}.json'
    engine=None
    with reservation(args.campaign,decision,args.stage,begun=BEGUN,requested=args.max_seconds) as allocation,c._RunLock(directory):
        try:
            data=c.receipt();old=c.load(directory/'manifest.json')
            manifest=c.prepare(directory,data,old['source_commit'])
            engine=Engine(directory,data,manifest,allocation)
            outer=execution_manifest(engine,decision,args.source_commit)
            if args.stage=='prefinal': result=prefinal(engine,outer,decision)
            else:
                c.require((directory/'final_observations.npz').exists(), 'Export final data after forecast freeze first.')
                engine.rates=decision['protected_rates']
                result=c.final_evaluation(engine,None)
                result=dict(status='final_completed',result_sha256=c.sha(directory/'final_evaluation.json'))
            c.require(c.source_hashes()==manifest['source_sha256'],'Frozen numerical source changed.')
            c.require({p:c.sha(ROOT/p) for p in SOURCES}==outer['source_sha256'],'Controller source changed.')
        except (Exception,KeyboardInterrupt) as exc:
            paused=isinstance(exc,(TimeoutError,KeyboardInterrupt))
            result=dict(status='paused' if paused else 'failed',reason=str(exc),traceback=traceback.format_exc(),
                        scientific_acceptance=False,blind_h3_acceptance=False,automatic_retry=False)
            if not paused: c.write(directory/'SAME_DAY_FAILED.json',result)
        finally:
            if engine is not None: engine.close()
        result.update(new_jobs=0 if engine is None else len(engine.store.events),
                      numerical_manifest_identity=c.load(directory/'manifest.json')['identity'])
        c.write(path,result)
    c.write(directory/'same_day_closed_ledger.json',c.load(args.campaign),replace=(directory/'same_day_closed_ledger.json').exists())
    print(json.dumps(dict(report=str(path),**result)),flush=True)
    if result['status']=='failed': raise SystemExit(2)


if __name__=='__main__': main()

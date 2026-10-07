"""Explicit 25/49-evaluation continuation of the preserved terminal H2 run.

Only optimizer caps and execution provenance differ. Native model jobs keep
exactly their predecessor identities. The old failure/result is evidence.
"""
from time import perf_counter, time
BEGUN=perf_counter()-.25
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from root.f4 import run_h2_same_day as s
c=s.c
np=c.np
require=c.require
load=c.load
write=c.write
curve_key=c.curve_key
penalty=c.penalty
ordered_sum=c.ordered_sum
SOURCES=('root/f4/run_h2_cap_extension.py','root/f4/h2_cap_extension_decision_20261007.json',
         'root/H2_CAP_EXTENSION_20261007.md')

def optimizer_family(engine, pieces):
    """Deterministically reconstruct COBYQA state by exact objective-call replay."""
    from scipy.optimize import minimize
    require(pieces in (2,4), 'Unknown fit family.')
    calls_path = engine.directory/f'extension_optimizer_{pieces}_calls.json'
    result_path = engine.directory/f'extension_optimizer_{pieces}_result.json'
    saved = load(calls_path) if calls_path.exists() else dict(identity=engine.manifest['identity'], calls=[])
    require(saved['identity'] == engine.manifest['identity'], 'Optimizer trace identity changed.')
    calls = saved['calls']; cursor = 0; cap = {2:24, 4:48}[pieces]
    registry_path = engine.directory/'extension_candidate_registry.json'
    registry = load(registry_path) if registry_path.exists() else dict(identity=engine.manifest['identity'], families={'2':[], '4':[]})
    require(registry['identity'] == engine.manifest['identity'], 'Candidate registry changed.')
    for family, keys in registry['families'].items():
        require(len(keys) == len(set(keys)) <= {'2':24, '4':48}[family], 'Candidate budget registry invalid.')
    require(1+sum(len(k) for k in registry['families'].values()) <= 73, 'Distinct pair ceiling exceeded.')
    if result_path.exists():
        result = load(result_path)
        require(result['identity'] == engine.manifest['identity'], 'Optimizer result identity changed.')
        require(result['converged'] is True, 'Saved optimizer did not converge; no automatic new fit.')
        return result

    def objective(logits):
        nonlocal cursor
        candidate = dict(pieces=pieces, logits=np.asarray(logits, dtype=float).tolist())
        key = curve_key(candidate)
        if cursor < len(calls):
            row = calls[cursor]
            require(row['logits'] == candidate['logits'] and row['key'] == key,
                'COBYQA call order changed during deterministic resume.')
            require(math.isfinite(row['score']) and row['penalty'] == penalty(candidate)
                and row['objective'] == -row['score']+row['penalty'], 'Cached objective algebra changed.')
            cursor += 1
            return row['objective']
        engine.check(search=True)
        keys = registry['families'][str(pieces)]
        if key != 'linear' and key not in keys:
            require(len(keys) < cap and 1+sum(len(k) for k in registry['families'].values()) < 73,
                'Declared distinct curve-pair budget exhausted.')
            keys.append(key)  # Reserve candidate before any partial model work.
            write(registry_path, registry, replace=registry_path.exists())
        rows = engine.replay(candidate, 'base', end=72, search=True)
        score = ordered_sum(r['log_evidence'] for r in rows)
        reg = penalty(candidate); value = -score+reg
        require(math.isfinite(value), 'Nonfinite objective; never substitute infinity.')
        calls.append(dict(logits=candidate['logits'], key=key, score=score, penalty=reg, objective=value))
        write(calls_path, saved, replace=calls_path.exists()); cursor += 1
        print(json.dumps(dict(stage='fit', pieces=pieces, objective_calls=cursor,
            new_pairs=len(keys), objective=value)), flush=True)
        return value

    result = minimize(objective, np.zeros(2*(pieces-1)), method='COBYQA',
        bounds=[(-4.,4.)]*(2*(pieces-1)), options=dict(maxfev=cap+1,
            initial_tr_radius=.5, final_tr_radius=.02, scale=False))
    require(cursor == len(calls), 'Optimizer stopped before consuming its preserved trace.')
    converged = bool(result.success) and int(result.status) == 0 and math.isfinite(float(result.fun))
    value = dict(identity=engine.manifest['identity'], pieces=pieces,
        candidate=dict(pieces=pieces, logits=np.asarray(result.x).tolist()),
        objective=float(result.fun), nfev=int(result.nfev), status=int(result.status),
        message=str(result.message), converged=converged,
        algorithmic_local_termination_only=True, scientific_acceptance=False)
    write(result_path, value)
    require(converged, f'COBYQA family {pieces} did not reach its trust-radius termination: {result.message}')
    return value


class Engine(s.Engine):
    def check(self, *, search=None):
        require(not (self.directory/'EXTENSION_FAILED.json').exists(), 'Terminal extension; no retry.')
        super().check(search=search)


def execution(engine, decision, source_commit):
    require(s.re.fullmatch('[0-9a-f]{40}',source_commit) is not None, 'Immutable controller commit required.')
    predecessor=load(engine.directory/'same_day_execution.json')
    require(decision['identity']==c.digest({k:v for k,v in decision.items() if k!='identity'})
            and decision['predecessor_execution_identity']==predecessor['identity']
            and decision['maxfev']=={'2':25,'4':49}
            and decision['global_pair_cap']==73, 'Unregistered method change.')
    for manifest in (predecessor,load(engine.directory/'budgeted_execution.json')):
        require(all(c.sha(ROOT/p)==h for p,h in manifest['source_sha256'].items()), 'Predecessor source changed.')
    provenance=load(engine.directory/'extension_origin.json')
    require(provenance['predecessor_manifest_identity']==engine.manifest['identity']
            and provenance['decision_identity']==decision['identity'], 'Wrong extension origin.')
    for name,h in provenance['preserved_predecessor_metadata'].items():
        require(c.sha(engine.directory/name)==h, 'Preserved terminal evidence changed.')
    outer=dict(schema='f4-h2-cap-extension-execution-v1',source_commit=source_commit,
        source_sha256={p:c.sha(ROOT/p) for p in SOURCES},decision_identity=decision['identity'],
        numerical_manifest_identity=engine.manifest['identity'],
        predecessor_execution_identity=predecessor['identity'],maxfev=decision['maxfev'],
        global_pair_cap=73,original_manifest_caps_are_predecessor_only=True,
        full73_scenario_admitted=False,scientific_acceptance=False)
    outer['identity']=c.digest(outer)
    path=engine.directory/'extension_execution.json'
    if path.exists(): require(load(path)==outer,'Controller changed during extension.')
    else:
        write(path,outer)
        with zipfile.ZipFile(engine.directory/'extension_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in SOURCES:z.writestr(p,(ROOT/p).read_bytes())
    return outer


def prefinal(engine, outer, decision):
    from root.f4.measure_h2_compact_rates import work_graph
    c.load_probe(engine)
    engine.rates=decision['protected_rates']
    graph=work_graph(engine.rates,48,24,engine.allocation['before'])
    engine.validation_remaining=graph['protected_validation_seconds']
    families={1:c.validated_family(engine,c.LINEAR)}
    require(abs(engine.validation_remaining-decision['remaining_validation_seconds'])<=1e-7,
            'Validation reserve changed.')
    engine.allocation['ledger']['validation_reserve_seconds']=max(
        engine.allocation['ledger']['validation_reserve_seconds'],engine.validation_remaining)
    for pieces in (2,4):
        fit=optimizer_family(engine,pieces)
        families[pieces]=c.validated_family(engine,fit['candidate'])
    selected=c.select_families(families)
    selected.update(schema='f4-h2-frozen-selection-v1',identity=engine.manifest['identity'],
        execution_identity=outer['identity'],execution_manifest_sha256=c.sha(engine.directory/'extension_execution.json'),
        predecessor_execution_identity=outer['predecessor_execution_identity'],
        curves={str(p):families[p]['candidate'] for p in families},
        metrics={str(p):{g:c.selection_metrics(families[p]['streams'][g],engine.data['observations'])
                        for g in ('base','fine')} for p in families},
        comparisons={str(p):families[p]['comparison'] for p in families},
        cutoff=engine.data['packet']['policy']['split']['holdout_start'],
        final_observations_decoded=0,scientific_acceptance=False,blind_h3_acceptance=False)
    path=engine.directory/'selection.json'
    if path.exists():require(load(path)==selected,'Frozen model selection changed.')
    else:write(path,selected)
    c.forecast_week(engine,families,selected)
    return dict(status='prefinal_completed',selected_family=selected['selected_family'],
                final_observations_decoded=0,scientific_acceptance=False,blind_h3_acceptance=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--stage',choices=('prefinal','final'),default='prefinal')
    args=p.parse_args()
    budget=load(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    decision=load(ROOT/'root/f4/h2_cap_extension_decision_20261007.json')
    directory=args.run.resolve()
    require(not any((directory/name).exists() for name in ('STOP','FAILED.json','SAME_DAY_FAILED.json','EXTENSION_FAILED.json')),
            'Preserved stop/failure; do not restart.')
    s.validate_campaign(load(args.campaign),budget)
    count=len(list(directory.glob('extension_report_*.json')))+1
    engine=None
    with s.reservation(args.campaign,budget,'cap-extension-'+args.stage,begun=BEGUN,requested=600.) as allocation,c._RunLock(directory):
        try:
            data=c.receipt();old=load(directory/'manifest.json')
            manifest=c.prepare(directory,data,old['source_commit'])
            engine=Engine(directory,data,manifest,allocation)
            outer=execution(engine,decision,args.source_commit)
            if args.stage=='prefinal':result=prefinal(engine,outer,decision)
            else:
                require((directory/'final_observations.npz').exists(),'Final data export required after forecast freeze.')
                engine.rates=decision['protected_rates'];c.final_evaluation(engine,None)
                result=dict(status='final_completed',result_sha256=c.sha(directory/'final_evaluation.json'))
            require(c.source_hashes()==manifest['source_sha256'],'Frozen numerical sources changed.')
            require({p:c.sha(ROOT/p) for p in SOURCES}==outer['source_sha256'],'Controller sources changed.')
        except (Exception,KeyboardInterrupt) as exc:
            paused=isinstance(exc,(TimeoutError,KeyboardInterrupt))
            result=dict(status='paused' if paused else 'failed',reason=str(exc),traceback=traceback.format_exc(),
                        scientific_acceptance=False,blind_h3_acceptance=False,automatic_retry=False)
            if not paused:write(directory/'EXTENSION_FAILED.json',result)
        finally:
            if engine is not None:engine.close()
        result.update(new_jobs=0 if engine is None else len(engine.store.events),
                      numerical_manifest_identity=load(directory/'manifest.json')['identity'])
        write(directory/f'extension_report_{count:03d}.json',result)
    write(directory/'extension_closed_ledger.json',load(args.campaign),replace=(directory/'extension_closed_ledger.json').exists())
    print(json.dumps(result),flush=True)
    if result['status']=='failed':raise SystemExit(2)


if __name__=='__main__':main()

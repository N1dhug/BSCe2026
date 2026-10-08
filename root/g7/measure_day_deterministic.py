"""One native24h deterministic objective; own archived M254 MAP origin.

Standalone one-origin timing only: not a whole-graph compatible rate packet.
"""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import scipy
from day_control import deterministic_job, check_sources
from receipt_recovery import guarded_store_class, audit_store
from src.method.bm_1_g7_jobs import NodeJobStore, digest
from src.method.bm_1_g8 import G8BankOrigin


def main(output):
    output=Path(output);reg=json.loads((output/'registration.json').read_bytes());source=Path(reg['source_root']);manifest=reg['source_sha256'];check_sources(source,manifest)
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==reg['runner_sha256']
    raw=Path(reg['origin_path']).read_bytes();origin=G8BankOrigin.from_verified_bytes(raw,reg['origin_sha256'])
    assert origin.time==12 and origin.backend=='mixture' and origin.conditional_mass.shape==(5,257)
    contract=dict(schema='g7-standalone-deterministic-rate-v1',origin=origin.to_record(),source_sha256=manifest,prices=reg['prices'],horizon=[12,36],solver_max_step=1/96,objective_max_step=1/96,versions=dict(numpy=np.__version__,scipy=scipy.__version__),whole_graph_compatible=False)
    control=SimpleNamespace(evaluator=SimpleNamespace(origin=origin),contract_id=digest(contract))
    Guard=guarded_store_class(NodeJobStore,native_source_path=source/'src/method/bm_1_g7_jobs.py',native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    store=Guard(output/'campaign',prior_debit=0,ceiling=6,max_new_seconds=6)
    start=perf_counter();value=deterministic_job(control,reg['prices'],store,seconds=4,deadline=start+4);elapsed=perf_counter()-start
    receipts=list((store.root/'nodes').glob('*.json'));assert len(receipts)==1
    with store.lock():audit_store(store,full=True)
    row=json.loads(receipts[0].read_bytes());payload=row['payload'];assert np.isfinite(value) and abs(payload['J']-value)<1e-12 and np.isfinite(payload['state']).all() and np.isfinite(payload['demand']).all()
    assert row['job']['kind']=='g7-day-deterministic-v1' and row['job']['selected_node']==origin.map_origin().selected_node
    result=dict(schema='g7-standalone-deterministic-rate-result-v1',complete=True,raw_reviewed=True,seconds=elapsed,horizon_hours=24,rate_unit='deterministic objective',prices=reg['prices'],J=value,selected_node=row['job']['selected_node'],initial_state=payload['initial_state'],receipt_sha256=hashlib.sha256(receipts[0].read_bytes()).hexdigest(),payload_sha256=row['payload_sha256'],contract_sha256=control.contract_id,native_ode_solves=1,whole_graph_compatible=False,scientific_accept=False,note='Native exact day deterministic_job including its job setup, ODE and receipt write. One-origin archived timing; cannot populate integrated16-rate admission context before all four origin identities and same frozen context are registered.')
    (output/'rate_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])

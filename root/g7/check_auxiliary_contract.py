"""One cheap ownership/deadline check; no solver import, transport or old test replay."""
import hashlib
import json
from pathlib import Path
import tempfile
import time

from restore_auxiliary import canonical, sha, verified_receipt
from run_auxiliary_measurement import job_deadline


def check():
    started=time.perf_counter();checks=[]
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'node.json'
        job=dict(contract='original-owned-contract',index=2,rho=.25,prices=[.5,.5])
        payload=dict(index=2,prices=[.5,.5],mass=[[.4,.6]])
        receipt=dict(schema='g7-owned-node-v1',job=job,payload=payload,
                     payload_sha256=hashlib.sha256(canonical(payload)).hexdigest())
        path.write_bytes(canonical(receipt));original=path.read_bytes()
        assert verified_receipt(path,sha(path),job)==payload
        assert path.read_bytes()==original;checks.append('valid read-only receipt retains exact bytes')
        for name,expected in [('wrong price',dict(job,prices=[.4,.5])),('wrong origin',dict(job,contract='foreign'))]:
            try:verified_receipt(path,sha(path),expected)
            except ValueError:checks.append(name+' rejected')
            else:raise AssertionError(name)
        receipt['payload']['mass'][0][0]=.3;path.write_bytes(canonical(receipt))
        try:verified_receipt(path,sha(path),job)
        except ValueError:checks.append('rehash of altered payload cannot evade nested identity')
        else:raise AssertionError('payload accepted')
    try:
        with job_deadline(time.perf_counter()-1.,RuntimeError):pass
    except RuntimeError:checks.append('expired job deadline rejected')
    else:raise AssertionError('deadline accepted')
    # Alarm handler/timer must have been restored; no model work is scheduled.
    return dict(schema='g7-auxiliary-contract-check-v1',passed=True,checks=checks,
                native_solver_imports=0,new_forwards=0,seconds=time.perf_counter()-started)


if __name__=='__main__':print(json.dumps(check(),indent=2))

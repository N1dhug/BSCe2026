"""One process-lifecycle control; no model, data, JIT or optimizer work."""
from time import perf_counter, time
BEGUN = perf_counter() - .25
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from root.f4 import run_h2_same_day as s
c = s.c

def check():
    program = '''import subprocess,sys,time,json
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(4)'])
try:
    time.sleep(4)
except KeyboardInterrupt:
    time.sleep(.05)
    alive=child.poll() is None
    child.terminate();child.wait(timeout=1)
    print(json.dumps(dict(parent_received_interrupt=True,child_not_signalled=alive,child_joined=True)))
    sys.exit(0 if alive else 2)
else:
    child.terminate();child.wait(timeout=1);sys.exit(3)
'''
    command = ['timeout', '--foreground', '--preserve-status', '--signal=INT',
               '--kill-after=1s', '.3s', sys.executable, '-c', program]
    result = subprocess.run(command, capture_output=True, text=True, timeout=3)
    c.require(result.returncode == 0, 'Soft-stop control failed: '+result.stderr)
    evidence = json.loads(result.stdout)
    c.require(all(evidence.values()), 'Parent/child signal scope or cleanup failed.')
    return dict(status='passed', scope='parent-only SIGINT, caught interrupt, live child cleanup and preserved exit status',
                evidence=evidence, native_propagations=0, model_jobs=0,
                utility_version=subprocess.check_output(['timeout','--version'],text=True).splitlines()[0])

if __name__ == '__main__':
    campaign=c.OUT/'f4_h2_same_day_campaign_20261007.json'
    decision=c.load(c.ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    with c._RunLock(campaign.with_suffix('.lock')):
        ledger=s.validate_campaign(c.load(campaign),decision)
        native=c.ordered_sum(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))
        c.require(900.-native >= 5., 'Insufficient lifecycle-control allowance.')
        entry=dict(label='native-development|soft-stop-lifecycle-control',status='reserved',started_utc_seconds=time(),charged_seconds=5.)
        ledger['entries'].append(entry)
        ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        c.write(campaign,ledger,replace=True)
        timer=s.threading.Timer(max(.01,4.-(perf_counter()-BEGUN)),lambda:s.os._exit(124));timer.daemon=True;timer.start()
        try:
            result=check();result.update(source_sha256=c.sha(Path(__file__)),elapsed_seconds=perf_counter()-BEGUN)
            c.write(c.OUT/'f4_h2_soft_stop_control_20261007.json',result)
            print(json.dumps(result))
        finally:
            entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            c.write(campaign,ledger,replace=True);timer.cancel()

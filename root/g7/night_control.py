"""Registered G7-only study. Resume completed units, never retry interrupted ones.

Run from repository root with PYTHONPATH=.; --output is a persistent run folder.
Atomic unit checkpoints and a write-ahead time reservation survive process loss.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import scipy

from src.method.bm_1_g7_matched import MatchedDeterministicControl
from src.method.bm_1_g7_energy import CachedSearch, affine_box_range, box_quadratic, energy_gap, precision


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, default=serial, allow_nan=False, separators=(',', ':'))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def restore(f):
    return {k: np.asarray(v) if k in ('time_hours','state','midpoints','demand','residual','target','block_prices') else v for k,v in f.items()}


def run(packet_path, protocol_path, out):
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads(Path(protocol_path).read_text())
    packet_bytes = Path(packet_path).read_bytes()
    if hashlib.sha256(packet_bytes).hexdigest() != protocol['packet_sha256']:
        raise ValueError('Immutable packet changed.')
    for path, digest in protocol['source_sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Registered source changed: '+path)
    packet = json.loads(packet_bytes)
    control = MatchedDeterministicControl(packet)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report_path = out/'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['protocol'] != protocol:
            raise ValueError('Resume protocol differs.')
        # An interrupted reservation remains charged at its full cap.
        for entry in report['ledger']:
            if entry['status'] == 'running':
                entry['status'] = 'interrupted-no-retry'
        write(report_path, report)
    else:
        report = dict(schema='g7-night-control-v1', protocol=protocol,
                      versions=dict(numpy=np.__version__,scipy=scipy.__version__),
                      ledger=[],units={}, status='running',
                      stochastic_accept=False, historical_ranking=False, causal_accept=False)
        write(report_path, report)
    started = time.perf_counter()
    def alarm(*_): raise TimeoutError('Unit time reservation exhausted; no automatic retry.')
    old_signal = signal.signal(signal.SIGALRM, alarm)

    def unit(key, cap, fn):
        if key in report['units']:
            return json.loads((out/report['units'][key]).read_text())
        if any(e['key']==key for e in report['ledger']):
            return None
        charged = sum(e['charged_seconds'] for e in report['ledger'])
        if charged+cap > protocol['run_cap_seconds']:
            report['stop_reason']='Insufficient remaining registered reservation.'
            write(report_path,report); return None
        entry=dict(key=key,status='running',reserved_seconds=cap,charged_seconds=cap)
        report['ledger'].append(entry); write(report_path,report)
        t=time.perf_counter(); signal.setitimer(signal.ITIMER_REAL,cap)
        try:
            value=fn()
            write(out/(key+'.json'),value)
            report['units'][key]=key+'.json'
            entry['status']='complete'
        except Exception as exc:
            value=None; entry['status']='failed-no-retry'; entry['error']=repr(exc)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            entry['charged_seconds']=time.perf_counter()-t
            write(report_path,report)
            print(key,entry['status'],round(entry['charged_seconds'],3),'s',flush=True)
        return value

    try:
        for name, model in control.models.items():
            selected={}
            if model['family']=='BM1':
                g=model['g']; desired=2*model['x_origin']-1
                eq=float(np.interp(desired,list(reversed(g['values'])),list(reversed(g['knots']))))
                d=min(.15,eq/2,(1-eq)/2)
                starts=[np.full(4,eq),eq+d*np.array([1,-1,-1,1])]
                for neutral in (False,True):
                    mode='neutral' if neutral else 'free'
                    fits=[]
                    for i,start in enumerate(starts):
                        def search(start=start,neutral=neutral):
                            searcher=CachedSearch(control,name,60,400,time.perf_counter()+29)
                            result,forward=searcher.search(start,neutral)
                            return dict(search=result,forward=forward,start=start)
                        value=unit(name+'_'+mode+'_start'+str(i),30,search)
                        if value is not None and value['search']['success'] and (not neutral or abs(value['search']['energy_gap'])<=1e-8):
                            fits.append(value)
                    if fits:
                        selected[mode]=min(fits,key=lambda v:v['search']['objective'])
            else:
                def build_affine():
                    base=control.evaluate(name,np.zeros(4),60)
                    probes=[control.evaluate(name,np.eye(4)[j],60) for j in range(4)]
                    H=np.column_stack([p['residual']-base['residual'] for p in probes])
                    ds=np.column_stack([p['demand']-base['demand'] for p in probes])
                    ss=np.stack([p['state']-base['state'] for p in probes],axis=-1)
                    lo,hi=affine_box_range(base['demand'],ds)
                    slo,shi=affine_box_range(base['state'],ss)
                    if lo.min()<0 or hi.max()>1 or slo[:,0].min() < -1 or shi[:,0].max()>1:
                        raise ValueError('WM full box is not feasible; do not omit path constraints.')
                    a=np.array([energy_gap(p)-energy_gap(base) for p in probes])
                    return dict(H=H,c=base['residual'],a=a,b=-energy_gap(base),
                                demand_box_extrema=[lo.min(),hi.max()],
                                F_box_extrema=[slo[:,0].min(),shi[:,0].max()],
                                base=base,probes=probes,actual_forwards=5)
                aff=unit(name+'_affine',10,build_affine)
                if aff is not None:
                    for neutral in (False,True):
                        mode='neutral' if neutral else 'free'
                        def solve(neutral=neutral):
                            qp=box_quadratic(aff['H'],aff['c'],(np.array(aff['a']),aff['b']) if neutral else None)
                            f=control.evaluate(name,qp['prices'],60)
                            err=float(np.max(np.abs(f['residual']-(np.array(aff['c'])+np.array(aff['H'])@qp['prices']))))
                            if err>1e-12 or qp['kkt_residual']>1e-8:
                                raise ValueError('Independent affine/KKT gate failed.')
                            return dict(search=dict(qp,success=True,actual_forwards=1,affine_error=err),forward=f)
                        v=unit(name+'_'+mode+'_qp',10,solve)
                        if v is not None: selected[mode]=v
            for mode,coarse in selected.items():
                def validate(coarse=coarse,mode=mode):
                    c=restore(coarse['forward'])
                    f=control.evaluate(name,c['block_prices'],120)
                    check=precision(c,f,mode=='neutral')
                    dt=np.diff(f['time_hours']); delta=f['state'][-1]-f['state'][0]
                    balance=model['parameters'][0]*delta[0] if model['family']=='BM1' else 24*delta[0]
                    euler_correction=0. if model['family']=='BM1' else .5*dt[0]*delta[1]
                    return dict(forward=f,precision=check,actual_forwards=1,
                                terminal_change=delta,energy_gap=energy_gap(f),
                                continuous_terminal_balance=balance,
                                euler_midpoint_correction=euler_correction,
                                balance_residual=energy_gap(f)-balance-euler_correction,
                                selected_search=coarse['search'])
                unit(name+'_'+mode+'_fine',5,validate)
        final_keys=[n+'_'+m+'_fine' for n in control.models for m in ('free','neutral')]
        final=[json.loads((out/report['units'][k]).read_text()) for k in final_keys if k in report['units']]
        report['all_six_precision_pass']=len(final)==6 and all(v['precision']['passed'] for v in final)
        report['status']='complete' if report['all_six_precision_pass'] else 'partial-or-rejected'
        report['new_compute_seconds']=sum(e['charged_seconds'] for e in report['ledger'])
        report['current_invocation_body_seconds']=time.perf_counter()-started
        report['g_debit_before_overhead']=protocol['prior_g_debit_seconds']+report['new_compute_seconds']
        write(report_path,report)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_signal)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.packet,a.protocol,a.output)
    print(json.dumps({k:r[k] for k in ('status','new_compute_seconds','all_six_precision_pass')}))
    raise SystemExit(0 if r['status']=='complete' else 2)

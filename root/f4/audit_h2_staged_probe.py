"""Read-only audit of the registered nonlinear prefix; no native/FV replay."""
from time import perf_counter
BEGUN=perf_counter()-.25
import argparse
import io
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'


def audit(directory):
    from src.exe.run_model import bm_1_h2_compact as c
    from src.exe.run_model.bm_1_history import _beta_mass
    from src.method.bm_1_interval_forecast import compare_interval_moments, compare_interval_distributions
    from root.f4.measure_h2_compact_rates import compare_states, work_graph
    import numpy as np
    manifest=c.load(directory/'manifest.json'); report=c.load(directory/'nonlinear_probe.json')
    c.require(manifest['identity']==c.digest({k:v for k,v in manifest.items() if k!='identity'}), 'Manifest digest changed.')
    c.require(report['identity']==manifest['identity'] and report['candidate']==c.PROBE and report['complete'], 'Wrong probe.')
    source=c.source_hashes()
    c.require(source==manifest['source_sha256'], 'Audit source differs from executed source.')
    with zipfile.ZipFile(directory/'source.zip') as archive:
        c.require(set(archive.namelist())==set(source),'Source archive members differ.')
        for p,h in source.items():
            c.require(c.hashlib.sha256(archive.read(p)).hexdigest()==h,'Source archive bytes differ: '+p)
    data=c.receipt(); obs=c._read_array_checkpoint((directory/'observations.npz').read_bytes())
    c.require(c.fingerprint(obs)==c.fingerprint(data['observations']), 'Frozen observations differ.')
    c.require(c.fingerprint(c._read_array_checkpoint((directory/'inputs.npz').read_bytes()))==c.fingerprint(data['inputs']), 'Frozen inputs differ.')
    c.require(c.load(directory/'packet.json')==data['packet'], 'Frozen packet differs.')
    jobs={}; timings={}
    for name,h in report['job_hashes'].items():
        path=directory/'jobs'/name
        c.require(c.sha(path)==h,'Raw job hash differs.')
        saved=c._read_array_checkpoint(path.read_bytes()); token=saved['token']
        c.require(saved['checksum']==c.fingerprint({k:v for k,v in saved.items() if k!='checksum'}), 'Raw payload checksum differs.')
        c.require(name==c.digest(token)+'.npz' and token['identity']==manifest['identity']
            and token['candidate']==c.curve_key(c.PROBE) and token['branch']=='prefinal','Wrong job identity.')
        key=(token['grid'],token['hour'],token['kind'])
        c.require(key not in jobs and saved['elapsed_seconds']>0.,'Duplicate or invalid timing.')
        jobs[key]=saved; timings[key]=saved['elapsed_seconds']
    expected={(g,h,'prediction' if h<24 else 'observed') for g in ('base','fine') for h in range(27)}
    expected|={(g,h,'moments') for g in ('base','fine') for h in range(24,27)}
    expected|={(g,h,'cdf') for g in ('base','fine') for h in (25,26)}
    c.require(set(jobs)==expected,'Incomplete registered 64-job prefix.')
    rows={g:[] for g in ('base','fine')}; maxima=dict(mass=0.,conditional=0.,overlap=0.)
    for grid in rows:
        initial=jobs[(grid,0,'prediction')]['prior_log_state']
        p=data['packet']['prior']; expected_mass=_beta_mass(np.linspace(0.,1.,len(initial)+1),alpha=p['alpha'],beta=p['beta'])
        c.require(np.max(np.abs(np.exp(initial)-expected_mass))<=1e-13,'Initial Beta cell mass differs.')
        logs=initial
        for hour in range(27):
            seen=hour>=24; saved=jobs[(grid,hour,'observed' if seen else 'prediction')]
            c.same_state(saved['prior_log_state'],logs); result=saved['result']; c.state_check(result['log_state'])
            maxima['mass']=max(maxima['mass'],result['maximum_abs_mass_error'])
            moments=None
            if seen:
                c.require(saved['token']['value']==float(obs['values'][hour-24]),'Wrong observed value.')
                c.same_state(result['log_posterior_mass'],result['log_state'])
                maxima['conditional']=max(maxima['conditional'],result['maximum_relative_conditional_guard'])
                maxima['overlap']=max(maxima['overlap'],result['maximum_overlap_log_difference'])
                moment=jobs[(grid,hour,'moments')]; c.same_state(moment['prior_log_state'],logs)
                moments=moment['result']; c.state_check(moments['log_state'])
                weights=np.exp(moments['log_state']); m=moments['conditional_mean']; v=moments['conditional_variance']
                mean=float(weights@m); variance=float(weights@(v+(m-mean)**2))
                c.require(abs(mean-moments['interval_mean'])<=1e-13 and abs(variance-moments['interval_variance'])<=1e-13,'Moment algebra differs.')
                if hour in (25,26):
                    cf=jobs[(grid,hour,'cdf')]; c.same_state(cf['prior_log_state'],logs)
                    law=c.decode_distribution(cf['result']['distribution']); law.audit()
                    for q in cf['result']['distribution']['quantiles']:
                        c.require(law.quantile(q['probability'])==q,'Saved CF quantile reconstruction differs.')
            logs=result['log_state']
            rows[grid].append(dict(hour=hour,log_state=logs,observed=seen,
                log_evidence=result['log_evidence'] if seen else 0.,moments=moments))
    c.require(maxima['mass']<=1e-10 and maxima['conditional']<=1e-8 and maxima['overlap']<=1e-7,'Original guard failed.')
    comparison=c.compare_streams(rows['base'],rows['fine'],{'prefix':(24,27)})
    c.require(comparison==report['prefix_comparison'],'Reported prefix comparisons differ.')
    distributions=[]
    for hour in (25,26):
        distributions.append(compare_interval_distributions(
            c.decode_distribution(jobs[('base',hour,'cdf')]['result']['distribution']),
            c.decode_distribution(jobs[('fine',hour,'cdf')]['result']['distribution'])))
    rates={}
    for grid in rows:
        rates[grid]={}
        for kind in ('prediction','observed','moments','cdf'):
            ordered=[v for (g,h,k),v in sorted(timings.items()) if g==grid and k==kind]
            warm=ordered if kind=='cdf' else ordered[1:]
            rates[grid][kind]=max(data['rates'][grid][kind],max(warm))
    c.require(rates==report['rates'],'Reported rates differ.')
    return dict(status='passed',schema='f4-h2-staged-probe-raw-audit-v1',source_commit=manifest['source_commit'],
        source_files=len(source),raw_jobs=len(jobs),manifest_identity=manifest['identity'],
        comparison=comparison,guards=maxima,distributions=distributions,rates=rates,
        native_propagations=0,full_window_precision=False,scientific_acceptance=False,blind_h3_acceptance=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from src.exe.run_model import bm_1_h2_compact as c
    with c.reservation(c.OUT/'f4_h2_remaining_budget.json',90.,'staged-probe-raw-audit',native=True,begun=BEGUN):
        result=audit(args.directory)
        c.write(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

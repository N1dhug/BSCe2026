"""Read saved F rate/precision evidence; never rerun uploaded code or kernels."""
from time import perf_counter
BEGUN = perf_counter()-.25
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'files/output/bm_1/runs/estimation'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ordered_sum(values):
    total = 0.
    for value in values:
        total += value
    return total


def archive_members(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        require(len(names)==len(set(names)), 'Duplicate archive members.')
        require(all(not n.startswith('/') and '..' not in n.split('/') for n in names), 'Unsafe archive path.')
        return {n:z.read(n) for n in names}


def audit(path, evidence_path, known_ledger_path):
    import numpy as np
    import pandas as pd
    from scipy.special import logsumexp, betainc
    from src.tools.bm_1_reports import _read_array_checkpoint
    from src.params.bm_1 import conservative_mass_transfer
    from src.method.bm_1_interval_forecast import (IntervalCDFPlan, IntervalDistribution,
        interval_cdf_plan, compare_interval_distributions)
    raw = path.read_bytes(); files = archive_members(raw)
    manifest, report, ledger, graph = [json.loads(files[n]) for n in
        ('manifest.json', 'report.json', 'ledger.json', 'work_graph.json')]
    evidence = json.loads(evidence_path.read_text())
    require(manifest['source_commit']==evidence['source_commit'], 'Wrong declared Git source.')
    require(digest({k:v for k,v in manifest.items() if k!='identity'})==manifest['identity'], 'Manifest digest differs.')
    sources = archive_members(files['source.zip']); hashes = manifest['source_sha256']
    require(set(sources)==set(hashes)==set(evidence['source_git_blobs']) and len(sources)==64, 'Incomplete source closure.')
    matches = {}
    for name, content in sources.items():
        require(hashlib.sha256(content).hexdigest()==hashes[name], 'Original source hash changed: '+name)
        choices = {'original_bytes':content, 'exact_checkout_lf':content.replace(b'\r\n',b'\n')}
        match = [label for label,b in choices.items() if hashlib.sha1(
            b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==evidence['source_git_blobs'][name]]
        require(bool(match), 'Source differs from pinned Git blob: '+name)
        matches[name] = match[0]
    ready = json.loads(sources['root/f4/h2_compact_data_ready_20261007.json'])
    packet = json.loads(files['packet.json'])
    require(digest({k:v for k,v in packet.items() if k!='identity'})==packet['identity']
        ==ready['packet_identity']==manifest['packet_identity'], 'Packet identity changed.')
    require(packet==json.loads((ROOT/ready['packet_path']).read_text()), 'Packet differs from frozen data receipt.')
    require(packet['policy']==json.loads(sources['root/f4/h2_compact_data_policy_20261006.json']), 'Source/data policy differs.')
    for name,h in manifest['packet_input_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h, 'Frozen input differs: '+name)
    require(manifest['packet_input_sha256'][ready['audit_path']]==ready['audit_sha256'], 'Audit export hash still changed.')
    require(manifest['input_workbook_sha256']==packet['input_sha256']['input'], 'Demand provenance differs.')
    require(report['status']=='completed' and report['prefix_precision_passed'] is True
        and manifest['candidate']==dict(id='linear',pieces=2,logits=[0.,0.])
        and manifest['no_optimizer'] is True and manifest['no_final_Y'] is True, 'Wrong pilot/acceptance scope.')
    env = manifest['environment']
    require(env['workers']==8 and env['blas_threads']==env['numba_threads']==1
        and env['numba']=='0.62.1' and env['llvmlite']=='0.45.1' and env['fastmath'] is False, 'Execution policy changed.')
    for flag in ('historical_fit_ready','numerical_likelihood_ready','runtime_admitted',
        'blind_h3_acceptance','scientific_acceptance','full_window_precision_passed','nonlinear_rates_measured'):
        require(report[flag] is False, 'Unsupported acceptance: '+flag)
    observations = _read_array_checkpoint(files['observations.npz'])
    require(hashlib.sha256(files['observations.npz']).hexdigest()==manifest['observation_sha256'], 'Observation hash changed.')
    times = pd.date_range(packet['policy']['split']['train_start'],packet['policy']['split']['holdout_start'],freq='h',inclusive='left')
    require(len(times)==72 and observations['times']==[t.isoformat() for t in times], 'Observation chronology differs.')
    seen, values, group = [observations[k] for k in ('observed','values','group_kW')]
    require(seen.dtype.kind=='b' and seen.shape==values.shape==group.shape==(72,)
        and np.array_equal(np.isfinite(values),seen) and np.array_equal(np.isfinite(group),seen), 'Observation missing policy differs.')
    np.testing.assert_array_equal(values[seen],group[seen]/packet['frozen_scales']['power_kW'])
    selected = np.flatnonzero(seen[:48])[:3].tolist()
    require(selected==observations['selected_train_indices'] and observations['pilot_end_hour']==24+selected[2]+1
        and observations['train_complete']==int(seen[:48].sum()) and observations['selection_complete']==int(seen[48:].sum())
        and all(observations[k]==0 for k in ('final_values_decoded','prior_values_decoded','imputed_values')), 'Pilot boundary/counts differ.')
    n = packet['numerics']; timings = json.loads(files['timings.json'])
    jobs = {row['checkpoint'].replace('\\','/'):_read_array_checkpoint(
        files[row['checkpoint'].replace('\\','/')]) for row in timings}
    require(set(files)==set(jobs)|{'comparisons.json','manifest.json','observations.npz',
        'packet.json','report.json','source.zip','timings.json','ledger.json','work_graph.json'}, 'Unexpected receipt members.')
    require(len(jobs)==len(timings)==64, 'Wrong completed job count.')
    state, cumulative = {}, dict(base=0.,fine=0.)
    def normalized(logs, size):
        require(logs.shape==(size,) and not np.isnan(logs).any() and not np.isposinf(logs).any()
            and abs(float(logsumexp(logs)))<=1e-10, 'Full state shape/normalization failed.')
    for grid in ('base','fine'):
        size=n[grid+'_cells'];mass=np.diff(betainc(20.,20.,np.linspace(0.,1.,size+1)))
        with np.errstate(divide='ignore'):state[grid]=np.log(mass)
    comparisons=[]; first=set(); guard_max=dict(mass=0.,conditional=0.,overlap=0.)
    cursor=0
    for hour in range(observations['pilot_end_hour']):
        moments={};laws={}
        for grid in ('base','fine'):
            size=n[grid+'_cells'];observed=hour>=24 and seen[hour-24]
            kinds=(['moments']+(['cdf'] if hour-24 in selected[1:] else [])+['observed']) if observed else ['prediction']
            for kind in kinds:
                row=timings[cursor];name=f'jobs/{cursor:04d}.npz';cursor+=1;d=jobs[name]
                require(row['checkpoint'].replace('\\','/')==name and hashlib.sha256(files[name]).hexdigest()==row['checkpoint_sha256'], 'Raw checkpoint hash differs.')
                require(d['manifest_identity']==manifest['identity'] and d['grid']==row['grid']==grid
                    and d['rate_type']==row['rate_type']==kind and d['hour']==row['hour']==hour, 'Job order/identity differs.')
                require(row['cold']==((grid,kind) not in first and kind!='cdf')
                    and math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0., 'Invalid timing/cold accounting.')
                first.add((grid,kind));normalized(d['prior_log_state'],size)
                if hour==0:
                    # Beta CDF implementations differ slightly between the PC
                    # SciPy version and the review environment. Verify cell
                    # probabilities, then retain the original full log state.
                    # All state-continuation and pilot precision gates below
                    # remain exact or use their existing declared tolerances.
                    require(np.max(np.abs(np.exp(d['prior_log_state'])-np.exp(state[grid])))<=1e-13,
                        'Initial cell probabilities differ from declared Beta(20,20).')
                else:
                    np.testing.assert_array_equal(d['prior_log_state'],state[grid])
                if kind=='cdf':
                    f=d['forecast'];normalized(f['log_state'],size)
                    require(f['identity']==manifest['identity'] and f['hour']==hour, 'Forecast identity differs.')
                    finite=np.isfinite(moments[grid]['log_state'])
                    require(np.array_equal(finite,np.isfinite(f['log_state'])) and np.max(np.abs(
                        f['log_state'][finite]-moments[grid]['log_state'][finite]))<=1e-7,'Saved moment/CF states disagree.')
                    distribution=f['distribution'];plan=IntervalCDFPlan(**distribution['plan'])
                    require(plan==interval_cdf_plan(packet['physical']['sigma_y']), 'CDF analytic policy changed.')
                    cf=distribution['characteristic_real']+1j*distribution['characteristic_imag']
                    require(cf.shape==(plan.frequencies+1,) and np.isfinite(cf).all()
                        and abs(cf[0]-1.)<=1e-10 and np.all(np.abs(cf)<=1.+1e-10),'Saved characteristic guard failed.')
                    law=IntervalDistribution(plan,cf);laws[grid]=law
                    quantiles=[law.quantile(q) for q in (.05,.5,.95)]
                    require(quantiles==distribution['quantiles'], 'Saved CDF quantiles do not reconstruct.')
                    law.audit([q[k] for q in quantiles for k in ('left','right')])
                else:
                    logstate=d['posterior_log_state'] if kind=='observed' else d['log_state']
                    normalized(logstate,size)
                    require(0.<=d['maximum_abs_mass_error']<=1e-10,'Saved neutral mass guard failed.')
                    guard_max['mass']=max(guard_max['mass'],d['maximum_abs_mass_error'])
                    if kind=='observed':
                        np.testing.assert_array_equal(logstate,d['log_posterior_mass'])
                        require(math.isfinite(d['log_evidence']) and 0.<=d['maximum_relative_conditional_guard']<=1e-8
                            and 0.<=d['maximum_overlap_log_difference']<=1e-7,'Saved likelihood guard failed.')
                        guard_max['conditional']=max(guard_max['conditional'],d['maximum_relative_conditional_guard'])
                        guard_max['overlap']=max(guard_max['overlap'],d['maximum_overlap_log_difference'])
                        cumulative[grid]+=d['log_evidence'];require(cumulative[grid]==d['cumulative_log_score'],'Score prefix does not reconstruct.')
                    if kind=='moments':
                        cm,cv=d['conditional_mean'],d['conditional_variance'];p=np.exp(logstate)
                        require(cm.shape==cv.shape==logstate.shape and np.isfinite(cm).all() and np.isfinite(cv).all()
                            and np.all(cv>=0.),'Saved conditional moments invalid.')
                        mean=float(p@cm);variance=float(p@(cv+(cm-mean)**2))
                        require(abs(mean-d['interval_mean'])<=1e-12 and abs(variance-d['interval_variance'])<=1e-12
                            and abs(d['observation_variance']-variance-packet['physical']['sigma_y']**2)<=1e-12
                            and abs(d['observation_mean']-mean)<=1e-12,'Saved marginal moments do not reconstruct.')
                        moments[grid]=d
                    else:state[grid]=logstate
        b,f=state['base'],state['fine'];bp,fp=np.exp(b),np.exp(f)
        be,fe=np.linspace(0.,1.,len(b)+1),np.linspace(0.,1.,len(f)+1)
        comparison=dict(hour=hour,state_mean_difference=abs(float(bp@((be[:-1]+be[1:])/2.))-float(fp@((fe[:-1]+fe[1:])/2.))),
            conservative_state_L1_difference=float(np.abs(bp-conservative_mass_transfer(fe,fp,be)).sum()),
            pilot_total_log_score_difference=abs(cumulative['base']-cumulative['fine']))
        require(comparison['state_mean_difference']<=.002 and comparison['conservative_state_L1_difference']<=.05
            and comparison['pilot_total_log_score_difference']<=.05,'Raw state/total score precision failed.')
        saved=report['comparisons'][hour]
        for k,v in comparison.items():require(abs(v-saved[k])<=1e-12,'Saved comparison differs: '+k)
        if moments:
            for k in ('interval_mean','interval_variance'):
                error=abs(moments['base'][k]-moments['fine'][k])
                require(error<=.002 and abs(error-saved['moments'][k])<=1e-12,'Raw moment comparison differs.')
        if laws:
            errors=compare_interval_distributions(laws['base'],laws['fine'])
            for k in ('maximum_cdf_difference','maximum_quantile_difference'):
                require(abs(errors[k]-saved['distributions'][k])<=1e-12,'Raw CDF/quantile comparison differs.')
        comparisons.append(comparison)
    require(cursor==len(jobs) and cumulative==report['window_scores']
        and report['comparisons']==json.loads(files['comparisons.json']), 'Incomplete raw chronology or score.')
    rates={g:{k:max(r['elapsed_seconds'] for r in timings if r['grid']==g and r['rate_type']==k and not r['cold'])
        for k in ('prediction','observed','moments','cdf')} for g in ('base','fine')}
    require(rates==report['rates'] and ordered_sum(r['elapsed_seconds'] for r in timings if r['cold'])==report['cold_job_seconds'], 'Warm maximum/cold sums differ.')
    known=json.loads(known_ledger_path.read_text())
    require(ledger['entries'][:-1]==known['entries'] and ledger['entries'][-1]['label']=='native-development|compact-historical-rate'
        and all(e['status']=='closed' for e in ledger['entries']) and ledger['charged_seconds']==425.8919726690074+
        ordered_sum(e['charged_seconds'] for e in ledger['entries']), 'PC ledger prefix/total differs.')
    require(manifest['native_seconds_before']==ordered_sum(e['charged_seconds'] for e in known['entries']
        if e['label'].startswith('native-development|')) and manifest['grant_seconds']==300.
        and 0.<=manifest['launcher_seconds']<=60. and 0.<=report['elapsed_seconds_before_ledger_close']<=
        ledger['entries'][-1]['charged_seconds']<240., 'Pilot grant/startup/stop accounting differs.')
    # Independent arithmetic for all 12 jobs; no candidate division by workers.
    train,selection=observations['train_complete'],observations['selection_complete'];b,f=rates['base'],rates['fine']
    costs=[49*train*b['observed'],49*(48-train)*b['prediction'],49*24*b['prediction'],
        3*selection*b['observed'],3*(24-selection)*b['prediction'],3*(train+selection)*f['observed'],
        3*(72-train-selection)*f['prediction'],3*24*f['prediction'],336*(max(b['observed'],b['prediction'])+max(f['observed'],f['prediction'])),
        408*(b['moments']+f['moments']),336*(b['moments']+f['moments']),14*(b['cdf']+f['cdf'])]
    require(len(graph['jobs'])==len(costs)==12 and all(abs(j['planned_seconds']-1.5*c)<=1e-10
        for j,c in zip(graph['jobs'],costs)), 'Saved work-graph job price differs.')
    total=ledger['charged_seconds']+1.5*ordered_sum(costs)+600.+60.
    validation=1.5*ordered_sum(costs[3:])+300.+60.
    require(abs(total-graph['projected_total_seconds'])<=1e-9 and abs(max(7200.,validation)-graph['protected_validation_seconds'])<=1e-9
        and graph['within_ten_hour_ceiling']==(total<=36000.) and graph['within_six_hour_target']==(total<=21600.)
        and graph['runtime_admitted'] is False,'Independent whole-graph budget differs.')
    return dict(schema='f4-h2-compact-rate-raw-audit-v1',passed=True,source_commit=manifest['source_commit'],
        review_zip_sha256=hashlib.sha256(raw).hexdigest(),source_files_verified=64,source_matches=matches,
        packet_identity=packet['identity'],raw_jobs_verified=len(jobs),chronological_hours=observations['pilot_end_hour'],
        observed_hours=3,train_complete_hours=train,selection_complete_hours=selection,rates=rates,
        pc_invocation_charged_seconds=ledger['entries'][-1]['charged_seconds'],pc_H_charged_seconds=ledger['charged_seconds'],
        projected_total_hours=total/3600.,protected_validation_seconds=validation,maximum_guards=guard_max,
        maximum_state_mean_difference=max(c['state_mean_difference'] for c in comparisons),
        maximum_state_L1_difference=max(c['conservative_state_L1_difference'] for c in comparisons),
        maximum_total_log_score_difference=max(c['pilot_total_log_score_difference'] for c in comparisons),
        linear_prefix_precision_verified=True,full_window_precision_verified=False,nonlinear_rates_measured=False,
        historical_fit_ready=False,runtime_admitted=False,blind_h3_acceptance=False,scientific_acceptance=False,
        model_jobs_replayed=0,limitations=['Workbook not included; complete fixed MED raw household means cannot be independently recomputed.',
            'Linear 27-hour prefix only; no full TRAIN/selection/final or nonlinear accuracy/cost admission.',
            'Nominal physical/noise/prior assumptions; unknown previous final-week exposure.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--git-evidence',type=Path,default=ROOT/'root/f4/h2_compact_rate_git_evidence_20261007.json')
    parser.add_argument('--known-ledger',type=Path,default=OUT/'f4_h2_frozen_export_ledger_after_20261007.json')
    args=parser.parse_args();require(not args.output.exists(),'Preserve previous raw review.')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
    from src.exe.run_model.bm_1_h2_native import native_reservation
    with native_reservation(OUT/'f4_h2_remaining_budget.json',76.,'historical-rate-raw-review',begun=BEGUN):
        result=audit(args.archive,args.git_evidence,args.known_ledger)
        result['elapsed_seconds_before_ledger_close']=perf_counter()-BEGUN
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('passed','raw_jobs_verified','projected_total_hours','model_jobs_replayed')}))


if __name__=='__main__':
    main()

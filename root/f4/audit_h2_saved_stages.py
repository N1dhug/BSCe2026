"""Reconstruct retained validation and frozen forecasts from immutable jobs only."""
import numpy as np


def retained_stages(directory, data, jobs, candidate_by_key):
    from src.exe.run_model import bm_1_h2_compact as c
    from src.exe.run_model.bm_1_history import _beta_mass
    from src.method.bm_1_interval_forecast import compare_interval_moments, compare_interval_distributions
    from root.f4.measure_h2_compact_rates import compare_states

    def initial(candidate, grid):
        if c.curve_key(candidate)=='linear':
            return data['jobs'][(grid,0,'prediction')]['prior_log_state']
        p=data['packet']['prior']; n=data['packet']['numerics'][grid+'_cells']
        with np.errstate(divide='ignore'):
            return np.log(_beta_mass(np.linspace(0.,1.,n+1),alpha=p['alpha'],beta=p['beta']))

    def job(key,grid,branch,hour,kind,prior):
        saved=jobs.get((key,grid,branch,hour,kind))
        if saved is not None:
            c.same_state(saved['prior_log_state'],prior)
            return saved['result']
        if key=='linear' and branch=='prefinal' and (grid,hour,kind) in data['jobs']:
            raw=data['jobs'][(grid,hour,kind)];c.same_state(raw['prior_log_state'],prior)
            if kind=='cdf':return raw['forecast']
            result=dict(raw);result['log_state']=raw['posterior_log_state'] if kind=='observed' else raw['log_state']
            return result
        return None

    def moment_algebra(result):
        w=np.exp(result['log_state']);mean=float(w@result['conditional_mean'])
        variance=float(w@(result['conditional_variance']+(result['conditional_mean']-mean)**2))
        c.require(abs(mean-result['interval_mean'])<=1e-13
            and abs(variance-result['interval_variance'])<=1e-13,'Predictive moment algebra differs.')
        c.require(abs(result['observation_mean']-mean)<=1e-13
            and abs(result['observation_variance']-variance-data['packet']['physical']['sigma_y']**2)<=1e-13,
            'Observation noise was not added to the predictive interval variance.')

    def stream(candidate,grid):
        key=c.curve_key(candidate);logs=initial(candidate,grid);rows=[]
        for hour in range(96):
            seen=hour>=24 and bool(data['observations']['observed'][hour-24])
            result=job(key,grid,'prefinal',hour,'observed' if seen else 'prediction',logs)
            if result is None:break
            moments=job(key,grid,'prefinal',hour,'moments',logs) if hour>=72 else None
            if hour>=72:
                c.require(moments is not None,'A completed selection update lacks its pre-update moments.')
                moment_algebra(moments)
            logs=result['log_state'];rows.append(dict(hour=hour,log_state=logs,observed=seen,
                log_evidence=result['log_evidence'] if seen else 0.,moments=moments))
        return rows

    retained={1:c.LINEAR}
    for pieces in (2,4):
        path=directory/f'optimizer_{pieces}_result.json'
        if path.exists() and c.load(path)['converged']:
            retained[pieces]=c.load(path)['candidate']
    families={};partial={}
    for pieces,candidate in retained.items():
        streams={grid:stream(candidate,grid) for grid in ('base','fine')}
        if any(len(v)!=96 for v in streams.values()):
            partial[str(pieces)]={g:len(v) for g,v in streams.items()};continue
        comparison=c.compare_streams(streams['base'],streams['fine'],{'train':(24,72),'selection':(72,96)})
        families[pieces]=dict(candidate=candidate,streams=streams,comparison=comparison)
    c.require(1 in families,'The audited mandatory linear reference is missing.')
    c.require(families[1]['comparison']==c.load(directory/'linear_full_window.json')['comparison'],
        'Linear summary no longer matches its raw streams.')
    report=dict(validated_families=sorted(families),partial_retained_streams=partial,
        comparisons={str(p):v['comparison'] for p,v in families.items()},selection_verified=False,
        open_forecast_verified=False,handoff_verified=False)
    selected_path=directory/'selection.json'
    if not selected_path.exists():return report
    selected=c.load(selected_path);expected=c.select_families(families)
    c.require(all(selected[k]==v for k,v in expected.items()),'Saved selection rule/rank differs from raw likelihoods.')
    c.require(selected['curves']=={str(p):v['candidate'] for p,v in families.items()}
        and selected['comparisons']==report['comparisons'],'Selected curves/comparisons differ.')
    metrics={str(p):{g:c.selection_metrics(v['streams'][g],data['observations']) for g in ('base','fine')} for p,v in families.items()}
    c.require(selected['metrics']==metrics and selected['final_observations_decoded']==0,'Selection metric/data-boundary changed.')
    report.update(selection_verified=True,selected_family=selected['selected_family'],metrics=metrics)
    path=directory/'forecast_frozen.npz'
    if not path.exists():return report
    frozen=c._read_array_checkpoint(path.read_bytes())
    c.require(frozen['checksum']==c.fingerprint({k:v for k,v in frozen.items() if k!='checksum'})
        and frozen['selection_sha256']==c.sha(selected_path)
        and frozen['identity']==selected['identity']
        and frozen['final_observations_decoded']==frozen['assimilated_observations']==0,
        'Forecast freeze identity/checksum/data-boundary changed.')
    chosen={1,selected['selected_family']};keys={c.curve_key(families[p]['candidate']) for p in chosen}
    c.require(set(frozen['outputs'])==set(frozen['origins'])==set(frozen['models'])==keys,'Frozen model set differs.')
    for pieces in chosen:
        candidate=families[pieces]['candidate'];key=c.curve_key(candidate);rebuilt={}
        for grid in ('base','fine'):
            logs=families[pieces]['streams'][grid][-1]['log_state']
            c.same_state(logs,frozen['origins'][key][grid]);rows=[]
            for hour in range(96,264):
                moment=job(key,grid,'open',hour,'moments',logs)
                c.require(moment is not None,'Incomplete frozen forecast moments.');moment_algebra(moment)
                full=job(key,grid,'open',hour,'cdf',logs) if hour-95 in c.HORIZONS else None
                if hour-95 in c.HORIZONS:
                    c.require(full is not None,'Incomplete frozen horizon distribution.')
                    law=c.decode_distribution(full['distribution']);law.audit()
                    for q in full['distribution']['quantiles']:
                        c.require(law.quantile(q['probability'])==q,'Frozen CF quantile differs.')
                    active=np.isfinite(moment['log_state']);other=full['log_state']
                    c.require(np.array_equal(active,np.isfinite(other))
                        and np.max(np.abs(moment['log_state'][active]-other[active]))<=1e-7,'Moment/CF state differs.')
                rows.append(dict(hour=hour,horizon=hour-95,moments=moment,distribution=full))
                logs=moment['log_state']
            c.require(c.fingerprint(rows)==c.fingerprint(frozen['outputs'][key][grid]),'Frozen forecast contains different raw outputs.')
            rebuilt[grid]=rows
        checks=[]
        for base,fine in zip(rebuilt['base'],rebuilt['fine']):
            row=dict(horizon=base['horizon'],moments=compare_interval_moments(base['moments'],fine['moments']),
                state=compare_states(base['moments']['log_state'],fine['moments']['log_state']))
            if base['distribution'] is not None:
                row['distribution']=compare_interval_distributions(c.decode_distribution(base['distribution']['distribution']),
                    c.decode_distribution(fine['distribution']['distribution']))
            checks.append(row)
        c.require(frozen['models'][key]==dict(candidate=candidate,checks=checks),'Frozen precision summary differs.')
    report['open_forecast_verified']=True
    path=directory/'conditional_handoff.npz'
    if path.exists():
        handoff=c._read_array_checkpoint(path.read_bytes())
        c.require(handoff['checksum']==c.fingerprint({k:v for k,v in handoff.items() if k!='checksum'})
            and handoff['frozen_forecast_sha256']==c.sha(directory/'forecast_frozen.npz')
            and handoff['selected_family']==selected['selected_family']
            and handoff['selected_curve_key']==c.curve_key(selected['curves'][str(selected['selected_family'])])
            and handoff['origin_timestamp']==selected['cutoff']
            and handoff['origin_time']==4. and handoff['time_unit']=='day','Handoff identity/origin differs.')
        for key,grids in frozen['origins'].items():
            c.require(handoff['origins'][key]['candidate']==frozen['models'][key]['candidate'],'Handoff curve differs.')
            for grid,logs in grids.items():c.same_state(logs,handoff['origins'][key]['grids'][grid]['log_state'])
        report['handoff_verified']=True
    return report

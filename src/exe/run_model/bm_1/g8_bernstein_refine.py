"""One separately authorized own M254 refinement; no solver/model changes.

Fresh directory only. Two-hour pilot is part of the actual 24-hour replay.
900 s includes pilot, comparisons, actual JSON resume and export (30 s reserve).
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
from src.exe.run_model.bm_1.g8_bernstein import SolverAudit, save_partial, sha, write
from src.exe.run_model.bm_1.g8 import summarize, compare, TOLERANCES
from src.exe.run_model.bm_1.g8_basis import node_comparison, projection_max
from src.method.bm_1_g8 import bank_filter, context, G8BankOrigin, RHO_VALUES
from src.method.bm_1_g8_bernstein import BASIS_KIND
from src.params.bm_1 import BM1ObservationSeries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--fv513',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+870.
    inputs={str(x.resolve()):sha(x) for x in [args.fixture,args.coarse,args.fv513,args.manifest]}
    expected=json.loads(args.manifest.read_text());source={};source_audit=[]
    for name,m in list(sys.modules.items()):
        if name.startswith('src.') and getattr(m,'__file__',None):
            path=Path(m.__file__).resolve();rel=str(path.relative_to(Path.cwd()))
            data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            source[rel]=sha(path)
            if expected.get(rel)!=blob:raise ValueError('Imported source differs from registered branch: '+rel)
            source_audit.append(dict(path=rel,git_blob=blob,sha256=source[rel]))
    source[str(Path(__file__).resolve().relative_to(Path.cwd()))]=sha(Path(__file__))
    registration=dict(study='G8 own M254 513 refinement',source_parent='a368090719ddf97f0ae1c18443191acd66961f5c',
        basis_kind=BASIS_KIND,degree=254,components=255,cells=513,max_step=1/192,cfl=.4,
        rho_values=RHO_VALUES,observation='Same saved synthetic point-demand hours; sigma_y=.03; GL3; missing6,17',
        budget_seconds=900,compute_seconds=870,export_reserve_seconds=30,pilot_seconds=60,
        admission='Two completed hours within60s; elapsed + 1.25*max(last2 hour times)*(remaining hours+2 resume hours)+10s <=870s. Continue same replay; no reset.',
        aggregate='1800s overall proposed panel ceiling retained. Earlier measured runs about345s plus conservatively reserved300s for unmeasured probes/archive: debit650s. New cap900s leaves250s, not a G7 admission.',
        tests='No old tests repeated. Actual own fine replay, fixed paired gates and full-case serialized tail are the checks.',
        tolerances=TOLERANCES,input_sha256=inputs,source_sha256=source,
        stop='TimeoutError interrupts long solve; preserve complete hours. No retry, larger noise, degree search or looser gates.')
    write(out/'registration.json',registration);write(out/'source_audit.json',source_audit)
    report=dict(registration=registration,full_filter_accept=False,mixture_to_G7_ready=False,
        historical_accept=False,parameter_recovery_accept=False,records_complete=0)
    records=[];audit=None;hour_times=[];admitted=False;last=None
    def timeout(_signum,_frame):raise TimeoutError('Registered refinement deadline/admission exhausted.')
    signal.signal(signal.SIGALRM,timeout)
    def arm(at):signal.setitimer(signal.ITIMER_REAL,max(.001,at-time.perf_counter()))
    arm(min(deadline,start+60.))
    try:
        with np.load(args.fixture,allow_pickle=False) as f:
            obs=BM1ObservationSeries(f['times'],f['values'],'hour',f['observed'])
            truth=(f['truth_states'],f['truth_demand'])
        if not np.array_equal(obs.times,np.arange(1.,25.)) or not np.array_equal(np.where(~obs.observed)[0],[5,16]):
            raise ValueError('Fixture contract differs.')
        # Compare-only references; they never become initial state or coefficients.
        with np.load(args.coarse,allow_pickle=False) as z:
            if not np.array_equal(z['times'],obs.times) or not np.array_equal(z['observed'],obs.observed):
                raise ValueError('Coarse reference times/mask differ.')
        run_start=time.perf_counter();last=run_start
        def configure(projector):
            nonlocal audit
            audit=SolverAudit(projector.projection_solver,projector.dx)
            projector.projection_solver=audit
        def checkpoint(record):
            nonlocal last,admitted
            now=time.perf_counter();hour_times.append(now-last);last=now;records.append(record)
            write(out/('origin_'+str(int(record['time']))+'.json'),record['origin'].to_record())
            write(out/'checkpoint_latest.json',record['origin'].to_record())
            estimate=1.25*max(hour_times[-2:])*(24-len(records)+2)+10.
            write(out/'progress.json',dict(completed_hours=len(records),hour_seconds=hour_times,
                  cumulative_seconds=now-start,remaining_estimate_seconds=estimate,projection_audit=audit.record()))
            print('M254513 hour',int(record['time']),'elapsed',round(now-start,2),
                  's remaining+resume estimate',round(estimate,2),'s',flush=True)
            if len(records)==2:
                admitted=now+estimate<=deadline
                write(out/'pilot.json',dict(seconds=now-run_start,remaining_estimate_seconds=estimate,admitted=admitted))
                if not admitted:raise TimeoutError('Pilot estimate cannot fit registered refinement+resume allowance.')
                arm(deadline)
        error=None
        try:
            _,model,parameters,_=bank_filter(513,obs,1/192,basis_kind=BASIS_KIND,
                  deadline=deadline,on_step=checkpoint,configure_projector=configure)
        except (TimeoutError,RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            error=dict(type=type(exc).__name__,message=str(exc))
            model,_,parameters,_,_,_=context(513)
        signal.setitimer(signal.ITIMER_REAL,0.)
        report.update(records_complete=len(records),run_seconds=time.perf_counter()-run_start,
            error=error,pilot_admitted=admitted,hour_seconds=hour_times,
            projection_audit=None if audit is None else audit.record())
        if records:
            save_partial(out,'bank_m254_513',records,model,parameters,report['run_seconds'],truth)
            # Regenerate summaries from frozen arrays only, never rerun a reference.
            for name,path in [('bank_m254_257',args.coarse),('bank_fv513',args.fv513)]:
                with np.load(path,allow_pickle=False) as z:
                    fields=('times','edges','predicted_mass','posterior_mass','log_evidence','observed',
                            'log_weights','conditional_mass','conditional_log_evidence','prior_log_weights')
                    arrays={k:z[k] if k=='edges' else z[k][:len(records)] for k in fields}
                if not np.array_equal(arrays['times'],obs.times[:len(records)]) or not np.array_equal(arrays['observed'],obs.observed[:len(records)]):
                    raise ValueError('Comparison reference time/mask differs.')
                np.savez_compressed(out/(name+'.npz'),**arrays)
                s=summarize(model,parameters,arrays['times'],arrays['edges'],arrays['predicted_mass'],
                    arrays['posterior_mass'],arrays['log_evidence'],arrays['observed'],
                    (truth[0][:len(records)+1],truth[1][:len(records)+1]))
                write(out/(name+'.json'),s)
            report['conditional_checks']={name:node_comparison(out,'bank_m254_513',name)
                 for name in ('bank_m254_257','bank_fv513')}
            report['marginal_checks']={name:compare(out,'bank_m254_513',name,bank=True)
                 for name in ('bank_m254_257','bank_fv513')}
            report['bayes_projection_l1']=projection_max([r['posterior_projection'] for r in records])
            missing=[r for r in records if not r['observed']]
            report['missing_posterior_change']=max((float(np.max(abs(r['posterior_mass']-r['predicted_mass']))) for r in missing),default=0.)
            report['missing_weight_change']=max((float(np.max(abs(r['log_weights']-r['prior_log_weights']))) for r in missing),default=0.)
        paired_pass=(len(records)==24 and error is None and admitted
            and all(x['passed'] for x in report['conditional_checks'].values())
            and all(x['all_pass'] for x in report['marginal_checks'].values())
            and report['projection_audit']['maximum_projection_cell_mass_l1']<=TOLERANCES['projection_l1']
            and report['bayes_projection_l1']<=TOLERANCES['projection_l1']
            and report['missing_posterior_change']<=TOLERANCES['checkpoint']
            and report['missing_weight_change']<=TOLERANCES['checkpoint'])
        report['paired_pass']=paired_pass
        if paired_pass:
            arm(deadline)
            origin=G8BankOrigin.from_record(json.loads((out/'origin_22.json').read_text()))
            tail=BM1ObservationSeries(obs.times[22:],obs.values[22:],'hour',obs.observed[22:])
            t=time.perf_counter();resumed,_,_,_=bank_filter(513,tail,1/192,basis_kind=BASIS_KIND,
                 origin=origin,deadline=deadline)
            errors={field:float(np.max(abs(np.array([r[field] for r in resumed])-
                  np.array([r[field] for r in records[22:]]))))
                  for field in ('conditional_mass','conditional_coefficients','log_weights','log_evidence','conditional_log_evidence')}
            report['resume']=dict(seconds=time.perf_counter()-t,errors=errors,
                       passed=max(errors.values())<=TOLERANCES['checkpoint'])
            write(out/'resumed_origin_24.json',resumed[-1]['origin'].to_record())
            report['full_filter_accept']=report['resume']['passed']
            report['mixture_to_G7_ready']=report['full_filter_accept']
    except (TimeoutError,RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        report['failure']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        report['total_wall_seconds']=time.perf_counter()-start
        report['source_unchanged']=all(sha(Path(path))==value for path,value in source.items())
        report['inputs_unchanged']=all(sha(Path(path))==value for path,value in inputs.items())
        report['within_wall_budget']=report['total_wall_seconds']<=900.
        valid=report['source_unchanged'] and report['inputs_unchanged'] and report['within_wall_budget']
        report['full_filter_accept']=bool(report['full_filter_accept'] and valid)
        report['mixture_to_G7_ready']=report['full_filter_accept']
        write(out/'summary.json',report)
    print(json.dumps({k:report.get(k) for k in ('records_complete','run_seconds','total_wall_seconds',
          'error','failure','paired_pass','resume','full_filter_accept')},indent=2),flush=True)


if __name__=='__main__':main()

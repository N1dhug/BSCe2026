"""One registered M254 recursive filter, with a 300 s cumulative wall budget.

Linux/POSIX SIGALRM can interrupt a long Python LS call. No retry/new budget.
References are comparisons only; every run starts from the original prior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import unittest
import numpy as np
import scipy
from src.exe.run_model.bm_1.g8 import plain, TOLERANCES, retain_bank, compare
from src.exe.run_model.bm_1.g8_basis import node_comparison, projection_max
from src.method.bm_1_g8 import context, bank_filter, G8BankOrigin, RHO_VALUES
from src.method.bm_1_g8_basis import BERNSTEIN_BASIS_KIND as BASIS_KIND
from src.params.bm_1 import BM1ObservationSeries


def write(path,record):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(plain(record),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class SolverAudit:
    """Transparent G-only wrapper; delegates the unchanged optimization.

    Retain only the last external gap check for a solve (including fallback).
    Internal calls made by the wrapped solver bypass this wrapper.
    """
    def __init__(self,solver,dx):
        self.solver=solver;self.dx=dx;self.calls=0;self.seconds=0.;self.pending=None
        self.checked=0;self.max_l1=0.;self.max_scaled_gap=0.;self.max_mass_error=0.
    def __getattr__(self,name):return getattr(self.solver,name)
    def flush(self):
        if self.pending is not None:
            l1,gap,mass=self.pending
            self.checked+=1;self.max_l1=max(self.max_l1,l1)
            self.max_scaled_gap=max(self.max_scaled_gap,gap)
            self.max_mass_error=max(self.max_mass_error,mass);self.pending=None
    def solve(self,target,initial):
        self.flush();self.calls+=1;t=time.perf_counter()
        try:return self.solver.solve(target,initial)
        finally:self.seconds+=time.perf_counter()-t
    def optimality_gap(self,target,coefficients):
        gap,gradient=self.solver.optimality_gap(target,coefficients)
        error=self.solver.matrix@coefficients-target
        self.pending=(float(np.sum(abs(error))*np.sqrt(self.dx)),
                      float(gap/max(1.,float(np.max(abs(gradient))))),
                      abs(float(np.sum(coefficients))-1.))
        return gap,gradient
    def record(self):
        self.flush()
        return dict(solver_calls=self.calls,checked_projections=self.checked,
                    active_set_seconds=self.seconds,maximum_projection_cell_mass_l1=self.max_l1,
                    maximum_scaled_gap=self.max_scaled_gap,maximum_coefficient_mass_error=self.max_mass_error,
                    factorizations=self.factorizations,cache_hits=self.cache_hits,
                    scope='All completed primary/fallback solutions; interrupted solve excluded from error maxima.')


def save_partial(out,name,records,model,parameters,seconds,truth):
    """Export even before the first missing hour; old retainer assumes full day."""
    from src.exe.run_model.bm_1.g8 import summarize
    arrays={field:np.array([r[field] for r in records]) for field in
            ('time','observed','predicted_mass','posterior_mass','log_evidence','log_weights',
             'conditional_mass','conditional_coefficients','conditional_log_evidence','prior_log_weights')}
    arrays['times']=arrays.pop('time');arrays['edges']=records[-1]['origin'].x_edges
    arrays['basis']=records[-1]['origin'].basis
    np.savez_compressed(out/(name+'.npz'),**arrays)
    n=len(records)
    summary=summarize(model,parameters,arrays['times'],arrays['edges'],arrays['predicted_mass'],
          arrays['posterior_mass'],arrays['log_evidence'],arrays['observed'],(truth[0][:n+1],truth[1][:n+1]))
    mask=~arrays['observed']
    summary.update(seconds=seconds,completed_hours=n,
        missing_weight_change=0. if not mask.any() else float(np.max(abs(arrays['log_weights'][mask]-arrays['prior_log_weights'][mask]))),
        maximum_conditional_mass_error=float(np.max(abs(arrays['conditional_mass'].sum(-1)-1.))),
        minimum_conditional_mass=float(arrays['conditional_mass'].min()),
        posterior_projection=[r['posterior_projection'] for r in records])
    write(out/(name+'.json'),summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--fv257',type=Path,required=True)
    parser.add_argument('--fv513',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+270.
    files={'fixture':args.fixture,'fv257':args.fv257,'fv513':args.fv513}
    hashes={str(Path(m.__file__).resolve().relative_to(Path.cwd())):sha(Path(m.__file__))
            for name,m in list(sys.modules.items()) if name.startswith('src.') and getattr(m,'__file__',None)}
    hashes[str(Path(__file__).resolve().relative_to(Path.cwd()))]=sha(Path(__file__))
    test_path=Path('tests/test_bm1_g8_bernstein.py');hashes[str(test_path)]=sha(test_path)
    registration=dict(study='G8 own recursive supervisor Bernstein M254',source_parent='c64198bd8749982143e55a0781a1af6ff2dc5b81',
        basis_kind=BASIS_KIND,degree=254,components=255,cells=257,max_step=1/96,cfl=.4,
        rho_values=RHO_VALUES,prior='Exact degree-elevated Beta(20,20); equal node weights',
        observation_contract='Synthetic point demand; hours; sigma_y=.03; GL3; right price side; missing 6,17',
        budget_seconds=300,compute_deadline_seconds=270,export_reserve_seconds=30,pilot_seconds=60,
        pilot_admission='After two completed hours: elapsed + remaining hours * max(last two hour times) <= remaining computation budget; otherwise stop. Pilot is actual replay, no reset.',
        tests='Only new Bernstein prior/checkpoint tests, and existing localized resume compatibility test; counted in budget.',
        refinement_admission='Only if full 257 run passes reused gates and 513+two-hour resume is estimated to fit remaining budget with factor two safety margin; no retuning.',
        hard_timeout='POSIX SIGALRM raises TimeoutError (not caught as solver RuntimeError fallback). No retry.',
        tolerances=TOLERANCES,source_sha256=hashes,
        input_files={k:dict(path=str(p.resolve()),sha256=sha(p)) for k,p in files.items()},
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        excluded='No FV posterior initialization, historical observations, parameter recovery, new M search, noise changes or G7.')
    write(out/'registration.json',registration)
    report=dict(registration=registration,runs={},full_filter_accept=False,historical_accept=False,
                parameter_recovery_accept=False,mixture_to_G7_ready=False)
    def timeout(_signum,_frame):raise TimeoutError('Registered M254 numerical wall deadline exhausted.')
    signal.signal(signal.SIGALRM,timeout)
    def arm(when):signal.setitimer(signal.ITIMER_REAL,max(.001,when-time.perf_counter()))
    arm(min(deadline,start+30.))
    try:
        test_start=time.perf_counter()
        suite=unittest.defaultTestLoader.loadTestsFromNames([
            'tests.test_bm1_g8_bernstein',
            'tests.test_bm1_g8_basis.G8BasisTests.test_one_node_and_resume_preserve_custom_basis'])
        test=unittest.TextTestRunner(verbosity=2).run(suite)
        report['tests']=dict(seconds=time.perf_counter()-test_start,run=test.testsRun,passed=test.wasSuccessful())
        write(out/'tests.json',report['tests'])
        if not test.wasSuccessful():raise RuntimeError('New contract/compatibility test failed; no scientific replay.')
        with np.load(args.fixture,allow_pickle=False) as f:
            obs=BM1ObservationSeries(f['times'],f['values'],'hour',f['observed'])
            truth=(f['truth_states'],f['truth_demand'])
        if (not np.array_equal(obs.times,np.arange(1.,25.)) or not np.array_equal(np.where(~obs.observed)[0],[5,16])):
            raise ValueError('Saved fixture contract differs.')
        # Reused references are copied/cropped only for comparing completed hours.
        run_start=time.perf_counter();pilot_deadline=min(deadline,run_start+60.)
        arm(pilot_deadline)
        records=[];audit=None;hour_times=[];last=run_start;admitted=False
        def configure(projector):
            nonlocal audit
            audit=SolverAudit(projector.projection_solver,projector.dx)
            projector.projection_solver=audit
        def checkpoint(record):
            nonlocal last,admitted
            now=time.perf_counter();hour_times.append(now-last);last=now;records.append(record)
            write(out/'checkpoint_latest.json',record['origin'].to_record())
            write(out/('origin_'+str(int(record['time']))+'.json'),record['origin'].to_record())
            write(out/'progress.json',dict(completed_hours=len(records),hour_seconds=hour_times,
                  cumulative_seconds=now-start,projection_audit=audit.record()))
            estimate=(24-len(records))*max(hour_times[-2:])
            print('M254 hour',int(record['time']),'elapsed',round(now-start,2),'s remaining estimate',round(estimate,2),'s',flush=True)
            if len(records)==2 and not admitted:
                admitted=now+estimate<=deadline
                write(out/'pilot.json',dict(seconds=now-run_start,remaining_estimate_seconds=estimate,admitted=admitted))
                if not admitted:raise TimeoutError('Pilot estimates full replay exceeds remaining registered budget.')
                arm(deadline)
        error=None
        try:
            _,model,parameters,_=bank_filter(257,obs,1/96,basis_kind=BASIS_KIND,
                   deadline=deadline,on_step=checkpoint,configure_projector=configure)
        except (TimeoutError,RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            error=dict(type=type(exc).__name__,message=str(exc))
            model,_,parameters,_,_,_=context(257)  # FV context only; no extra projection.
        finally:signal.setitimer(signal.ITIMER_REAL,0.)
        run=dict(seconds=time.perf_counter()-run_start,completed_hours=len(records),hour_seconds=hour_times,
                 error=error,projection_audit=None if audit is None else audit.record(),pilot_admitted=admitted)
        if records:
            save_partial(out,'bank_m254_257',records,model,parameters,run['seconds'],truth)
            for key in ('fv257','fv513'):
                with np.load(files[key],allow_pickle=False) as z:
                    crop={k:(z[k][:len(records)] if k!='edges' else z[k]) for k in z.files}
                np.savez_compressed(out/('bank_'+key+'.npz'),**crop)
                # compare reads summary JSON: produce it from saved arrays, not a replay.
                from src.exe.run_model.bm_1.g8 import summarize
                summary=summarize(model,parameters,crop['times'],crop['edges'],crop['predicted_mass'],crop['posterior_mass'],crop['log_evidence'],crop['observed'],
                                  (truth[0][:len(records)+1],truth[1][:len(records)+1]))
                write(out/('bank_'+key+'.json'),summary)
            run['conditional_checks']={key:node_comparison(out,'bank_m254_257','bank_'+key) for key in ('fv257','fv513')}
            run['marginal_checks']={key:compare(out,'bank_m254_257','bank_'+key,bank=True) for key in ('fv257','fv513')}
            run['bayes_projection_l1']=projection_max([r['posterior_projection'] for r in records])
        if not (out/'pilot.json').exists():
            write(out/'pilot.json',dict(seconds=run['seconds'],completed_hours=len(records),admitted=False,error=error))
        report['runs']['257']=run
        report['refinement']=dict(performed=False,reason='Incomplete 257 replay' if len(records)<24 else 'Separate admission/result required; no automatic budget extension.')
        report['checkpoint_resume']=dict(short_contract_passed=report['tests']['passed'],full_case_performed=False)
        primary_pass=(len(records)==24 and error is None
                      and all(v['passed'] for v in run.get('conditional_checks',{}).values())
                      and all(v['all_pass'] for v in run.get('marginal_checks',{}).values())
                      and run['projection_audit']['maximum_projection_cell_mass_l1']<=TOLERANCES['projection_l1'])
        fine_estimate=2.*run['seconds']*(1.+2./24.)
        report['refinement']['estimated_seconds_with_margin']=fine_estimate
        if primary_pass and time.perf_counter()+fine_estimate<=deadline:
            fine=[];fine_audit=None;fine_start=time.perf_counter();arm(deadline)
            def fine_configure(projector):
                nonlocal fine_audit
                fine_audit=SolverAudit(projector.projection_solver,projector.dx)
                projector.projection_solver=fine_audit
            def fine_checkpoint(record):
                fine.append(record)
                write(out/'fine_checkpoint_latest.json',record['origin'].to_record())
                if record['time']==22.:write(out/'fine_origin_22.json',record['origin'].to_record())
                print('M254 fine hour',int(record['time']),'remaining wall budget',round(deadline-time.perf_counter(),2),'s',flush=True)
            fine_error=None
            try:
                _,fine_model,fine_parameters,_=bank_filter(513,obs,1/192,basis_kind=BASIS_KIND,
                    deadline=deadline,on_step=fine_checkpoint,configure_projector=fine_configure)
            except (TimeoutError,RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                fine_error=dict(type=type(exc).__name__,message=str(exc))
                fine_model,_,fine_parameters,_,_,_=context(513)
            fine_run=dict(seconds=time.perf_counter()-fine_start,completed_hours=len(fine),error=fine_error,
                          projection_audit=None if fine_audit is None else fine_audit.record())
            if len(fine)==24 and fine_error is None:
                save_partial(out,'bank_m254_513',fine,fine_model,fine_parameters,fine_run['seconds'],truth)
                fine_run['conditional_checks']=node_comparison(out,'bank_m254_513','bank_fv513')
                fine_run['marginal_checks']=compare(out,'bank_m254_513','bank_fv513',bank=True)
                fine_run['refinement_checks']=compare(out,'bank_m254_257','bank_m254_513',bank=True)
                fine_run['conditional_refinement']=node_comparison(out,'bank_m254_257','bank_m254_513')
                origin=G8BankOrigin.from_record(json.loads((out/'fine_origin_22.json').read_text()))
                tail=BM1ObservationSeries(obs.times[22:],obs.values[22:],'hour',obs.observed[22:])
                resume_start=time.perf_counter()
                resumed,_,_,_=bank_filter(513,tail,1/192,basis_kind=BASIS_KIND,origin=origin,deadline=deadline)
                errors={field:float(np.max(abs(np.array([r[field] for r in resumed])-np.array([r[field] for r in fine[22:]]))))
                        for field in ('conditional_mass','conditional_coefficients','log_weights','log_evidence')}
                report['checkpoint_resume'].update(full_case_performed=True,seconds=time.perf_counter()-resume_start,
                              errors=errors,passed=max(errors.values())<=TOLERANCES['checkpoint'])
                report['full_filter_accept']=bool(fine_run['conditional_checks']['passed'] and fine_run['marginal_checks']['all_pass']
                       and fine_run['refinement_checks']['all_pass'] and fine_run['conditional_refinement']['passed']
                       and fine_run['projection_audit']['maximum_projection_cell_mass_l1']<=TOLERANCES['projection_l1']
                       and report['checkpoint_resume']['passed'])
                report['mixture_to_G7_ready']=report['full_filter_accept']
            elif fine:
                save_partial(out,'bank_m254_513',fine,fine_model,fine_parameters,fine_run['seconds'],truth)
            report['runs']['513']=fine_run
            report['refinement'].update(performed=True,reason='Primary gates and conservative runtime admission passed.')
        elif primary_pass:
            report['refinement']['reason']='Runtime estimate does not leave room for refinement and resume within same budget.'
        # A partial replay or missing refinement is deliberately not accepted.
    except (TimeoutError,RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        report['failure']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        report['total_wall_seconds']=time.perf_counter()-start
        write(out/'summary.json',report)
    print('M254 report',out/'summary.json','full filter accepted:',report['full_filter_accept'],flush=True)

if __name__=='__main__':main()

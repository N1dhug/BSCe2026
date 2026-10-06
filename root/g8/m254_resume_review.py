"""Bounded post-review: actual M254 full-case JSON resume, no new budget.

One two-hour tail, starting from the saved five-node hour22 checkpoint.
Reused 42-component results are comparisons only, never filter state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
from src.exe.run_model.bm_1.g8_bernstein import write
from src.exe.run_model.bm_1.g8 import TOLERANCES
from src.method.bm_1_g8 import G8BankOrigin, bank_filter
from src.method.bm_1_g8_bernstein import BASIS_KIND
from src.params.bm_1 import BM1ObservationSeries


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference42',type=Path,required=True)
    args=parser.parse_args();out=args.run
    old=json.loads((out/'summary.json').read_text())
    prior_wall=old['total_wall_seconds'];start=time.perf_counter()
    cap=min(60.,270.-prior_wall)
    if cap<=0:raise ValueError('No remaining registered compute budget.')
    registration=dict(study='Actual full-case JSON resume addendum',reason='257 full replay passed all reused gates; fine replay not admitted. Check same model/own conditional state on its actual saved tail.',
        previous_wall_seconds=prior_wall,total_budget_seconds=300,compute_budget_seconds=270,
        resume_cap_seconds=cap,cutoff=22.,tail=[23.,24.],basis_kind=BASIS_KIND,
        inputs={str(p):sha(p) for p in [out/'summary.json',out/'origin_22.json',out/'bank_m254_257.npz',args.reference42]},
        script_sha256=sha(Path(__file__)),new_filter_or_degree_search=False)
    write(out/'resume_registration.json',registration)
    report=dict(registration=registration,full_filter_accept=False,mixture_to_G7_ready=False)
    def timeout(_signum,_frame):raise TimeoutError('Cumulative M254 resume allowance exhausted.')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,cap)
    try:
        reg=json.loads((out/'registration.json').read_text())
        with np.load(reg['input_files']['fixture']['path'],allow_pickle=False) as f:
            tail=BM1ObservationSeries(f['times'][22:],f['values'][22:],'hour',f['observed'][22:])
        origin=G8BankOrigin.from_record(json.loads((out/'origin_22.json').read_text()))
        records,_,_,_=bank_filter(257,tail,1/96,basis_kind=BASIS_KIND,origin=origin,deadline=start+cap)
        with np.load(out/'bank_m254_257.npz',allow_pickle=False) as saved:
            errors={field:float(np.max(abs(np.array([r[field] for r in records])-saved[field][22:])))
                    for field in ('conditional_mass','conditional_coefficients','log_weights','log_evidence')}
            with np.load(args.reference42,allow_pickle=False) as ref:
                if not np.array_equal(saved['times'],ref['times']) or not np.array_equal(saved['edges'],ref['edges']):
                    raise ValueError('42-component reference has different times/grid.')
                observed=saved['observed']
                report['reference42_comparison']=dict(
                    maximum_conditional_mass_l1=float(np.max(np.sum(abs(saved['conditional_mass']-ref['conditional_mass']),axis=-1))),
                    maximum_marginal_mass_l1=float(np.max(np.sum(abs(saved['posterior_mass']-ref['posterior_mass']),axis=-1))),
                    maximum_weight_difference=float(np.max(abs(np.exp(saved['log_weights'])-np.exp(ref['log_weights'])))),
                    maximum_log_evidence_difference=float(np.max(abs(saved['log_evidence'][observed]-ref['log_evidence'][observed]))),
                    scope='Reused development reference comparison, not independent truth.')
        report['resume']=dict(errors=errors,passed=max(errors.values())<=TOLERANCES['checkpoint'])
        write(out/'resumed_origin_24.json',records[-1]['origin'].to_record())
    except (TimeoutError,RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        report['failure']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        report['additional_wall_seconds']=time.perf_counter()-start
        report['cumulative_wall_seconds']=prior_wall+report['additional_wall_seconds']
        write(out/'post_review.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

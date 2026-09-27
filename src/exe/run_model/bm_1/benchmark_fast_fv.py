"""Compare the reference and opt-in fast FV paths on identical problems.

Run python -B -m src.exe.run_model.bm_1.benchmark_fast_fv for repeated fixed-price
measurements. Add --optimize for the slower complete 24-variable Powell A/B
comparison. This deliberately does not claim a speedup for mixture or MC.
"""
import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from src.exe.run_model.bm_1.complex_target_multi_price_experiment import (
    HOURLY_EDGES, REFERENCE_PRICES_24, build_complex_deterministic_target,
    build_stochastic_model, _parameter_template,
)
from src.exe.run_model.bm_1.f2_reference_case import beta_initial_cell_mass, SIGMA_X
from src.method.bm_1_fast_fv import FastFVStochasticPriceEvaluator
from src.method.bm_1_stochastic_optimization import FVStochasticPriceEvaluator, multistart_powell


def build_pair(model, n_cells):
    edges, _, _ = model.fokker_planc_reference.grid(n_cells)
    args = (model, _parameter_template(HOURLY_EDGES), HOURLY_EDGES,
            SIGMA_X, beta_initial_cell_mass(edges))
    controls = dict(objective_max_step=1/96, solver_max_step=1/96,
                    breakpoints=HOURLY_EDGES[1:-1])
    return (FVStochasticPriceEvaluator(*args, **controls),
            FastFVStochasticPriceEvaluator(*args, **controls))


def run_benchmark(n_cells=(201,1001), repeats=7, optimize=False, progress=print):
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    target, _, _ = build_complex_deterministic_target()
    model = build_stochastic_model(target)
    records = []
    for n in n_cells:
        slow, fast = build_pair(model, n)
        a, b = slow.evaluate(REFERENCE_PRICES_24), fast.evaluate(REFERENCE_PRICES_24)
        mass_difference = float(np.max(np.abs(a.cell_mass - b.cell_mass)))
        np.testing.assert_allclose(a.cell_mass,b.cell_mass,atol=2e-12,rtol=2e-11)
        np.testing.assert_allclose(a.objective.total,b.objective.total,atol=2e-12,rtol=2e-11)
        timings = {'reference':[], 'fast':[]}
        # Alternate order to reduce thermal/order bias. Setup is outside timers.
        for repeat in range(repeats):
            order = [('reference',slow),('fast',fast)]
            if repeat % 2:
                order.reverse()
            for name,evaluator in order:
                started=perf_counter()
                evaluator(REFERENCE_PRICES_24)
                timings[name].append(perf_counter()-started)
        old,new = median(timings['reference']),median(timings['fast'])
        item = dict(cells=n,reference_seconds=old,fast_seconds=new,speedup=old/new,
                    maximum_cell_mass_difference=mass_difference,
                    bitwise_identical_mass=bool(np.array_equal(a.cell_mass,b.cell_mass)),
                    objective_difference=abs(a.objective.total-b.objective.total),
                    timings=timings)
        records.append(item)
        if progress:
            progress(f"{n} cells: {old:.6f} -> {new:.6f} s; {old/new:.2f}x; "
                     f"max mass difference={mass_difference:.3e}")
    output = dict(python=platform.python_version(),platform=platform.platform(),
                  numpy=np.__version__,scipy=scipy.__version__,repeats=repeats,
                  price_intervals=24,solver_max_step=1/96,objective_max_step=1/96,
                  fixed_price=records)
    if optimize:
        slow,fast=build_pair(model,201)
        coarse=np.repeat(REFERENCE_PRICES_24.reshape(12,2).mean(axis=1),2)
        starts=np.array([coarse,REFERENCE_PRICES_24])
        results={}
        for name,evaluator in [('reference',slow),('fast',fast)]:
            if progress:
                progress(f"Full 24-variable Powell comparison: {name}")
            started=perf_counter()
            fit=multistart_powell(evaluator,starts,options=dict(xtol=5e-5,ftol=1e-9,maxiter=100),progress=progress)
            results[name]=dict(seconds=perf_counter()-started,prices=fit.best.x.tolist(),
                               objective=float(fit.best.fun),evaluations=fit.objective_evaluations,
                               cache_hits=fit.cache_hits,successes=sum(bool(r.success) for r in fit.results))
        x0,x1=results['reference']['prices'],results['fast']['prices']
        results['price_max_difference']=float(np.max(np.abs(np.array(x0)-x1)))
        results['reference_cost_difference']=abs(slow(x0)-slow(x1))
        results['speedup']=results['reference']['seconds']/results['fast']['seconds']
        output['optimization']=results
        if progress:
            progress(f"Full optimization speedup={results['speedup']:.2f}x; "
                     f"price max difference={results['price_max_difference']:.3e}; "
                     f"reference cost difference={results['reference_cost_difference']:.3e}")
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=int,nargs='+',default=[201,1001])
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--optimize',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('files/output/bm_1_fast_fv_benchmark.json'))
    args=parser.parse_args()
    result=run_benchmark(args.cells,args.repeats,args.optimize,lambda s:print(s,flush=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'Saved numerical benchmark: {args.output}')


if __name__=='__main__':
    main()

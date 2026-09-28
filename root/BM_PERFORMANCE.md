# BM1 performance review: opt-in fast FV evaluation

Reviewed main: `542bb6bebd6b26a0f966b1428b9b5fb2d0f1db4d`.
Latest complex-target experiment: `67766c38082ada631009a972d4301eb168219f0f`.
Date: 27 September 2026.

## Result

The measured improvement is in the deterministic numerical evaluation of the
stochastic BM1 finite-volume objective. It is not a claimed speedup of all
repository code, the mixture solver, Monte Carlo, or real-data fitting.

| Controlled measurement | Reference | Fast FV | Speedup |
|---|---:|---:|---:|
| One 24-price objective, 201 cells, median of 7 | 0.091729 s | 0.007876 s | 11.65x |
| One 24-price objective, 1001 cells, median of 7 | 0.249825 s | 0.019422 s | 12.86x |
| Complete 24-variable Powell optimization, 201 cells, two identical starts | 463.447 s | 35.579 s | 13.03x |

Fixed-price measurements alternate execution order and exclude constructor,
imports and target construction. The full optimization is one A/B pair in
the same environment, not a universal hardware speed guarantee. It excludes
final MC and plotting. Both optimization runs made **4319 actual objective
evaluations**, had zero cache hits, and returned exactly the same best prices
and cost, **J = 0.00575901191445379**. The two starts were the pairwise-averaged
reference lifted to 24 hours, and the target-generating hourly reference.
This is not the same warm-start sequence as the full 2/4/6/12/24 experiment.

For the fixed-price 201- and 1001-cell comparisons, the entire returned cell
mass histories were bit-for-bit identical; maximum absolute difference and
objective difference were zero. This is observed agreement in the tested
cases, not a promise of bitwise equality for every LAPACK platform/problem.

Local benchmark environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
Linux x86_64, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MPLBACKEND=Agg.

## Measured bottlenecks

Profiling the reference FV objective identified Python scalar loops in
`finite_volume_1d._solve_tridiagonal`, repeated model/input evaluation and
validation, and recomputation of demand/moments in both the model wrapper
and the objective layer. In a profile of 30 objective calls, the Python
tridiagonal routine consumed 1.289 of 4.983 profiled seconds. Profiled times
include instrumentation overhead and are not the unprofiled benchmark above.

A separate M=30 dynamic-mixture profile spent 7.492 of 8.888 profiled seconds
inside the generic active-set solver. It made 50,528 `matrix_rank` calls and
3,900 least-squares KKT solves. That is a distinct future optimization target;
this change does not replace the shared active-set implementation.

## What the fast evaluator changes

`src/control/bm_1.py` adds `FastFVStochasticPriceEvaluator`, with the same
constructor inputs as the reference evaluator plus `batch_size=128` and
`vectorized=True`.

1. Prepare the existing FV integration grid once per evaluator.
2. Evaluate face drift and demand over bounded blocks of times using NumPy
   broadcasting and the existing BM model functions.
3. Assemble the same nearest-neighbor transition rates and use SciPy LAPACK
   `dgtsv` instead of scalar Python Thomas elimination.
4. During optimizer calls, accumulate only the needed demand moments instead
   of constructing density histories, state moments and duplicate demand
   output that the optimizer discards.
5. Return a full requested cell-mass history only when `.evaluate(prices)`
   is explicitly called. `.evaluate_cost(prices)` returns cost/diagnostics;
   `__call__` uses this cost-only route.

The state evolution is still sequential in time. Only coefficient evaluation
is batched; no time steps are solved independently.

## Mathematical equivalence and safeguards

The existing generator and implicit update are unchanged mathematically:

$$m^{n+1}=(I-h_nQ(t_n,U))^{-1}m^n.$$

The fast path uses the same face drift, center diffusion, zero-flux boundaries,
left coefficient times, step widths, midpoint cost weights, and

$$J=\sum_k w_k\left[(\mathbb E[D_k]-A_k)^2+\operatorname{Var}(D_k)\right].$$

No coarser grid, larger step, lower precision, fewer starts/paths, relaxed
optimizer tolerance, altered random seed, clipping or renormalization is
introduced. LAPACK error codes, nonfinite coefficients, materially negative
mass and excessive probability-mass drift cause explicit failures. The
in-place LAPACK RHS belongs to the evaluator, not to the caller's initial law.

LAPACK can pivot, so small floating-point differences from the original
Thomas solver are possible. Regression comparisons use tight numerical
tolerances rather than a universal bitwise-equality requirement.

Callbacks must be deterministic pure functions. The fast broadcast mode
requires the callbacks to support NumPy time/state broadcasting. Use
`vectorized=False` for the scalar-time callback convention. The evaluator's
model and discretization are fixed over its lifetime; create a new evaluator
if grids/settings change. Time-dependent baselines are evaluated at every
original time, not frozen to one value per price interval.

The shared FV solver, model equations, validators, active-set solver, package
initializers and WM sources are unchanged. The old reference remains the
default. No new library dependency beyond the existing SciPy is needed.

## Memory and longer horizons

A separate seven-day **forward cost evaluation** with 168 hourly price
intervals and 1001 cells measured peak traced allocations after setup:

| Evaluator | Peak traced allocations | J |
|---|---:|---:|
| Reference | 23.4567 MiB | 0.1469202494680164 |
| Fast, cost-only | 9.8477 MiB | 0.1469202494680164 |

This is `tracemalloc` allocation peak, not whole-process RSS, and it excludes
imports/setup. It is not a seven-day optimization benchmark. Cost-only mode
avoids storing every internal density, but still stores time grids/moment
vectors and bounded coefficient blocks. A small problem may use more
temporary block storage than a scalar loop; no universal memory factor is
claimed. Real operational datasets have not been benchmarked yet.

## Verification performed

- Baseline main plus the latest complex-target tests: **113/113 passed**.
- New fast-FV regression module: **13/13 passed**.
- Full local suite: **126/126 passed**, 33.699 s test time.
- GitHub Python 3.12 integration run: **126/126 passed**, 30.276 s test time.
- Full complex-target 2/4/6/12/24 fast experiment, final 20,000-path MC and
  plotting completed successfully; local process wall time 85.274 s.
- Full paired 24-variable optimization and fixed-price equivalence results
  are given above. Existing unrelated full reference executables were not
  all rerun in this performance task.

New tests cover multiple grids/noise strengths, random and extreme prices,
time-varying baseline, zero flexibility, small capacity, batch-size
independence, scalar-time callbacks, non-mutation of initial masses, injected
LAPACK failures, injected mass loss, and a complete small Powell comparison.

## Local commands

Existing commands retain the original evaluator. Opt in explicitly:

```powershell
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.optimize complex --fast-fv
python -B -m src.exe.run_model.bm_1.optimize multi --fast-fv
```

Monte Carlo settings are unchanged. The readability revision makes plots opt-in
with `--save` and writes them under ignored `files/output/bm_1/runs/graphs/`.

Run the short repeated local benchmark:

```powershell
python -B -m src.exe.run_model.bm_1.verify benchmark
```

Add `--optimize` to repeat the slower complete original-versus-fast
24-variable Powell experiment. Add `--save` to write the timing report to
`files/output/bm_1/runs/fast_fv_benchmark.json`.

The complex-target script/tests were already written on the experiment branch
but were not yet in the reviewed main. They are carried into this branch;
only the opt-in evaluator flag changes the experiment itself.

## Further opportunities, not implemented here

**Large price vectors:** measured speed now comes from cheaper objective
calls, not fewer calls. An analytic/discrete-adjoint gradient with a suitable
bounded optimizer is the next algorithmic candidate. For fixed time/price
edges, differentiating the existing step gives

$$(I-h_nQ_n)s_{n+1}=s_n+h_n(\partial_uQ_n)m_{n+1}.$$

A gradient implementation must also differentiate demand in the cost and
handle upwind/demand branch changes correctly. It must be checked against
finite differences before replacing Powell. No additional speed factor or
global-optimality claim is made.

**Mixture QP:** a BM-only simplex-specialized solver, carrying the active set
or reusing reduced factorizations, targets the measured repeated-rank/KKT
work. High-degree conditioning makes an unverified generic solver swap risky.

**Long-horizon MC:** simulate independent path cohorts and accumulate pathwise
costs/moments, rather than retaining all state/demand/increment arrays. Preserve
the common-random-number coupling between candidate policies. This remains
separate from the implemented FV memory reduction.

**Parallelism:** independent starts, scenarios or path cohorts may be run in
parallel after measuring process overhead and controlling BLAS threading.
Successive time steps or days cannot simply be treated as independent when
the battery state/law carries across their boundaries.

The old FV/MC discretization discrepancy, local nature of Powell, and target
model assumptions remain; making an evaluation faster does not resolve them.

## External API references

The implementation uses the official SciPy LAPACK `dgtsv` interface:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgtsv.html

Gradient-based optimizers accept an explicit Jacobian through SciPy minimize:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

All speed/memory values above are this project's measured experiments, not
performance claims taken from these external documents.


## Post-F3-F audit

The later, separately tested F1-A–F3-F audit is documented in
`BM_AUDIT_F1_F3.md`. It addresses the dynamic-mixture bottleneck identified
above, preserves the shared solvers, and includes fixed-input before/after
benchmarks plus numerical and full-control regressions. The earlier timings
above are retained as historical evidence, not overwritten with new timings.

# BM F1/F2 consolidation audit

Reviewed baseline: `0746b60b8907d0b2cf9dffe710575b6f24eb092f` (F2-I).
Review date: 27 September 2026.

## Conclusion and scope

The controlled F1/F2 workflow still runs after consolidation. The review found
real documentation/architecture problems and several input-safety gaps, but
no change to the model equations was needed to reproduce the established
reference results. This is not a proof of universal numerical accuracy or
of global optimality of the stochastic prices.

The hard boundary is **no modification of files imported by WM**, directly
or through package initializers. The source import audit found 60 files in
that transitive closure, including `src/model/bm_1.py`,
`src/diagnostics/bm_1.py`, common solvers/validators and package `__init__.py`
files. All 60 are byte-for-byte unchanged. A separate check of all 22
WM-named Python sources also found no changes. The WM data fitting and
prediction applications were not rerun; they are outside this BM review.

## Documentation consolidation

- `F1.md` now explains the deterministic model, exact benchmark, ODE
  breakpoints, price parametrization, objective and feasibility certificates.
- `F2.md` is the single current A–I account, including limitations and code
  ownership. It distinguishes mathematical assumptions, implementation
  choices and the evidence supplied by the experiments.
- Old `F2_A.md` through `F2_I.md` and `F2_B_convergence.md` remain as short
  links to stable anchors in `F2.md`. Original text remains in Git history.

The old D/E/F/G Markdown contained missing backslashes and control characters
from improperly escaped LaTeX strings. The consolidated equations correct
those formatting errors. No existing Overleaf file is modified.

## Mathematics: checked and clarified

The review checked the implemented demand/drift signs and logistic factor,
constant-input arcsinh solution, inverse-price and capacity bounds, SDE units,
Brownian scaling, conservative FV rates, stochastic cost decomposition and
paired Monte Carlo comparison. The main formulas agree with their stated
model assumptions. Important qualifications are now explicit:

1. **F1 quadrature.** Restarting the ODE at a price jump does not make global
   Simpson integration over sampled costs jump-aware. The historical F1
   target also interpolates sampled demand across its jump. Its near-zero
   discrete benchmark is not an exact continuous-time tracking certificate.
   The shared APIs remain unchanged; a BM-only opt-in midpoint evaluator
   handles explicitly supplied discontinuities without changing old results.
2. **Output and projection grids.** EM/FV/mixture output times are inserted
   into integration grids. A different output grid can change the numerical
   steps and mixture projection schedule. Full independence is not claimed.
3. **Mixture error interpretation.** Dynamic mixture uses explicit Euler and
   averaged-center drift at faces; pure FV uses implicit Euler and direct
   face drift. Their gap therefore includes more than repeated projection.
4. **Beta representation.** The initial law uses CDF cell masses, but the
   basis still uses normalized center samples. The continuous variance-floor
   formula is not an exact formula for those discrete moments. No adaptive
   or CDF-integrated basis is silently substituted.
5. **Conditioning.** A fixed basis/grid gives a fixed full Hessian, independent
   of the target density. The target changes the active set. The normal
   matrix can lose small singular directions in floating point; a measured
   condition number near 1e17 does not accurately resolve a true value
   potentially near 1e22. Sensitivity is not mathematical non-uniqueness.
6. **Stochastic objective and optimization.** The target in the decomposition
   is deterministic. Zero process noise does not remove initial uncertainty.
   Powell multistart provides a numerical candidate, not a global optimum
   proof. The original I benchmark changes both initial and process
   uncertainty relative to the deterministic reference. Its paired MC gain
   supports a better decision, while its roughly 4% FV–MC absolute-cost
   difference remains a numerical-accuracy limitation.

The full derivations and conventions are in `F1.md` and `F2.md`. Statements
above are review findings, not claims that all limitations were removed.

## Code changes

### Reusable code has one owner

`src/method/bm_1_beta_mixture.py` owns the projected-mixture stepping code,
extracted from the F2-F executable without changing its numerical recursion.

`DynamicMixturePriceEvaluator` now lives in
`src/method/bm_1_stochastic_optimization.py`, not in the I executable.
The repeated demand-grid evaluation is shared through
`src/method/bm_1_stochastic_objective.py`.

Old executable helper imports remain aliases, preserving the existing tests
and command-line entry points. The new method modules do not import
executables and are not added to shared package initializers.

### Safe reductions in repeated work

Fixed weighted projection matrices and Hessian conditioning are computed
once per mixture evaluator, rather than once per candidate price vector.
Bounded multistart Powell memoizes only **exactly repeated** price vectors
inside one call. There is no rounding, no persisted cross-problem cache and
no change to optimization stopping tolerances. `cache=False` remains available.

The I executable now prints stage, start and evaluation progress immediately.
It also prints the spread of successful-start objective values and the
number of actual evaluations/cache hits.

### Safety and small correctness changes

- EM/FV/convergence/legacy demo modules no longer simulate or save figures
  merely because they are imported; execution remains under `main()`.
- Evaluators copy caller-owned grid and initial-mass arrays.
- The FV price evaluator accepts additional known target/baseline
  discontinuities through `breakpoints`, in addition to price edges.
- Fractional Monte Carlo path counts and fractional mixture-evaluator degrees
  are rejected rather than silently truncated; Beta cell edges must span
  the actual unit interval.
- Dynamic output histories are initialized with NaNs, so missed writes cannot
  look like valid numerical results.
- Raw mass conservation and projected probability constraints are checked
  inside the reusable stepping routine, including optimizer calls, not only
  at the end of the demonstration executable. No density repair is added.
- The H cumulative-cost plot places an interval's accumulated cost at its
  right edge rather than its midpoint. Calculated costs are unchanged.

`deterministic_midpoint_objective` is an explicit optional alternative for
jump-aware deterministic cost evaluation. It does not replace the shared
F1 `tracking_error` or `model.optimize` functions.

## What was deliberately not changed

No global `fokker_planc` rename, solver replacement, lower resolution, fewer
MC paths, altered random seeds or relaxed tolerances. The independent exact
formulas in F1 regression tests are kept independent. Shared historical
FV/Beta code and all WM-imported files are untouched.

Tracked Python cache files and the existing WM2 invalid-escape docstring
warning remain outside this narrowly scoped change. Global cache cleanup
would be a separate repository-wide housekeeping decision, not a silent BM
change. Full WM fitting/data workflows are not covered by the test claims.

## Verification actually performed

### Regression tests

- Unmodified source: **91/91** tests pass, both in a fresh GitHub Python 3.12
  environment and locally.
- Refactored source: **104/104** tests pass locally; the 13 additional tests
  cover import safety, compatibility aliases, layer direction, array
  ownership, validation, setup reuse, exact caching, raw-mass failures and
  jump-safe deterministic quadrature.
- GitHub runs the same full suite on Python 3.12 using the read-only BM review
  workflow. It explicitly uses Bash with `pipefail`, so piping logs through
  `tee` cannot turn a failed test command into a successful CI result.

The first cross-environment run exposed a portability issue in the **new
import-safety test**: mocking `Figure.savefig` before pyplot initialized
broke pyplot's docstring setup. The test now initializes pyplot first. This
was a test-harness issue, not a changed BM simulation, and the final CI result
is the relevant gate.

### Full reference runs

All 14 executables below were run to completion on both baseline and refactor,
using the same environment, seeds, numerical defaults and single-thread BLAS.
Every process returned exit code zero. After excluding paths and new progress
messages, every pre-existing printed numerical result matched.

| Executable | Before [s] | After [s] |
|---|---:|---:|
| deterministic_reference | 2.237 | 2.151 |
| optimization_reference | 2.416 | 2.328 |
| optimization_feasibility_reference | 8.977 | 9.752 |
| em_reference | 2.347 | 2.316 |
| fp_reference | 2.042 | 2.126 |
| em_convergence | 3.020 | 3.008 |
| mc_vs_fp_reference | 2.581 | 2.602 |
| beta_projection_reference | 3.426 | 3.476 |
| beta_dynamic_reference | 16.504 | 16.806 |
| beta_conditioning_reference | 21.545 | 20.484 |
| stochastic_objective_reference | 20.762 | 20.249 |
| em_price_step | 3.588 | 3.532 |
| fp_price_step | 4.561 | 4.102 |
| stochastic_optimization_reference | 201.707 | 197.848 |

These are single-run wall times, not statistically controlled performance
benchmarks. Small increases also occur in unchanged code. They do not support
a claim of a precise speedup or a universal no-slowdown guarantee. The full I
run was slightly faster and its FV multistart avoided 25 exact repeated
solves (737 actual evaluations). No cheaper approximation was introduced.

Local comparison environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8, Linux, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MPLBACKEND=Agg`, `PYTHONDONTWRITEBYTECODE=1`.

### Full-precision output comparisons

A separate before/after snapshot comparison checked **42 numerical arrays**:
ODE and EM outputs, FV densities/moments, M=20/30/38 dynamic densities,
coefficients and moments, small H objective/path costs, and fixed-price FV
and mixture evaluations. All 42 matched **bit-for-bit** (`array_equal`);
maximum absolute difference was exactly zero. This is stronger than comparing
only rounded terminal output, but still concerns the tested cases.

Source parsing, absence of control characters in the consolidated notes,
compatibility links and the WM source boundary were also checked.

## How to use the reviewed code

The existing commands remain valid. From the repository root:

```powershell
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.stochastic_optimization_reference
```

The full optimization now reports progress. The branch is a reviewable
refactor, not another F2 model extension; merge only after reviewing its diff.

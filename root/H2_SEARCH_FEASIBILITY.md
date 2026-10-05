# F4-H2: bounded inexpensive interval search

Declared 6 October 2026 before computation. This belongs to stochastic F;
it neither changes nor uses the deterministic G fitting strategy.

## Fixed first-hour trial

Use the existing synthetic hourly development observations, physical
parameters, hourly held B/U, Beta(2,3) prior and sigma_y=.04. Evaluate exactly
three shared monotone curve pairs with endpoints 1,-1 and one inner knot:

| Candidate | f(.5) | g(.5) |
|---|---:|---:|
| linear | 0 | 0 |
| two_saved | .30 | -.25 |
| two_near | .32 | -.23 |

Each candidate starts from its own discretization of the same prior. No
posterior from a different candidate or a later hour is reused. The trial
is not optimization, truth recovery, historical fitting or H3 validation.

The inexpensive path is the existing positive joint X/accumulator method,
not an endpoint observation or Gaussian at a mean trajectory. Compare two
already investigated discretizations: 161 cells, h=1/768 day, da=1/256;
321 cells, h=1/1536, da=1/512. Both use the fitted FV generator and a 4096
accumulator-node limit. Their earlier absolute likelihood failures remain
valid; this trial investigates candidate differences, not a relabelled pass.

The reference uses the unchanged terminal-wise tilt kernel at 1921 cells,
h=1/24576, 64 frequencies and period 2. Reuse the saved two_saved first hour
from the archived PR #81 day after its exact raw/input/source validation.
Only the two other reference hours are new. Existing guards 1e-8 and 1e-7
remain unchanged. The 2561/1921 day comparison supports the saved reference
on its own fixture; it is not a continuum error certificate for new curves.

For every search discretization report absolute score errors, all signed
pairwise score differences and their errors, ranking, conservatively mapped
posterior L1, state means, diagnostics, complete marginal states and runtime.
Local search feasibility requires score-difference errors <=.05, state-mean
errors <=.002, posterior L1 <=.05 and unchanged ranking for every pair.
Absolute score errors use the same .05 threshold and are reported separately.
Gaps <=.05 are explicitly marked unresolved at that tolerance. Even a pass
does not establish ranking for a training stream or throughout curve space.

The complete trial has a cumulative 900-second limit, with 870 seconds for
compute admission and 30 seconds reserved for saving/stopping. New units run
in subprocesses with a timeout set to the remaining total budget; a timeout
kills the unfinished unit and preserves completed units. Resume requires an
identical source/input/design/environment manifest. An interrupted unfinished
invocation conservatively charges intervening wall time instead of resetting
the budget. All remaining H computation targets 4–6 hours with a 10-hour ceiling.

## Execution and necessary checks

```bash
python -B -m src.exe.run_model.bm_1_interval_search_pilot \
  --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json \
  --cn-reference files/output/bm_1/runs/estimation/h2_interval_positive_cn_control_20261005.json \
  --terminal-reference files/output/bm_1/runs/estimation/f4_h2_terminal_tilt_day_20261005.json.gz \
  --run-dir files/output/bm_1/runs/estimation/f4_h2_search_feasibility_20261006
```

Freeze and commit source before running. Preserve report.json, manifest.json,
budget.json and unit JSON files. Source/input checks, the actual trial and its
raw comparisons cover the new orchestration. Parse the new Python file; use
the existing saved-reference integrity test. Numerical kernels are unchanged;
no full suite, old day replay, Windows or LaTeX job is needed.

If the trial fails, keep its negative evidence and stop this search path.
If it passes locally, the next bounded step must examine a recursive stream
and its timing before integrating the historical fitter. No historical fit
is admitted solely on a single-hour ranking. A new method decision is required
if candidate differences or recursive states cannot be trusted within budget.

## First-hour result and declared recursive continuation

The first-hour trial completed on clean local commit `6cafe95` in 63.24 seconds,
including worker startup and orchestration. The 321-cell method passed all
declared local gates: maximum pairwise score-difference error .0062424,
maximum state-mean error .0007823, maximum posterior L1 .0372581. Its median
kernel-only reference/search time ratio was 105.63 on these three hours.
This is not a whole-stream or historical-fit speedup. The 161-cell method
failed posterior L1 (.06899–.07643), even though ranking and score differences
passed. No numerical gate was loosened and no search result became accepted ML.

Before historical integration, declare one recursive prefix of eight hours
at joint321 for the identical three candidates. It includes hourly baseline
changes, the first price change at hour 6 and the missing observation at
index 7. Carry each method/candidate's own state; reset only the accumulator.
Reuse all six matching first-hour joint321/terminal1921 results and the saved
two_saved reference prefix. Compute only hours 1–7 for the two other precise
candidates and the inexpensive continuations. Check every completed prefix's
score differences, ranking and posterior with the unchanged .05/.002/.05
gates. Stop at the first failed prefix, numerical guard or remaining budget.
The first-hour trial and this continuation share the same 900-second ceiling,
including process startup, with 30 seconds reserved for saving/stopping.
No subsequent fit or larger grid panel is admitted by this continuation.

```bash
python -B -m src.exe.run_model.bm_1_interval_search_stream \
  --pilot-dir files/output/bm_1/runs/estimation/f4_h2_search_feasibility_20261006 \
  --run-dir files/output/bm_1/runs/estimation/f4_h2_search_stream_20261006
```

The real recursive check covers the new continuation wiring. Existing
independent numerical references are reused, rather than repeating them.

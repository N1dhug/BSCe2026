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

## Recursive negative result and next numerical decision

The continuation stopped exactly at the first failed prefix, hour six.
Combined first-hour/recursive charge: 387.36 seconds (6.46 minutes).
The largest score-difference error across all six prefixes was .0341578,
with unchanged ranking. At the failing endpoint, two_near posterior L1
was .0503623 > .05; means still passed. Maximum absolute score error
across the prefixes was .0841519. Thus joint321 with implicit Euler does
not pass this declared recursive feasibility gate. Hours 6–7 were not run;
the price change and missing-observation coverage remain open.

An inexpensive method decision follows this failure, rather than a larger
grid panel: align only the joint method's state step with the positive
Crank–Nicolson already used by the precise reference. Keep cells=321,
h=1/1536, da=1/512, all three curves, noise and all gates unchanged.
An isolated `PositiveCNJointSolver` subclass reuses the existing fitted
generator and cached LAPACK implicit half-step. Production solvers and
historical fitting remain unchanged. No new precise interval is computed.

For column generator Q, define H=I+hQ/2 and M=I-hQ/2. If every diagonal
entry of H is nonnegative, H is nonnegative and column-conservative; M is
an M-matrix whose inverse is nonnegative and column-conservative. The
update M^-1 H therefore preserves nonnegative joint columns and their
mass. Reject an invalid positivity step, without clipping or adaptive
refinement. This is a time-discretization experiment, not a new observation
law or a proof of the full split/deposition method's accuracy.

Declare exactly three fresh six-hour joint-CN prefixes, from each curve's
original prior. Compare every prefix with all the already computed precise
hours, using the unchanged score-difference, ranking and posterior gates.
Cap this new cheap experiment at 60 seconds inside the original cumulative
900-second ceiling. One independent constant-reward Markov semigroup test
checks second-order state convergence, mass, nonnegativity and the rejection
of a negative explicit diagonal. No reference day, optimizer or broad
parameter panel is repeated. Both old failures remain archived.

```bash
python -B -m src.exe.run_model.bm_1_interval_joint_cn \
  --stream-dir files/output/bm_1/runs/estimation/f4_h2_search_stream_20261006 \
  --run-dir files/output/bm_1/runs/estimation/f4_h2_joint_cn_20261006
```

The CN experiment completed in 6.82 additional seconds but did not solve
the recursive posterior issue: hour-six maximum L1=.0503307. The original
implicit result was .0503623. Thus changing this state time scheme has little
effect on the failing quantity; it does not by itself justify acceptance.

Declare one final inexpensive accumulator-only diagnostic, not a new state
grid or precise panel: same CN prototype, cells=321, h=1/1536, all three
fresh six-hour prefixes and all saved precise results, but da=1/1024 rather
than 1/512. Positive nodal deposition adds dispersion, and the unchanged
diagnostic bound scales with da; this hypothesis can be checked without a
new precise replay. Require the same gates and record absolute score errors
too. Cap at another 60 seconds, charging both earlier stages and the CN
experiment to the same 900-second total. If this fails, stop numerical
experimentation and produce a bounded mathematical handoff. A local pass
still requires representative stream/runtime and selected-fit checks.

```bash
python -B -m src.exe.run_model.bm_1_interval_joint_cn \
  --stream-dir files/output/bm_1/runs/estimation/f4_h2_search_stream_20261006 \
  --run-dir files/output/bm_1/runs/estimation/f4_h2_joint_cn_a1024_20261006 \
  --spacing 0.0009765625
```

The accumulator-refined prefixes passed every declared gate for all three
curves at all six endpoints. Maximum candidate-difference error .00537955,
absolute score error .02705772, mean error .00013638 and posterior L1
.04978562. New six-hour replays took 2.457/3.633/2.971 seconds. Combined
charged computation is 405.93 seconds. The result supports only these fixed
prefixes; some reference candidate gaps are smaller than .05, so it does
not certify statistically distinct choices or a general ranking margin.

Declare one final cheap daily continuation of two_saved at these same
settings. Reuse its six already computed inexpensive endpoint rows and
continue the remaining 18 hours, including both missing observations and
price changes. Compare every endpoint and cumulative absolute score with
the saved 1921-cell day, retaining .05/.002/.05. No new precise day, seed,
optimizer or other candidate is introduced. Cap fresh work at 60 seconds
inside the same cumulative 900-second limit. This tests duration/runtime
on the one fixture and cannot admit historical fitting by itself.

```bash
python -B -m src.exe.run_model.bm_1_interval_joint_cn_day \
  --prefix-dir files/output/bm_1/runs/estimation/f4_h2_joint_cn_a1024_20261006 \
  --run-dir files/output/bm_1/runs/estimation/f4_h2_joint_cn_day_20261006
```

## Final daily result and revised scope

The cheap day continued from the six preserved endpoint rows and computed
18 new intervals in 9.028 seconds. It handled both missing observations and
hourly input changes. The first failed prefix is hour 7, coincident with the
first price change; that coincidence does not identify the cause. All 24
hours completed, so the report retains the complete negative evidence.

| Quantity | Error | Unchanged gate |
|---|---:|---:|
| Final cumulative score | 3.58901219 | .05 |
| Maximum endpoint state mean over the day | .00822776 | .002 |
| Maximum endpoint posterior L1 over the day | .26784242 | .05 |
| Final state mean | .00069221 | .002 |
| Final posterior L1 | .03133849 | .05 |

The final mean/L1 happen to pass while important intermediate states do not.
Do not infer recursive validity from the terminal state alone. Neither the
first-hour nor the six-hour local pass admits this surrogate for historical
search. Numerical/historical fit readiness and H2/H3 acceptance remain false.
Production numerical kernels, historical fitter and G were not modified.

Combined charged experimental time is 415.89 seconds (6.93 minutes), well
below 900 seconds. Two focused tests passed: archived-reference integrity and
the independent CN semigroup/positivity control. Budget a separate conservative
10-second allowance for these brief checks and the raw arithmetic audit; the
remaining-H ledger therefore reserves 425.89 seconds for this task. All new
Python files were parsed. Prefix score arithmetic was independently recomputed
from raw rows without FV. No full suite, Windows, LaTeX, original reference day
or historical fit was run. Runtime ratios above are local kernel observations,
including one archived comparison, not controlled whole-fit speedups.

Preserved reports in `files/output/bm_1/runs/estimation/` are lossless gzip
copies of each actual result: `f4_h2_search_feasibility_20261006.json.gz`,
`f4_h2_search_stream_20261006.json.gz`, `f4_h2_joint_cn_20261006.json.gz`,
`f4_h2_joint_cn_a1024_20261006.json.gz` and
`f4_h2_joint_cn_day_20261006.json.gz`. They contain their frozen designs,
source hashes, observations/inputs references, numerical diagnostics and
marginal states. Frozen local source snapshots are included separately;
local commit names in reports must not be mistaken for remote commit names.

**Windows update, 6 October:** the later synthetic eight-worker pilot completed
in 556.53 s. Its conservative 17-candidate projection is 5.50 hours for
24 synthetic hours, but 20.65 hours when extrapolated to 168 hours, including
spent time and a provisional two-hour reserve. The old table below is retained
as the earlier planning illustration and does not support run admission.
See [H2_PRECISE_BATCH.md](H2_PRECISE_BATCH.md). Historical throughput and the
actual F TRAIN/validation contract remain unresolved.

The earlier proposed route reduces the number of precise candidate evaluations
and parallelizes independent whole-stream replays. It does not assume a
surrogate is now accepted. Based on the saved synthetic 1921-cell rate only:

| Seven TRAIN days, eight workers | Equal-cost batch waves | Search hours | With a provisional 2-hour reserve |
|---|---:|---:|---:|
| 17 candidates | 3 | 3.68 | 5.68 |
| 25 candidates | 4 | 4.91 | 6.91 |
| 73 candidates | 10 | 12.27 | 14.27 |

These are planning illustrations, not approved runs or actual historical
measurements. Adaptive optimizer dependencies and candidate costs can worsen
them; a seven-day TRAIN is not silently selected. The reserved two hours
must be checked against the real precision/validation protocol, and the
already used 6.93 minutes must also be counted. A short representative
parallel pilot and frozen data contract precede any fit. See the concrete
[next-task handoff](H2_SEARCH_NEXT_HANDOFF.md). There is no demonstrated
historical H completion time yet, and no new lengthy run has been started.

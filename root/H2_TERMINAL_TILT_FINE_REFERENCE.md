# F4-H2: one 2561-cell reference

5 October 2026, declared before the run. This is stochastic F's continuation
of [the completed cross control](H2_TERMINAL_TILT_CROSS_CONTROL.md), independent
of G. The cross control isolated a .00216009 time change and a .08538749
spatial change; only the latter failed the unchanged .05 score gate.

## Fixed design and acceptance scope

Run exactly one 24-hour reference at 2561 cells, h=1/49152 day, 64 frequencies,
period 2 and the existing fixed terminal-center bank. Reuse the saved 1921-cell
PR #81 day at h=1/24576; no old interval is recomputed. The fixture, shared
two-piece f/g, sigma_y=.04, physical parameters, Beta(2,3) prior, observation
stream and original two missing intervals are unchanged. No historical
observations, fit, parameter search or new seed are introduced.

The generator-only audit already showed that 2561 cells at h=1/24576 violates
positive CN (minimum explicit diagonal -.337639), whereas the smaller declared
h has margin .331181. Positivity requires `1+(h/2)*Q_jj >= 0`. This justifies
the explicitly changed step; it is not silent adaptation.

Compare total log likelihood, final state mean and conservatively transferred
terminal mass with the saved reference, using exactly .05/.002/.05. This is
a **joint time/state refinement**. The 1281-cell time control does not certify
time error at 2561 cells. Passing this one comparison is local development
evidence, not a continuous-model error bound, a convergence-order claim,
universal parameter-domain coverage, historical-backend adoption or H2/H3
acceptance. Numerical fit readiness stays false.

Estimated work is about 28 minutes from the saved workload. Each invocation
has <=900 s admission budget, checked between intervals; an admitted interval
finishes. Actual log-state checkpoints are atomic after each hour. Resume
requires unchanged source, all three inputs and environment; completed hours
are reused. If an invocation pauses, preserve a compressed checkpoint on
GitHub before continuation, without changing local frozen source/HEAD.

## Implementation and minimum checks

`bm_1_interval_terminal_tilt --stage fine` uses the existing runner and archived
reference validator. The exact report/input fingerprints, 35 relevant BM
source files and byte-identical terminal-update prefix are checked. Unrelated
G/WM changes remain outside F equivalence. Only stage declaration and final
comparison orchestration change; the numerical kernels are untouched.

Parse the two changed Python files and verify the unchanged kernel prefix.
Use the actual full replay/checkpoint/resume as the stage integration check.
Reuse the already passed exact/positive/recursive-tail and reference-integrity
tests; no new mirrored unit case, old replay panel, full suite, Windows run or
LaTeX build is required for this declaration-only extension.

```bash
python -B -m src.exe.run_model.bm_1_interval_terminal_tilt --stage fine \
  --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json \
  --cn-reference files/output/bm_1/runs/estimation/h2_interval_positive_cn_control_20261005.json \
  --terminal-reference files/output/bm_1/runs/estimation/f4_h2_terminal_tilt_day_20261005.json.gz \
  --output files/output/bm_1/runs/estimation/f4_h2_terminal_tilt_fine_20261005.json
```

At a budget pause (exit 3), repeat the same command with `--resume` and identical
frozen source/inputs. Completed evidence is never overwritten. A completed
negative precision result returns exit 2 and is preserved. No broad follow-up
panel follows automatically. Results and the bounded next decision will be
appended after the frozen run.

## Completed result

The complete 24-hour replay passed on 5 October 2026. The original missing
hours, indices 7 and 19, propagated the state without a score contribution.
All 22 observed updates resolved every reachable terminal cell. No floor,
clipping, historical observation or parameter fit was introduced.

| Quantity | Saved C: 1921 cells, h=1/24576 | New D: 2561 cells, h=1/49152 |
|---|---:|---:|
| Total log likelihood | -95.37892393456096 | -95.40947855361053 |
| Replay runtime, seconds | 630.8714348370 | 1503.2969252920 |

| C/D comparison | Absolute difference | Unchanged gate | Result |
|---|---:|---:|---|
| Total log likelihood | .030554619049567577 | .05 | pass |
| Final state mean | .000013957999987379566 | .002 | pass |
| Conservatively transferred final mass L1 | .0011751598181428788 | .05 | pass |

The signed score change D-minus-C is -.030554619049567577. The maximum
state-mean difference over all 24 endpoints is .00012120064806198094; this
is descriptive and does not replace the declared final-mean gate.
Maximum conditional guard is 2.1066210961633687e-9 (<1e-8), maximum resolved
contour overlap log discrepancy 6.75709088682197e-9 (<1e-7), and maximum
absolute mass error 2.282618538629322e-13. Finite log masses down to
-6534.720975137246 were retained and used by subsequent updates.

This passes one joint state/time comparison on one fixed development fixture.
It does not isolate state error at 2561 cells, establish a continuum error
bound or convergence order, validate a parameter domain, or overturn the
older A/C and B/C failures. C can serve as the less costly local development
reference supported by this comparison. It is not a historical-fit default.
`numerical_fit_readiness`, `scientific_acceptance` and
`rigorous_roundoff_certificate` remain false; historical values read: zero.

## Preserved source, checkpoint and raw evidence

The frozen local source commit was
`ed15959416bca150a5b56845067dab07cccb0542`, with `dirty=false`. The same
source was archived before the run in GitHub commit
`a8343e735dce05eb82f7fea92309b5e0290b8c8b`. The 35 relevant BM dependencies
and terminal-update prefix matched the saved PR #81 reference. They also
matched current main before integration; unrelated WM/G changes were retained.

The first <=900 s invocation admitted and finished 15 hours, then returned
exit 3. Their replay work was 940.9759518859719 s; the budget is checked
between hours, so the last admitted hour may finish after 900 s. The exact
checkpoint was compressed and archived in GitHub commit
`f560dcb59b7ddae50c5030672664db67b8d66324` before continuation. The identical
command with `--resume` completed the last nine hours and returned exit 0;
that invocation took 563.1405922129998 s. Total replay work was 1503.30 s
(25.1 minutes), rather than the estimated 28 minutes. Archive/idle overhead
is not included in replay runtime.

The final report records `resumed_intervals=15`, one old replay reused and
zero old reference intervals recomputed. The first 15 final rows have the
same digest as the archived checkpoint, including their actual log states:
`37c700197053d9bc1fb5f7c0869eec0366bb0d675cd8867a78530e85647aa4ce`.

Artifacts in `files/output/bm_1/runs/estimation/`:

- `f4_h2_terminal_tilt_fine_20261005.json.gz`: lossless full raw report,
  including all log states, inputs, source/environment and diagnostics.
  Uncompressed SHA256:
  `7f54f58f3e16c3030a3f8274b840bda3416ecb884c939e6bf735e6dd7960034a`.
- `f4_h2_terminal_tilt_fine_20261005.partial_001.checkpoint.json.gz`:
  preserved 15-hour checkpoint. Uncompressed SHA256:
  `36e12acd586ef8e7af956868436369761b281d83ccdd07562c11cef1b8ebe0e8`.

Actual checks: the two changed Python files were parsed before the run;
the existing CI syntax block also found and parsed both. Reference/kernel
equivalence and the complete real checkpoint/resume prefix were verified.
No numerical kernel changed and no full suite, old replay panel, Windows run
or LaTeX build was repeated. Existing independent reference tests are reused.

## Next bounded runtime control (not yet run)

Even C takes 10.5 minutes per day, too costly for routine repeated pooled
historical fitting. First test one representation reduction on one paired
hour, not a new grid/day panel or fit. Use C's saved prior immediately before
hour index 6, the observed hour with its largest conditional guard
(2.0960332675534518e-9). Keep 1921 cells, h=1/24576, the same physical inputs,
f/g, noise, centers and log prior. Compare the existing K=64, L=2 with K=48,
L=1.5 in the same process; cap the complete pair at a declared 120 s admission
budget, retaining failure rather than adapting settings. No midnight reset.

For each declared tilt, z=y-sigma_y^2*alpha=center, so distance from the
reward range [0,1] is at most 1. At L=1.5 the minimum alias gap is .5.
With sigma_y=.04 the existing worst-case analytic alias bound is
2.34760e-33 and truncation bound 4.09899e-15 (versus 5.04671e-15 at 64/2).
The existing roundoff allowance is approximately 1.41733e-13 for both; it
remains heuristic. The Gaussian Fourier cutoff K/L stays 32. This justifies
testing the same likelihood representation with 49 rather than 65 complex
columns; it does not promise a wall-time gain or universal tilt coverage.

Preserve all conditional/overlap guards and .05/.002/.05 gates. Also compare
the paired finite terminal log-joints against the existing 1e-7 overlap
tolerance and record both score/posterior differences and measured runtimes.
Reuse existing exact/positive reference evidence; do not add mirrored tests.
One paired hour is development evidence, not a whole-day speedup or backend
acceptance. If it fails, save the failure and reconsider rather than launch
a panel. Even a useful reduction leaves actual time variation/breakpoints,
historical log-state integration and data/split/exposure/baseline choices open.
Sol is sufficient for this fixed representation/runtime control. A new
discretization or likelihood method would warrant a separate mathematical
review; the completed Gaussian cancellation review need not be repeated.

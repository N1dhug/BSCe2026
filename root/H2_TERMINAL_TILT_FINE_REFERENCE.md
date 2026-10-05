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

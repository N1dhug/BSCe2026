# H2 interval characteristic controls

Updated 5 October 2026. Positive CN passes the original linear and new
four-piece/wide-noise fixtures. The two-piece/narrow-noise fixture fails all
five declared replays because conditional Fourier densities cancel.
Historical fitting remains blocked. The historical fitter still uses the
conservative joint state/accumulator implementation from PR #72.

## Representation and probability guards

The experiment evolves the terminal-state transform of the hourly average
with the fitted-FV state generator. Gaussian evidence uses a periodic Fourier
series and the exact exponential-tilting identity

\[
\phi_\sigma(y-A)=e^{-(y-c)^2/(2\sigma^2)}
 e^{(y-c)(A-c)/\sigma^2}\phi_\sigma(c-A).
\]

The tilt is s=(y-c)/sigma². A neutral state column checks ordinary probability.
Alias, truncation and heuristic roundoff bounds retain the relative guard 1e-8.
Negative Fourier densities still abort; no clipping, probability floor,
substituted objective or production fallback is introduced.

The original unscaled control failed in subnormal terminal-state tails.
The continuation stores each positive transform amplitude as a logarithm and
each Fourier column relative to that amplitude. Positive triangular recurrences
supply diagonal scales; real LAPACK then solves the same implicit equation
after diagonal similarity scaling. Its zero-frequency correction is retained.
This is an algebraic representation, not normalization of a prediction.
True disconnected zeros remain zero. Log posteriors continue across intervals;
ordinary arrays in development reports can show underflowed zeros. These
reports are not resumable production state checkpoints.

## Isolating the first-order time error

All six balanced implicit-Euler controls now complete. The saved
[balanced panel](../files/output/bm_1/runs/estimation/h2_interval_balanced_characteristic_control_20261005.json)
took 453.72 s. Its 1281/2561-cell total-score difference is 0.133850,
above the existing 0.05 limit. At 1281 cells, 64 versus 96 frequencies changes
the score by 1.78e-14. Stable transforms alone do not establish time accuracy.

Two fixed-state controls reuse those results instead of rerunning the panel.
The [time control](../files/output/bm_1/runs/estimation/h2_interval_balanced_time_control_20261005.json)
took 306.14 s:

| Comparison | Total log-likelihood difference | Development tolerances |
|---|---:|---|
| 1281 cells, h=1/6144 versus 1/12288 | 0.112179 | Fail |
| 1281 cells, h=1/12288 versus 1/24576 | 0.056413 | Fail |
| 1281/2561 cells at the same h=1/12288 | 0.021671 | Pass |

The last spatial reference uses 96 frequencies rather than 64; the independent
frequency difference above is recorded. These results identify time
integration as the dominant unresolved error on this fixed case.

## Positive Crank–Nicolson development panel

For the column-mass generator Q, the state step is

\[
S_h=(I-hQ/2)^{-1}(I+hQ/2).
\]

The explicit half is accepted only when every entry is nonnegative,
including 1+h Q_ii/2 >= 0. Its positive mixtures are evaluated in log space.
The inverse implicit half has the usual M-matrix positivity. Neutral state
prediction uses the same CN step. Oversized steps abort rather than adapting
the declared grid, clipping masses or repairing a posterior.

Frozen reward / CN state / frozen reward splitting is second order on this
registered case: B and U are constant within each hour. Left-frozen fields
with arbitrary time variation are not claimed second order. Historical
baseline/price adapters need their own time-field and integration controls.

The predeclared panel uses period 4 and 64 frequencies, without fitting:

| Cells | Maximum step (days) | Runtime (s) | Total log-likelihood |
|---:|---:|---:|---:|
| 1281 | 1/12288 | 113.95 | -22.646708 |
| 1281 | 1/24576 | 237.00 | -22.646790 |
| 1921 | 1/24576 | 350.12 | -22.662981 |

All three comparisons pass the existing limits: total score 0.05, terminal
mean 0.002 and terminal mass L1 0.05.

| Comparison | Score difference | Terminal mean difference | Terminal mass L1 |
|---|---:|---:|---:|
| Fixed state, time refinement | 0.00008256 | 9.57e-8 | 0.000003550 |
| Fixed time, state refinement | 0.01619114 | 0.000014951 | 0.002218061 |
| Candidate versus combined reference | 0.01627370 | 0.000015047 | 0.002219248 |

Maximum neutral mass error is 1.58e-13; maximum relative series guard is
1.29e-11. The [raw CN panel](../files/output/bm_1/runs/estimation/h2_interval_positive_cn_control_20261005.json)
took 701.08 s. Its state refinement ratio is 1921/1281, not a doubling.
The finest replay is a development reference, not exact continuous-state
truth. This is one constructed 24-hour stream with fixed physical/noise
parameters and linear f/g, not simulated recovery or historical validation.
No accepted speedup against an equally accurate historical backend is claimed.

## Exact operator reuse and bounded new controls

Interval-local LRU caches hold at most sixteen exact implicit factorizations
and reward phases. Factor keys contain the actual three matrix bands; phase
keys contain the actual time step and reward bytes. There is no rounded step
or assumption that the generator key also describes the reward. Reward and
unkeyed state fields are still evaluated at every original frozen-left time.
The existing explicit generator-key contract is retained with bounded storage.
Disabling the new reuse provides the same-arithmetic reference.

Three paired one-hour measurements, with alternating order and BLAS=1, give
median 4.8893 s without reuse and 3.9805 s with reuse (1.23x). Every conditional
transform, log-amplitude and neutral mass array is bitwise identical. One pair
was slower with reuse; this is a small timing sample, not whole-day or historical
fit throughput. [Raw timing](../files/output/bm_1/runs/estimation/h2_interval_operator_runtime_20261005.json).

The linear case at period 2 / 32 frequencies matches the saved period 4 / 64
candidate: score difference 8.53e-14, terminal mean 1.11e-16, L1 7.25e-16.
Its replay took 51.07 s; both period and column count changed, so this is not a
matched operator-cache timing. [Raw period control](../files/output/bm_1/runs/estimation/h2_interval_period_control_20261005.json).

Two predeclared joint curve/noise fixtures reuse the 24 observations and missing
mask, physical parameters, hourly B/U and Beta(2,3). One shared curve pair is
fixed throughout each replay; no optimizer or day-specific curve selection.
The fixtures do not isolate the effect of curve shape from noise width.

| Fixture | f values | g values | sigma_y | Base period/frequencies |
|---|---|---|---:|---|
| Two pieces, knots 0,.5,1 | 1,.3,-1 | 1,-.25,-1 | .04 | 2 / 64 |
| Four pieces, knots 0,.25,.5,.75,1 | 1,.7,.2,-.55,-1 | 1,.45,-.1,-.8,-1 | .16 | 3 / 48 |

Each has five replays: candidate 1281/h12288, fixed-state h24576, fixed-time
1921/h24576, doubled frequency at candidate resolution, and period 4 at the same
maximum frequency. Existing score/mean/L1 limits remain .05/.002/.05.

All five wide-noise comparisons pass:

| Comparison | Score difference | Terminal mean difference | Terminal mass L1 |
|---|---:|---:|---:|
| Fixed state, time | 0.000004166 | 1.30e-10 | 3.89e-7 |
| Fixed time, state | 0.002234488 | 3.19e-6 | 0.001858843 |
| Combined | 0.002238654 | 3.19e-6 | 0.001858841 |
| Frequencies 48/96 | 0 | 0 | 0 |
| Periods 3/4, frequencies 48/64 | 2.19e-14 | 0 | 2.41e-15 |

Maximum mass error 1.74e-13 and relative-series guard 3.47e-14. The candidate
took 65.78 s; all five replays 574.48 s.
[Raw wide-noise control](../files/output/bm_1/runs/estimation/h2_interval_four_wide_control_20261005.json).

All five narrow-noise replays abort in the first hour before any observation
update: conditional densities have 184–304 negative cells and minima about
-1e-14 to -1e-13. Finer grids, doubled frequencies and the larger period do not
resolve it. The panel took 38.89 s. No failed comparison is relabeled a pass.
[Raw narrow-noise control](../files/output/bm_1/runs/estimation/h2_interval_two_narrow_control_20261005.json).

An analytic three-state witness with zero dynamics confirms exact Gaussian
conditional tails about 3.28e-27 while the integrated Fourier densities are
negative about -6.66e-15. Direct closed-form phases with 128 frequencies and
period 2 remove state/time error and reduce the truncation bound to 9.94e-58;
computed tails are positive about 1e-15 and inaccurate. The existing relative
guard rejects that weighted case (0.01397 > 1e-8). Log-amplitude storage solves
underflow, not conditional Fourier cancellation; increasing frequencies alone
does not fix floating-point cancellation.
The [integrated witness](../files/output/bm_1/runs/estimation/h2_interval_cancellation_witness_20261005.json)
and [exact-phase witness](../files/output/bm_1/runs/estimation/h2_interval_exact_phase_cancellation_20261005.json)
are preserved. The concrete next review is [Astra XHigh](H2_INTERVAL_ASTRA_HANDOFF.md).

The new control CLI checkpoints after each whole replay under an exclusive
lock, preserving numerical failures. Resume checks source, design, environment,
input fingerprints and the saved-row checksum, and skips completed units.
It does not export production log-state checkpoints or portable cross-source
resume. Completed raw reports are immutable. Matching experiment source for
local snapshot 77cc66b is [ecbb8d9](https://github.com/N1dhug/BSCe2026/commit/ecbb8d972de7d8d40a84e25846c50955736a7319).

## Tests, source and next work

Eleven short transform tests cover the independent finite-state matrix-exponential/Fourier
reference, plain/tilted identity, rare Gaussian evidence, strict failure guards,
dense implicit/CN matrix identities, below-float-range amplitudes, true zeros,
and second-order CN score convergence, bitwise cache parity with changing
fields/rewards and eviction, and the exact-phase narrow-tail failure. Three
runner tests cover frozen registration, persisted failures and pause/resume
without duplicated work or corrupted checkpoint reuse. All 36 focused
interval/history tests pass. Windows CI runs all fourteen new controls on its
native numerical stack; the earlier nine also passed float64 log scans.

Raw reports retain their local source snapshots and SHA256 maps. Matching
experiment kernel/CLI/test blobs are archived in the following repository
commits; unrelated contemporary WM changes are preserved from main:

| Control | Local snapshot | Experiment source archive |
|---|---|---|
| Balanced transform | 4a38c06 | [154b13c](https://github.com/N1dhug/BSCe2026/commit/154b13c11817b41e19389a7cddc9df675f0083a6) |
| Fixed-state time controls | 14426c0 | [801485c](https://github.com/N1dhug/BSCe2026/commit/801485c2eca33dab6d523f1f9ed5f83d0b6487cc) |
| Positive CN | ec26452 | [654aabe](https://github.com/N1dhug/BSCe2026/commit/654aabeed473f867e6f380021ec2610efbff2703) |

The [original negative control](../files/output/bm_1/runs/estimation/h2_interval_characteristic_control_20261005.json)
is preserved. Plain unshifted Fourier inversion still fails its rare-evidence
relative-series guard; the guards have not been weakened.

Next: resolve conditional Fourier cancellation using the saved analytic
references before another large panel or optimizer. An interval adapter must
preserve log-state checkpoint information and actual time-varying fields. Population,
split/exposure audit and frozen historical price/scale/baseline/prior/physical
parameters also remain prerequisites. No historical observations or holdout
values were read, no optimizer was run, and no H2/H3, model-selection or
forecast acceptance is claimed.

## Reproduction

The new fixtures and analytic witnesses are saved; no rerun is needed for
review. A new bounded invocation uses a new output path and frozen source:

```bash
python -B -m src.exe.run_model.bm_1_interval_curve_control \
  --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json \
  --cn-reference files/output/bm_1/runs/estimation/h2_interval_positive_cn_control_20261005.json \
  --case four-wide --output NEW_REPORT.json --blas-threads 1
```

Other cases: runtime, linear-period, two-narrow. `--resume` requires the original
checkpoint and unchanged source/input/environment. Exit 2 records a declared
numerical failure; it does not trigger a retry or historical fallback.

Use a new output path for each invocation. These commands are synthetic controls.

```powershell
python -B -m src.exe.run_model.bm_1_interval_characteristic --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json --output files/output/bm_1/runs/estimation/interval_balanced.json --blas-threads 1

python -B -m src.exe.run_model.bm_1_interval_characteristic --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json --time-refinement-reference files/output/bm_1/runs/estimation/h2_interval_balanced_characteristic_control_20261005.json --output files/output/bm_1/runs/estimation/interval_time_control.json --blas-threads 1

python -B -m src.exe.run_model.bm_1_interval_characteristic --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json --positive-cn --output files/output/bm_1/runs/estimation/interval_positive_cn.json --blas-threads 1
```

Exit 2 means an unresolved numerical guard. Exit 0 means that the declared
replays completed; inspect comparison results and acceptance flags separately.
Inputs and source hashes are checked before atomic publication.
Baseline model: [H2_HOURLY_INTERVALS.md](H2_HOURLY_INTERVALS.md).

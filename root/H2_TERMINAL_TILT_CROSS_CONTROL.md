# F4-H2: one cross-resolution control

5 October 2026. Sol continuation of the scoped Astra review in
[H2_TERMINAL_TILT_REVIEW.md](H2_TERMINAL_TILT_REVIEW.md). This experiment
belongs to stochastic F, independently of G. The design below is committed
before the numerical run.

## Declared experiment and interpretation

The saved PR #81 day compared A=(1281 cells, h=1/12288 day) with
C=(1921 cells, h=1/24576 day). Both completed, but the absolute total
log-likelihood difference .08754758 exceeds the unchanged .05 gate.

Run only B=(1281 cells, h=1/24576 day), keeping the entire previous fixture:
shared two-piece f/g, sigma_y=.04, physical parameters, prior, observations,
two missing intervals, 64 frequencies, period 2 and the fixed terminal-center
bank. No A or C interval is recomputed. The replay uses the existing kernel
and atomic actual-log-state checkpoints, BLAS=1, <=900 s per invocation.
An already admitted interval finishes; pause is not acceptance.

For signed score differences,

\[
\Delta_t=\ell_B-\ell_A,\qquad
\Delta_x=\ell_C-\ell_B,\qquad
\ell_C-\ell_A=\Delta_t+\Delta_x.
\]

A/B changes time at a fixed state grid; B/C changes space at the fixed fine
time step. These are controlled differences at these settings, not unique
components of the unknown continuous-model error. Two comparisons do not
establish a convergence rate, universal optimizer-domain accuracy or a
rigorous likelihood enclosure. Preserve the .05 score / .002 final mean /
.05 final mass-L1 gates for both comparisons. Retain the original failed
A/C comparison and keep numerical fit readiness false.

## Reuse and checks

The CLI `bm_1_interval_terminal_tilt --stage cross` requires
`--terminal-reference` pointing to the saved PR #81 JSON or JSON.gz.
It verifies the exact uncompressed report SHA256, fixture and input hashes,
the explicit 35-file BM dependency list, and the byte-identical terminal-update
kernel prefix. Unused G/WM code is outside the equivalence comparison; changes
there must not couple F to those tracks. The two edited orchestration files
are whole-file exceptions; their numerical update prefix still matches PR #81.
The new manifest fingerprints all three input files and saves this equivalence
evidence. Existing source/input/environment resume guards remain in force.

One focused reuse test covers accepted original evidence and rejection of
changed evidence, inputs or a numerical dependency, while permitting unrelated
G/WM source changes. The unchanged kernel's
previous exact-Gaussian/one-jump/recursive-tail tests are reused. No old panel,
full suite, Windows run, historical fit or LaTeX build is requested.

## Results, 5 October 2026

Only B was run: 24 intervals, including the original missing observations
at indices 7 and 19. A and C were reused; zero reference intervals were
recomputed. B's total log likelihood is -95.293536448336. Its replay took
431.77 s, and the complete invocation 433.43 s (BLAS=1). No resume was needed.

| Comparison | Absolute score difference | Final mean difference | Final mass L1 | Pass |
|---|---:|---:|---:|---|
| A/B: time at 1281 cells | .0021600944 | 8.778e-7 | 3.216e-5 | Yes |
| B/C: space at h=1/24576 | **.0853874862** | 3.905e-5 | .00255857 | **No** |
| Saved A/C: combined | **.0875475806** | 3.993e-5 | .00256929 | **No** |

The limits remain .05/.002/.05. Both signed score differences are negative,
so .0021600944+.0853874862=.0875475806: approximately 97.5% of the observed
A/C change belongs to the fixed-time spatial comparison. This diagnoses the
dominant difference between these settings; it is not a percentage breakdown
of the unknown continuous-model error. The finer time step alone cannot
resolve the failed score gate.

All intervals passed the terminal-contour diagnostics. Maximum selected
conditional guard was 2.075e-9 (limit 1e-8), maximum resolved overlap
discrepancy 3.670e-9 (limit 1e-7), and maximum mass discrepancy 9.015e-14.
Finite log-posterior values down to -3581.97 were retained. No clipping/floor
or observation/model change was used. The completed negative precision
result returns exit 2; `numerical_fit_readiness` remains false.

The focused reuse test passed; no unchanged numerical test
or old panel was repeated. Changed-Python syntax passed locally; the actual
full replay also exercised the new stage, checkpoint writes and final
comparisons. Source remained clean and byte-frozen throughout. After the run,
input-path separators were normalized for Linux/Windows evidence reuse; the
same focused test covers backslash-separated keys. That compatibility-only
guard change does not change any fingerprint value or numerical kernel, and
the numerical experiment source above remains archived. No numerical replay
was repeated.

### Raw evidence and reproducibility

Numerical experiment source:
[ffdf295](https://github.com/N1dhug/BSCe2026/commit/ffdf2957a265a14822c5594cac2b67b4ea0c252a).
The raw report retains clean local snapshot
`a948808fda3760b24abd3b064ba9b66cb6079093`, all Python-source SHA256 values,
three input fingerprints and the explicit BM/source-equivalence proof.
The 35 BM dependencies match the remote base; unrelated WM imports are not
a claim of identical entire local/remote checkouts.

Lossless full raw report, with every log-state checkpoint row:
`files/output/bm_1/runs/estimation/f4_h2_terminal_tilt_cross_20261005.json.gz`.
Uncompressed SHA256:
`1a3674bba976e9bd4ee86d1b8a64488417df6fee44286b70ba8a9908f5b71424`.
Original JSON and atomic checkpoint remain locally. Historical observations
read: zero; scientific/H2/H3 acceptance remains false.

Reproduction command (choose a new output name; completed evidence is never
overwritten):

```bash
python -B -m src.exe.run_model.bm_1_interval_terminal_tilt --stage cross \
  --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json \
  --cn-reference files/output/bm_1/runs/estimation/h2_interval_positive_cn_control_20261005.json \
  --terminal-reference files/output/bm_1/runs/estimation/f4_h2_terminal_tilt_day_20261005.json.gz \
  --output files/output/bm_1/runs/estimation/f4_h2_terminal_tilt_cross_reproduction.json
```

This command is for reproduction, not a request to repeat the completed study.

## Next bounded choice and cost

Next, use one 2561-cell reference with h=1/49152 day on the identical fixture,
compared with saved C; do not rerun A/B/C. Keep 64 frequencies, period 2,
the same contours and all three gates. A smaller time step is necessary to
retain positive CN at that spatial resolution; it is not silent adaptation.

For generator Q, the explicit CN half requires
`1+(h/2)*Q_jj >= 0`, hence `h*max(-Q_jj) <= 2`. A cheap generator-only audit
at all 24 frozen hourly inputs (no likelihood replay) gives:

| Cells | h (days) | Maximum exit rate | Minimum explicit diagonal |
|---:|---:|---:|---:|
| 1921 | 1/24576 | 37062.78226 | .245956 |
| 2561 | 1/24576 | 65747.61266 | **-.337639** |
| 2561 | 1/49152 | 65747.61266 | .331181 |

Actual time steps are no larger than the declared h. Therefore merely
increasing cells at h=1/24576 would violate the existing positivity guard.
The smaller declared h has a positive margin on this fixture. This audit
does not score the new grid or establish its likelihood accuracy.

Scaling saved C's work by `(2561/1921)*2` suggests approximately 1682 s
(28 minutes) for the new day, with uncertain contour/runtime behavior. Keep
<=900 s invocations and atomic log-state checkpoints, then resume if needed.
This is a development reference, not a practical historical-fit default.
If its precision comparison passes, use it to guide a bounded runtime
improvement before adopting a historical backend. If it fails, preserve the
negative evidence and reconsider spatial discretization/runtime deliberately;
do not launch a refinement panel or relax the tolerance. Ordinary refinement
and orchestration remain Sol work; a new likelihood/discretization method
decision may justify another Astra review. Time variation/breakpoints,
historical log-state integration and data/split/baseline decisions remain open.

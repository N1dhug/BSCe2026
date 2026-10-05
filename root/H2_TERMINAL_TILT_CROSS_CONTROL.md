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

Results and the next bounded choice will be appended after the frozen run.

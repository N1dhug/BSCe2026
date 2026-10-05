# F4-H2: terminal-wise exponential tilting

5 October 2026. Scoped mathematical development in **F**, independent of the
deterministic G chain. The method and bounded design were committed before
the BM runs; measured results follow below. This does not release historical
fitting or H2/H3 acceptance.

## Diagnosis and choice

The saved exact three-state witnesses rule out time/state refinement as a
solution to the narrow-noise cancellation. A global exponential tilt may
resolve total evidence while an individual terminal state's conditional
Gaussian convolution is still far into its tail. Log amplitudes preserve the
weight of that state, but cannot recover digits lost inside its Fourier sum.
Positive Fourier output alone does not establish accuracy.

Choose a **finite bank of exact exponential tilts**, with a separate resolved
representation selected for every reachable terminal cell. Retain the existing
positive CN/Strang state propagation and its frozen time/state grid. This
directly targets cancellation without rebuilding the joint accumulator solver.
The alternative joint nodal method remains useful independent evidence, but
its saved likelihood errors and dispersion do not justify adopting it at the
coarse settings. More frequencies alone cannot repair roundoff cancellation.

The preconditioning idea is supported by Lunde, Kleppe and Skaug,
[Saddlepoint-adjusted inversion of characteristic functions](https://arxiv.org/abs/1811.05678),
section 2 (2020 revision). They invert an exponentially tilted, standardized
density near its mean, retaining the inversion correction. Our implementation
uses a declared finite bank instead of solving a saddlepoint equation per
terminal state; it is not their full algorithm or a saddlepoint approximation.

## Exact identity for this interval model

Let A be normalized average demand on one interval, with 0 <= A <= 1, and
let mu_j(da) be its joint measure with terminal state j, conditional on the
previous observations. The desired unnormalized update is

\[
q_j(y)=\int\phi_\sigma(y-a)\,\mu_j(da).
\]

For any real alpha and center c, define

\[
M_j=\int e^{\alpha(a-c)}\mu_j(da),\qquad
d\nu_j(a)=e^{\alpha(a-c)}\mu_j(da)/M_j.
\]

Completing the square gives the exact identity

\[
q_j(y)=M_j e^{\sigma^2\alpha^2/2-\alpha(y-c)}
\mathbb E_{\nu_j}\{\phi_\sigma(y-\sigma^2\alpha-A)\}.
\]

Use alpha=(y-c)/sigma^2. The final expectation is now a Gaussian convolution
evaluated at c. The factor outside it is handled in logarithms. Different
centers give the **same q_j**, so choosing a different center for each j does
not mix parameter models, reset the process, or change the observation law.
For zero generator and constant reward r_j, choosing c near r_j makes this
conditional convolution well scaled even when q_j is astronomically small.

The identity also holds for the discretized process: every positive CN state
matrix is a conservative transition and each reward half-step adds a bounded
increment. Each contour uses identical transitions, priors and reward steps.
Only the transform representation differs. The Feynman-Kac construction for
additive functionals is the existing one; see
[Fitzsimmons and Pitman (1999)](https://www.stat.berkeley.edu/~pitman/kac.pdf).

## Error policy and recursive tails

The declared center order is (0.5, 0.25, 0.75, 0, 1). After each contour,
keep the best resolved representation of each reachable terminal cell. Stop
when all are resolved; exhausting the bank with unresolved cells is a failure.
This explicitly changes the number of contours, never the time step or model.

For each conditional Fourier sum d_j, compute the existing analytic periodic
alias and series-truncation bounds plus the existing floating-point allowance
64*eps*(1+2*sum(abs(coefficients)))/period. With their sum delta, require
d_j > delta and delta/(d_j-delta) <= 1e-8. This is stronger than checking only
total evidence. A negative or unresolved contour row is not used; another
exact representation must resolve that same state. No mass is clipped or
floored. Resolved overlapping log-joint values must agree within 1e-7, or the
whole update fails. This overlap check is additional diagnostic evidence.

The alias/truncation bounds follow from bounded A and |conditional CF| <= 1.
**The roundoff allowance is heuristic:** it does not certify all error in
phase evaluation, row balancing or repeated LAPACK/Thomas solves. We therefore
claim checked numerical stability on the references, not rigorous enclosure
of the continuous-state likelihood. Time/state errors remain separate.

If every component update really has multiplicative error in
[1-epsilon_i,1+epsilon_i], positivity of the next exact interval kernel
preserves the corresponding bounds for unnormalized recursive weights.
After n observations, evidence lies between the products of those factors;
its log error is bounded by sum(-log(1-epsilon_i)). Normalized log-posterior
error is bounded by sum(log((1+epsilon_i)/(1-epsilon_i))). This explains why
componentwise accuracy is useful even for tiny tails. It is a conditional
error argument, not a certificate for the heuristic floating-point allowance.

Finite log-posteriors are passed to the next interval, including values below
the ordinary floating-point range. A genuine disconnected zero stays -inf.
Missing observations propagate the log state without a Gaussian update.

## Bounded controls declared before BM runs

1. Exact original Gaussian witness: rewards (.05,.5,.95), prior (.2,.5,.3),
   y=.6, sigma=.04. Also a recursive zero-generator case starting at log
   masses (0,-900,-950), then y=1.5 twice, to expose accidental tail deletion.
   Score/log-posterior tolerance 2e-8; disconnected zeros retained.
2. Independent positive one-jump reference: state 0 jumps to absorbing state
   1 at rate .8, rewards (.1,.9), y=.55, sigma=.04. A third state is disconnected.
   Integrate the positive jump-time density with Gauss-Legendre 96/160.
   Compare CN at 128/256 steps: fine log-joint error <2e-4 and error ratio >3.5.
3. Reuse the exact saved two-piece/narrow BM fixture from PR #76. First its
   previously failing first hour, then (only if successful) the same full
   constructed day. Two settings: 1281 cells, h=1/12288 day; 1921 cells,
   h=1/24576 day. Both use 64 frequencies, period 2 and the fixed center bank.
   This is one combined time/state comparison, not five separate studies.
   Preserve the existing .05 total score / .002 mean / .05 mass-L1 tolerances.
   No fit, new observations, wide-noise rerun or large parameter grid.

The CLI caps an invocation at 900 seconds, checked between intervals; a
currently admitted interval finishes. It checkpoints each completed pilot
interval atomically with actual log state, input/source hashes and environment.
Resume reuses completed intervals only with the same manifest. These are
synthetic pilot checkpoints, not the historical fitter's production adapter.

## Remaining scope

The finite bank can fail for other distributions, including strongly separated
conditional mixtures. Failure must remain explicit. There is no guaranteed
universal contour coverage or justified noise lower bound. General time-varying
fields remain frozen-left, so no general second-order claim is made.
Historical integration, its data/split/baseline decisions, optimizer-domain
readiness, and the full shared f/g fit remain open. G's fits are separate.

There is a concrete limitation even for an unlimited bank of real tilts.
Suppose one terminal cell has equal conditional atoms at A=0 and A=1,
y=0.5 and sigma=0.04. For B=A-0.5, the tilted conditional Gaussian density at
the shifted observation is

\[
d_\alpha=\phi_\sigma(0.5)
\frac{\exp(-\sigma^2\alpha^2/2)}{\cosh(\alpha/2)}
\leq\phi_\sigma(0.5).
\]

Every real tilt still evaluates a tiny density. No center-selection rule alone
can universally remove cancellation for separated conditional mixtures. A
future positive component decomposition or independently controlled positive
reference would need its own argument. This prototype deliberately fails if
its finite bank cannot resolve a cell; it does not silently repair such cases.

## Measured results, 5 October 2026

All results below use BLAS=1 and the frozen experiment source. The three new
focused unit tests passed: exact Gaussian/recursive-tail/disconnected-zero
behavior, positive one-jump integration with time refinement, and explicit
failure without prior mutation when the contour/frequency budget is inadequate.
No existing numerical kernel changed. No full suite, old accepted panel,
Windows run, historical fit or LaTeX build was repeated.

| Control | Observed error or result | Requirement |
|---|---:|---:|
| Original exact Gaussian witness, log score | 4.44e-16 | <2e-8 |
| Original witness, maximum log-posterior error | 1.21e-10 | <2e-8 |
| Two recursive below-float updates, maximum log-posterior error | 2.06e-10 | <2e-8 |
| Positive one-jump reference, fine log-joint error | 9.37e-7 | <2e-4 |
| One-jump coarse/fine error ratio | 4.00 | >3.5 |
| First BM hour, score / final mean / mass L1 | 2.51e-6 / 5.35e-7 / .00216338 | .05 / .002 / .05 |
| Full BM day, score / final mean / mass L1 | **.08754758** / 3.993e-5 / .00256929 | **.05** / .002 / .05 |

The exact-reference report took 0.462 s and the paired first hour 40.69 s.
Both full-day replays completed 24 intervals, including the original two
missing observations (indices 7 and 19). The coarse/fine replay times were
225.70/630.87 s; total numerical work was 856.57 s. Their log likelihoods were
-95.291376353943 and -95.378923934561. Each observed interval used two or
three contours. The largest selected relative conditional guard was 2.10e-9
(limit 1e-8), largest resolved overlap discrepancy 2.93e-9 (limit 1e-7),
and largest mass discrepancy 1.39e-13. Finite log-posterior values down to
-5065.72 survived; no tail clipping or probability floor was used.

**The full-day combined time/state precision gate fails.** Successful
completion resolves the saved cancellation witness on this fixture; it does
not establish continuous-state likelihood precision or fit readiness. The
largest signed per-hour coarse-minus-fine score contributions occur at indices
13 (.02221122) and 8 (.02148109). These include propagated prior differences;
they do not isolate a local error or distinguish time from space.

The actual day run was deliberately paused after its first coarse interval
(exit 3), then resumed with unchanged source, inputs and environment. The
completed report records `resumed_intervals=1`. Its saved first log posterior
and score are bitwise identical to the separate first-hour control. All 48
completed interval units retain their log states in the final report; the
failed precision gate returns exit 2 with `status=completed`, not a crash.

### Source and raw evidence

The numerical source is archived in
[808f751](https://github.com/N1dhug/BSCe2026/commit/808f751082101ecd17e64e9abe655160fde76d0b).
Raw reports retain local snapshot `fa709d31a6892751412ace3b8b512c867a189f2d`
and a full Python-source SHA256 map. Their `dirty=true` records an untracked
documentation-copy temporary file, `root/F4.md.h2sync`, not modified numerical
source. It was removed after the frozen run. The imported F numerical
dependencies and new prototype match the archived source; this does not claim
identical entire local and remote checkouts (unused WM imports differ).

Files under `files/output/bm_1/runs/estimation/`:

- `f4_h2_terminal_tilt_references_20261005.json`
- `f4_h2_terminal_tilt_hour_20261005.json`
- `f4_h2_terminal_tilt_day_20261005.json.gz` (lossless gzip of the complete
  raw JSON, including all log states and source/input fingerprints).

The uncompressed day JSON SHA256 is
`b35a376d28659e221c2ee65ca9b8ddee98e5d9c0ca2a57ab3642d5cb0b45676c`.
Compression changes no experimental value. The original raw JSON and atomic
checkpoint were retained locally. Historical observations read: zero.

### Next bounded task

The cancellation diagnosis and scoped method review are complete. Routine
implementation can continue on Sol with this explicit experiment: add **one**
cross-resolution replay of the same day at 1281 cells and h=1/24576 day,
keeping the prior, observations, f/g, noise, contours and frequency/period fixed.
Compare it with the two saved replays, without recomputing either: coarse vs
cross isolates time at 1281 cells; cross vs fine isolates space at h=1/24576.
The smaller step preserves the coarse grid's positive-CN admissibility.
Budget one invocation at <=900 s with atomic log-state checkpoints; the
observed work suggests roughly 450 s, not a guaranteed timing. Preserve all
three existing numerical tolerances and report any failure. Do not launch a
refinement panel or accept the finer grid just because it completed.

After that diagnosis, choose a bounded accuracy/runtime correction. Before a
historical fit, independently cover time variation/breakpoints and the actual
historical log-state adapter, and settle the data/split/exposure/baseline/
parameter choices. Revisit Astra for a new mathematical method decision or a
conditional-mixture failure; no repeat Astra review of the resolved exact
Gaussian witness is needed. None of these tasks belongs to G.

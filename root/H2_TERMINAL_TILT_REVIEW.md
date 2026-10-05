# F4-H2: terminal-wise exponential tilting

5 October 2026. Scoped mathematical development in **F**, independent of the
deterministic G chain. This document records the method and bounded design
before its BM runs. It does not release historical fitting or H2/H3 acceptance.

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

Results will be appended after the frozen runs, including negative outcomes.

# F4-H2: exact-kernel acceleration and bounded adaptive estimation

Mathematical decision, 6 October 2026, after the Windows raw audit. This is
the Astra review requested by Johan. **The implementation and speed are not
yet established.** This document supersedes the next-action sections of
H2_SEARCH_NEXT_HANDOFF.md and H2_PRECISE_BATCH.md; their recorded results remain
valid. F is stochastic BM1, independent of G and WM.

## Decision and scope

Keep the sequential interval likelihood, existing fitted FV generator,
positive Crank–Nicolson/Strang discretization and terminal contour selection.
Replace the repeated row-balanced LAPACK solve by a compiled positive Thomas
recurrence in log-amplitude/conditional-transform coordinates. Distribute
Fourier columns of one candidate across eight persistent workers. This makes
parallelism available inside an adaptive objective evaluation; candidate days
remain strictly sequential. Use BLAS=1 and no nested native thread pool.

If a bounded equivalence/runtime check supports the complete H budget, use
bounded COBYQA on the scalar penalized interval log likelihood. The fixed
17-point coordinate panel becomes archived development evidence, not a fit.
Neither the old 73-candidate proposal nor the new search ceiling is admitted
before the runtime/data gates below. No coarse solver, point-observation
substitution, noise inflation, shortened TRAIN or omitted curve family is
part of this decision.

The result of this review is an algebraic implementation specification and a
go/no-go protocol. It does **not** show that the 10-hour goal is attainable.
If the compiled implementation fails that protocol, retain H as incomplete;
do not consume the remaining budget on automatic alternative solvers.

## Population and chronological windows

Johan delegated population/period selection on 6 October and requires MED to
remain separate if used. Choose the fixed mean of **all 13 MED households**
with electric heating, using the ordered membership in the existing metadata
audit. No UDEN household enters this fit or its baseline. A missing member
makes that hour's group mean missing; the denominator never changes.

| Role | Half-open local interval, Europe/Copenhagen | Hours |
|---|---|---:|
| TRAIN | [2022-02-07 00:00+01:00, 2022-02-14 00:00+01:00) | 168 |
| MODEL_SELECTION | [2022-02-14 00:00+01:00, 2022-02-21 00:00+01:00) | 168 |
| Planned final evaluation | [2022-02-21 00:00+01:00, 2022-02-28 00:00+01:00) | 168 |

These are interval-START windows: the observation for 13 February 23:00–24:00
belongs to TRAIN even though it becomes available at the selection boundary.
Filter availability still uses interval END. Do not apply a start-window
mask to endpoint timestamps and lose/transfer that boundary observation.
The training week covers weekdays and a weekend. Every candidate has one
common f/g pair and its own continuous state from the prior origin.

The final week is **not yet certified blind**. Prior G exposure included
January/early February; no exhaustive audit for the final week is available.
The current loader's `require_blind_holdout` gate remains in force. A dates-only
decision or an empty exposure list cannot bypass it. Do not read final-week
values while preparing the contract or developing this implementation.

The machine-readable decision in `f4/h2_method_decision_20261006.json` is
deliberately not an executable history-fit config. It records verified MED
IDs, selected dates and method choices, but supplies no invented column names,
baseline, price area/scales, physical/noise parameters, prior origin or
exposure evidence. Resolve exact flattened columns from the existing header
audit before config export. The one-hour kWh/hour-start and timezone choices
retain their existing working-assumption status.

## Why an optimizer switch or ordinary least squares is insufficient

For the candidate parameters theta, the observation law is

\[
 A_i=\Delta_i^{-1}\int_{s_i}^{t_i}D_\theta(v,X_v)\,dv,
 \qquad Y_i=A_i+\epsilon_i,\quad \epsilon_i\sim N(0,\sigma_y^2).
\]

With the candidate's previous posterior, define the positive joint kernel
\(K_{i,\theta}(y)\) on endpoint state. Then

\[
 q_i=K_{i,\theta}(y_i)p_{i-1},\quad
 Z_i=\mathbf1^Tq_i,\quad p_i=q_i/Z_i,
 \qquad J(\theta)=-\sum_{i\in O}\log Z_i+\lambda R(\theta).
\]

Missing observations apply only the state transition and contribute no score.
Changing theta changes all subsequent posteriors. Optimizer candidates must
therefore replay the full TRAIN; caching another theta's posterior is invalid.

In general \(E[\phi_\sigma(y-A)]\ne\phi_\sigma(y-E[A])\). An ODE least-squares
fit or a Gaussian residual at the predicted mean would change this objective,
and even a Gaussian moment approximation has parameter-dependent variance
and a log-variance term. It is not ordinary unweighted least squares. A formal
sum-of-squares encoding of a shifted negative log likelihood would still need
the same expensive filter and would not produce a computational saving.

The old Windows envelope gives 6.12 hours for just **one** parallel wave of
week-long candidates, before subsequent model-selection/final streams. The
17-candidate plan has three such waves and a 20.65-hour total with its old
reserve. These are extrapolations, not measured historical times. Reducing
17 to eight candidates does not establish fit convergence or fund all H work.

## Algebraic replacement for the implicit state step

Use column probability masses. For the frozen tridiagonal Markov generator
Q, let \(s=h/2\), \(M=I-sQ\), and \(E=I+sQ\). The state step is
\(M^{-1}E\). Q has nonnegative off-diagonals and zero column sums; M is a
nonsingular M-matrix. The explicit part additionally requires \(E\ge0\),
exactly as in the existing positive-CN guard. No new time step is chosen.

Write M's diagonal as \(d_j\), lower band \(-l_j\) and upper band
\(-u_j\), with \(l_j,u_j\ge0\) on the link from j to j+1. Its positive
Thomas factors are

\[
 v_0=d_0,\qquad \beta_j=l_j/v_j,\qquad
 v_{j+1}=d_{j+1}-\beta_j u_j>0.
\]

For a transformed right-hand side b, forward elimination and backward
substitution give **the same linear solve**:

\[
 F_0=b_0,\quad F_{j+1}=b_{j+1}+\beta_jF_j,
 \qquad z_{n-1}=F_{n-1}/v_{n-1},\quad
 z_j=(F_j+u_jz_{j+1})/v_j.
\]

Represent each row as \(b_j=e^{a_j}c_j(\omega)\), where \(a_j\) is the
log of its positive zero-frequency amplitude and \(c_j(0)=1\). For positive
addition define

\[
 L=\operatorname{logaddexp}(a,b),\qquad
 c=e^{a-L}c_a+e^{b-L}c_b.
\]

Use this operation in the forward recurrence, with the second log-amplitude
\(\log\beta_j+\log F_j(0)\), and in the backward recurrence, with
\(\log u_j+\log z_{j+1}(0)\). Division by the positive pivot subtracts
\(\log v_j\) from the amplitude and leaves c unchanged. Thus each transform
update is a convex combination in exact arithmetic. For a valid incoming
conditional characteristic function, \(|c_j(\omega)|\le1\) is preserved by
the state solve. This is a structural property, not a floating-point proof.

The explicit E product uses the same positive-add operation on up to three
neighbor contributions. Existing reward half-steps remain

\[
 a_j\leftarrow a_j+\alpha(r_j-c)h/(2\Delta),\qquad
 c_j(\omega)\leftarrow c_j(\omega)
       e^{i\omega(r_j-c)h/(2\Delta)}.
\]

Here c in the exponent is the contour center, not the conditional transform.
Alpha, contour order, Fourier frequencies, span, reward and the terminal
Gaussian inversion remain exactly those of `terminal_tilt_update`.

Implementation details that preserve the identity:

- Zero coefficients have log -infinity. A sum of two true zeros remains zero;
  never evaluate -infinity minus -infinity. An inactive transform can be set
  to zero while its amplitude remains -infinity. One active addend passes
  through directly. Do not floor a small finite log-amplitude.
- Use stable scalar logaddexp, float64/complex128, `fastmath=False` and
  positive-pivot/sign/finite guards. Avoid the cumulative-log-product
  subtraction used by `_log_scan`; the local recurrence needs no longdouble.
- Keep the existing zero-frequency amplitude correction if accumulated
  floating-point c(0) differs from one: add log(c(0)) to the row amplitude and
  divide its transforms by c(0), after checking reality/positivity. This
  preserves the represented transform; it is not a probability repair.
- Check probability, transform modulus, conditional Fourier guard 1e-8 and
  contour overlap 1e-7. A negative Fourier candidate remains unresolved; the
  declared contour bank must resolve it or fail. This recurrence does not
  cure the previously proved separated-mixture limitation of that bank.
- Retain the independent neutral mass evolution and mass guard 1e-10.
  Normalization occurs only at Bayes. Missing-hour propagation keeps its own
  full log-state, with no observation-contour work.

The old `_balanced_state_step` constructs a scaled matrix and real/imaginary
RHS arrays and calls `dgtsv` at every substep. The replacement applies the
already positive factors directly and can fuse traversal and buffer reuse in
compiled loops. Complexity remains O(cells × frequencies × steps × contours).
Lower allocations/interpreter overhead are a plausible opportunity, **not a
profiled bottleneck or a measured speedup**. Existing 1.23x operator-cache
evidence is not evidence for this different implementation.

## Native implementation and parallelism

Use an explicitly selected experimental kernel, with the old kernel retained
as a reference. Numba nopython compilation is a practical first implementation;
freeze a compatible Numba/llvmlite version with the actual PC environment.
Do not upgrade NumPy/SciPy silently or install packages in this review. Record
compiler/version/CPU, source and settings in the new run manifest. Compilation,
startup and cache loading count in the campaign ledger.

Compile the complete substep loop for a frozen B/U segment, not just a tiny
helper called thousands of times from Python. Reuse factors/phases only for
identical operator-defining inputs and exact h. Hour/input boundaries remain
explicit; the final shortened step must not be rounded into a cache key.
If an observation crosses a held-input boundary, preserve the original
observation duration Delta in every reward half-step and carry the accumulator
transform across segments; do not restart or renormalize it per segment.

For adaptive search use **one candidate at a time**, with eight persistent
processes handling disjoint positive Fourier indices. For 64 frequencies,
worker b owns indices 8b+1 through 8b+8, b=0,...,7; every worker additionally
carries index zero for its amplitude correction. Each worker starts a contour
from the same complete log prior, uses identical positive factors and visits
all substeps in order. No cross-frequency coupling occurs until inversion.
Gather exactly indices 0,...,64 in canonical order, retaining zero once;
require identical zero-frequency/amplitude/neutral results across workers
(same operations/environment; a mismatch is a failure, not an average).

Contour admission/selection stays sequential in its original order. Reassemble
and apply its guards before the next contour. After the Bayes update only the
parent's complete posterior enters the next hour. A partial set of frequency
blocks cannot create an hourly checkpoint. A worker failure stops the unit.
The workers' duplicated scalar work and IPC may negate gains; measure this
schedule rather than dividing a serial timing by eight. Eight frequency
workers are the whole pool; there are not eight simultaneous optimizers too.

The old 8-candidate pilot remains immutable. Frequency blocks need a new
execution manifest, never a relaxed old source/worker/resume check. Reuse the
common ledger and completed matching NEW hourly checkpoints. Do not attach
new numerical code to an old active run.

## Conditional adaptive search after kernel admission

Preserve the existing drop-softmax parameterization and endpoints. Two-piece
f/g has two independent logits; four-piece has six. Use one linear start per
family, bounds [-4,4], fixed physical/noise/prior/baseline inputs, and

\[
 R=\sum_{q\in\{f,g\}}\sum_{j\text{ inner}}
       (q(z_j)-(1-2z_j))^2,\qquad\lambda=2.
\]

This is a fixed design regularizer, equivalent to a quadratic preference with
scale 0.5 on each inner value (within the monotone family), not a calibrated
Bayesian uncertainty analysis. It is chosen before historical observations;
there is no lambda grid and TRAIN does not select its value. More inner values
receive more total regularization under the existing sum convention.

Use `scipy.optimize.minimize(method='COBYQA')` on the scalar J, with initial
trust-region radius 0.5, final radius 0.02, `scale=False`, and no objective
target. These are declared local search resolutions, not parameter confidence
or a global optimum certificate. Model-based derivative-free search avoids
unjustified tiny finite differences through contour decisions and FV error.
The demand map also has a derivative kink at zero flexibility when B != 0.5;
standard smooth convergence results cannot be claimed automatically.

The **outer ceiling** is one shared linear TRAIN replay, at most 16 new
two-piece and 32 new four-piece full TRAIN evaluations: at most 49 distinct
curve pairs. Corresponding library-call ceilings are 17 and 33 including each
linear start. Reuse the identical linear pair across families by physical
curve/input identity, while recording both family memberships. Budget all
unfinished attempts/restarts and posterior reconstructions as actual work;
library `nfev` alone is insufficient. No multi-start or automatic cap increase.

Restarting the deterministic optimizer may replay its call sequence against
cached objectives with the same SciPy version/options. Store exact call order,
candidate identities, objective components, stop reason and complete hourly
states; if the call sequence diverges, fail rather than fabricate optimizer
continuation. Reuse an incumbent's matching posterior instead of refitting it.

A cap/timeout is `budget_exhausted`, not convergence. A trust-radius success
is at most algorithmic local termination. Both nonlinear families must be
actually fitted and reviewed; an unfinished four-piece search cannot be
silently replaced by linear and reported as completed H. Numerical failures
are failures, never +infinity objective values or silently discarded points.

Before fitting, check CN positivity over the declared physical condition and
search domain. A cheap sufficient bound is available: for each held B,
\(-B\Xi/C\le a(x)\le(1-B)\Xi/C\). Insert these endpoint drifts into the
existing fitted-flux formula with the same D and D-gradient; the right-going
rate increases and the left-going rate decreases with drift. Their respective
maxima bound every column's outgoing rate. Require h/2 times that bound <=1
at each cell. This is conservative and entails no likelihood replay. If it
fails, do not let the optimizer repair grids or return substituted costs.

## Selection, precision and final use

Fit on TRAIN only. Continue each of the three fitted/reference states through
MODEL_SELECTION with frozen curves. Rank primarily by unpenalized sequential
predictive Y-score on the same observed hours; report interval-demand RMSE/MAE
and the existing forecast requirements separately. Predict each interval
before assimilating its Y; a mean evaluated only at its endpoint is not the
interval prediction. Shared physical-unit Jacobians do not affect rankings.

Compare each family's retained solution with the existing finer discretization
on its full origin→selection stream, maintaining the fine replay's own state.
This lets the fine TRAIN part check the retained fit and the fine selection
part check ranking; do not reset fine state at a window boundary. Retain the
existing absolute .05 total-score / .002 state-mean / .05 mass-L1 gates, now
applied to each declared window and saved endpoint, plus all per-hour guards.
Do not multiply .05 by the number of days. Grid comparisons are empirical
precision evidence, not rigorous continuum error bounds.

Choose among completed families only after precision review. Require stable
ranking across the checked discretizations to claim a resolved improvement.
For score gaps <=0.1, or a rank change, record numerical ambiguity; the fixed
deployment tie rule prefers fewer pieces and makes no resolved-gain claim.
The 0.1 threshold is twice the .05 comparison tolerance, not a confidence
interval or proof that gaps above it are accurate. A failed individual
precision gate stops acceptance, even if the linear model looks preferable.

Freeze the family, curves, physical inputs, scales and evaluation protocol
before final observations are opened. Continue the selected model and linear
reference into the final week (deduplicate if linear wins); keep matching
coarse/fine states. The final score never triggers a refit. Record price counts
and posterior-state occupancy by knot interval, and mark weak coverage. Broad
profiles/bootstrap stay deferred as previously agreed; do not report joint
physical/curve recovery or causal demand response from this conditional fit.
Blind status, predictive coverage and required forecast/handoff checks remain
separate prerequisites; a score comparison alone does not finish H3 or I/J.

## Whole-campaign budget and falsifiable runtime gate

The existing charged wall-time is **1024.631120639875 s**; remaining absolute
budget is **34975.368879360125 s**. Use the latest ledger across PC/chat, not
the 468.09 s seed. The method review adds no numerical runs or ledger charge.
The next synthetic implementation/equivalence/runtime stage gets **at most
900 additional seconds total**, including compilation and all resumes, with
60 seconds of that allocation reserved for orderly stop/save. This is a
maximum development allowance, not a historical search command.

For historical admission, measure a complete interval with the **actual
eight-frequency-worker schedule** and representative declared inputs/curves.
Let tau_b and tau_f be conservative measured end-to-end per-hour times at
base/fine settings, including normal IPC/checkpoint work, and L the number of
prior-lead units. Use the maximum observed unit times and the existing 1.5
planning factor; none is a hard bound on unmeasured future tails.

For the ceiling above, an explicit sequential workload is:

| Work | Base-grid units | Fine-grid units |
|---|---:|---:|
| Up to 49 full TRAIN candidates | 49(168+L) | 0 |
| Three MODEL_SELECTION continuations | 3×168 | 0 |
| Three retained-curve precision streams to selection end | 0 | 3(336+L) |
| Final week, winner and linear, worst case | 2×168 | 2×168 |
| Total before additional forecasts/controls | 9072+49L | 1344+3L |

The last two rows are reserves; no final observations are read at admission.
Completed identical work can be subtracted only by validated checkpoints.
Include forecast means/bands, declared forecast horizons, any necessary near-
competitor checks, historical pilot, compilation/startup and export in explicit
additional jobs. Unknown work is not zero. The executable graph and rates
must be complete before admission; the current data contract cannot supply it.

For remaining jobs, require

\[
 T_{charged}+T_{startup}+1.5(N_b\tau_b+N_f\tau_f)
       +T_{other}+T_{stop}\le36000\text{ seconds}.
\]

Keep at least the existing 7200-second validation reserve, **increase it to
the priced remaining validation work when larger**, and do not let search
spend it. The new adapter must support that explicit larger reservation; the
old runner's fixed reserve/projection is not sufficient. Pilot work counts
once in the ledger and is removed from future work only when reusable.

For perspective only: use L=0 (an optimistic illustration, not an allowed
origin), spend the full next 900 s, reserve 3600 s for additional work plus
60 s stopping, and assume tau_f=(2×2561/1921) tau_b. This ratio is an operation-
count illustration, not a measured fine-grid multiplier. The resulting totals
are approximately 6.82 h at tau_b=1 s, 9.46 h at 1.5 s, and 12.10 h at 2 s.
The six-hour target needs about 0.84 s/base hour; the ten-hour ceiling about
1.60 s, with these optimistic assumptions. Actual prior lead, forecast jobs
and measured fine costs can tighten them. These are **required performance
levels, not promised speedups**. Do not compare them directly with 84.12 s
per candidate-hour under the different old eight-candidate schedule.

Enforce the campaign wall deadline in parent and workers, preserving completed
hours and aborting unfinished frequency blocks before the stop allowance.
Reservation/crash accounting, STOP and source/input/environment checks remain.
Watch actual pace during every candidate and stop new admissions when the
remaining mandatory work no longer fits. A hard timeout limits spend; it
does not guarantee completion or make a stopped search acceptable.

## Bounded handoff to Sol High

1. Implement the isolated positive-recursion kernel and eight-frequency-worker
   adapter, keeping the old reference unchanged. Reuse curve geometry, input
   freezing, log checkpoints and campaign machinery; add the manifest version
   and variable validation reserve needed by this schedule.
2. Use one focused algebra check covering an asymmetric positive generator,
   zero links/true zeros and sub-float revival. Compare a small well-scaled
   case against a direct dense solve; retain the saved Gaussian/one-jump
   witnesses. This addresses the new recurrence, not a broad solver test suite.
3. Within the single 900-second development allocation, reuse old raw outputs
   and run only the NEW implementation on the saved narrow two-piece day and
   the saved Windows linear/four-piece eight-hour prefixes. Select the first
   four-piece pilot candidate in manifest order. These exercise recursion,
   the price change and missingness without another model/grid panel. Do not
   recompute old reference trajectories. Stop immediately on a failed guard,
   equivalence check, or budget. A partially completed check is not a pass.
4. Require same-grid cumulative score and every finite log-posterior component
   to agree with the saved matched reference within 1e-7, with identical true-
   zero support and preserved .05/.002/.05 refinement evidence. Contour-choice
   equality is not required near a threshold; the selected joint result and
   all guards are. A mismatch is investigated, never relabelled as speed gain.
   Match mathematical inputs and preserve original provenance, including the
   already audited Windows CRLF conversion; do not loosen execution resumes.
5. If the new method is equivalent and plausibly fast enough, finish the F
   data contract, resolve full hourly/forecast task counts, then prepare a
   bounded representative historical rate check and the whole-job admission
   report. Charge it to the same ledger. Only after this gate implement/enable
   the prescribed optimizer and H3 continuation. No automatic long run follows
   a synthetic success. Johan receives the concrete PC commands when ready.

The Astra-specific mathematical choice is complete. Implementation, rates,
historical inputs and scientific acceptance are explicitly outstanding. No
numerical source, old archive, executable configuration or ledger is modified
by this documentation deliverable; no model replay, numerical test, historical
observation access, Windows job or LaTeX build was performed.

## Sources and evidence

- Repository definitions: `src/model/base/bm.py`, `src/params/bm_1_interval.py`,
  `src/solver/finite_volume.py`, `src/validation/bm_1/interval_characteristic.py`,
  `src/validation/bm_1/interval_terminal_tilt.py`; raw evidence and provenance
  in [H2_PRECISE_BATCH.md](H2_PRECISE_BATCH.md).
- Nocedal and Wright, *Numerical Optimization*, second edition, supplied PDF:
  sections 9.1–9.3 (numerical error, interpolation models and pattern search)
  and chapter 10 (least-squares structure). These motivate the optimization
  distinction; the log-Thomas construction above is derived from this repo's
  positive discretization, not attributed to that book.
- [SciPy scalar minimization / COBYQA](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html)
  and the [SciPy 1.17 optimizer index](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.html):
  bounded scalar optimization, explicit evaluation ceiling and trust radii.
- [Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
  and [compatibility](https://numba.readthedocs.io/en/stable/user/installing.html):
  compiled loops, strict floating-point settings and actual environment checks.
  Their illustrative speedups are not F4-H2 runtime evidence.

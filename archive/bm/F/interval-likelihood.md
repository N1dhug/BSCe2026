# F: interval-likelihood, positive operatorer og terminale tilts

<a id="h2-terminal-tilt-review"></a>
<a id="h2-terminal-bank-extension-review"></a>
<a id="h2-interval-characteristic-control"></a>
<a id="h2-interval-astra-handoff"></a>
<a id="h2-hourly-intervals"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../root/BM_GUIDE.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

Et interval har middelreward $A=\Delta^{-1}\int_s^t D(v,X_v)dv$ og
observation $Y=A+\epsilon$, $\epsilon\sim N(0,\sigma_y^2)$.
En punktlikelihood eller en Gaussian omkring $E[A]$ er en anden model.
Kun intervalakkumulatoren nulstilles; hele tilstandsfordelingen fortsætter.

## Joint likelihood and numerical method

Reset only A_s=0, with dA=D dt/(t_i-s_i), and evolve the joint state/accumulator
law. Its forward equation and end-state update are

\[
\partial_t p=\mathcal L_X^*p-\partial_a\{D(t,x)p/(t_i-s_i)\},
\qquad\pi_i(x)\propto\int p(x,a,t_i)\phi_{\sigma_y}(y_i-a)\,da.
\]

This preserves dependence between hourly consumption and terminal state.
A Gaussian around E[D], or an endpoint point likelihood, is another model.
The kernel reuses the existing conservative fitted/upwind FV generator and
cached LAPACK solves. Frozen-left half reward transport / implicit state /
half transport is first-order overall. Positive linear nodal deposition
conserves mass and first moments but adds dispersion. Common translation is
exact (one accumulator node for Xi=0). Accumulator support grows; no tails
are clipped, reflected or truncated. A declared node budget aborts instead.
Prediction is not renormalized; only Bayes normalizes. State continues across
missing intervals, gaps and midnight; missing measurements get NaN evidence.
Every shared f/g candidate starts a whole-stream replay from its declared prior.
Point/H1 observations and the separate G work retain their existing paths.


## Positiv Crank–Nicolson
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


## Den eksakte terminale tilt og dens begrænsning
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

## Historisk afgrænsning ved tilt-reviewet, 5. oktober 2026

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


## Positiv Thomas-rekursion i logamplituder
Følgende udledning ændrer repræsentationen af samme diskrete solve. Den
historiske implementering blev efterfølgende kontrolleret mod den gamle.
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

## Hvorfor ni konturer løste kandidat 28 — uden universalitetsclaim
## What failed

Candidate 28 failed at the first observed TRAIN hour (index 24), after its
own 24 h prediction-only prior. The five old centers left 25 active cells,
indices 847--871, unresolved. Their state centers span x=0.441176--0.453670.
All 3,655 completed old jobs and the 27 complete four-piece objective calls
are preserved. The two-piece fit remains converged and previously validated.

The saved rerun reproduces the old five unresolved counts exactly:
1120, 1120, 872, 872, 25. After adding centers 0.125, 0.375, 0.625 and 0.875,
the counts are 25, 25, 25, 0. The last new center resolves the remaining cells.
No likelihood or posterior probability was clipped, floored or discarded.

## Mathematics and why this change is legitimate

Write R in [0,1] for the normalized interval-average demand, and q_j(dr) for
its joint measure with terminal state cell j, conditional on the saved prior.
With Gaussian measurement error sigma, the unnormalized update is

    L_j(y) = integral phi_sigma(y-r) q_j(dr).

For a contour center c, set lambda=(y-c)/sigma^2. Let

    M_j = integral exp(lambda*(r-c)) q_j(dr),
    q_j^lambda(dr) = exp(lambda*(r-c)) q_j(dr) / M_j.

Completing the square gives the exact identity

    L_j(y) = M_j * exp(-lambda*(y-c)+sigma^2*lambda^2/2)
             * E_{q_j^lambda}[phi_sigma(c-R)].

The same identity applies to the positive discrete path measure generated by
the frozen FV/CN/reward steps. The log-amplitude recurrence carries log(M_j),
and the conditional characteristic function evaluates the expectation using
the Gaussian Fourier series. Changing c changes the conditioning and numerical
cancellation of this representation, not the target likelihood, noise level,
response curves, prior or state/time discretization.

For Fourier density d and the existing absolute error allowance a, the guard
requires a/(d-a) <= eta, eta=1e-8. Thus d >= a*(1+1/eta) is required.
Here a is approximately 1.46779e-13: truncation 5.04671e-15 plus roundoff
allowance 1.41733e-13, with negligible alias bounds. The old representations
of the 25 cells lie below the approximately 1.46779e-5 density threshold;
several are dominated by cancellation near machine precision. The new 0.875
contour gives densities 0.94746--1.12576 for those cells.

Their combined resulting posterior mass is approximately 1.35648e-43. They
still retain finite log probabilities (-102.2314 to -101.5744); small mass is
not used as permission to remove them. This preserves the full-state contract.

The roundoff allowance remains a diagnostic bound, not verified interval
arithmetic for every FV/Thomas operation. Multiple agreeing contours do not
prove state/time-grid convergence or universal robustness of a finite bank.

## Implemented representation

New optional adapter: src/method/bm_1_interval_contour_extension.py.
It calls the unchanged native update with centers

    (0.5, 0.25, 0.75, 0, 1, 0.125, 0.375, 0.625, 0.875).

All old centers occur first, in their exact old order. The native routine
stops as soon as all cells pass. Therefore any old successful update performs
the same operations and exits with the same payload before reaching a new
center. This is the compatibility argument for reusing successful old jobs;
actual imports will still need source, token and prior-chain checks and an
explicit adoption record. Old identities must not simply be relabeled.

No production runner or frozen source was edited. The separate diagnostic
uses the exact saved prior and operator settings and the same original native
function with the extended settings. It records every completed contour.
Instrumentation only saves returned arrays and reconstructs diagnostics.

Appending a finite midpoint bank is selected because it directly resolves
the observed representation gap without increasing every successful job's
cost. Looser guards, discarding small cells, replacing a failed objective
with infinity, or refining every old grid would not be this correction.
Another unresolved case will remain a failure; there is no unbounded retry.


## Hvad kontrollerne betyder
Sammenligningerne mellem grids, tidsstep, konturer og uafhængige små
Markovreferencer er samlet i [numerisk evidens](numerical-evidence.md).
Et bestået finite-state/Gaussian-test beviser ikke historisk likelihoodpræcision.
Ingen negativ celle fjernes, ingen probability floor indføres, og en udtømt
konturbank skal fortsat fejle eksplicit. Kode: `interval_native`,
`interval_terminal_tilt`, `interval_characteristic` og `bm_1_interval_contour_extension`.

## Konkrete kode- og kontrolindgange

[native operator](../../../src/validation/bm_1/interval_native.py), [terminale tilts](../../../src/validation/bm_1/interval_terminal_tilt.py), [tilt-tests](../../../tests/test_bm1_interval_terminal_tilt.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.

# F4-H2: compact historical design and interval forecasts

Astra mathematical review, 6 October 2026, after the verified native v2 PC run.
This is the active implementation handoff. **Choose a declared 48-hour TRAIN,
24-hour MODEL_SELECTION and the same 168-hour final week.** Keep the native v2
likelihood and the 49-pair COBYQA ceiling. Compute inexpensive interval moments
for every forecast hour; full observation distributions are required at the
seven existing horizons. The complete conservative *illustration* is 8.76 h,
not a measured runtime or admission. The 4–6 h target is not established.

No numerical model run or historical measurement was read in this review.
H remains charged 1460.5315267789392 s; native development uses
435.9004061390642/900 s, leaving 464.0995938609358 s across all machines/resumes.
The machine-readable decision is `f4/h2_compact_method_decision_20261006.json`.
It is deliberately not a runnable history-fit configuration.

## Explicit change of experimental scope

Johan delegated the population and training interval and rejects a total
above ten hours. The seven-day TRAIN was our design choice. At the saved
warm maximum, its 49-pair base work alone illustrates 15.38 h. Merely
switching optimizers, reducing an arbitrary call cap or recompiling again
cannot fund the mandatory fine streams. Select a smaller complete experiment
before opening measurements, rather than running an undeclared subset.

| Role | Half-open interval, Europe/Copenhagen | Hours |
|---|---|---:|
| Prediction-only prior lead | 17 February 00:00 to 18 February 00:00 | 24 |
| TRAIN | Friday 18 February 00:00 to Sunday 20 February 00:00 | 48 |
| MODEL_SELECTION | Sunday 20 February 00:00 to Monday 21 February 00:00 | 24 |
| FINAL_HOLDOUT | Monday 21 February 00:00 to Monday 28 February 00:00 | 168 |

All dates are in 2022 with UTC offset +01:00; masks apply to interval START,
while information availability applies to interval END. Use every valid
observation in the declared TRAIN. One common f/g pair spans Friday and
Saturday with a continuous candidate-specific state through midnight.
Sunday is one fixed chronological selection block. Final dates are unchanged.
No earlier measurements are assimilated in the 24-hour lead; each candidate
propagates its declared prior under its own model and frozen B/U.

This is a deliberately limited historical case study, not completion of the
superseded seven-day TRAIN experiment or a claim that two days are statistically
equivalent to a week. It covers a weekday and a weekend day, but cannot establish
weekday-wide stability, monthly generalization or practical identification of
six curve logits. Sunday-only model choice can be unrepresentative. The final
week tests the frozen result; it cannot be used to repair these choices.
A new split/exposure record must identify this revision explicitly. If final
exposure cannot be ruled out, label it development evidence and leave blind
H3 acceptance false; do not search for a favorable replacement period.

MED remains the fixed mean of the 13 previously audited electric-heating
households, separate from UDEN. Any missing member makes the group hour
missing. The archived metadata audit verifies IDs and group meaning, but does
not supply the exact flattened demand column names. No measurement values
were decoded to choose these periods. Preserve the old decision JSON and all
synthetic/reference/PC archives; they describe their original scopes.

## Estimator and acceptance

The original stochastic BM1 demand map, f(x), g(u), diffusion, boundaries,
interval-average Gaussian observation law and full log-state propagation stay
in place. The original linear family remains the reference; the two- and
four-piece monotone families use the existing drop-softmax coordinates, with
2 and 6 free logits, respectively. No G/WM basis or estimator is substituted.

For every candidate theta and observed hour i, use

\[
 q_i=K_{i,\theta}(y_i)p_{i-1},\quad Z_i=\mathbf1^Tq_i,
 \quad p_i=q_i/Z_i,\qquad
 J(\theta)=-\sum_{i\in O_{TRAIN}}\log Z_i+2R(\theta).
\]

Missing hours use only the state transition and have no score. Changing theta
requires its own complete prior lead/TRAIN replay. Do not carry another
candidate's posterior or reset state daily. The same fixed penalty R sums
squared inner-knot departures from the linear curve; on 48 rather than 168
hours it has relatively greater influence. This is an explicit regularized
conditional fit, not evidence that the original seven-day estimate is retained.

Retain SciPy 1.17.0 COBYQA, bounds [-4,4], one linear start per nonlinear family,
initial/final trust radii 0.5/0.02, scale=False and no objective target. Allow
one shared linear TRAIN replay, at most 16 new two-piece and 32 new four-piece
pairs (library maxfev 17/33, including the cached linear start). No multistart,
automatic extension or replacement by the old coordinate panel. Preserve call
order and exact curve/input/source/environment identities for deterministic
cached replay. Count interrupted work and necessary state reconstruction too.

COBYQA's default interpolation set uses 2n+1 points; a call ceiling is not a
termination or convergence guarantee. Require finite valid results and the
declared trust-radius termination from both nonlinear fits. A cap, callback
stop, timeout or numerical failure leaves the fit incomplete; never return a
fabricated infinite objective or present its incumbent as a converged model.
Algorithmic local termination is not a global optimum or identification proof.

All three retained families continue into selection. Rank by unpenalized
sequential predictive Y log-score; record interval-mean RMSE/MAE on the same
observed hours. Run each retained family's own fine prior→TRAIN→selection
stream. Keep total-score difference <=0.05 **per declared window**, maximum
state-mean difference <=0.002 and endpoint mass L1 <=0.05, with the existing
conservative cell-overlap comparison. Do not scale the score tolerance by days.
All hourly mass, conditional and overlap guards remain 1e-10, 1e-8 and 1e-7.
An individual precision failure blocks acceptance. A ranking reversal or score
gap <=0.1 records numerical ambiguity; the existing tie rule prefers fewer
pieces without claiming resolved improvement. A linear outcome is valid only
after both nonlinear families have actually completed the required work.

Report price-bin counts, missingness, daily score components and posterior
state occupation by knot interval. These are limited coverage/identifiability
diagnostics, not independent-day bootstrap intervals or joint parameter recovery.
The broad profiles and population studies remain deferred as already agreed.

## Cheap interval moments: derived from the same discrete operator

The legacy `forecast_fv` returns endpoint demand moments. Hourly energy cannot
be scored against those endpoint values. Add an F-only interval-moment adapter;
retain the legacy interface and its results for their original meaning.

For one hour [s,t], Delta=t-s, set

\[
 A_v=\Delta^{-1}\int_s^v r(X_w,w)\,dw,\qquad
 M_k(v)_j=E[A_v^k\mathbf1_{X_v=j}],\quad k=0,1,2.
\]

For the frozen FV generator Q and R=diag(r/Delta), the Markov-reward equations
are

\[
 \dot M_0=QM_0,\quad \dot M_1=QM_1+RM_0,\quad
 \dot M_2=QM_2+2RM_1.
\]

Use **the existing positive CN/Strang operator**, not a new PDE time solver.
For a reward half-step a_j=r_j h/(2 Delta),

\[
 M_0'=M_0,\quad M_1'=M_1+aM_0,\quad
 M_2'=M_2+2aM_1+a^2M_0,
\]

where the right-hand M1 is its pre-update value. The state step applies the
same positive CN matrix to each Mk. These equations are the first two
transform derivatives at zero of the existing split reward/transition law.
They give moments of that **same discretized interval law**, not a new
Gaussian closure and not an exact continuous-SDE error certificate.

For stable implementation retain log M0 and conditional mean m and variance v
of A given each intermediate state. Initially A=0, so m=v=0. A reward shift adds
a to m and leaves v unchanged. For every positive two-term Thomas mixture
with weights w0,w1 supplied by the existing `_mix`, use

\[
 m=w_0m_0+w_1m_1,\qquad
 v=w_0v_0+w_1v_1+w_0w_1(m_0-m_1)^2.
\]

This is the law of total variance. Apply the same nested order for three-term
explicit additions as in native v2. Division by a positive pivot changes only
log amplitude. True-zero branches retain the original handling. This avoids
subtracting nearly equal raw second moments and propagates two real fields
instead of a Fourier bank. Independent neutral mass/sign checks remain.
Do not floor log tails, clip moments, renormalize outside Bayes, or smooth the
original demand-map kink.

At the hour end, with probabilities p_j=exp(log M0_j),

\[
 \mu_A=\sum_jp_jm_j,\qquad
 V_A=\sum_jp_j\{v_j+(m_j-\mu_A)^2\},\quad
 \mu_Y=\mu_A,\quad V_Y=V_A+\sigma_y^2.
\]

Only the interval accumulator moments reset for the next hour; full log-state
continues. Across a B/U change *within* an hour, carry m,v and keep the original
Delta. Across missing observations, propagate state without an update. These
are plug-in moments under frozen physical/curve inputs, not uncertainty in
estimated parameters or future prices. Export all 168 hourly means/variances;
do not call mean +/- 1.645 standard deviations a 90% band without a Gaussian
law. Base/fine mean and variance differences must each be <=0.002 in normalized
units, in addition to the existing state probability/precision gates.

## Open-week distributions only at the seven declared horizons

Clone the selected and linear post-selection log-states at 21 February 00:00,
one clone per grid. The open branch never consumes final Y. A separate final
filter branch supplies the preserved 168-hour prequential log-score and its
own fine comparison. The two branches cannot share post-origin posteriors.
Deduplicate winner/linear only when the physical curves and full inputs agree.

The open branch has hourly moments throughout the week. For h in
(1,6,12,24,48,72,168), also calculate the distribution of the hour [h-1,h],
using its saved pre-hour open state and one **untilted** characteristic pass.
These are hourly-average/energy targets at horizon h, not total energy over
[0,h]. All horizons share one forward prefix per curve/grid; no seven repeated
week propagations. State/reward evolution is the same native operator.

For endpoint j, the existing kernel returns log p_j and conditional transform
c_j(omega). Marginalize the endpoint and convolve observation noise:

\[
 \chi_Y(\omega)=e^{-\sigma_y^2\omega^2/2}
                 \sum_j e^{\log p_j}c_j(\omega).
\]

The next open prior is the returned **full untilted log-state**, never an
observation-conditioned posterior or log of an underflowed ordinary mass.
A global characteristic function suffices for a marginal CDF; resolving every
endpoint's tiny conditional density by multiple observation-dependent contours
is unnecessary for this output. The filtering likelihood still uses its full
terminal-tilt guards and contour bank.

Since A is in [0,1], choose a=-r, b=1+r, L=b-a, with
L=max(2,1+16 sigma_y), r=(L-1)/2. Then the Gaussian mass outside [a,b] is at
most epsilon_alias=2 Phi(-r/sigma_y). For omega_n=2 pi n/L and
beta=2 pi^2 sigma_y^2/L^2, the truncated periodic CDF is

\[
 F_N(y)=\frac{y-a}{L}+\frac2L\operatorname{Re}
 \sum_{n=1}^N\chi_Y(\omega_n)
       \frac{e^{-i\omega_n a}-e^{-i\omega_n y}}{i\omega_n}.
\]

Wrapping changes the CDF by at most the outside probability: event indicators
agree inside [a,b]. Using |chi_A|<=1, the omitted Fourier contribution is bounded by

\[
 \epsilon_{trunc}\le\frac2\pi\sum_{n>N}\frac{e^{-\beta n^2}}n
 \le\frac{E_1(\beta N^2)}\pi
 \le\frac{e^{-\beta N^2}}{\pi\beta N^2}.
\]

Choose the smallest N>=64 with this last bound <=1e-10; require alias <=1e-10.
Freeze L,N from declared sigma_y before final observations are accessible and
price their actual work. Small sigma_y may require more than 64 frequencies;
the illustrative timing below assumes 64. Do not silently change settings
inside the final run or reuse an unpriced 64-frequency rate for a larger bank.

The bounds concern wrapping/truncation for the ideal discrete transform.
They do not certify floating-point or continuous-SDE error. Preserve the
computed chi(0) in the implemented constant term, check its mass error, and
never repair the distribution by clipping or renormalization. Guard finite
values, CDF range/monotonicity at saved evaluation points and quantile brackets;
check base/fine maximum CDF difference <=0.002 on 8N+1 equally spaced
points in [a,b], added quantile bracket endpoints and observed Y locations
after forecast freeze. Require 0.05/0.5/0.95 quantile differences <=0.002;
bisect to absolute width 0.0002 with at most 100 iterations, otherwise
record the unresolved bracket. Report a
failed/ambiguous inversion explicitly. Gaussian constant-reward and saved
finite-state cases provide small independent checks. Do not interpret these
empirical grid comparisons as a universal CDF roundoff proof.

Freeze and save all open forecasts before opening final Y. Use a seven-day
seasonal-naive price forecast: repeat the hourly price week immediately before
the origin with the same frozen F scales. This accesses past prices only;
baseline B remains the MED-only history-based baseline fitted before prior
origin. Final prequential filtering may use contemporaneous realized inputs
under its separate declared availability contract. Open-forecast errors also
include price-forecast error; no known-future-price/oracle claim is made.

Report all-hour mean RMSE/MAE and the seven declared marginal bands/PIT values.
Seven coverage outcomes are descriptive, not a calibration certificate. A
single final week does not establish population coverage; the old population
acceptance requirements remain prerequisites for any such later claim. No
168-hour simultaneous/path band is claimed from these marginal calculations.

## Complete remaining-work graph

Use one 8-worker pool with BLAS=1; no second optimizer pool and no division of
candidate counts by eight. Scalar moments/prior lead may use one worker.
The following conservative graph assumes all declared data hours are observed,
three distinct retained families and a nonlinear winner distinct from linear.
Rates are maximum end-to-end measured times for each actual job type/grid,
including normal IPC/checkpoint/export work. Apply the factor 1.5 once.

| Job | Base units | Fine units | Rate type |
|---|---:|---:|---|
| 49 full 48-hour TRAIN candidates | 2352 | 0 | observed likelihood |
| Each candidate's 24-hour prediction lead | 1176 | 0 | neutral propagation |
| Three 24-hour selection continuations | 72 | 0 | observed likelihood |
| Three retained TRAIN+selection precision streams | 0 | 216 | observed likelihood |
| Their fine prior leads | 0 | 72 | neutral propagation |
| Final prequential winner and linear | 336 | 336 | observed likelihood |
| Selection/final pre-update interval moments | 408 | 408 | real moments |
| Open-week hourly interval moments | 336 | 336 | real moments |
| Seven open distribution horizons, two curves | 14 | 14 | global CF/CDF |

TRAIN candidates do not need moments for optimization. Retained checkpoints
supply filter states before each scored hour for the separate mean/variance
pass. Fine trajectories keep their own states; they never initialize from a
coarse posterior. Open-week states are computed by the moment recursion and
its full log-amplitude, with all saved forecast horizons coming from that path.
Matching completed work is removed from future counts only after raw identity
validation. Count failed/interrupted attempts, discarded pilot conditions and
optimizer reconstruction as actual charged work.

Let b_g,p_g,m_g,c_g be the four measured rates at grid g. With N=49,T=48,S=24,
H=168,L=24,K=7, the priced future work before auxiliary jobs is

\[
 1.5\{(NT+3S+2H)b_b+[3(T+S)+2H]b_f
       +NLp_b+3Lp_f+(3S+4H)(m_b+m_f)+2K(c_b+c_f)\}.
\]

Add the latest shared ledger, all unfinished auxiliary work and 60 seconds for
orderly stop. Reserve 600 seconds for the explicitly itemized auxiliary jobs:
300 for new focused controls/nonreusable historical rate checks, preparation,
imports/startup; 300 for final diagnostics/export/handoff. These allocations
are unmeasured caps, not evidence that the jobs complete within them. The
implementation must list and price each subjob; excess work fails admission.
Required work cannot disappear into an assumed zero or be relabeled optional.

Reserve the **entire** remaining validation graph plus its 300 seconds and
stop allowance before search can spend time. Retain the minimum 7200-second
reserve, but increase it whenever the priced validation is larger. Under the
illustration it is **14431.28 seconds, about 4.01 hours**, so the old two-hour
reserve is insufficient. Do not add the reserve twice to the total: it is a
protected allocation within the same work graph and ledger.

### Conditional numerical illustration, not runtime admission

Reuse the saved v2 maximum observed-hour rate b_b=4.0682129 s and the saved
missing-hour maximum p_b=0.5607367 s. For illustration only, set m_b=p_b,
c_b=1.5420629+0.3321637=1.8742266 s (maximum warm contour plus maximum observed
outside-contour wall time), and multiply each fine rate by 2*2561/1921.
Moment/CDF rates and this fine multiplier are **unmeasured proxies**. The
single-contour sum is not an actual historical open-forecast timing.

| Component | Illustrative planned hours |
|---|---:|
| TRAIN candidates | 3.987 |
| All prior leads | 0.320 |
| Selection + retained fine streams | 1.098 |
| Final prequential base/fine streams | 2.088 |
| All predictive/open interval moments | 0.637 |
| Fourteen distributions per grid | 0.040 |
| Already charged + auxiliary allocation + stop | 0.589 |
| **Total** | **8.759** |

Doubling the unknown moment rates raises the illustration to 9.397 h. Actual
historical narrow-noise/curve behavior, longer work or slower fine/CDF rates
can still make admission fail. The 4–6-hour target would need substantially
lower actual work than this conservative illustration; do not promise it.
There is no claim that shortening TRAIN alone guarantees optimizer completion.

## Narrow implementation and go/no-go handoff

Return to Sol High. The mathematical choice is complete; no further Astra
review or blanket permission is required to implement this declared design.

1. Reuse the existing historical boundary/curve geometry, native likelihood,
   checkpoints and cumulative ledger. Add the isolated conditional-moment
   recurrence and open-interval distribution adapter. Keep legacy numerical
   kernels, G and WM unchanged; all new settings/source paths get their own
   immutable manifests. Expose posterior versus open-state ownership clearly.
2. Use one focused development check of the *new* moment/CDF identities:
   asymmetric positive finite-state CN/Strang recurrence against an independent
   direct matrix calculation, true-zero/tiny-log support, a within-hour input
   change retaining Delta, and constant reward plus Gaussian noise. Reuse saved
   native/FV evidence; do not rerun the completed 40-hour timing cases. No broad
   grid/solver/Windows suite. Any actual new numerical work uses the remaining
   464.10 native seconds, including imports/JIT, with 60 seconds stop reserve.
3. Finish the F-only data packet before historical likelihood work. Reuse the
   audited 13 IDs; obtain exact header mapping from the existing header report.
   Freeze F price role/area/scales, MED-only earlier baseline, Beta prior and
   C,k,Xi,sigma_x,sigma_y with real provenance. Do not infer physical parameters
   from synthetic fixtures or copy G's estimated condition automatically.
   The selected prior origin is 17 February; baseline and scale fit cutoffs
   must be earlier. The legacy incomplete hourly config has obsolete numerics
   and is not this study's config. Exposure audit remains blocking.
4. Once new identities and the data contract pass, use a **single bounded
   representative rate gate** for the actual base/fine observed, neutral,
   moment and forecast-CDF job types. Use registered pre-final conditions,
   reuse its exact saved work when possible and charge all nonreusable pilot
   work within the explicit auxiliary allocation. Do not launch an unbounded
   pilot just to price it. Enforce a parent/worker deadline and checkpoint.
5. Admit the optimizer only if the complete graph with measured rates and all
   data/numerical prerequisites fits the remaining ten-hour cap while protecting
   the actual validation reserve. Otherwise retain `historical_fit_ready=False`
   and report the specific missing/too-expensive work. No automatic further
   period shortening, call-cap increase, solver replacement or omitted family.
6. Freeze winner/protocol and complete pre-final fine/forecast checks before
   final Y is decoded. Final data can score the saved forecasts and feed the
   separately declared prequential filter; it cannot choose/refit a model or
   tune the numerical protocol. Final failures remain failures. Export the
   conditional curve artifact only with its actual scope and acceptance flags.

The short study can complete its own declared H2/H3 comparison only after these
gates and real results. This review does not mark historical H2/H3, operation-
wide weekly forecasting, the original longer study, all-month validation or
F4-I/J complete. Later broader claims retain their own evidence requirements.

## Sources and attribution

- Johan/Thomas, *FP2025*, section 3.2.1, printed pp. 5–6: the existing BM1
  monotone f/g, logistic flexibility and bounded demand construction. This
  review retains that model and the original linear reference.
- DTU *Lecture08*, slides 5–7: stochastic state/observation distinction and
  time-scaled process noise. Its linear Gaussian formulas do not justify
  replacing this nonlinear interval filter by ordinary least squares.
- Nocedal & Wright, *Numerical Optimization*, 2nd ed., chapter 9, pp. 220–225:
  objective-evaluation noise, model-based derivative-free trust regions and
  interpolation costs. This motivates the retained optimizer class, not a
  guarantee for BM1's kinked discrete objective or its limited call budget.
- [SciPy COBYQA options](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html)
  and [COBYQA method/termination documentation](https://www.cobyqa.com/stable/ref/generated/cobyqa.minimize.html).
  The installed SciPy 1.17.0 wrapper was also inspected; no package upgrade.
- The conditional mean/variance recurrence and the Gaussian CDF tail bounds
  above are derived here from the already implemented positive split operator
  and elementary probability/Fourier inequalities. They are implementation
  specifications pending the declared checks, not results asserted by the
  supplied textbook or changes to the supervisor's legacy basis.

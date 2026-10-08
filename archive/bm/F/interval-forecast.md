# F: intervalmomenter, CDF og åben prognose

<a id="h2-interval-forecast-implementation"></a>
<a id="h2-compact-estimation-review"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../root/BM_GUIDE.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

De matematiske afsnit nedenfor beskriver den implementerede F-adapter.
Formuleringer om at tilføje den er bevaret som udledningens metodebeskrivelse;
de er ikke en ny opgave. Hvert horizon-output er den pågældende times
gennemsnit, ikke akkumuleret energi siden origin. Filtrering og åben prognose
har forskellige post-origin informationsgrænser.

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


## Implementeret interface og uafhængig kontrol
## Implemented interfaces

The isolated F module is `src/method/bm_1_interval_forecast.py`.

| Interface | Meaning |
|---|---|
| `native_interval_moments` | Same native positive CN/Strang operator; full log amplitude plus conditional mean/variance. Adds observation noise only to the reported Y variance. |
| `interval_cdf_plan` | Immutable noise/span/frequency plan with analytic alias and truncation bounds, each <=1e-10. Small noise can require N>64; price the actual N. |
| `native_interval_distribution` | Existing **untilted** native CF, endpoint marginalization and Gaussian observation convolution. Returns full log state, CDF audit and 0.05/0.5/0.95 quantile brackets. |
| `compare_interval_moments` | Base/fine interval mean and variance gate, each <=0.002 in normalized units. The caller retains its separate state-grid gates. |
| `compare_interval_distributions` | Base/fine CDF and quantile gates, each <=0.002; declared 8N+1 grid, quantile endpoints and optional post-freeze observed locations. |
| `forecast_open_intervals` | One chronological prefix, 168 hourly moments by default and distributions at 1/6/12/24/48/72/168. No observations argument or Bayes updates. |
| `interval_forecast_checkpoint` | Payload for existing real-array `.npz` I/O, preserving -inf true zeros, finite underflowed log tails and split real/imaginary CF arrays. |

Each hour describes [h-1,h], not cumulative week energy. Reward half-steps
retain the original hour's Delta across input changes. Accumulator fields reset
only at hour boundaries; full log state continues. No moment subtraction,
Gaussian closure, flooring, clipping, probability renormalization or CDF shape
repair is introduced. Outside the chosen support, the *compact periodic-law
CDF* is defined as zero/its actual mass; this is not clipping an invalid CDF.
Its approximation allowances do not certify floating-point/continuous-SDE error.

The caller owns immutable manifest validation, future input availability,
parent/worker deadline, eight-worker mapper, common ledger, state-grid precision
and exported physical units. In particular, these library functions do not
start workers, admit a budget or establish holdout blindness. An interrupted
week continues from its last complete hour using `completed_intervals`, the
saved boundary log state and only remaining operators. Verify source/input/
grid/plan identities first; preserve earlier rows/distributions in their own
checkpoints. A callback can save each complete payload through the existing
`_atomic_report` with a `.npz` filename. Existing checkpoints are not overwritten.

The filtering likelihood, native v2 kernel, legacy endpoint `forecast_fv`,
G and WM source are unchanged. No historical measurements have been decoded.

## Focused evidence and shared time

Runner: `root/f4/check_h2_interval_forecast.py`; independent control:
`src/validation/bm_1/interval_forecast.py`. The **final** immutable evidence is
`files/output/bm_1/runs/estimation/f4_h2_interval_forecast_control_final_20261006.json`.
Its 57 source hashes and Python/NumPy/SciPy/Numba/llvmlite environment are saved.

| Check | Result |
|---|---:|
| Independent dense raw moments vs log conditional moments | maximum error 1.11e-16 |
| Within-hour input-change continuation vs combined operator | exact |
| Independent dense characteristic transform | maximum error 3.51e-16 |
| Analytic Gaussian CDF, sigma=0.01 and N=196 | maximum error 4.65e-11 |
| Analytic Gaussian quantiles | maximum error 5.45e-5, bracket width <=0.0002 |
| True zero and finite log mass -900 | preserved |
| Hourly accumulator reset and caller prior ownership | passed |
| Hour-boundary resume and real-array checkpoint roundtrip | exact |
| Deliberate precision violation, invalid prior and deadline stop | rejected |

The first invocation found the environment's pinned Numba dependency missing;
that failed import, restoring Numba 0.62.1/llvmlite 0.45.1 and both source-specific
controls were all charged. The first completed control remains archived; the
second adds the required moment gate, provenance and checkpoint/resume checks.
No completed old 40-hour native/FV cases or broad suites were repeated.

Total new charge: **33.87002325399999 s**, including installation/import/JIT/export.
Shared H ledger: **1494.4015500329392 s**. Native development:
**469.7704293930642/900 s**, leaving **430.2295706069358 s** across machines/resumes,
including the mandatory 60-second invocation stop reserve. The native seed
extends the exact previous PC/chat history. The ledger snapshot is
`files/output/bm_1/runs/estimation/f4_h2_interval_forecast_ledger_after_20261006.json`.
The final control itself charged 3.07888727100044 s. These are small-control
times, not measured BM grid/forecast-hour rates.


Den aktive kommandos datoer og outputkontrakt står i [F4_RUN](../../../root/F4_RUN.md).
Det historiske H-forløbs egne valgte posteriorer må ikke forveksles med den
selvstændige F4-kørsels nye deklarerede Beta-prior. Syv marginale bånd er
hverken et simultant ugebånd eller en dækningskalibrering.

## Konkrete kode- og kontrolindgange

[forecast-adapter](../../../src/method/bm_1_interval_forecast.py), [F4-output-tests](../../../tests/test_bm1_f4_runner.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.

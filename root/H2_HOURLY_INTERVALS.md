# H2: hourly interval observations

Updated 5 October 2026. Johan declares one-hour consumption, provisionally kWh,
with hour 03 meaning [03:00,04:00). Europe/Copenhagen is a working timezone.
These are **working assumptions**, not independently verified meter metadata.
Population, frozen scales/baseline, price/proxy and split/exposure remain open.

For frozen power scale P in kW and elapsed duration Δh in hours, energy E in
kWh becomes y=E/(P Δh). The normalized observation model is

\[
Y_i=(t_i-s_i)^{-1}\int_{s_i}^{t_i}D(v,X_v)\,dv+\epsilon_i,
\quad\epsilon_i\sim N(0,\sigma_y^2).
\]

One hour is 1/24 day. In one hour, kWh and mean kW have equal numerical values
while retaining interval meaning. Baseline uses the power-scale unit; sigma_y
uses **normalized average power**. C and sigma_x retain the day contract.
Gaussian observations are not clipped. Availability is at the interval end;
prefixes exclude partial intervals. Last TRAIN ends at MODEL_SELECTION start,
before selection measurements are read. UTC elapsed time preserves DST;
ambiguous/nonexistent naive workbook hours still fail rather than being inferred.

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

## Development evidence

The short panel uses 24 constructed hourly observations (two missing), fixed
physical/noise parameters and Beta(2,3). It is not simulated truth or recovery.
Two-/four-piece fits have ceilings 6/8; both are negative budget stops. No
TRAIN winner is selected. Saved runtime was 10.87 s with BLAS=1; daily replays
ranged from 0.23 s (81 cells, h=1/384 day, da=1/128) to 5.97 s
(321,1/1536,1/512). Maximum mass error was below 3.34e-15.

**All initial BM comparisons failed** the predeclared development tolerances:
total log-likelihood 0.05, terminal mean 0.002, terminal mass L1 0.05. Base
score differed by 6.55 and the combined 161-cell replay by 2.18. Mass
conservation does not establish likelihood accuracy; coarse settings cannot
be recommended for historical fitting. The extra bounded refinement uses
321/641/1281 cells, h=1/1536,1/3072,1/6144, da=1/512,1/1024,1/2048, without
another optimizer run. Reports are linked from the H2 contract.

The finer panel took 407.88 s. Daily 321/641/1281-cell replays took
5.49/49.60/352.54 s; maximum mass error was 1.05e-14. Against 1281, total
score errors were 1.368 and 0.416, both above 0.05. Terminal mean/L1 passed
their development tolerances (641: 0.000243 / 0.00897). This is a negative
likelihood-readiness result. The next task is a more efficient accurate
interval kernel/numerical design, not a full historical optimizer run.

An independent three-state tilted-matrix/Fourier Gaussian reference agrees
at quadrature orders 96/160. Joint time/accumulator resolutions 64/256/512
give score errors 0.0151/0.00781/0.00434 and posterior maximum errors
0.0241/0.00699/0.00358. This checks the additive law on a fixed finite state
space, not the continuous-state historical problem. Tests also cover constant
Gaussian evidence, within-hour jumps, missing/gap causality, full/likelihood-only
identity, DST hour-start and energy conversion, and bounded pause/resume.
No historical values were read; no H2/H3/model-selection/forecast acceptance.

## Commands

```powershell
python -B -m src.exe.run_model.bm_1.curves history-config --hourly-assumptions --output files/output/bm_1/runs/estimation/h2_hourly_config.json --blas-threads 1
python -B -m src.exe.run_model.bm_1.curves interval-pilot --output files/output/bm_1/runs/estimation/interval_development.json --blas-threads 1
python -B -m src.exe.run_model.bm_1.curves interval-pilot --refine --output files/output/bm_1/runs/estimation/interval_refinement.json --blas-threads 1
```

Use new output paths; reports are preserved. The config records working
assumptions but remains incomplete. `history-fit` rejects missing prerequisites
before measurement access. Interval replay needs explicit accumulator spacing
and node budget; quadrature_order pertains only to the preserved point path.
Source hashes and a weighted remaining/total ETA are recorded; source changes
abort report publication. A passing declaration check is not evidence.

Mathematical reference: [Fitzsimmons and Pitman (1999)](https://www.stat.berkeley.edu/~pitman/kac.pdf)
on additive Markov functionals/Feynman-Kac. The joint PDE and control above
are derived for this model; the paper is not prototype acceptance evidence.
API reference: [SciPy dgtsv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgtsv.html).
Existing normalization and limitations: [BM_REFERENCE.md](BM_REFERENCE.md).

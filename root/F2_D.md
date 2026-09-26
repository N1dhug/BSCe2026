# F2-D: Monte Carlo versus pure finite-volume reference

F2-D compares two independent numerical descriptions of the same stochastic
BM1 model:

- Euler-Maruyama Monte Carlo paths;
- the pure one-dimensional finite-volume Fokker-Planck reference from F2-C.

Only BM-specific files are changed in this step. Shared method, solver and tool
files used by Water Models are left untouched.

## 1. Shared BM1 reference case

The common controlled experiment is now defined in
`src/exe/run_model/bm_1/f2_reference_case.py`.

The shared values are

$$
T=[0,1]	ext{ day},qquad B=0.3,qquad C=1,qquad k=2,qquad Xi=1,
$$

$$
sigma_X=0.5 mathrm{day}^{-1/2},
$$

with hourly output and the price change

$$
0.75longrightarrow0.25
$$

at 12.5 h.

The original F2-B EM reference keeps its deterministic point initial state
$X_0=0.5$. F2-C/F2-D use the shared smooth comparison law

$$
X_0simoperatorname{Beta}(20,20).
$$

For finite volume, exact cell probabilities are computed from Beta CDF
differences. For Monte Carlo, initial states are sampled directly from the same
Beta law.

## 2. Same stochastic model

Both methods approximate

$$
dX_t
=
a(t,X_t),dt
+
sigma_X X_t(1-X_t),dW_t,
$$

with the same BM1 drift, model parameters, price signal, baseline, output times
and price breakpoint.

The Monte Carlo simulation uses `boundary="none"`. Raw Euler-Maruyama
boundary violations remain diagnostic failures rather than being reflected or
clipped.

The finite-volume route uses the mass-conserving F2-C reference and performs
no clipping or renormalization.

## 3. Distribution comparison

Monte Carlo states are binned on exactly the finite-volume cell edges. Hence

$$
m_i^{MC}(t)
$$

and

$$
m_i^{FV}(t)
$$

refer to the same cells.

The comparison reports

$$
mathbb E[X_t],qquad
operatorname{Var}(X_t),qquad
mathbb E[D_t],
$$

as well as three distribution distances.

Cell-mass L1 distance:

$$
L_1(t)
=
sum_i
left|
m_i^{MC}(t)-m_i^{FV}(t)
ight|.
$$

Discrete CDF sup distance:

$$
d_{infty}^{CDF}(t)
=
max_i
left|
sum_{jle i}m_j^{MC}
-
sum_{jle i}m_j^{FV}
ight|.
$$

On the uniform grid, a discrete approximation of the one-dimensional
Wasserstein-1 distance is

$$
W_1(t)
approx
Delta x
sum_i
left|
F_i^{MC}-F_i^{FV}
ight|.
$$

These distribution metrics avoid introducing a kernel-density bandwidth or a
second set of histogram bins.

## 4. Monte Carlo uncertainty

For $M$ paths,

$$
operatorname{SE}(ar X_t)
=
rac{s_X(t)}{sqrt M},
$$

and the same standard-error calculation is used for expected demand.

These intervals quantify Monte Carlo sampling error. They do not include EM or
finite-volume discretization error, so disagreement outside a Monte Carlo
interval is not by itself proof that either method is incorrect.

## 5. BM-only repository changes

- `src/exe/run_model/bm_1/f2_reference_case.py`: common F2 reference inputs
  and Beta initial-law conversion/sampling.
- `src/exe/run_model/bm_1/em_reference.py`: reuses the common inputs while
  preserving the original point-initial F2-B experiment.
- `src/exe/run_model/bm_1/fp_reference.py`: reuses the common inputs and
  shared Beta cell masses.
- `src/exe/run_model/bm_1/mc_vs_fp_reference.py`: F2-D comparison.
- `src/diagnostics/bm_1.py`: BM-only cell-mass comparison diagnostics.
- `tests/test_bm_1_f2_mc_vs_fp.py`: shared-case, metric and small integration
  tests.

No WM model file and no shared method/solver/tool implementation is changed.

## 6. Run

From the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_mc_vs_fp.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.em_reference
python -B -m src.exe.run_model.bm_1.fp_reference
python -B -m src.exe.run_model.bm_1.mc_vs_fp_reference
```

The F2-D executable writes

`files/output/graphs/bm_1/bm_1_mc_vs_fp_reference.png`.

The main comparison uses 50,000 Monte Carlo paths and 201 finite-volume cells.
The result should be interpreted by inspecting moment differences,
distribution distances, raw EM boundary diagnostics and FVM mass diagnostics
together rather than by using one arbitrary scalar pass/fail threshold.

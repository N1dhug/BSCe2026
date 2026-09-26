# F2-B verification: coupled Euler-Maruyama convergence

This note verifies the Euler-Maruyama machinery introduced in F2-B before the
project proceeds to the finite-volume Fokker-Planck reference. No BM1 model
equation is changed.

## 1. Strong convergence

For an SDE

$$
dX_t = a(t,X_t)\,dt + b(X_t)\,dW_t,
$$

Euler-Maruyama with step size $h$ is

$$
X_{n+1}^{h}
=
X_n^{h}
+
a(t_n,X_n^{h})h
+
b(X_n^{h})\Delta W_n,
\qquad
\Delta W_n\sim\mathcal N(0,h).
$$

A numerical method has strong order $p$ when a pathwise error measure satisfies

$$
\left(\mathbb E|X_T-X_T^h|^2\right)^{1/2}
=
O(h^p).
$$

Under the standard regularity assumptions Euler-Maruyama has strong order
$p=1/2$. Therefore halving $h$ should asymptotically reduce the RMS pathwise
error by approximately

$$
2^{-1/2}\approx 0.707.
$$

The observed local order is estimated from two successive errors by

$$
p_h
=
\frac{\log(E_h/E_{h/2})}{\log 2}.
$$

A second estimate is obtained as the slope in a least-squares fit of

$$
\log E_h = c + p\log h.
$$

## 2. Why the Brownian paths must be coupled

Strong convergence compares numerical approximations of the **same random
realization**. Fine Brownian increments are generated once on the finest grid.
Coarser increments are sums of the fine increments:

$$
\Delta W_k^{(c)}
=
\sum_{j=0}^{r-1}
\Delta W_{rk+j}^{(f)}.
$$

This is exactly the `coarsen_brownian_increments` construction added in F2-B.
Independent Brownian paths would mix discretization error with the difference
between two different random realizations and therefore would not be a strong
convergence test.

## 3. Exact logistic-diffusion benchmark

BM1 itself has no known closed-form solution, so the implementation is first
checked on an SDE with the same diffusion structure.

Let

$$
Y_t = \operatorname{logit}(x_0)+\sigma W_t,
\qquad
X_t = \operatorname{sigmoid}(Y_t).
$$

Ito's formula gives

$$
dX_t
=
\frac12\sigma^2X_t(1-X_t)(1-2X_t)\,dt
+
\sigma X_t(1-X_t)\,dW_t.
$$

Thus the diffusion is exactly

$$
b(x)=\sigma x(1-x),
$$

while the exact solution is available from the transform above. The script
uses the coupled levels

$$
h,\quad h/2,\quad h/4,\quad h/8
$$

with 1 h as the coarsest step and measures

$$
E_h
=
\left[
\frac1M
\sum_{m=1}^{M}
\left(
X_T^{h,(m)}-X_T^{(m)}
\right)^2
\right]^{1/2}.
$$

This is the primary implementation check for the expected EM strong order
near $1/2$.

## 4. BM1 self-convergence

For BM1 there is no exact terminal state available. Instead the same Brownian
paths are used on four levels and successive approximations are compared:

$$
E_h^{\mathrm{self}}
=
\left[
\frac1M
\sum_{m=1}^{M}
\left(
X_T^{h,(m)}-X_T^{h/2,(m)}
\right)^2
\right]^{1/2}.
$$

If the method is in its asymptotic regime and has order $p$, then

$$
E_h^{\mathrm{self}}
\propto h^p.
$$

The BM1 study uses a constant admissible price and the canonical linear
responses to isolate the SDE time-discretization error from input
discontinuities. The coarsest BM1 step is 15 min and the levels are
15, 7.5, 3.75 and 1.875 min.

This is a self-convergence diagnostic, not a proof of the exact BM1 error.
Pre-asymptotic orders need not equal $1/2$ exactly.

## 5. Boundary handling

Every level uses `boundary="none"` and records the raw proposal diagnostics
introduced in F2-B. The convergence study aborts if any proposal leaves
$[0,1]$. Reflection or clipping would change the numerical scheme and could
hide a step-size problem.

## 6. Repository layout

- `src/diagnostics/stochastic.py`: reusable RMS error and observed-order
  calculations.
- `src/diagnostics/__init__.py`: exposes the stochastic diagnostics.
- `src/exe/run_model/bm_1/em_convergence.py`: exact benchmark and BM1
  self-convergence experiment.
- `tests/test_bm_1_f2_em_convergence.py`: focused regression tests.

The BM1 model, Euler-Maruyama solver, validation layer and Fokker-Planck code
are unchanged by this verification step.

## 7. Run

From the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_em_convergence.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.em_convergence
```

The experiment writes

`files/output/graphs/bm_1/bm_1_em_convergence.png`.

The exact logistic benchmark is expected to give a fitted strong order close
to $1/2$. The BM1 coupled errors should decrease under refinement; their local
orders are interpreted as empirical self-convergence evidence rather than an
exact theoretical order measurement.

## 8. Sources

The Euler-Maruyama formula, Brownian-path coupling idea and the distinction
between strong and weak convergence follow Higham (2001), *An Algorithmic
Introduction to Numerical Simulation of Stochastic Differential Equations*,
especially Sections 4-5. The exact logistic transform is used here as an
algorithmic benchmark because it preserves the F2 diffusion form
$\sigma x(1-x)$.

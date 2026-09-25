# F2-B: controlled Euler-Maruyama and Monte Carlo reference

F2-B turns the stochastic BM1 path simulation from F2-A into a controlled numerical reference. It does **not** change the BM1 SDE and does not modify the Fokker-Planck or beta-mixture solver.

## Mathematical reference

For

$$dX_t=a(t,X_t)\,dt+b(X_t)\,dW_t,$$

Euler-Maruyama uses

$$X_{n+1}=X_n+a(t_n,X_n)h_n+b(X_n)\Delta W_n,$$

with $h_n=t_{n+1}-t_n$ and $\Delta W_n\sim\mathcal N(0,h_n)$. For BM1,

$$a(t,x)=\frac{D(x,U(t),B(t))-B(t)}{C},\qquad b(x)=\sigma_Xx(1-x).$$

Known discontinuities in deterministic inputs must be integration boundaries because the EM coefficients are evaluated at the left endpoint of each numerical interval.

## Output and integration grids

`bm_1.euler_maruyama_solver` now distinguishes the returned output grid from the internal EM grid:

- `t_eval`: returned times;
- `breakpoints`: known input discontinuities inserted into the internal grid;
- `max_step`: maximum internal EM step.

The generic `euler_maruyama.build_time_grid` includes every output time and breakpoint and subdivides the remaining intervals until their length is at most `max_step`. Omitting the new controls preserves the previous grid behaviour.

## Wiener increments and coupling

Random-number generation is separated from the EM recursion. `brownian_increments(t, n_paths, seed)` generates increments with the correct stepwise variances, while `solve_paths(..., brownian_increments=dW)` accepts supplied increments directly.

For a fine/coarse comparison, the same Brownian path is obtained by summing fine increments:

$$\Delta W_k^{(c)}=\sum_{j=0}^{r-1}\Delta W_{rk+j}^{(f)}.$$

`coarsen_brownian_increments` implements this. Reusing only the same seed is not a mathematical coupling guarantee.

## Initial states

The EM route accepts either one shared initial state or one initial state per Monte Carlo path. All BM1 initial states must lie in $[0,1]$. The existing scalar stochastic entry remains in place for the current Fokker-Planck initializer, so F2-B does not silently change the FP initial-law interpretation.

## Raw boundary diagnostics

Every EM step first forms the uncorrected proposal

$$\widetilde X_{n+1}=X_n+a_nh_n+b_n\Delta W_n.$$

When diagnostics are requested, violations of $[0,1]$ are counted **before** optional `reflect` or `clip` handling. The output reports lower/upper violations, affected paths, per-step counts and the minimum/maximum raw proposal.

The controlled reference uses `boundary="none"` and stops if a raw proposal leaves $[0,1]$. Gaussian increments have unbounded support, so no positive fixed step can guarantee interval preservation for every realization; the diagnostic makes this finite-sample numerical issue visible instead of hiding it.

## Monte Carlo uncertainty

For $M$ independent paths,

$$\bar X_n=\frac1M\sum_{m=1}^M X_n^{(m)},\qquad
s_X^2(t_n)=\frac1{M-1}\sum_{m=1}^M\left(X_n^{(m)}-\bar X_n\right)^2,$$

and the estimated standard error of the mean is

$$\operatorname{SE}(\bar X_n)=\frac{s_X(t_n)}{\sqrt M}.$$

The reference script computes the same quantities for demand. These standard errors quantify Monte Carlo sampling uncertainty only; they do not include EM time-discretization error.

## Repository integration

- `src/method/euler_maruyama.py`: generic time-grid construction, Wiener increments/coarsening, EM recursion and raw proposal diagnostics.
- `src/validation/method/euler_maruyama.py`: generic validation for grids, initial states, increments and diagnostics.
- `src/model/bm_1.py`: BM1 wrapper with separate output/integration controls.
- `src/validation/model/bm_1.py`: BM1 time controls and vector initial-state validation for EM; scalar FP route remains separate.
- `src/exe/run_model/bm_1/em_reference.py`: controlled F2-B Monte Carlo reference.
- `tests/test_bm_1_f2_em.py`: regression tests for the F2-B numerical machinery.

## Run from repository root

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_em.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.em_reference
python -B -m src.exe.run_model.bm_1.em_price_step
```

The reference uses hourly output, a maximum internal step of 15 minutes, a deliberately non-hourly price breakpoint at 12.5 h, 20,000 paths and explicitly generated Wiener increments. It writes `files/output/graphs/bm_1/bm_1_em_reference.png`.

## What remains after F2-B

F2-B provides the machinery for a coupled EM convergence study but does not yet claim an observed strong or weak convergence order. The next larger step is F2-C: extract a pure one-dimensional finite-volume Fokker-Planck reference. F2-D can then compare Monte Carlo and Fokker-Planck using the same initial law, inputs and output times before beta-mixture error or stochastic optimization is assessed.

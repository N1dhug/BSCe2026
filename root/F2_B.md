# F2-B: controlled Euler-Maruyama and Monte Carlo reference

F2-B turns the existing stochastic BM1 path simulation into a controlled
numerical reference. It does **not** change the BM1 SDE fixed in F2-A and does
not modify the Fokker-Planck/beta-mixture solver.

## 1. Numerical equation

For

$$dX_t=a(t,X_t)\,dt+b(X_t)\,dW_t,$$

Euler-Maruyama on an internal integration grid gives

$$X_{n+1}=X_n+a(t_n,X_n)h_n+b(X_n)\Delta W_n,$$
$$\Delta W_n\sim\mathcal N(0,h_n).$$

For BM1,

$$a(t,x)=\frac{D(x,U(t),B(t))-B(t)}{C},\qquad
b(x)=\sigma_Xx(1-x).$$

The deterministic and stochastic increments are therefore evaluated at the
left endpoint of each EM interval. Known discontinuities in deterministic
inputs must be integration boundaries; otherwise one numerical step would use
one side of a discontinuous input across both sides of the jump.

## 2. Output grid versus integration grid

`bm_1.euler_maruyama_solver` now accepts

- `t_eval`: times returned to the caller;
- `breakpoints`: known input discontinuities that must be internal boundaries;
- `max_step`: maximum length of an internal EM step.

The generic method `euler_maruyama.build_time_grid` constructs an integration
grid containing every `t_eval` value and every breakpoint, and subdivides each
remaining interval until its length is at most `max_step`.

If the new controls are omitted, `t_eval=self.t`, there are no extra
breakpoints and `max_step=np.inf`. Thus the old output/integration grid is
preserved by default.

For the BM1 wrapper, `t_eval` must span the model interval. This keeps the
meaning of `init_cond` unchanged: it is the state at `T[0]`.

## 3. Supplied Wiener increments and Brownian coupling

Random-number generation is separated from the EM recursion. The generic
method provides

```python
dW = method.brownian_increments(t, n_paths, seed)
```

and `solve_paths(..., brownian_increments=dW)` accepts these increments
explicitly. A seed and supplied increments are mutually exclusive so the source
of randomness is never ambiguous.

For strong-convergence comparisons, coarse and fine solutions must use the
same Brownian path. If a coarse interval contains `r` fine intervals, then

$$\Delta W_k^{(c)}=\sum_{j=0}^{r-1}\Delta W^ý{rk+j}^{(f)}.$$

`coarsen_brownian_increments` performs this sum. Reusing only the same seed is.not mathematical coupling guarantee; explicit increment aggregation is.

## 4. Initial states

The generic EM method now accepts either one shared initial value or exactly
one initial value per Monte Carlo path. The BM1 EM entry accepts the same form
and additionally enforces that all initial states lie in `[0,1]`.

The scalar-only `stochastic_inputs` route remains in place for Fokker-Planck,
so this change does not silently reinterpret the current FP mixture initializer. A
common random initial law can therefore be supplied to Monte Carlo later
without pretending that the current FP beta initializer is already identical.

## 5. Raw boundary diagnostics

Euler-Maruyama does not inherit the continuous SDE's interval invariance at a
finite time step. Every step first forms the raw proposal

$$\widetilde X_{n+1}=X_n+a_nh_n+b_n\Delta W_n.$$

F2-B records violations of `[0,1]` **before** `reflect` or `clip` is applied.
With `return_diagnostics=True`, the method returns counts by step, total lower
and upper violations, the number of affected paths, and the minimum/maximum raw
proposal.

The controlled reference script uses `boundary="none"`. If any raw proposal
leaves `[0,1]`, the script stops rather than hiding the issue with a correction.
The existing `em_price_step.py` remains a separate demonstration of the older
reflection choice.

This does not prove that a particular time step is universally safe. Gaussian
increments have unbounded support, so no positive fixed EM step can guarantee
pathwise interval preservation for every realization. The diagnostics instead
make the finite-sample numerical issue visible.

## 6. Monte Carlo uncertainty

For `M` independent paths, the sample mean is

$$\bar X_n%=\frac1M\sum_{m=1}^M X_n^{(m)},$$

and the unbiased sample variance is

$$s_X^2(t_n)=\frac1{M-1}\sum_{m=1}^M(X_n^{(m)}-\bar X_n)^2.$$

The Monte Carlo standard error of the estimated mean is

$$\operatorname{SE}(\bar X_n)=\frac{s_X(t_n)}{\sqrt M}.$$

The reference script reports the same quantities for demand. These standard
errors quantify sampling uncertainty only; they do not measure EM time-step
error. Time-discretization error must be studied separately with coupled
Brownian paths and grid refinement.

## 7. Repository integration

| Location | F2-B responsibility |
| --- | --- |
| `src/method/euler_maruyama.py` | Internal grid construction, Wiener generation/coarsening, EM recursion and raw proposal diagnostics. |
| `src/validation/method/euler_maruyama.py` | Generic validation of grids, path initial states and supplied increments. |
| `src/model/bm_1.py` | BM1 wrapper with separate output/integration controls; model equations remain unchanged. |
| `src/validation/model/bm_1.py` | BM1 interval/state validation; scalar FP entry remains separate. |
| `src/exe/run_model/bm_1/em_reference.py` | Controlled F2-B reference experiment. |
| `tests/test_bm_1_f2_em.py` | Regression tests for F2-B numerical controls and coupling. |

The deterministic validation logic reuses a shared `time_controls` helper; its
accepted inputs and numerical behavior are intended to remain unchanged.

## 8. Run and review

From the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_em.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.em_reference
```

The reference experiment uses hourly output, a maximum internal step of 15 min,
an intentionally non-hourly price breakpoint at 12.5 h, 20,000 paths and
explicitly generated Wiener increments. It writes
`files/output/graphs/bm_1/bm_1_em_reference.png`.

A valid reference run should report zero raw interval violations. If it does
not, do not switch on reflection to make the warning disappear; first refine
the integration grid or reconsider the chosen reference parameters.

## 9. What F2-B does not establish

F2-B provides the machinery required for stochastic path validation, but it
does not by itself establish an observed strong/weak convergence order for BM1.
It also does not validate the current Fokker-Planck solver or make the MC and FP
initial laws identical.

The next natural work is:

1. use coupled fine/coarse Brownian paths for an EM convergence study;
2. extract a pure one-dimensional finite-volume Fokker-Planck reference (F2-C);
3. compare MC and FP with the same initial law and inputs (F2-D);
4. only then isolate beta-mixture error and stochastic-objective error.

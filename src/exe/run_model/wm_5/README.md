# WM5 nonlinear deterministic response / moving-horizon study

## Model

The deterministic WM5 is

```
    dF/dt = R
    dR/dt = r_F F + sum_i a_i phi_i(R) + sum_j b_j psi_j(U)
    I(t) = O(t) + R(t)
```

`O(t)` is the **same** fitted statistical baseline used by WM4. Signed
`F,R,U` are interpreted on [-1,1]. A response is supplied through a
`response_function(name, value, derivative, derivative_bounds)`, with
`phi(0)=0`, `phi'(x)>=0`, and its derivative bounds **certified over
the continuous interval [-1,1]**. The runtime samples the certificate to
catch errors but cannot formally prove arbitrary user-defined callables.

All coefficients `a_i,b_j <= 0` and `r_F<0`, yielding damping,
negative price sensitivity, and restoring flexibility. Linear inequalities
enforce WM4-equivalent expected equilibrium bounds, for all signed prices
and states: `F*(U) in [-1,1]` at `R=0`, and `R*(F,U) in [-1,1]` for the
rate equilibrium. Other linear inequalities enforce a conservative form of
the **Euler-discretized Jacobian Schur stability** for all `R in [-1,1]`,
using the supplied global derivative envelopes. This does **not** project
realized transient trajectories into [-1,1], just as WM4 equilibrium
restrictions do not impose hard path constraints. Stability guarantees
apply to this discrete model on the certified domain, not arbitrary
out-of-domain dynamics.

The library provides centered linear, cubic, quintic, exponential
`expm1(x)` and signed logarithmic `sign(x)*log1p(abs(x))` responses.
Candidates always include a linear feature when adding nonlinear rate
or price terms, so WM5 nests WM4. Coefficients may be zero.

## Parameter fitting

Given `R_obs=I_obs-O`, fix **R0=R_obs[0]** and reconstruct

```
    F_k = F0 + dt * sum_{j<k} R_obs[j].
```

The **parameter QP** minimizes the *rate-transition* squared residual
`sum (Delta R_obs - dt*(r_F*F + g(R_obs) + h(U)))**2`, subject to the
linear physical and numerical constraints above. Parameter fitting never
simulates candidate nonlinear trajectories.

The **initial-state QP** optimizes only `F0 in [-1,1]`. Since the
features of `R_obs` and `U` do not vary with `F0`, the residual
is affine in `F0`, so this is a scalar convex QP with a closed-form,
clipped least-squares solution. The solver alternates the two QPs until
convergence. Individual blocks are convex, but the *joint* optimization
is not necessarily convex due to the bilinear `r_F*F0` term. Saved
`state_converged` flags distinguish convergence from iteration limits.

**Fair WM4 control:** The executable calls WM4's existing alternating QP
with `R0` fixed to the **same observed** value, via equal lower/upper
`R0` bounds. It initializes both models with `F0=0`. Baseline,
normalization, time discretization, transition mask, observation mask,
window timing and state bounds match.

## Moving-horizon evaluation

Historical parameter fits use the preceding 24 hours for each hourly
prediction origin. Future oracle performance uses the next 24 hours
with the parameters held fixed. WM4's oracle optimizes **both future
initial states** using its exact linear least-squares QP. WM5's oracle
also optimizes both states in [-1,1], but it requires bounded nonlinear
least squares with analytic sensitivity and multiple starting points;
its oracle solution is **not guaranteed to be globally optimal**.
These future-optimal initial states are retrospective diagnostic upper
bounds, not deployable forecasts.

Run from repository root:

```bash
python -m src.exe.run_model.wm_5.moving_horizon --max-origins 3 --variants linear_linear,poly3_linear,exp_linear,log_linear
python -m src.exe.run_model.wm_5.moving_horizon --stride 24 --variants all
python -m src.exe.run_model.wm_5.moving_horizon --stride 1 --variants all
```

The full hourly experiment with all nonlinear variants is computationally
expensive. The default samples all hourly origins; start with `--stride 24`
or `--max-origins` for checks. Use `--variants` to select a
comma-separated subset, `--groups ALL,MED`, and adjust
`--max-state-iter`, `--state-tol`, `--oracle-starts`, and
`--oracle-nfev` as necessary.

Configurations: `linear_linear`, `poly3_linear`,
`poly5_linear`, `exp_linear`, `log_linear`,
`linear_poly3`, `linear_exp`, `linear_log`,
`poly3_poly3`, `exp_exp`, `log_log`, `poly3_exp`.
All are compared against the separate WM4 `wm4_linear` control.

Output folder: `files/output/wm_5/moving_horizon/`.

- `fit_parameters.csv`: per-model fitted coefficients, training MSE,
  objective, iterations, convergence and runtime, plus oracle diagnostics.
- `oracle_horizon_errors.csv`: one record per successful 24h
  prediction, with 24 horizon errors.
- `matched_oracle_summary.csv`: pooled RMSE, baseline RMSE,
  training RMSE, fitting time and convergence on **identical forecast
  origins** where *all* requested model variants succeeded.
- `matched_oracle_horizon.csv`: RMSE/MAE/bias for every forecast hour
  on the same paired evaluation cohort.
- `oracle_summary_all_available.csv` and
  `oracle_horizon_all_available.csv`: unpaired, available-data summary,
  which is **not** appropriate for direct ranking if failure rates differ.
- `skipped.csv`: fit/oracle errors. Plots in group subdirectories
  compare best-performing variants and the WM4 control.

The baseline RMSE in matched results should be identical for all
variants within each group. WM4 and WM5 linear/linear should also
agree up to numerical solver tolerances and different implementation
details; discrepancies should be investigated before interpreting
nonlinear improvements.

The price-optimization problem for nonlinear WM5 is not automatically
convex. This implementation focuses on deterministic simulation,
parameter/state QPs, and retrospective prediction quality. It does
not introduce price optimization or stochastic diffusion.

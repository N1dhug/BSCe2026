# WM4 moving-horizon deterministic fitting experiments

Run from the repository root, on `thomas_main`, with the project input workbooks and weather data configured:

```bash
python -m src.exe.run_model.wm_4.moving_horizon_qp
python -m src.exe.run_model.wm_4.moving_horizon_numerical
python -m src.exe.run_model.wm_4.compare_oracle_predictions
```

For a quick *real-data* smoke run (no synthetic datasets or synthetic-data tests):

```bash
python -m src.exe.run_model.wm_4.moving_horizon_qp --max-origins 3 --f0-steps 5
python -m src.exe.run_model.wm_4.moving_horizon_numerical --max-origins 3
python -m src.exe.run_model.wm_4.compare_oracle_predictions --max-origins 3
```

Both fitters use exactly 24 hourly observations in each trailing window, advancing one hour per origin (`--stride 1`). They fit normalized ALL (MED + UDEN together) and MED-only aggregations independently. The seasonal, price and input normalizations are inherited from `aligned_wm4_data`. The baseline `O` is estimated using strictly earlier-than-calendar-day observed data; it is reused for all windows originating on that day. No future hourly input observations enter the fitting QP or numerical objective. Because the inherited full-season min/max normalization uses the 2022 calibration period, the experiment is a consistent retrospective benchmark, not a fully causal live-data deployment.

By default the internal deterministic Euler time step is one quarter-hour (`--steps-per-hour 4`); set `--steps-per-hour 60` to reproduce the minute-resolution convention used by existing WM4 daily executables. The 24 observations occur at indices `[0, sph, ..., 23*sph]`; the unobserved hour-24 endpoint is not scored. In the QP, transitions beyond the last observed hour are excluded. The fit and oracle use identical grids and the same historical baseline at each prediction origin. `fit_bias` and `bias` are *predicted minus observed*.

## QP initialization and structural intercept

The standard deterministic model is
`dF/dt=R; dR/dt=r_F F+r_U U+r_R R; I=O+R`.

With `fit_intercept=True` it becomes
`dR/dt=r_0+r_F F+r_U U+r_R R`. The constant `r_0` is **not** the rate initial condition `R_0`. An intercept-free fitter fixes `r_0=0`, and both models use the signed equilibrium restrictions. The QP includes the additional linear constraints
`r_F-r_U+|r_0|<=0` and `r_R-r_F-r_U+|r_0|<=0`
(written as four linear inequalities).

The QP executable saves `zero` and `observed` initializations (fixed `F_0=0`; `R_0=0` or measured `I_0-O_0`), and `zero_intercept` and `observed_intercept` ablations. Only QP performs the `F_0` sensitivity sweep: it evaluates a configurable symmetric grid in `[-1,1]` with observed `R_0` **separately within each window**. The additional `grid_best` row selects the minimum free-run in-window MSE among that window's grid candidates. No state is propagated to the next window, and `grid_best` is an in-sample diagnostic rather than a deployable state estimator.

The optional intercept null analysis tests `H0: E[r_0_hat]=0`, based on the `observed_intercept` fits. It reports IID t-tests for reference and HAC/Newey-West standard errors with default lag 24 for overlapping hourly windows. A non-significant mean test does **not** demonstrate the structural term is unnecessary; in-window loss comparisons and the out-of-sample parameter/oracle benchmarks should also be examined. Set `--equivalence-margin M` for an additional practical-equivalence TOST around zero.

## Numerical fitter

`wm_4.fit_params_numerical` calls `src.params.fit_params_numerical.numerical_params`. It minimizes the deterministic, freely simulated 24-observation input error with SciPy SLSQP and exact forward state/parameter sensitivities. The initial state is fixed at `[F_0,R_0]=[0,0]` in the executable, with no state-sensitivity runs or alternating optimization. It fits both intercept-free and optional-intercept parameterizations. The parameter solver is warm-started from the constrained QP estimate; its reported `fit_time` **includes** QP initialization time, and `numerical_solve_time` is available separately in the fitter return object.

## Oracle comparison

`compare_oracle_predictions.py` reads the saved parameter CSVs, and optimizes **both** `F_0,R_0 in [-1,1]` against the next 24 hourly observations, independently for each fixed fitted parameter vector. It reuses `optimized_qp_params.optimize_prediction_initial_state`: conditional on fixed WM4 coefficients, the consumption trajectory is affine in initial state, so the oracle is another convex constrained least-squares QP.

The primary apples-to-apples comparison is `qp/zero` against `numerical/zero` with `r_0=0`, restricted to common hourly prediction origins. Additional intercept and QP-F0 scenarios are saved separately. Outputs include fitted-parameter CSV files, fitting MSE/RMSE/MAE/bias, QP transition residual loss, convergence/runtime diagnostics, fitted intercept distributions and HAC tests, F0 sensitivity charts, oracle state trajectories, 24-hour per-origin scores, *pooled* aggregate RMSE, empirical 20/40/60/80% quintile boundaries of per-window RMSE, and RMSE by forecast hour.

**Interpretation:** Oracle-state performance uses future observations and is an optimistic upper-bound-style diagnostic, not a feasible operational forecast. The two parameter fitters have distinct objective functions: direct consumption error versus reconstructed state-rate residuals. Both must be assessed against identical oracle state selection and matched data windows. The current experiment intentionally omits synthetic-data tests.

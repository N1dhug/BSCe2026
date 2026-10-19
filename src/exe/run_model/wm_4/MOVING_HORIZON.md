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

## Sequential QP optimization, initialization sensitivity and structural intercept

The default deterministic model is
`dF/dt=R; dR/dt=r_F F+r_U U+r_R R; I=O+R`.
For the **main QP variants** `zero` and `observed`, the executable now calls
`optimized_qp_params.fit(..., fit_initial_state=True)`. Every 24-hour window
iterates between (i) a constrained QP fit for `r_F,r_U,r_R` at fixed
`[F_0,R_0]`, and (ii) a bounded QP fit for **both** initial states
`F_0,R_0 in [-1,1]`, at the fitted coefficients. Iteration continues until
the state and rate-residual objective changes fall below `--state-tol`, or
`--max-state-iter` is reached.

`zero` initializes this alternating fit from `[F_0,R_0]=[0,0]`.
`observed` initializes from `[0,I_0-O_0]`. Neither seed fixes
the initial state during the final fit. The saved `F_0_seed`, `R_0_seed`
and fitted `F_0`, `R_0` columns distinguish starting guesses from results.

The **F0 sensitivity test** re-runs the entire alternating parameter/state
optimization from a configurable symmetric grid of **starting guesses**
`F_0_seed in [-1,1]`, always with `R_0_seed=I_0-O_0`. These guesses
do not constrain the final fitted `F_0`: they probe sensitivity to starting
conditions, convergence basins, and optimization stability. Each fitting window
is processed independently, with no state propagated from any earlier window.
The `f0_grid` rows record all converged solutions and optimizer diagnostics.
The `grid_best` row chooses the smallest **final QP transition loss** among
the seeds, which matches the alternating fit objective; it does not choose
the smallest deterministic consumption MSE. The seed-response plot now
shows the final parameters **and fitted states** versus initial `F_0_seed`,
and the saved per-window spread measures variability across seeds in the
*same* 24-hour window (rather than confounding it with across-window changes).
A full sensitivity sweep for every hourly origin is the default; to reduce
the substantial additional QP compute, use `--sensitivity-stride 24`
to evaluate it once every 24 origins, while still fitting the main sequential
QP parameters hourly.

**Separate fixed-state intercept ablations** are retained to study the
drift constant. The variants `zero_fixed` and `observed_fixed` fit
the intercept-free model at the specified fixed initial states; the
`zero_intercept` and `observed_intercept` variants allow the additional
constant `r_0` in `dR/dt` while retaining the same fixed starting states.
The intercept QP includes
`r_F-r_U+|r_0|<=0` and `r_R-r_F-r_U+|r_0|<=0`,
enforced as four linear inequalities. The null analysis compares **only**
`observed_fixed` against `observed_intercept` on matched fitting windows,
using HAC/Newey-West standard errors with configurable lag 24. It **does not**
compare a fixed-state intercept fit against the freely optimized-state
sequential fit, because `r_0` is structurally confounded with the origin
of freely optimized `F_0`. A non-significant zero-mean intercept test does
not alone prove the constant is structurally unnecessary; the model-fit and
oracle benchmarks are also relevant. Set `--equivalence-margin M` to test
a user-chosen practical equivalence band around zero.

## Numerical fitter

`wm_4.fit_params_numerical` calls `src.params.fit_params_numerical.numerical_params`. It minimizes the deterministic, freely simulated 24-observation input error with SciPy SLSQP and exact forward state/parameter sensitivities. The initial state is fixed at `[F_0,R_0]=[0,0]` in the executable, with no state-sensitivity runs or alternating optimization. It fits both intercept-free and optional-intercept parameterizations. The parameter solver is warm-started from the constrained QP estimate; its reported `fit_time` **includes** QP initialization time, and `numerical_solve_time` is available separately in the fitter return object.

## Oracle comparison

`compare_oracle_predictions.py` reads the saved parameter CSVs, and optimizes **both** `F_0,R_0 in [-1,1]` against the next 24 hourly observations, independently for each fixed fitted parameter vector. It reuses `optimized_qp_params.optimize_prediction_initial_state`: conditional on fixed WM4 coefficients, the consumption trajectory is affine in initial state, so the oracle is another convex constrained least-squares QP.

The primary comparison is `qp/zero` (alternating QP initialized at `[0,0]`, with **optimized historical initial state**) against `numerical/zero` (top-down parameter fit with **fixed historical initial state** `[0,0]`) with `r_0=0`, restricted to common hourly prediction origins. They optimize different parameter-fitting objectives and impose different historical-state fitting conditions by design; they are not a controlled comparison of parameter optimizers under identical state assumptions. For a fixed-state comparator, use the `zero_fixed` QP variant. Additional intercept and QP-F0 scenarios are saved separately. Outputs include fitted-parameter CSV files, fitting MSE/RMSE/MAE/bias, QP transition residual loss, convergence/runtime diagnostics, fitted intercept distributions and HAC tests, F0 sensitivity charts, oracle state trajectories, 24-hour per-origin scores, *pooled* aggregate RMSE, empirical 20/40/60/80% quintile boundaries of per-window RMSE, and RMSE by forecast hour.

**Interpretation:** Oracle-state performance uses future observations and is an optimistic upper-bound-style diagnostic, not a feasible operational forecast. The two parameter fitters have distinct objective functions: direct consumption error versus reconstructed state-rate residuals. Both must be assessed against identical oracle state selection and matched data windows. The current experiment intentionally omits synthetic-data tests.

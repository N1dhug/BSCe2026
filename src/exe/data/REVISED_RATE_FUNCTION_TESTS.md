# Reconstructed flexibility-rate function tests (Data chapter)

Run from the repository root:

```bash
python -m src.exe.data.revised_rate_function_tests --group ALL --backlog 4
python -m src.exe.data.revised_rate_function_tests --group MED --backlog 4
```

To check sensitivity to differencing and historical backlog:

```bash
python -m src.exe.data.revised_rate_function_tests --group ALL --backlog 4 --difference central
python -m src.exe.data.revised_rate_function_tests --group ALL --backlog 9
```

For a short smoke run: `--max-days 45`. Other options include `--start-day`, `--end-day`, `--train-fraction 0.75`, `--hac-days 14`, `--alpha 0.05`, `--gain-threshold-pct 1`, and `--output DIRECTORY`. No statistical-output baseline is used.

## Motivation

The revised dynamical model uses fixed response functions in

```
dF/dt = R,      dR/dt = theta_F F + theta_R g(R) + theta_U h(U).
```

We compare the affine specification against a finite, declared family of nonlinear candidate terms to determine whether nonlinear functions add *conditional information* about the reconstructed rate derivative. This is a model-specification study, not proof of linearity or a nonlinear constrained WM4 fitting/simulation sweep.

For each target day d, only N strictly preceding *corresponding weekdays* contribute to fitting either the weekly or temperature-adjusted weekly baseline. Target-day demand is not imputed. Define at hourly times j=0,...,23

```
R[d,j] = D[d,j] - B[d,j]
C[d,j] = sum_{l=0}^{j-1} R[d,l] * 1h
y[d,j] = R[d,j+1] - R[d,j]                    (forward, j=0..22)
```

As a sensitivity check `--difference central` uses `y[d,j]=(R[d,j+1]-R[d,j-1])/2` for j=1..22. All observations remain *actual hourly measurements* (no sub-hour interpolation pseudo-replication). The unknown day-specific initial flexibility F0 enters as an intercept alpha_d:

```
H0 (affine):   y[d,j] = alpha_d + theta_F*C[d,j] + theta_R*R[d,j] + theta_U*U[d,j] + error
H1 (expanded): same affine terms + gamma' * nonlinear_features(R[d,j], U[d,j])
```

Daily demeaning is algebraically equivalent to profiling out all alpha_d via the Frisch-Waugh-Lovell theorem. There is one *shared coefficient vector* estimated over all training days within each baseline.

Candidates are tested *separately against the same affine null*. For X=R or U, let z=X/q90(|X_train|), evaluated using a scale determined on training dates only. The fixed single-term candidates are z^2, z^3, z|z| and tanh(2z), each added beyond the linear term already in H0. The grouped candidates are (R^2,R^3), (U^2,U^3), and all four polynomial terms. This covers quadratic asymmetry, cubic nonlinearities, sign-sensitive curvature, and saturation, without attempting every possible response function.

Train/test days use one chronological 75/25 split. All tests use the **same target-day cohort for both baselines**, and identical split dates. Within-day demeaning is also used in held-out evaluation: this optimally removes an unknown new-day intercept using the *entire* held-out day. Consequently, the held-out RMSE measures conditional derivative fit, **not** an operational prediction.

For each candidate, the joint null is `H0: gamma=0`. Its Wald statistic uses a Bartlett/Newey-West covariance built from **daily score vectors**, with intraday covariance and up to 14 *calendar-day* lagged cross-products. The chi-square p-values are asymptotic/approximate. Holm adjustment controls the familywise error rate within each baseline over the tested candidate models. A candidate is provisionally informative if both its Holm-adjusted p-value is below 0.05 and its chronological held-out derivative RMSE decreases by at least 1% relative to affine. The threshold is a *declared practical-effect filter*, not a formal equivalence test.

## Outputs

The default path is `files/output/data/revised_rate_functions/{all|med}/backlog_04_forward/`. The executable creates:

- `reconstructed_rate_derivative.csv`: exact hourly reconstruction with D, B, R, C, U, and derivative.
- `function_tests.csv`: candidate, joint HAC Wald chi-square, raw and Holm-corrected p-values, training and held-out RMSE, percentage held-out gain, identification status, decision.
- `function_coefficients.csv`: fitted coefficients and HAC standard errors for each candidate model.
- `affine_heldout_residuals.csv`: held-out affine residuals, conditional on per-day intercept.
- `nonlinear_heldout_improvements.png`: two panels (weekly, temperature-adjusted) with held-out gains against affine.
- `affine_rate_residual_diagnostics.png`: binned held-out affine residual means versus R and U, for both baselines.
- `skipped_days.csv`: rejected missing/misaligned days and reasons.
- `run_report.json`: provenance, calendar split, candidate definitions and limitations.

### Interpretation limits

1. Measurement/baseline errors enter R on both sides of the regression; finite-differencing R creates mechanically correlated derivative noise. Thus coefficient estimates and their p-values need not identify the structural dynamical laws causally, even with day-score HAC.
2. Estimation of historical baselines produces *generated regressors*; the reported HAC variance does not propagate the baseline-estimation uncertainty.
3. Models with large extra terms can overfit hourly derivatives, and the held-out assessment optimizes day intercepts retrospectively. Dynamical model selection ultimately requires a matched full-state oracle and, preferably, operational forecast comparison.
4. Failure to reject an extra term **does not establish linearity**; the tested response families and observed variable ranges are finite. Conversely, small p-values do not imply useful forecast improvements.
5. The holdout is a *screening* split for candidate ranking. If candidates are selected after viewing it, use an additional untouched test period for a confirmatory claim. Overlapping historical baseline training windows create dependence; the day-HAC approximation only partially addresses this.

The executable deliberately does **not** alter the source WM4 estimation procedure, nonlinear response functions, or stored oracle prediction results.

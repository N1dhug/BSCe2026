# WM4 oracle initial-state audit (full [F0,R0])

The definition of a target-day **oracle initial state** is the
jointly optimized vector [F0,R0], not F0 fitted against a fixed
historical R0. For *fixed historical* WM4 theta, deterministic
predictions are affine in [F0,R0]. The globally solvable
bounded two-variable least-squares QP is already implemented by
src.params.wm_4.optimized_qp_params.optimize_prediction_initial_state
and the shared active_set solver.

## Updated implementations

The following executables previously used only an optimized F0 with
a fixed historical R0, or consumed an F0-only oracle result:

- sweep_weekday_horizon.py: every N/H/baseline uses joint [F0,R0]
  least squares for each horizon, with one historical theta fit per N.
- analyze_oracle_8weekday.py: N8/H7 reports and reconstructs both
  initial states and every hourly contribution from the full oracle.
- predict_all_moving_horizon.py: 24-hour target-day oracle fits both
  initial states, historical R0 now diagnostic only.
- sweep_fixed_responses.py: the nonlinear WM5 g(R)/h(U) comparison
  now jointly fits both initial states per horizon. Unlike linear
  WM4, this state-only problem can be NONCONVEX; the two-dimensional
  bounded grid/multi-start search does not certify global optimality.
- analyze_origin_7hour_rmse.py: rolling origin 00..17 jointly fits
  [F0,R0] for each seven-hour window using the existing WM4 QP.
  Also exposes the optimized R0 and historical reference R0.

analyze_price_identification.py consumes the revised N8/H7 oracle
results rather than its prior F0-only prediction-error statistics.
The fitted theta_U values are historical and remain independent
of target initial-state optimization, but forecast-error associations
MUST be recomputed.

## Already full two-component oracles (no replacement needed)

These WM4 executables already call the existing full-state
optimized_qp_params.optimize_prediction_initial_state method:
- compare_oracle_predictions.py
- compare_initial_state_oracle_day.py
- predict_all_days_oracle_diffusion.py
- predict_all_days_oracle_price_effect.py
- predict_all_days_adaptive_traceback.py (optional oracle mode)
- predict_all_days_brute_force.py (optional oracle mode)
- predict_all_days_fixed_tracebacks.py
- predict_all_days_i0_initial_state.py
- predict_all_days_initial_state_uncertainty.py
- predict_all_days_optimized.py (optional oracle mode)
- predict_all_days_recent_r0.py (full-state reference oracle)
- predict_all_days_shared_initial_state.py
- predict_all_days_uncertainty.py
- analyze_initial_state_uncertainty_horizon.py
- analyze_initial_state_uncertainty_horizon_observed_p0.py
- analyze_parameter_initial_state_correlation.py
- compare_baselines.py
- fit_oracle_mpc_diffusion.py
- test_initial_f0_sensitivity.py

Other executables that show oracle statistics read saved outputs
or call the already-full-state paths; they do not independently
implement a target-state oracle.

predict_all_days_recent_r0.py also includes a specifically named
_pseudo_oracle_fixed_r0() method for a controlled fixed-R0
ABlation. That intentionally restricted comparator is NOT a full
oracle and must not be relabeled or replaced; its full-state
reference oracle already optimizes both components.

## Regeneration and comparability

The previously published N8/H7 aggregate RMSE of 0.07896767 was
from an F0-only fit with historical median R0, not the full oracle.
The previous weekday/horizon rankings, nonlinear response rankings,
price-vs-forecast-error association plots, and rolling cumulative
RMSE plots must be regenerated. Do not mix their previous output
CSVs with full-state runs.

Core revised scripts version their saved configurations or verify
required oracle_R0 fields, rejecting obsolete results during
resume and report-only operations.

For reproducibility, run:
    python -m unittest discover -s tests -p "test_wm4_weekday_horizon_sweep.py"
    python -m unittest discover -s tests -p "test_wm4_oracle_8weekday.py"
    python -m unittest discover -s tests -p "test_wm4_joint_moving_horizon.py"
    python -m unittest discover -s tests -p "test_wm4_fixed_responses.py"
    python -m unittest discover -s tests -p "test_wm4_origin_7hour_rmse.py"

Warning: the full-state oracle uses target observations for both
components. It is a retrospective best-fit benchmark, NEVER a
historical-only forecast.

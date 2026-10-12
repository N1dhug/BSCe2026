BM boundary study: logistic f and g
===================================

The standalone runner calls the existing BM1/BM2 fitters, posterior updates,
deterministic predictors and finite-volume solvers. It does not edit them.

From the repository in PowerShell, retrieve the runner without changing branch:

  git fetch origin feature/bm-logistic-boundary-study-20261012
  $runner = Join-Path $env:USERPROFILE "Downloads\bm_boundary_study.py"
  git show origin/feature/bm-logistic-boundary-study-20261012:tools/bm_boundary_study.py | Set-Content -LiteralPath $runner -Encoding utf8
  if ($LASTEXITCODE -ne 0) { throw "Could not read runner from Git." }

Inspect existing data and reusable fits first:

  python -B $runner --repo . --study pilot --preflight

Run the pilot:

  python -B $runner --repo . --study pilot --workers 8 --fit-missing

Pilot: 5 May 2022, BM1/BM2 daily + MPC at 00:00, 08:00 and 16:00.
This gives 8 ordinary forecasts plus 24 controlled scenarios, 32 jobs total.
Six selected jobs also have finer-grid/time numerical checks.
BLAS is fixed to one thread per process. No new fitting occurs in preflight.
With --fit-missing, only missing fits are estimated, using the existing native
five-window fitters and the reference run's iteration/evaluation budgets.
New fits retain their convergence flags; a completed fit does not certify
first-order convergence.

Default reference and fit roots:
  Prepared inputs:
    files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08
  BM1 fits, in priority order:
    files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08
    files/output/bm_1/runs/fg_week_joint_2022-05-02_2022-05-08
  BM2 fits:
    files/output/bm_2/runs/week_2022-05-02_2022-05-08

Use --reference PATH, --bm1-fit-root PATH [PATH ...], and
--bm2-fit-root PATH [PATH ...] if your results are elsewhere.
Fits must be canonical logistic fit.json outputs from the existing week
runners, with corresponding settings.json and prepared/day.pkl identities.
The linear/piecewise multistart study is not a logistic reference.

Pilot output:
  files/output/bm_1/runs/boundary_logistic_pilot

Inspect completion.json, numerical_refinement.csv and plots before the week.
The numerical tolerances are pilot screening limits, not a convergence proof.
If a refinement fails, investigate its report before interpreting boundary
occupancy or forecast calibration.

Run the full week after assessing the pilot:
  python -B $runner --repo . --study week --workers 8 --fit-missing

Week: 2-8 May 2022, both models, daily forecasts and all 24 MPC origins.
350 ordinary forecasts + 160 controlled scenarios = 510 jobs.
Output: files/output/bm_1/runs/boundary_logistic_week

Outputs:
  all_boundaries.csv                State/demand moments, boundary probability,
                                    forecast errors and conditional intervals.
  forecast_by_boundary_risk.csv     RMSE, bias, coverage by predicted boundary risk.
  scenario_summary.csv             Expected boundary occupancy and final moments.
  numerical_refinement.csv         Selected coarse/fine comparisons.
  bm1/bm2_boundary_overview.png     State/price response and forecast-risk views.
  bm1/bm2_boundary_distributions.png Selected distributions after 1, 8 and 24 hours.
  jobs/.../report.json              Solver diagnostics, fit provenance and flags.
  jobs/.../state_law.npz            State marginal masses and demand supports.
  completion.json                  Complete versus partial run status.

Daily convention: five equal deterministic paths; process sigma is zero.
Daily intervals measure spread between initial-state paths.
MPC convention: native smoothed historical prior updated with I(origin),
native historical sigma calibration, then an eight-hour conditional forecast.
These intervals describe the native demand law; they do not add observation
noise as a separate future measurement component.

Controlled scenarios use one historical daily fit per model on 5 May.
Prices and starting states are changed without fitting target-day demand.
They have no RMSE or coverage score against counterfactual observations.
Pilot uses X0=0.01/0.99, U=0/1 and zero/estimated noise. BM2 additionally
uses historical-mean and outward-pointing R0.
The week expands this to five X0 values and four price paths.
Sigma is calibrated once per model for controlled scenarios, then reused.

Boundary widths: 0.025, 0.05 and 0.10.
For finite-volume laws, partial cells are integrated over the boundary strip.
For deterministic paths, indicators are evaluated at the actual states.
No probability clipping or renormalization is introduced in propagation.
Noisy point initializations use the nearest native grid cell; grid refinement
therefore also checks initial-state discretization.
BM2 exact blocked-boundary probability is available for deterministic paths.
For finite-volume laws it is left missing because cell masses do not identify
exact boundary contact.
The first sample below 50% boundary probability is an occupancy diagnostic,
not a stochastic first-passage time.
All state labels refer to the model's inferred latent state.

Resume with exactly the same command and output folder. Completed jobs are
identified by fingerprints and file hashes. Changed code/data/settings require
a new output folder. To stop after active jobs, create an empty STOP file in
the output folder. Remove that marker before resuming.

Validation during preparation:
  Short zero-noise equilibrium and stochastic mass/symmetry checks passed for
  both models. Partial-cell probability integration and native Bayes update
  checks passed. Ordinary replay predictions matched the current native
  BM1/BM2 predictors at floating-point precision on saved pilot fits.
  Some older BM2 CSV forecasts differ from recomputation with the current
  source; this is recorded in report.json rather than changing the old files.
  The final process-pool/scenario end-to-end check was interrupted by loss of
  the local execution connection. No full pilot/week result is claimed.

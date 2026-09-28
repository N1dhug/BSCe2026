# BM1 — code and run guide

F1, F2 and F3-A through F3-F are implemented and locally validated.
Run all commands from the repository root.

The post-F3-F maintenance candidate and its test/performance evidence are
documented in `BM_AUDIT_F1_F3.md`. The audit keeps the model, reference grids
and outer-optimizer settings fixed; its measured gains mainly concern dynamic
mixture projection. Windows validation is required before merging the patch.


## Where the code belongs

| Location | Role |
|---|---|
| `src/model/base/bm.py`, `src/model/bm_1.py` | Battery equations and forward-solver interfaces. |
| `src/control/bm_1.py` | Supplied parameters/initial conditions, common deterministic/FV/mixture evaluator construction, bounded price search and paired MC. |
| `src/control/objective.py` | Demand moments and jump-safe deterministic/stochastic tracking costs. |
| `src/method/` | Numerical methods: EM, FP, sampled quadrature and `beta_mixture.py` for projected density evolution. |
| `src/solver/` | Low-level finite-volume and active-set solvers shared with WM. |
| `src/params/` | Parameter estimation, as used by WM; price decisions are not parameter estimates. |
| `src/validation/control.py` | Price-control input checks. |
| `src/validation/model/`, `method/`, `solver/` | Input contracts for the corresponding layer. |
| `src/validation/bm_1/` | Controlled scenarios and numerical reference studies; no executable or plotting imports. |
| `src/diagnostics/` | Feasibility, distribution and error metrics. |
| `src/tools/bm_1_data.py` | Historical input preparation and result serialization helpers. |
| `src/exe/run_model/bm_1/` | Four command-line entrypoints: `verify`, `optimize`, `refine`, `real_data`. |
| `tests/test_bm1_*.py` | Regression tests grouped by topic; never run automatically by a simulation. |

Read the battery model for the equations, then `control/bm_1.py` from top to
bottom: problem inputs, results, initial law, forward evaluators, price search
and MC assessment. Scenario constants/builders live in `validation/bm_1`.
`control/objective.py` owns the F3-C terminal-state and price-jump cost terms.

WM uses `method/fokker_planc.py` too. The generic numerical methods and shared
package initializers are unchanged. Reference FV and fast FV are independent
implementations of the same discretization, not old/new compatibility copies.
The old `method/bm_1_*` and `validation/method/bm_1.py` files are removed, with
all current imports updated. No forwarding modules are kept.

## Normal runs

All four entrypoints print a short numerical report and write no files by default.
`--verbose` adds diagnostics/progress without repeating the simulation.
`--save` requests figures/results and imports matplotlib only when needed.

```powershell
python -B -m src.exe.run_model.bm_1.verify --help
python -B -m src.exe.run_model.bm_1.optimize --help

python -B -m src.exe.run_model.bm_1.verify deterministic
python -B -m src.exe.run_model.bm_1.verify feasibility
python -B -m src.exe.run_model.bm_1.optimize deterministic

python -B -m src.exe.run_model.bm_1.verify em
python -B -m src.exe.run_model.bm_1.verify fv
python -B -m src.exe.run_model.bm_1.verify em-convergence
python -B -m src.exe.run_model.bm_1.verify mc-fv
python -B -m src.exe.run_model.bm_1.verify projection
python -B -m src.exe.run_model.bm_1.verify mixture
python -B -m src.exe.run_model.bm_1.verify conditioning
python -B -m src.exe.run_model.bm_1.verify objective
python -B -m src.exe.run_model.bm_1.optimize stochastic

python -B -m src.exe.run_model.bm_1.optimize contract
python -B -m src.exe.run_model.bm_1.optimize contract --optimize
python -B -m src.exe.run_model.bm_1.optimize augmented --terminal-weight 0.01 --smoothing-weight 0.001
python -B -m src.exe.run_model.bm_1.optimize sensitivity
python -B -m src.exe.run_model.bm_1.optimize sensitivity --grid
python -B -m src.exe.run_model.bm_1.optimize decision --no-mc
python -B -m src.exe.run_model.bm_1.optimize decision
python -B -m src.exe.run_model.bm_1.optimize uncertainty
python -B -m src.exe.run_model.bm_1.optimize mixture-control --no-mc
python -B -m src.exe.run_model.bm_1.optimize mixture-control --mc-paths 20000
python -B -m src.exe.run_model.bm_1.optimize mixture-validate --fv-reference-prices 0.730609 0.264415 --fv-design-prices 0.729329 0.263873 --mixture-prices 0.735904 0.263610 --mc-paths 20000 --mc-time-paths 5000
python -B -m src.exe.run_model.bm_1.optimize multi --fast-fv
python -B -m src.exe.run_model.bm_1.optimize complex --fast-fv
python -B -m src.exe.run_model.bm_1.refine
python -B -m src.exe.run_model.bm_1.refine --mc-paths 20000 --mc-time-paths 5000
python -B -m src.exe.run_model.bm_1.real_data --help
python -B -m src.exe.run_model.bm_1.verify benchmark --cells 201 1001 --repeats 7
```

The F3-C sensitivity command compares raw tracking cost, terminal MSE and price roughness across weights; weighted totals from different rows are not a common metric because the objective itself changes.

F3-D optimizes a deterministic point-state controller and the stochastic FV controller under the same price grid/objective, then cross-evaluates both policies. Paired MC validates the stochastic policy difference.

F3-E separates the two uncertainty sources into deterministic, initial-only, process-only and full cases, optimizes each independently and cross-evaluates the four returned policies. Compare policies within a fixed model row; objective levels across rows use different state laws.

F3-F optimizes FV at both 201 and 81 cells plus the degree-30 dynamic mixture at the matched 81-cell design grid. The final `mixture-validate` step re-optimizes only FV at the F3-B high-resolution settings (801 cells, objective step 1/192, FV step 1/384) and checks the already-fixed policies with common-random-number MC and coupled EM refinement. In the retained validation the mixture policy has high-resolution FV regret 2.33e-6, about 0.0356% of the reference objective, and the 20,000-path paired MC difference is -2.21e-6 with SE 6.34e-7 under the convention FV-reference minus candidate. F3-F is therefore closed for the controlled two-price benchmark; the policies are not numerically identical, but their achieved control quality is very close.

Mixture/conditioning and full multi-price optimization can be expensive. Their
numerical settings are not reduced just to make an output report shorter.
F3-B freezes the prices while refining space, time and quadrature. Finest-grid
zero differences are by construction, not zero true error. FV/MC agreement
within sampling uncertainty is not proof of an exact PDE solution.

## Files and retained evidence

| Output | Default destination with `--save` |
|---|---|
| Verification / optimization figures | `files/output/bm_1/runs/graphs/` |
| F3-B JSON and figure | `files/output/bm_1/runs/f3_b/` |
| Historical runs | `files/output/bm_1/runs/real_data/` |
| Timing report | `files/output/bm_1/runs/fast_fv_benchmark.json` |

`runs/` is ignored by Git. A model does not read saved results back as input.
The benchmark also saves when an explicit `--output` is supplied.
Retained F3-B evidence is in `files/output/bm_1/f3_b/`; the historical evidence,
including its 20,000-path MC comparison, is in `files/output/bm_1/real_data/`.
These files and the original input workbooks are not changed by code cleanup.
Updating evidence requires an explicit destination:

```powershell
python -B -m src.exe.run_model.bm_1.refine --mc-paths 20000 --mc-time-paths 5000 --save --output-dir files/output/bm_1/f3_b
```

`real_data --save --output-dir files/output/bm_1/real_data` explicitly updates
historical evidence. This remains a retrospective tracking diagnostic with
supplied model parameters, not parameter estimation or proof of causal savings.

## Validation while developing

Use the relevant topic test during development, then the complete suite before
merge. `-q` limits output, not coverage; use `-v` when diagnosing a failure.

```powershell
python -B -m unittest tests.test_bm1_control -q
python -B -m unittest discover -s tests -q
```

Public input checks and probability/solver-failure safeguards remain active.
Layout tests explicitly name the current core modules, check that removed files
stay absent and prevent core code from importing scenarios or executables.

Do not delete data, stashes, virtual environments or shared WM branches to clean
BM source. Python caches and local run output are ignored and can be regenerated.
The WM1/WM2 price-step demo import `src.exe.tool.input_builder` is a pre-existing
WM issue, outside this BM change. WM import checks do not execute every WM run.

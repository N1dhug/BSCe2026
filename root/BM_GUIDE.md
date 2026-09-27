# BM1 — F1–F3 code and run guide

F3-A and F3-B are implemented. F3-C is not started by this cleanup.
The readability revision keeps model equations, numerical schemes, random
seeds, optimizer settings and the established reference cases. It separates
calculations from reporting and makes saving an explicit choice.

## Layer responsibilities

| Location | Responsibility |
|---|---|
| `src/model/bm_1.py` | BM1 model and solver interface. |
| `src/method/bm_1_control.py` | Control problem, parameters/numerics, reference/fast FV, bounded optimization, initial laws and paired MC evaluation. |
| `src/method/bm_1_objective.py` | Demand moments and tracking objective. |
| `src/method/bm_1_mixture.py` | Conservative prediction and constrained Beta-mixture projection. |
| `src/validation/method/bm_1.py` | BM-only input checks. |
| `src/validation/bm_1/` | Repeatable numerical validation experiments and controlled scenario inputs; no plotting or executable imports. |
| `src/diagnostics/` | Mathematical error, feasibility and distribution metrics. |
| `src/tools/bm_1_data.py` | Historical data import and preparation. No estimation. |
| `src/exe/run_model/bm_1/` | Four command-line entrypoints: select a run, call reusable code, print/save/plot results. |
| `tests/test_bm1_*.py` | Tests grouped by topic, including layering/import regressions. |

`validation/method` checks input contracts. `validation/bm_1` validates numerical
behaviour against reference problems. These are deliberately different roles.
The shared FV/EM/active-set implementation, WM sources and shared package exports
are unchanged. Different FV implementations remain because one is an independent
implementation reference for the fast path, not a deprecated compatibility copy.

## Reading the code

Read `model/bm_1.py` for the equations, then `method/bm_1_control.py` for the
problem definition and evaluators. Scenario inputs and numerical studies are
in `validation/bm_1`; each executable's `main()` only selects a case and options.
Imports of whole scenario modules (`cases`, `deterministic`, `mixture`) make the
origin of constants and functions visible without long lists of imported names.
Equations and nontrivial numerical phases have short docstrings/comments.

Public boundaries still validate inputs. The fixed-price evaluator now validates
its candidate once, then passes it to a private construction helper. Probability
mass, finite coefficients, simplex constraints, raw boundary violations and solver
failures remain checked. Invalid-input demonstrations and repeated PASS/FAIL
checks were removed from ordinary F1 executable runs because the regression suite
already tests those contracts; the analytic/Euler and feasibility calculations
remain available. Repeated resampling, model builders and paired-MC loops are
shared instead of copied between scenarios.

Unit tests are never imported or automatically executed by a simulation.
Use a focused test file while working, and the full suite before merging. `-q`
reduces unittest output, not test coverage. Use `-v` to diagnose a failure.

## Commands from the repository root

```powershell
python -B -m unittest discover -s tests -q
python -B -m src.exe.run_model.bm_1.verify --help
python -B -m src.exe.run_model.bm_1.optimize --help
```

All four entrypoints default to a short numerical report and **no file writes**.
`--verbose` adds diagnostics, full price vectors and optimizer progress where
applicable; it does not run a second simulation. `--save` requests figures/JSON.
Matplotlib is imported only when figures are requested.

```powershell
python -B -m src.exe.run_model.bm_1.verify objective
python -B -m src.exe.run_model.bm_1.verify objective --verbose --save
python -B -m src.exe.run_model.bm_1.optimize contract --optimize
```

### F1 deterministic work

```powershell
python -B -m src.exe.run_model.bm_1.verify deterministic
python -B -m src.exe.run_model.bm_1.verify feasibility
python -B -m src.exe.run_model.bm_1.optimize deterministic
```

The deterministic comparison still includes the unsegmented solver call to
measure the effect of breakpoints; it is a numerical comparison, not a retained
obsolete implementation or wrapper module.

### F2 stochastic and mixture work

```powershell
python -B -m src.exe.run_model.bm_1.verify em
python -B -m src.exe.run_model.bm_1.verify fv
python -B -m src.exe.run_model.bm_1.verify em-convergence
python -B -m src.exe.run_model.bm_1.verify mc-fv
python -B -m src.exe.run_model.bm_1.verify projection
python -B -m src.exe.run_model.bm_1.verify mixture
python -B -m src.exe.run_model.bm_1.verify conditioning
python -B -m src.exe.run_model.bm_1.verify objective
python -B -m src.exe.run_model.bm_1.optimize stochastic
```

Mixture/conditioning/full optimization cases can be expensive. They retain the
original settings; structural cleanup is not a reason to silently reduce them.

### F3 control and refinement

```powershell
python -B -m src.exe.run_model.bm_1.optimize contract
python -B -m src.exe.run_model.bm_1.optimize contract --optimize
python -B -m src.exe.run_model.bm_1.optimize multi --fast-fv
python -B -m src.exe.run_model.bm_1.optimize complex --fast-fv
python -B -m src.exe.run_model.bm_1.refine
python -B -m src.exe.run_model.bm_1.refine --mc-paths 20000 --mc-time-paths 5000 --save
```

`refine` performs no price re-optimization. It still uses three frozen policies,
the same spatial/time/quadrature levels, and explicitly nested EM grids.
Its zero difference on the finest level is by construction, not zero true error.
FV/MC agreement within sampling uncertainty is not proof of an exact PDE solution.

### Historical input and performance

```powershell
python -B -m src.exe.run_model.bm_1.real_data --help
python -B -m src.exe.run_model.bm_1.verify benchmark --cells 201 1001 --repeats 7
```

Historical data remain a retrospective diagnostic with fixed reference model
parameters, not an estimated causal demand model. Input workbooks are untouched.
The benchmark runs the same computation before/after the fast implementation;
its JSON is output, not model input. The old root-level benchmark JSON is removed.
Timing outputs are written only with `--save` or an explicit `--output`. They go under ignored `files/output/bm_1/runs/`; old measured
performance remains documented in `root/BM_PERFORMANCE.md` and Git history.

## Evidence and local files

The committed F3-B JSON/figure live together under `files/output/bm_1/f3_b/`,
while the retained historical real-data evidence (including the 20k-path paired-MC validation) stays under `files/output/bm_1/real_data/`.
New BM1 run outputs are reproducible local output under the ignored
`files/output/bm_1/runs/` tree and are not versioned. No source, solver, model or test
reads the old timing JSON to run a calculation. The unused old EM/FP demo entrypoints
and superseded historical BM1 verification figures are removed; their controlled
replacements are the `verify`/`optimize` entrypoints.
The default save directories are `runs/graphs/` (verify/optimize), `runs/f3_b/`
(refine), and `runs/real_data/` (historical runs). Saved data are not read back
implicitly by a model. Updating intentionally retained evidence requires an
explicit destination, for example:

```powershell
python -B -m src.exe.run_model.bm_1.refine --mc-paths 20000 --mc-time-paths 5000 --save --output-dir files/output/bm_1/f3_b
```

`real_data --save --output-dir files/output/bm_1/real_data` likewise explicitly
updates historical evidence. A normal run cannot overwrite either retained set.
The evidence already committed to the repository is not changed by this revision.

Tracked Python bytecode caches are removed and ignored. The empty
`src/validation/exe` placeholder is removed; functional package initializers
are retained. Git does not track empty folders.

Branches, Git history, virtual environments, original data and user stashes are
not purged. Do not use `git clean -xfd`, `git reset --hard`, or `git stash clear`
to perform this cleanup. File deletions are part of the refactor commit and are
applied by a normal branch switch/merge. Python can recreate ignored caches.

## Verification of the consolidation

All numerical tests are migrated, not replaced by weaker tests. The obsolete
alias-identity test is removed because those modules intentionally no longer
exist. New tests enforce the four-entrypoint layout, dependency direction,
CLI import safety, repository-relative data paths, explicit MC initial sampler,
source parseability and WM model imports.

A before/after comparison is performed with identical seeds and numerical
settings: F3-A fixed and optimized controls, all F3-B modes, mixture conditioning,
objective/distribution comparisons, the complex target, synthetic historical
input and random policy evaluations. Timing and Python-object identities are
not compared. The cleanup report records the test and numerical comparison
results; no universal numerical accuracy or performance improvement is claimed.

Proceed in the normal workflow: local tests -> numerical smoke run -> review
of code/meaning -> merge. No F4 estimation is introduced here.

## Readability regression scope

No new executable or compatibility module was added. The original 154 tests are
retained; four reporting/validation tests cover quiet/verbose equivalence,
opt-in save paths, explicit JSON/figure saving, and one candidate check per FV
evaluation. These tests cost nothing in ordinary model execution.

WM1–WM4 and related parameter/validation/executable code were used as a style
reference, not modified. WM also performs substantial runtime validation, so
matching its readability does not mean deleting BM's numerical safeguards.
Old WM1/WM2 price-step demos still refer to `src.exe.tool.input_builder`; that
pre-existing WM path issue is outside this BM-only change. WM model imports are
checked, but this does not claim all WM workflows have been executed.

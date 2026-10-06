# H2: raw historical inputs and the next fitting pilot

**Active compact decision, 6 October:** the explicitly revised windows are
TRAIN 18–19 February 2022, MODEL_SELECTION 20 February and unchanged final
week 21–27 February. Prior origin is 17 February; its 24-hour lead is
prediction-only. Use the fixed mean of all 13 audited MED households.
See [the compact method review](H2_COMPACT_ESTIMATION_REVIEW.md) and
`f4/h2_compact_method_decision_20261006.json`. All hours in the new TRAIN
are used; this is a limited new experiment, not a subset passed off as the
old seven-day fit. Original decisions below are historical.
Exact columns, F price/scale/physical/noise/prior/baseline provenance and final
exposure still require completion. The compact decision is not executable.
The open forecast repeats only the prior seven days' hourly prices and uses
the pre-origin frozen MED-only baseline; its input assumptions/uncertainty are
reported separately from the contemporaneous final prequential filter.

**6 October decision:** use the fixed mean of the 13 audited MED households,
separately from UDEN. TRAIN is 7–13 February 2022, MODEL_SELECTION 14–20
February, and planned final evaluation 21–27 February. Windows are half-open
at local midnight and apply to interval starts; availability is at interval
end. See [H2_METHOD_REDESIGN.md](H2_METHOD_REDESIGN.md) and
`f4/h2_method_decision_20261006.json`. These choices are now recorded and must
not be requested again. Exact header-column mapping and the remaining price,
scale, earlier baseline/prior and physical/noise declarations still need
completion. Final-week exposure is not audited; blind-holdout validation
remains blocking. The decision JSON is not a runnable configuration and
does not fill evidence fields with invented provenance. The old 73-candidate
commands below describe an earlier pilot, not the active execution plan.

Updated 5 October 2026. H1a-v2 and all six conditional synthetic H1b fits are
complete on PR #62/#64. This H2 preparation does not repeat their studies or
claim historical fitting/validation. See [F4.md](F4.md) for scientific status.

## Hourly continuation, Johan 5 October 2026

Johan now declares one-hour consumption, provisionally kWh, where hour 03
means [03:00,04:00). These choices and Europe/Copenhagen are working assumptions.
A separate joint state/accumulator interval likelihood, hour-start energy
adapter and bounded shared-curve replay are implemented; point observations
retain their existing path. See [H2_HOURLY_INTERVALS.md](H2_HOURLY_INTERVALS.md)
for equations, controls and commands. The small synthetic panel conserves mass
and its independent finite-state control converges, but all initial BM grid
comparisons fail the declared development tolerances. No real measurements or
holdout values were read; no historical fit or H2/H3 acceptance is claimed.

Saved reproducible development inputs/results:
[short panel](../files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json),
[bounded finer panel](../files/output/bm_1/runs/estimation/h2_hourly_interval_refinement_20261005.json),
and [incomplete working-assumption config](../files/output/bm_1/runs/estimation/h2_hourly_assumptions_config_20261005.json).
The finer run took 407.88 s; 641 versus 1281 cells still changes total
log-likelihood by 0.416 (limit 0.05), while terminal mean/L1 pass their
development tolerances. Numerical readiness remains false. Next: reduce
likelihood error at bounded runtime before historical optimization; complete
the independent split/exposure and frozen-input prerequisites alongside it.

## Confirmed population metadata, 5 October 2026

A read-only join of the consumption header IDs with `data_installation`
establishes the workbook's labels:

- **MED:** all 13 installation IDs are categorised as
  `Parcel-, række- m.v. huse med elvarme` (houses with electric heating).
- **UDEN:** 140 installation IDs are categorised as houses without electric
  heating: 126 `Parcel-, række- m.v. huse uden elvarme` and 14
  `Parcelhuse u/elvarme`.
- The two NULL, seven ?, one postal/telecommunications and one street-lighting
  columns remain separate categories; they are not members of the MED group.

The exact MED members are saved in the
[metadata audit](../files/output/bm_1/runs/estimation/h2_workbook_metadata_audit_20261005.json). All 164 header IDs were
matched against installation metadata. The audited input workbook SHA256 is
`3ce534b533e0da472cd72ae599916fa6209eccd10ad4c021a76ac5cfba3586d4`,
identical to Johan's earlier header-inspection report. Only `Labels 2022`,
`data_installation`, installation IDs and the two demand header rows were
decoded; **zero observation values were read**. The audit is reproducible
from commit `9de6d0d2e6f878ff9611f12acb776cdfb9ceae47` and
[its successful run](https://github.com/N1dhug/BSCe2026/actions/runs/37322217260).

This resolves the meaning and membership of those tags. A fitting contract
must still declare the selected fixed population. The inspected metadata sheets
provide no definition of consumption units, point versus interval measurements,
demand timezone or hour start/end. The price/proxy decision, chronological
split/exposure audit, frozen scales and baseline also remain prerequisites.
Johan's later hourly choices are working assumptions, not workbook evidence.
`historical_fit_ready=False` and scientific acceptance remain unchanged.

## Short local inspection

When no calculation is using this checkout, select the H2 branch (or main after
integration), then run from the repository root:

```powershell
python -B -m src.exe.run_model.bm_1.curves history-inspect --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "H2 header inspection failed." }
```

The command reads workbook structure, the two consumption header rows and the
first price header row. It records file SHA256 digests and writes
`files/output/bm_1/runs/estimation/h2_header_inspection.json`. It does not decode
observation rows, instantiate the convenience loaders, normalize, impute, fit,
simulate or access final-holdout observations. The metadata report can be
atomically refreshed; it is not a scientific checkpoint. No worker pool or
numerical study is needed. Share the JSON for schema review. Column names alone
do not establish physical units or observation semantics.

## Decisions required before fitting

| Item | Required evidence/decision |
|---|---|
| Fixed population | Meaning of IDs/tags, membership and exact ordered measurement columns. Keep membership fixed; any missing member makes the mean missing. |
| Observations | Johan declares hourly energy, provisionally kWh. The separate interval likelihood is implemented; historical numerical validation remains required. Units/labels are working assumptions. |
| Time | Demand timestamp timezone, meaning of hour labels and DST representation. Duplicate or ambiguous local hours are not silently averaged or inferred. |
| Price | Actual consumer tariff, or explicitly declared spot proxy plus area. SpotPriceDKK cannot establish a consumer tariff. Document price units. |
| Split/exposure | Chronological TRAIN / MODEL_SELECTION / FINAL_HOLDOUT and an exposure audit. Existing plots/previews mean blind status cannot be assumed from dates. |
| Scale/baseline | Fixed physical and price bounds; training-only scale provenance; history-only baseline and sufficient pre-TRAIN history. No holdout-driven scaling/clipping. |

Distinguish the working assumptions above from actual data documentation.
Inspection deliberately reports `historical_fit_ready=False`. Do not replace
missing evidence with a nonempty citation string in a contract.

## Implemented data boundary

`src/tools/bm_1_history_io.py` reads explicit XLSX cells instead of using the
legacy whole-workbook/interpolating loaders. Timestamp metadata and raw XML
bytes are traversed to locate the window; only the chosen population's
measurement cells inside it are decoded as observations. Numeric values
outside TRAIN, including formula cells, cannot influence its output. Missing
hourly rows/cells stay NaN; no imputation or normalization is performed.
Formatting-only rows are ignored. Invalid selected values, selected formulas,
duplicate/off-grid times and ambiguous/nonexistent demand hours fail explicitly.
Excel day fractions use microsecond precision to avoid false off-grid times
from binary floating point. UTC price times retain the distinct DST hours.

`load_historical_training_workbooks` requires the existing
`BM1HistoricalCurveContract`, a strictly earlier prior origin, demand timezone
and price area. It returns the entire TRAIN measurement table and raw prices
from prior origin through TRAIN end, plus digests/contract provenance. Workbook
digests must remain unchanged while loading. The spot-file route accepts only
an explicit `spot_proxy` / `DKK/MWh` contract. Consumer tariff input needs its
own documented adapter. Declared units are prerequisites, not proof supplied
by the loader.

Callers separately supply the complete history-only baseline and frozen physical
parameters to `prepare_historical_curve_training`. It preserves the first TRAIN
observation, fixed-population missing mask and elapsed UTC model time.
`read_raw_power_window` can load an explicitly declared earlier baseline-history
window through the same boundary; the training loader does not choose that
history or construct a baseline. Neither route provides an H3 holdout evaluator.
The bounded historical optimizer invocation below now connects this boundary to
the existing shared-curve fitter.

## Bounded H2 pilot (continuation after PR #67)

Generate the configuration before supplying scientific metadata:

```powershell
python -B -m src.exe.run_model.bm_1.curves history-config --output files/output/bm_1/runs/estimation/h2_pilot_config.json
```

The template leaves population, units, observation semantics, timestamp timezone,
price interpretation, chronological split/audit, physical/noise parameters, beta
prior and evidence references unresolved. It does not invent values. Paths in
`files` are relative to the repository root (or explicit `--input-root`), not the
configuration's directory. Do not rerun this command over an existing config.

Supply a frozen baseline CSV with exactly two columns: `timestamp,baseline`.
Timestamps need explicit UTC offsets; values use the declared power unit and
fixed-population mean. Supply every hour from prior origin through TRAIN end.
No interpolation is performed, and excluded baseline values are not decoded.
Record its method, training-data cutoff and availability; `baseline_fitted_before`
must be no later than the prior origin. A nonempty reference or passing check
does not prove these statements. A real consumer tariff needs a separate adapter;
this route supports only spot-price proxy in DKK/MWh. Hourly interval energy or
average cannot be re-labelled as point data to pass it.

Once prerequisites are resolved, prepare TRAIN without evaluating a likelihood:

```powershell
python -B -m src.exe.run_model.bm_1.curves history-fit --config files/output/bm_1/runs/estimation/h2_pilot_config.json --run-dir files/output/bm_1/runs/estimation/h2_pilot --prepare-only --blas-threads 1
```

Then admit at most three new objective candidates as a runtime pilot:

```powershell
python -B -m src.exe.run_model.bm_1.curves history-fit --config files/output/bm_1/runs/estimation/h2_pilot_config.json --run-dir files/output/bm_1/runs/estimation/h2_pilot --resume --limit-candidates 3 --max-seconds 300 --blas-threads 1
```

The fixed default total is **73 distinct objective candidates**: one linear
reference, at most 24 two-piece candidates and at most 48 four-piece candidates.
Each family starts from its declared linear logit vector. Physical parameters,
noise, regularization and numerics remain fixed; no Cartesian parameter/curve
grid or daily fit is introduced. Every candidate replays the entire contiguous
TRAIN stream from its own prior; midnight does not reset state. The five-minute
deadline is cooperative between candidates: one admitted candidate and a restored
best-candidate replay may finish after it. Restored replay work is disclosed
separately and does not expand the distinct-candidate budget. A larger declared
TRAIN changes cost per candidate; use the small invocation before extending it.

Candidate traces are saved atomically after every new candidate. Source, config,
input digests, software versions and settings must match on resume. A `STOP` file
in the run directory pauses admission; remove it before resuming. Finished families
are checked against saved traces, geometry, evidence/masks and probability guards,
then reused without new FV work. Runtime progress uses the existing remaining/total
time estimates. Run from clean committed source, and do not switch branches or
edit inputs during a calculation.

Reports retain negative budget stops and mark scientific/numerical acceptance
false. TRAIN gains do not select a winner. Selected-curve time/space/quadrature
refinements, MODEL_SELECTION and H3 still follow; a completed pilot does not
complete H2/H3. No historical workbook or holdout observations were read during
this continuation; the runtime/resume tests use synthetic XLSX and a small real
FV likelihood.

## Next scientific step

Resolve the metadata above, declare the split and earlier baseline history,
freeze scale/regularization/model-selection protocol and a small total candidate
budget. Prepare full declared TRAIN streams and run a bounded pilot of one
shared f/g pair across them. Compare linear, two-piece and four-piece families
with the declared validation protocol and selected-curve numerics; do not
declare a winner from TRAIN likelihood alone. Physical parameter/noise conditions
must be explicit; conditional curve fitting is not joint recovery. Keep
FINAL_HOLDOUT inaccessible until the model and test procedure are frozen.

## Validation scope

Synthetic XLSX tests cover poisoned excluded measurements, preserved missingness,
explicit population/price contracts, selected formulas, numeric Excel dates/times,
duplicate/off-grid times, DST and handoff to the existing H adapter. Header CLI
tests verify that no scientific study starts. No historical observation workbook
was read in this session; these fixtures establish implementation behavior,
not metadata or H2/H3 scientific acceptance.

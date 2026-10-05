# H2: raw historical inputs and the next fitting pilot

Updated 5 October 2026. H1a-v2 and all six conditional synthetic H1b fits are
complete on PR #62/#64. This H2 preparation does not repeat their studies or
claim historical fitting/validation. See [F4.md](F4.md) for scientific status.

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
| Observations | Establish power units and point versus hourly average/energy. Current H likelihood accepts documented point power only; interval data require a separate validated likelihood. |
| Time | Demand timestamp timezone, meaning of hour labels and DST representation. Duplicate or ambiguous local hours are not silently averaged or inferred. |
| Price | Actual consumer tariff, or explicitly declared spot proxy plus area. SpotPriceDKK cannot establish a consumer tariff. Document price units. |
| Split/exposure | Chronological TRAIN / MODEL_SELECTION / FINAL_HOLDOUT and an exposure audit. Existing plots/previews mean blind status cannot be assumed from dates. |
| Scale/baseline | Fixed physical and price bounds; training-only scale provenance; history-only baseline and sufficient pre-TRAIN history. No holdout-driven scaling/clipping. |

These facts remain unresolved unless backed by actual data documentation.
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
history or construct a baseline. Neither route provides an H3 holdout evaluator
or a historical optimizer invocation yet.

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

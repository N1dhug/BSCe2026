# F4-H2: implemented interval forecast adapter, 6 October 2026

The first Sol implementation step of the [compact Astra design](H2_COMPACT_ESTIMATION_REVIEW.md)
is complete: conditional interval moments, seven-horizon marginal CDFs,
precision gates and open-prefix checkpoint continuation. The focused independent
control passes. Historical data/throughput admission, fitting and H2/H3
scientific acceptance remain **false**. The older 8.76-hour illustration is
still conditional; these small controls do not measure historical throughput.

## Implemented interfaces

The isolated F module is `src/method/bm_1_interval_forecast.py`.

| Interface | Meaning |
|---|---|
| `native_interval_moments` | Same native positive CN/Strang operator; full log amplitude plus conditional mean/variance. Adds observation noise only to the reported Y variance. |
| `interval_cdf_plan` | Immutable noise/span/frequency plan with analytic alias and truncation bounds, each <=1e-10. Small noise can require N>64; price the actual N. |
| `native_interval_distribution` | Existing **untilted** native CF, endpoint marginalization and Gaussian observation convolution. Returns full log state, CDF audit and 0.05/0.5/0.95 quantile brackets. |
| `compare_interval_moments` | Base/fine interval mean and variance gate, each <=0.002 in normalized units. The caller retains its separate state-grid gates. |
| `compare_interval_distributions` | Base/fine CDF and quantile gates, each <=0.002; declared 8N+1 grid, quantile endpoints and optional post-freeze observed locations. |
| `forecast_open_intervals` | One chronological prefix, 168 hourly moments by default and distributions at 1/6/12/24/48/72/168. No observations argument or Bayes updates. |
| `interval_forecast_checkpoint` | Payload for existing real-array `.npz` I/O, preserving -inf true zeros, finite underflowed log tails and split real/imaginary CF arrays. |

Each hour describes [h-1,h], not cumulative week energy. Reward half-steps
retain the original hour's Delta across input changes. Accumulator fields reset
only at hour boundaries; full log state continues. No moment subtraction,
Gaussian closure, flooring, clipping, probability renormalization or CDF shape
repair is introduced. Outside the chosen support, the *compact periodic-law
CDF* is defined as zero/its actual mass; this is not clipping an invalid CDF.
Its approximation allowances do not certify floating-point/continuous-SDE error.

The caller owns immutable manifest validation, future input availability,
parent/worker deadline, eight-worker mapper, common ledger, state-grid precision
and exported physical units. In particular, these library functions do not
start workers, admit a budget or establish holdout blindness. An interrupted
week continues from its last complete hour using `completed_intervals`, the
saved boundary log state and only remaining operators. Verify source/input/
grid/plan identities first; preserve earlier rows/distributions in their own
checkpoints. A callback can save each complete payload through the existing
`_atomic_report` with a `.npz` filename. Existing checkpoints are not overwritten.

The filtering likelihood, native v2 kernel, legacy endpoint `forecast_fv`,
G and WM source are unchanged. No historical measurements have been decoded.

## Focused evidence and shared time

Runner: `root/f4/check_h2_interval_forecast.py`; independent control:
`src/validation/bm_1/interval_forecast.py`. The **final** immutable evidence is
`files/output/bm_1/runs/estimation/f4_h2_interval_forecast_control_final_20261006.json`.
Its 57 source hashes and Python/NumPy/SciPy/Numba/llvmlite environment are saved.

| Check | Result |
|---|---:|
| Independent dense raw moments vs log conditional moments | maximum error 1.11e-16 |
| Within-hour input-change continuation vs combined operator | exact |
| Independent dense characteristic transform | maximum error 3.51e-16 |
| Analytic Gaussian CDF, sigma=0.01 and N=196 | maximum error 4.65e-11 |
| Analytic Gaussian quantiles | maximum error 5.45e-5, bracket width <=0.0002 |
| True zero and finite log mass -900 | preserved |
| Hourly accumulator reset and caller prior ownership | passed |
| Hour-boundary resume and real-array checkpoint roundtrip | exact |
| Deliberate precision violation, invalid prior and deadline stop | rejected |

The first invocation found the environment's pinned Numba dependency missing;
that failed import, restoring Numba 0.62.1/llvmlite 0.45.1 and both source-specific
controls were all charged. The first completed control remains archived; the
second adds the required moment gate, provenance and checkpoint/resume checks.
No completed old 40-hour native/FV cases or broad suites were repeated.

Total new charge: **33.87002325399999 s**, including installation/import/JIT/export.
Shared H ledger: **1494.4015500329392 s**. Native development:
**469.7704293930642/900 s**, leaving **430.2295706069358 s** across machines/resumes,
including the mandatory 60-second invocation stop reserve. The native seed
extends the exact previous PC/chat history. The ledger snapshot is
`files/output/bm_1/runs/estimation/f4_h2_interval_forecast_ledger_after_20261006.json`.
The final control itself charged 3.07888727100044 s. These are small-control
times, not measured BM grid/forecast-hour rates.

## Data packet and next step

`root/f4/h2_compact_data_packet_20261006.json` joins the **existing** user header
report with the audited MED IDs. All 13 exact `_MED` measurement columns and
their order now agree; UDEN is excluded. The original header bytes are archived
as `files/output/bm_1/runs/estimation/h2_header_inspection_archived_20261006.json`,
SHA256 `3be5eb7891c5fcf688cd7d43970934a04f68562ce3bdea9e5353cfc3adbfb5be`.
This resolves headers, not physical units, model inputs or historical acceptance.

Before any historical likelihood/rate gate, complete genuine F provenance for:
the price role/area and frozen scales, MED-only baseline fitted strictly before
17 February, fixed C/k/Xi/sigma_x/sigma_y, Beta prior and final-week exposure.
Nulls remain explicit; synthetic conditions and G estimates are not substituted.
Then price one bounded representative base/fine observed/neutral/moment/CDF
gate and the entire remaining work graph, including auxiliaries and the actual
validation reserve. No optimizer is admitted by this implementation.

Sol High remains sufficient. There is **no new local model command yet**:
the next requirement is the F data packet, then the single declared rate gate.
Do not rerun the completed forecast control or earlier PC timing commands.

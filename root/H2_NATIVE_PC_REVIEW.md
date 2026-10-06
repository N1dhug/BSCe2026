# F4-H2: PC raw review and native v2, 6 October 2026

**Review completed:** the active decision and Sol implementation handoff are in
[H2_COMPACT_ESTIMATION_REVIEW.md](H2_COMPACT_ESTIMATION_REVIEW.md). The new,
explicit 48/24/168-hour study supersedes the old period/work-graph choice;
its actual timing/data admission is still pending. Preserve all earlier
mathematics and evidence for their declared scopes. No completed timing run
should be repeated from the historical commands below.

**Completed v2 PC update:** all 40 units pass the raw review in 104.31 s.
The current next task and latest ledger are in
[the post-v2 mathematical review](H2_NATIVE_V2_NEXT_REVIEW.md).
The PC commands below are completed evidence; do not rerun them.

Johan's first native PC run is complete and raw-verified. The same-method
arithmetic optimization v2 also passes its Linux equivalence check. Historical
fitting is not admitted. This is the active handoff after
[H2_NATIVE_DEVELOPMENT.md](H2_NATIVE_DEVELOPMENT.md).

## First PC evidence

The original uploaded ZIP is preserved byte-for-byte in
`files/output/bm_1/runs/estimation/f4_h2_native_pc_pilot_20261006/` with SHA256
`38089ab4eed62e302bde43428fbae6673d8c93bd88ebb4d100046517d4c54c04`.
The reproducible reader is `root/f4/audit_h2_native_pc.py`; raw_audit.json,
the pre-PC ledger and exact executed-source bundle are beside the ZIP.
All 54 source hashes match exact Git/Windows LF/CRLF bytes. Manifest/input/
candidate/checkpoint identities, chronology, log-normalization, reconstructed
means, contour guards, all reference comparisons and full timings pass.

The run has **40 units, 36 observed updates and four missing predictions**,
eight frequency workers, BLAS=1, Numba=0.62.1/llvmlite=0.45.1 and fastmath=False.
It took **118.6677913 s**, charging 118.6747645 s including final saving.
Largest finite-log-state error is 9.46e-10 against 1e-7; conditional guard
2.10e-9 against 1e-8; overlap 1.54e-8 against 1e-7; mass error 1.19e-13 against
1e-10. No old FV trajectory was rerun and no historical measurement was read.

Python 3.12 changed float sum(), while this PC ran 3.11.7. Its narrow-day
score is exactly -95.37892393456146 under ordered binary64 addition; Python
3.12's sum returns -95.37892393456147, a 1.42e-14 difference. The read-only
auditor reconstructs the PC arithmetic exactly, with no tolerance/checksum
replacement. One-ulp wrong cumulative scores in either direction are rejected.
Original execution/resume validation is untouched. The independently generated
initial log-prior differs by only 1.14e-12 between libraries; mathematical
inputs and zero support match, and the original PC spec remains authoritative.
See [Python's documented sum change](https://docs.python.org/3.12/library/functions.html#sum).

## Budget implication

The first observed hour includes worker startup and takes 5.83485 s. Even
excluding that hour, the observed synthetic maximum is **3.54858 s/hour**.
The complete 49-pair plan has 9072 base units and 1344 fine units before
actual prior lead and additional forecast/control jobs. With the prescribed
1.5 factor, its **base work alone estimates 13.41 hours** at this warm rate.
The optimistic full illustration is 20.08 hours, using zero lead, an unmeasured
operation-count fine multiplier 2*2561/1921 and a 3600-second placeholder for
additional jobs. It is not actual historical throughput or a rigorous bound.
Unknown work cannot be zero; this illustration cannot admit a historical run.

49 is an upper limit, not a requested minimum, but a smaller arbitrary cap
is not evidence of optimizer convergence. Do not restart the old 17-coordinate
panel, omit required families/fine streams, silently shorten the chosen TRAIN
week or restore ceil(candidates/8) pricing. All candidates remain sequential.
The MED population/February dates, likelihood and scientific gates are unchanged.

## Same-method v2

Only repeated arithmetic in the isolated native kernel changed:

- A positive two-term sum returns complementary CF weights from one exponential;
  the smaller weight is computed directly. The log amplitude keeps finite tiny
  terms, even when an ordinary CF mixing weight underflows. Three-term explicit
  addition uses two nested positive mixtures; this changes floating-point
  rounding, not the discrete operator or the likelihood.
- Frozen ordinary neutral coefficients reuse the exact same math.exp results
  instead of recomputing them every substep.
- When the checked zero-frequency coefficient is exactly 1, log(1) and division
  by 1 are skipped. Non-unit correction and every guard remain unchanged.

The existing independent dense/Gaussian/one-jump check and the same matched
24+8+8 reference hours pass. Maximum finite-log-state difference is 9.19e-10,
cumulative score difference 1.98e-12. V2 took **80.72 s here**, versus the earlier
104.43 s Linux invocation. These are different invocations, not a paired PC
speedup experiment. V2 PC speed and historical/fine rates are still unknown.
Source, actual log checkpoints and control evidence are in
`f4_h2_native_v2_development_20261006.zip`; SHA256
`84f30cad364dcbf009cce9a6ed894acb1adc4e1a50ab13d3600ec0d6e08b5758`.
All 49 old F dependency files and G/WM code are unchanged. No broader suite,
new discretization/solver/grid panel, historical fit or LaTeX build was run.

## Shared time and next PC command

After importing the PC ledger and charging the read-only review, corrected
cross-version reader, v2 compilation/checks and source review, H is charged
**1352.8644959578778 s**. Native development uses **328.2333753180028/900 s**;
**571.7666246819972 s** remains across PC/chat/resumes. The updated native seed
exactly extends the PC history; the original pre-PC seed is separately preserved.
The current ledger also has exact matching Python 3.11/3.12 accounting sums.
The overall 4–6-hour target/10-hour ceiling and at least 7200-second validation
reserve remain binding; price actual mandatory work and increase the reserve.

After all other model processes stop, update the existing clean checkout from
main. Numba is already installed. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\root\f4\run_h2_native.ps1 `
  -RunDir 'files/output/bm_1/runs/estimation/f4_h2_native_v2_pc_20261006' `
  -Seconds 300
```

This uses a new manifest/checkpoint/archive name and preserves the first run.
At most five minutes includes a one-minute stop reserve; actual time is charged
to the same remaining allowance. It runs only the changed kernel on the same
declared comparisons, then creates `f4_h2_native_v2_pc_review_20261006.zip`.
Return that ZIP with its ledger. A pause is not a pass; source/environment locks
and the no-repeat rule for completed/failed runs still apply.

Next: use actual v2 PC timings and the completed independent F data contract
to price the whole estimator/validation graph. Actual rates may differ from
the deliberately narrow synthetic likelihood. If admission still fails,
deliberately revise the search/period design before optimizer enablement.
There is no automatic over-budget fit or acceptance after this timing check.

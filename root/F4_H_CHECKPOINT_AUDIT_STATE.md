# F4-H recovery and checkpoint audit — session state

Updated 2026-10-05. Branch `feature/f4-h-checkpoint-audit`, PR #64 targets
`feature/f4-h-resume` / PR #62. **Johan merges.** The newest result section in
`root/F4.md` overrides historical next-step paragraphs.

## Repository and ownership

- Last verified published H code: `928c1f3b58d067b478129cef788b6d9fe3672ac8`.
  This checkpoint also preserves the new main delta through `06942cf5cd29d457f595e611d5564b5c94bb9fdc`
  (PR #63) by applying its reviewed patch without changing the original worktree.
  Main and PR #62 remain untouched; Johan retains both PR merge decisions.
  The status file's containing commit identifies its current HEAD (a commit
  cannot contain its own hash). Verify with `git log -1 --oneline` and remote.
- Audit checkout: `/workspace/scratch/f4ee44dd2d5b/h-review`.
- Original checkout: `/workspace/scratch/a0b2c65d50d3/BSCe2026`, still clean at
  `e7d9089a4e9f6606c0877ddd6e268a61a645fca9`, branch `feature/f4-h-resume`.
  No source, HEAD or branch changes were made there. It owns the completed fit
  and frozen numerical dependency. No session-start process remains running.
- The isolated sparse worktree was required while the original H1b calculation
  still ran. Some local historical workbook/plot blobs were unavailable; they
  were not reconstructed or read. Original worktree/raw data were preserved.
- Remote publication uses GitHub Git-data APIs; exact tree/commit objects are
  verified locally. No synthetic source hash replaces a published commit.

## Status

| Task | State / evidence boundary |
|---|---|
| Integration / layout / Windows checkpoint base | DONE before this session, preserved from PR #61/#62 |
| H1a-v2 raw recovery | DONE before this session; independently rechecked all 96 raw units here, no FV |
| Real responsive fit pause/resume | DONE on e7d9089: stored 3+3 vs 6 traces and posteriors re-compared exactly here |
| Six H1b fits | DONE on original e7d9089; 675 distinct candidates; two 250-candidate budget stops remain |
| Selected-fit numerics | DONE on e398a5a; 12 units / 48 replays; every registered gate passed |
| Selected raw arithmetic review | DONE; 2,304 updates; score <=1.203e-15, posterior L1 <=2.232e-15; no FV |
| Checkpoint audit fixes | DONE and published on e398a5a; full local + CI verified |
| Previously staged TRAIN / array-index / completed-fit fixes | DONE on 928c1f3; 619-test CI run 324 passed |
| H2 / H3 | BLOCKED on actual metadata, explicit consumer-price/proxy interpretation and exposure audit |
| Windows study | No new mandatory long Windows run; Windows checkpoint CI passed on 928c1f3 |
| I/J final acceptance, demand-hop diagnosis, F5/F6 | Not advanced by these development studies |

## Recovery events and exact commands

The preexisting process ended with traces at 214 linear/4 and 249 nonlinear/4
candidates, without both final results. Before recovery: source fingerprint
matched the fit manifest; the original OS run lock was available; stored
responsive 3+3 versus 6-candidate traces and posteriors compared exactly.
The same clean source resumed, did 37 new candidates in 201.884 s, and finished:

```bash
# cwd: original checkout, pinned e7d9089; workers=3, BLAS=1
python -B -m src.exe.run_model.bm_1.curves fit --numerics-run files/output/bm_1/runs/estimation/h1_v2_parallel8 --run-dir files/output/bm_1/runs/estimation/h1_fit_parallel --resume --workers 3 --max-hours 8 --blas-threads 1
```

Source, settings, seeds, independent initial laws, two/four-piece family starts,
lambda=0.1, optimizer versions and total budget=250/family remain in the original
fit and numerical manifests. No fit candidate's conditional state was reused for
another candidate. Completed runs need no further mutating resume.

Own selected-check source was published and clean at e398a5a before these runs:

```bash
# cwd: audit checkout; one worker, BLAS=1
python -B -m src.exe.run_model.bm_1.curves validate-fit --numerics-run /workspace/scratch/a0b2c65d50d3/BSCe2026/files/output/bm_1/runs/estimation/h1_v2_parallel8 --fit-run /workspace/scratch/a0b2c65d50d3/BSCe2026/files/output/bm_1/runs/estimation/h1_fit_parallel --run-dir files/output/bm_1/runs/estimation/h1_selected_audit_e398a5a --max-hours 0.02 --blas-threads 1
# Pilot paused after 3 units / 84.704 s; a new process resumed on the same source.
python -B -m src.exe.run_model.bm_1.curves validate-fit --numerics-run /workspace/scratch/a0b2c65d50d3/BSCe2026/files/output/bm_1/runs/estimation/h1_v2_parallel8 --fit-run /workspace/scratch/a0b2c65d50d3/BSCe2026/files/output/bm_1/runs/estimation/h1_fit_parallel --run-dir files/output/bm_1/runs/estimation/h1_selected_audit_e398a5a --resume --max-hours 8 --blas-threads 1
```

Resume reused 3 units and computed 9 in 167.366 s, exit 0. Pilot throughput was
about 12.5 s/unit after prerequisite reading; the declared rough continuation
estimate was 160 s. No tolerances, grids or seeds changed. Runtime measurements
include contention and prerequisite I/O and are not a general speedup claim.
Never resume this directory on the later cleanup source; read it as immutable
evidence, or create a new independently declared run directory.

## Scientific findings and limits

Two-piece fits used 84 / 63 / 21 candidates (linear / nonlinear / Xi=0).
Four-piece fits used 250 / 250 / 7; the first two are budget-stopped.
TRAIN gains against linear are 0.149 / 8.754 / 0 and 1.151 / 30.599 / 0.
No model winner is declared from TRAIN. Nonlinear-case f/g grid RMSE is
0.4701/0.1513 for two pieces and 0.0547/0.0909 for four, descriptive only.
Known physical parameters, two development streams per case, and finite
optimizer budgets do not prove population recovery, identifiability or
forecast generalization. Xi=0 correctly supplies no curve information.

Selected-curve maximum refinement differences: prefix evidence 0.00509427,
linearly centered evidence 0.01230423, posterior L1 0.00500849.
Original v1 rejection and v2 numerical history remain preserved.

## Tests actually run

- e398a5a: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m unittest discover -s tests -q`
  — 617 tests, 656.996 s, exit 0, 2 platform skips. GitHub CI run 322 also
  passed regression, Windows launcher/locking and LaTeX notes.
- Before the checkpoint fix: regression reproductions accepted invalid NaN
  totals, masks and probability diagnostics, and orphaned raw scores inconsistent
  with the selected fit. All new regressions pass after the fix.
- After recovered fixes, on e398a5a plus this patch: unittest discovery patterns
  `test_bm1_selected*.py`, `test_bm1_curve*.py`, `test_bm1_response*.py`,
  `test_bm1_reports.py`, `test_bm1_history_contract.py`, `test_checkpoint_io.py`
  — 45 tests, 19.213 s, exit 0, 2 Windows-only skips.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m unittest discover -s tests -p test_bm1_data.py -q`
  — 16 tests, 0.551 s, exit 0.
- 928c1f3: GitHub run 324 (`37250598670`) ran the complete regression on its
  PR merge ref: **619 tests in 658.565 s, OK, 2 platform skips**. Windows
  launcher/actual locks and notes also passed. The reviewed PR head was 928c1f3.
- Main-integration targeted tests and final combined full CI are recorded in
  the next handoff checkpoint; do not infer their result from run 324.
- `git diff --check` passes.

## Review ledger

| Area | Finding / action |
|---|---|
| Selected fit metadata | Reject nonfinite totals, nonboolean/2-D masks and failed probability diagnostics |
| Selected raw-unit recovery | Every read binds base scores/final posterior to exact selected fit, even after crash before summary; valid units require zero new FV |
| Completed family recovery | Apply the same fit validator before using a finished family as a downstream start |
| Report NPZ I/O | Recovered writer/reader mismatch at index 10000; preserve archive format and test 10001 arrays |
| Historical TRAIN adapter | Recovered silent omission of first TRAIN point; explicit earlier hourly prior origin and complete B/U support required |
| BM/WM structure | Preserved model/params/method/execution/tools separation and existing shared Windows-safe I/O |
| Broader old F/G cleanup | Not claimed complete; large cosmetic refactors deferred to avoid the H critical path |

## Durable artifacts

All following uploads succeeded and local Library metadata was applied.
Every part has a per-file SHA256 manifest; archives were read back locally
and ZIP integrity plus every contained file hash were verified before upload.

- Preexisting numerical dependency: `H1a_v2_complete_90211e4_manifest.json`,
  `libfile_0c040030e4ac8191918fa80e497a6dfb` (48 original archive parts).
- Completed fit (all 17 files): `H1b_completed_e7d9089_20261005_part_01.zip`,
  `libfile_d3b58760d0ec81919a3994b864c82bb6`; manifest
  `libfile_4cd72f7567e08191a54d503be648f7b2`.
- Selected checks (14 files): `H1b_selected_e398a5a_20261005_part_01.zip`
  through `_part_06.zip`; manifest `libfile_ee0ef24882a48191bce1478be128643c`.
  Part IDs in order: `libfile_d5ebc5708ec08191b1b9d59cd79f4456`,
  `libfile_a01eed4ce25c81918ebfb125388ad4ac`,
  `libfile_136a6265e6dc8191a6266a9509bf9e07`,
  `libfile_0435b14bd3e88191bebe8cee6770b42f`,
  `libfile_1e7b0b094bbc81918381dcb72664010e`,
  `libfile_7229c89a8b308191b3a76c363e6824a0`.
- Consolidated audit script/results/logs index will be linked in the final
  checkpoint after publication; do not mistake pending evidence for uploaded.

Archives extract beneath `files/output/bm_1/runs/estimation`. Preserve dependency
summary bytes: its hash is part of the fit manifest; a mutating numerical resume
would change that dependency even if no FV work were repeated.

## Next concrete actions

1. Verify/publish this checkpoint and check full combined CI. PR #64 includes
   the current main G2 delta so it preserves PR #63 alongside H; it still targets
   PR #62's branch. Leave both PR merges to Johan.
2. Resolve actual H2 metadata without accessing FINAL_HOLDOUT observations:
   population/meter membership, physical units/scale, point vs interval meaning,
   baseline provenance, consumer tariff vs declared proxy, timezone/missingness,
   chronological TRAIN/MODEL_SELECTION split and exposure-audited final holdout.
   A metadata-reference string or an old preview is not proof of these facts.
   Code-only preflight found that legacy input/price constructors read the whole
   workbook/series and interpolate/ffill/bfill before date selection; reusing
   their processed output can leak across a split. Supply raw TRAIN-only inputs
   before preprocessing through a separate H contract, preserving WM/F3 semantics.
   No historical observations or FINAL_HOLDOUT values were opened for this audit.
3. Only after those inputs are documented, freeze H2 normalization, regularization,
   optimizer budgets and model-selection protocol; fit one shared f/g pair across
   all declared TRAIN streams. Keep FINAL_HOLDOUT inaccessible until model and
   test procedure are frozen. Continue to I/J only when those prerequisites pass.

No new long Windows study is required for this handoff. To check the code after
merging (from a clean checkout with no running calculation):

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch failed." }
git switch feature/f4-h-checkpoint-audit
if ($LASTEXITCODE -ne 0) { throw "Branch switch failed." }
git pull --ff-only origin feature/f4-h-checkpoint-audit
if ($LASTEXITCODE -ne 0) { throw "Git pull failed." }
python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Regression failed." }
```

Use `main` in those branch commands once Johan has merged and deleted the branch.
For an optional new Windows numerical reproduction, restore all referenced
archives, use a new selected-check directory, BLAS=1 and the guide's `validate-fit`
command. Its output is `manifest.json`, `summary.json`, `units/*.npz`; resume only
that exact source/environment. Send the summary and complete raw directory if
review is desired. Completed studies need not be recomputed for cleanup.

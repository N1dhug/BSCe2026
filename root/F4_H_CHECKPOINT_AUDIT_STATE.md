# F4-H checkpoint audit — session state

Independent recovery review, 2026-10-05. This complements PR #62; Johan merges.

## Repository state
- Branch: `feature/f4-h-checkpoint-audit`, based on published `e7d9089a4e9f6606c0877ddd6e268a61a645fca9`.
- Primary H1b run remains in `/workspace/scratch/a0b2c65d50d3/BSCe2026` on `feature/f4-h-resume`; observed live process 9145 uses three case workers. Do not change its source, HEAD, branch or dependency files.
- Isolated sparse worktree: `/workspace/scratch/f4ee44dd2d5b/h-review`. Required because the primary checkout has an active source-locked run. Historical input/plot blobs are unavailable in the local object database and are excluded; do not reconstruct them or access holdout observations.
- Remote PR #62 confirmed base `bf3ef6ed163b4512e03a8ac0f42de888f535c094`, head `e7d9089`; no merge performed.
- This initial status is the first publication on the audit branch. Later checkpoints list verified parent commits; the status file's containing commit is available from Git history.

## Task status
- DONE: locate and preserve the ongoing H1b run; verify current PR metadata and published source.
- DONE: reproduce missing nonfinite-score/mask/probability checks and missing selected-fit linkage after a crash before summary publication.
- DONE: implement focused validation and real tiny-fit recovery tests. Correct orphaned raw units resume with zero new FV replays; changed scores/posteriors are rejected.
- IN_PROGRESS: publish the tested fix and run the full regression before declaring the PR ready.
- BLOCKED: historical H2/H3 scientific claims still require actual metadata and a documented blind split.

## Existing evidence (not rerun in this session)
PR #62 reports H1a-v2 96 units / 384 replays, durable raw archives and exact responsive 3+3 vs 6 candidate resume parity. The parity file and raw H1a directory are present locally. This session has not independently revalidated the complete evidence yet. Existing work also stages report-array and TRAIN-origin fixes outside the locked checkout; avoid duplicating those edits.

## Tests / artifacts
Verified pushed parent: `2265e2f558496ed003290d23da7a3037f727b43b` (initial session state).
On that parent plus this patch: 32 targeted tests passed in 20.334 s, exit 0.
Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -c "import unittest; s=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p) for p in ['test_bm1_selected*.py','test_bm1_curve*.py','test_bm1_response*.py','test_bm1_reports.py']); r=unittest.TextTestRunner().run(s); raise SystemExit(not r.wasSuccessful())"`.
Full regression has not yet run on this patch. `git diff --check` passes.
Before the fixes, targeted tests reproduced six invalid-metadata acceptance failures and acceptance of shifted raw evidence after orphan-unit recovery. New tests use small actual synthetic fits only; they do not read historical data.
No new scientific runs or claims. The original H session owns its active fit files and their archival. See `root/F4.md` and PR #62 for preexisting evidence indices.

## Next concrete action
Publish this checkpoint, run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m unittest discover -s tests -q`, then create a small follow-up PR targeting the H recovery branch. Verify latest upstream before publication. This patch is independent of the primary session's staged report-array, completed-family and TRAIN-origin fixes. Do not change the primary source-locked run, resume its calculations on this new source, or edit existing source manifests. A new selected-check run may read the old fit as an immutable dependency after fitting finishes.

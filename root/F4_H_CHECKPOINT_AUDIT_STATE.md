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
- IN_PROGRESS: inspect selected-fit checkpoint validation and replay linkage, without rerunning the active fits.
- TODO: reproduce concrete validation defects, implement focused fixes, run tests, publish reviewable changes.
- BLOCKED: historical H2/H3 scientific claims still require actual metadata and a documented blind split.

## Existing evidence (not rerun in this session)
PR #62 reports H1a-v2 96 units / 384 replays, durable raw archives and exact responsive 3+3 vs 6 candidate resume parity. The parity file and raw H1a directory are present locally. This session has not independently revalidated the complete evidence yet. Existing work also stages report-array and TRAIN-origin fixes outside the locked checkout; avoid duplicating those edits.

## Tests / artifacts
No new tests or numerical studies have run in this audit checkpoint. No new raw outputs yet. The original H session owns its active fit files and their archival. See `root/F4.md` and PR #62 for the verified preexisting evidence index.

## Next concrete action
Review `src/validation/bm_1/response_fit_checks.py` and the `run_fit_checks` resume path. Add regression reproductions for nonfinite checkpoint metadata and consistency of recovered numerical units with the exact selected fit, then implement validation using existing contracts. Keep all changes on this branch; do not mutate the primary run checkout.

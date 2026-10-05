# F4-H2: next bounded task after the search trial

The positive joint surrogates have not passed the full declared development
stream. Do not integrate them into historical fitting, relax .05/.002/.05,
drop a candidate, or replace interval averages by point observations. The
isolated CN subclass is an experimental control only; production code is
unchanged. Reuse all positive and negative raw evidence described in
[H2_SEARCH_FEASIBILITY.md](H2_SEARCH_FEASIBILITY.md). G is independent.

The next concrete implementation should preserve the precise terminal-tilt
likelihood and limit the search itself. Reuse curve geometry, frozen historical
inputs and checkpoint machinery. Prepare a small predetermined candidate
batch around the linear start for both declared curve families; a bounded
coordinate/pattern search is a possible implementation, not a validated fit.
Every candidate must replay the entire contiguous declared TRAIN from its
own prior and carry its own log posterior. Parallelize candidates, never days
of the same state stream. Keep BLAS=1 and at most 8 workers on Johan's PC.

Use one aggregate wall-time budget, candidate ceiling and hourly log-state
checkpoints, including resumes, worker startup, repeats, selected-fit precision,
validation and existing study time. Subprocess admission must leave time to
stop/save before 10 hours. The target remains 4–6 hours. A complete batch must
retain failures and budget stops, and cannot imply optimizer convergence or
TRAIN-based model selection. No guessed historical metadata or observation
window may be introduced to make the runtime estimate fit the target.

Before a historical fit, resolve H's actual population, scale, prior origin,
physical/noise condition, baseline and chronological exposure/split contract.
G's configuration/observational exposure is not automatically H's contract.
Measure a short representative parallel pilot, including contour-bank behavior
and startup; then evaluate the total budget. No 73-candidate run is admitted.
The saved 630.87-second synthetic day is only a planning rate, not historical
Windows throughput or a guarantee across candidate curves. Changing the TRAIN
period needs an explicit scientific declaration; the seven-day example does
not select it. If the bounded precise plan cannot fit the real contract below
10 hours, reconsider the likelihood implementation in a mathematical review
rather than start a doomed fit. No further surrogate panel is automatic.

The measured failed trial cost 415.89 seconds; reserve another 10 seconds for
the brief verification checks and preserve 425.89 seconds in the remaining-H
ledger. Freeze the final model and validation protocol before
FINAL_HOLDOUT, validate selected curves with the precise method, and export
one common f/g pair only after the required H2/H3 decision. Wider profiles,
bootstrap and robustness panels remain postponed. This handoff implements
the reduction in scope; it does not claim H acceptance.

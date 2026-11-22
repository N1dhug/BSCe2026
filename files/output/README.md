# Generated results

This directory is the output root for retained BM1 and WM4 report executables.
The thesis figures in Chapters 3–5 are reproduced by current Python programs,
rather than taken from the old experimental output archives.

All previously tracked files in `files/output/` were removed in the
November 2026 output cleanup: none was an identified Chapter 3–5 figure
artifact with its corresponding active generation workflow.

Run an executable to recreate its output directory automatically. Relevant
examples:

- `data/preliminary_statistics/`: Chapter 3 descriptive statistics.
- `wm_4/corresponding_interval_fit/`: Chapter 4 historical and oracle examples.
- `wm_4/moving_horizon_predictions/`: Chapter 4 moving-horizon examples.
- `wm_4/moving_horizon/`: Chapter 5 deterministic moving-horizon sweeps.
- `wm_4/moving_horizon_diffusion/`: Chapter 5 stochastic calibration.
- `bm_1/fixed_beta_samples/`: Chapter 5 BM1 selected samples.
- `wm_4/selected_sample_nll/`: Chapter 5 WM4 selected-sample statistics.
- `bm_1/single_day/` and `wm_4/single_day/`: current single-day model diagnostics.

The removed `graphs/data/season/` and `graphs/data/weekly/` plots were
separate exploratory overviews, not the figures shown in the thesis.
Their generators may still exist, but running them recreates these folders.

Input data in `files/input/` were not modified. To recover historical
experimental outputs, check out the preceding Git revision rather than
re-running deleted prototype executables.

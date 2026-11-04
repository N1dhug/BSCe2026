# Analyze the best historical backlog in the completed WM4 N-k sweep

This executable reads saved `oracle_horizon_scores.csv` (one row per forecast origin, baseline, historical matching-weekday backlog N, and oracle-optimized prediction horizon k). **No model fitting or oracle reoptimization occurs.**

Run from the repository root, after the complete N=2..14 and k=1..24 sweep:

```bash
python -m src.exe.run_model.wm_4.analyze_weekday_backlog --group ALL
python -m src.exe.run_model.wm_4.analyze_weekday_backlog --group MED
```

Select a less optimistic range of oracle horizons, excluding k=1 and k=2, using:

```bash
python -m src.exe.run_model.wm_4.analyze_weekday_backlog --group ALL --min-horizon 3
```

The default input is `files/output/wm_4/weekday_horizon_sweep_expanded/all/` (or `med/`). Customize it with `--input DIRECTORY`. Customize the output folder with `--output DIRECTORY`.

## RMSE aggregation

For baseline B, historical backlog N, forecast origin o and horizon k, the existing sweep stores

```
MSE[B,N,o,k] = mean((D_prediction[j]-D_observed[j])**2 for j in range(k))
```

The *primary pooled RMSE* treats every predicted hourly observation from every selected k as one error:

```
pooled_rmse[B,N] = sqrt(sum_{o,k} k*MSE[B,N,o,k] / sum_{o,k} k)
```

This gives a 24-hour oracle prediction 24 times the observation weight of a 1-hour prediction. The alternative `equal_horizon_rmse` gives equal weight to each k by taking `sqrt(mean_{o,k}(MSE))`. They answer different questions; both are reported. `all_baselines` pools the three baseline scores with equal representation.

All comparisons use **exactly the same forecast dates** across every baseline, N and k in the selected sweep range. Incomplete days are excluded; a wholly missing baseline/N/k combination causes an explicit error rather than a misleading partial ranking.

## Results

The executable prints one full backlog ranking per baseline plus the combined ranking, showing:
- Pooled hourly RMSE (main ranking)
- Equal-horizon RMSE
- Percentage gap from the best backlog in that baseline
- Number of horizons (from k=1..24) for which that backlog achieves the lowest aggregate RMSE
- The smallest backlog within 1% of the minimum pooled RMSE

It writes `backlog_summary.csv`, `backlog_by_horizon.csv`, `best_backlog_by_horizon.csv`, and `backlog_comparison.png` into `backlog_analysis/` beneath the input directory. The PNG shows pooled RMSE against N for each baseline and all baselines combined, plus counts of the horizons won by each N.

All evaluation remains *retrospective oracle* performance: each horizon has its own two-state optimized initial condition. Short k=1..2 horizons can be almost entirely explained by two free initial-state variables, so compare the default result with `--min-horizon 3` or `--min-horizon 6` before drawing conclusions. Forecast origins and horizons overlap; the summary is descriptive, not an independent-observation statistical significance test.

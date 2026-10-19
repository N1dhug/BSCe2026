from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.exe.run_model.wm_4.moving_horizon_common import OUTPUT, GROUPS, baseline_before, fit_row, group_frame, origins, plot_parameter_time, save_table, window_data
from src.params.wm_4 import optimized_qp_params, qp_params


def fit_windows(group, start, end, stride, steps_per_hour, f0_grid, max_origins=0, fail_fast=False,
                max_state_iter=50, state_tol=1e-8, sensitivity_stride=1):
    """Fit alternating QP parameters and states; F0 grid points are *seeds*, not fixed states."""
    frame = group_frame(group)
    cache, rows, failures = {}, [], []
    timeline = origins(start, end, stride)
    if max_origins: timeline = timeline[:max_origins]
    if sensitivity_stride < 1: raise ValueError("sensitivity_stride must be at least 1.")

    for count, origin in enumerate(timeline, start=1):
        try:
            baseline = baseline_before(frame, origin, cache)
            model, data, _, observed = window_data(frame, origin - pd.Timedelta(hours=24), baseline, steps_per_hour)
            fixed = qp_params(model)
            sequential = optimized_qp_params(model)
            observed_R0 = float(data["I"][0] - data["O"][0])
            alternating_kwargs = {
                "fit_initial_state": True, "max_state_iter": max_state_iter, "state_tol": state_tol,
                "f_bounds": (-1.0, 1.0), "r_bounds": (-1.0, 1.0),
            }
            # The initial [F0, R0] is a seed; the alternating QP re-estimates both states.
            fit_zero = sequential.fit(data, [0.0, 0.0], **alternating_kwargs)
            fit_observed = sequential.fit(data, [0.0, observed_R0], **alternating_kwargs)
            rows.append(fit_row(group, origin, "zero", fit_zero, observed, steps_per_hour))
            rows.append(fit_row(group, origin, "observed", fit_observed, observed, steps_per_hour))

            # Separate fixed-state ablations are retained for the r_0 hypothesis test:
            # optimizing r_0 and unrestricted F_0 together creates exact location non-identifiability.
            fit_zero_fixed = fixed.fit(data, [0.0, 0.0])
            fit_observed_fixed = fixed.fit(data, [0.0, observed_R0])
            fit_zero_intercept = fixed.fit(data, [0.0, 0.0], fit_intercept=True)
            fit_observed_intercept = fixed.fit(data, [0.0, observed_R0], fit_intercept=True)
            for name, fit in (
                ("zero_fixed", fit_zero_fixed), ("observed_fixed", fit_observed_fixed),
                ("zero_intercept", fit_zero_intercept), ("observed_intercept", fit_observed_intercept),
            ):
                rows.append(fit_row(group, origin, name, fit, observed, steps_per_hour))

            # Independently initialize and fully converge the alternating QP from every F0 guess.
            # This is a basin-of-attraction / seed-sensitivity experiment, not a fixed-F0 profile.
            if (count - 1) % sensitivity_stride == 0:
                grid_rows = []
                for F0_seed in f0_grid:
                    try:
                        fit = (fit_observed if np.isclose(F0_seed, 0.0, atol=1e-12)
                               else sequential.fit(data, [float(F0_seed), observed_R0], **alternating_kwargs))
                        entry = fit_row(group, origin, "f0_grid", fit, observed, steps_per_hour, f0_grid=float(F0_seed))
                        grid_rows.append(entry)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        if fail_fast: raise
                        failures.append({"group": group, "origin": origin, "F_0_seed": float(F0_seed),
                                         "error": f"sequential sensitivity seed failed: {exc}"})
                rows.extend(grid_rows)
                # Selection uses the same transition-residual objective as the alternating QP.
                if grid_rows:
                    converged = [row for row in grid_rows if row["state_converged"]]
                    selected = dict(min(converged if converged else grid_rows, key=lambda row: row["qp_loss"]))
                    selected["variant"] = "grid_best"
                    rows.append(selected)
            if count % 24 == 0:
                print(f"[qp sequential] {group} {count}/{len(timeline)} origins; latest={origin}")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            failures.append({"group": group, "origin": origin, "error": str(exc)})
            print(f"[qp sequential] {group} {origin} skipped: {exc}")
    return rows, failures

def hac_inference(values, alpha=0.05, lag=24, equivalence_margin=None):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3: return {"n": n, "mean": np.nan, "hac_se": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(np.mean(values))
    centered = values - mean
    lag = min(int(lag), n - 1)
    lrv = float(centered @ centered) / n
    for offset in range(1, lag + 1):
        lrv += 2.0 * (1.0 - offset / (lag + 1.0)) * float(centered[offset:] @ centered[:-offset]) / n
    se = float(np.sqrt(max(lrv, 0.0) / n))
    z = mean / se if se > 0.0 else np.nan
    p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    crit = float(norm.ppf(1.0 - alpha / 2.0))
    std = float(np.std(values, ddof=1))
    iid_t = mean / (std / np.sqrt(n)) if std > 0.0 else np.nan
    row = {
        "n": n, "mean": mean, "std": std, "median": float(np.median(values)),
        "hac_lag": lag, "hac_se": se, "z_stat": z, "p_value": p,
        "ci_low": mean - crit * se, "ci_high": mean + crit * se, "alpha": alpha,
        "reject_mean_zero": bool(np.isfinite(p) and p < alpha),
        "iid_p_value": float(2.0 * t.sf(abs(iid_t), n - 1)) if np.isfinite(iid_t) else np.nan,
    }
    if equivalence_margin is not None:
        margin = float(equivalence_margin)
        if margin <= 0.0: raise ValueError("equivalence margin must be positive.")
        p_lower = float(norm.sf((mean + margin) / se)) if se > 0.0 else np.nan
        p_upper = float(norm.cdf((mean - margin) / se)) if se > 0.0 else np.nan
        p_tost = max(p_lower, p_upper) if np.isfinite(p_lower) and np.isfinite(p_upper) else np.nan
        row.update({"equivalence_margin": margin, "tost_p_value": p_tost,
                    "equivalent_to_zero": bool(np.isfinite(p_tost) and p_tost < alpha)})
    return row


def save_r0_analysis(frame, output, alpha, hac_lag, equivalence_margin):
    rows = []
    for group, data in frame.groupby("group"):
        pairs = data.loc[data["variant"].isin(("observed_fixed", "observed_intercept"))]
        free = pairs.loc[pairs["variant"] == "observed_fixed"].set_index("origin")
        intercept = pairs.loc[pairs["variant"] == "observed_intercept"].set_index("origin")
        matched = free[["fit_mse"]].join(intercept[["r_0", "fit_mse"]], how="inner", lsuffix="_free", rsuffix="_intercept")
        if len(matched) < 3: continue
        record = hac_inference(matched["r_0"], alpha=alpha, lag=hac_lag, equivalence_margin=equivalence_margin)
        record.update({
            "group": group, "mean_in_sample_mse_reduction": float((matched["fit_mse_free"] - matched["fit_mse_intercept"]).mean()),
            "intercept_smaller_fit_mse_pct": float(100.0 * (matched["fit_mse_intercept"] < matched["fit_mse_free"]).mean()),
        })
        rows.append(record)
        matched.reset_index().to_csv(output / f"{group.lower()}_r_0_observations.csv", index=False)

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        axes[0].plot(matched.index, matched["r_0"], linewidth=0.9)
        axes[0].axhline(0.0, linestyle="--"); axes[0].set_ylabel("Fitted drift intercept r_0")
        axes[0].set_title(f"{group}: dR = r_0 + r_F F + r_U U + r_R R"); axes[0].grid(True, alpha=0.25)
        axes[1].hist(matched["r_0"], bins=35)
        axes[1].axvline(0.0, linestyle="--", label="Null value")
        axes[1].axvline(record["mean"], linestyle="-", label="Sample mean")
        axes[1].set_xlabel("r_0"); axes[1].set_ylabel("Count"); axes[1].legend()
        fig.tight_layout(); fig.savefig(output / f"{group.lower()}_r_0_empirical.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(matched.index, matched["fit_mse_free"] - matched["fit_mse_intercept"])
        ax.axhline(0.0, linestyle="--")
        ax.set_title(f"{group}: in-window fit improvement from intercept (positive = better)")
        ax.set_ylabel("MSE without - MSE with r_0"); ax.set_xlabel("Origin"); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(output / f"{group.lower()}_r_0_fit_gain.png", dpi=180); plt.close(fig)
    save_table(rows, output / "r_0_null_summary.csv")
    return pd.DataFrame(rows)


def save_sensitivity(frame, output):
    """Analyse convergence *from* each F0 seed, including recovered state and parameters."""
    for group, data in frame.groupby("group"):
        grid = data.loc[data["variant"] == "f0_grid"].copy()
        if grid.empty: continue
        keys = ("r_F", "r_U", "r_R", "F_0", "R_0", "qp_loss", "fit_mse", "state_iterations")
        converged_grid = grid.loc[grid["state_converged"] == True].copy()
        if converged_grid.empty:
            print(f"[qp sequential] {group}: no converged grid runs; skipping seed sensitivity means.")
            continue
        summary = converged_grid.groupby("F_0_grid", as_index=False).agg({key: ["mean", "std"] for key in keys})
        summary.columns = ["F_0_grid"] + [f"{a}_{b}" for a, b in summary.columns.to_list()[1:]]
        summary.to_csv(output / f"{group.lower()}_f0_seed_sensitivity_summary.csv", index=False)

        # A per-window spread isolates initialization effects from genuine across-window variability.
        spread_rows = []
        for origin, subset_all in grid.groupby("origin"):
            subset = subset_all.loc[subset_all["state_converged"] == True]
            row = {"origin": origin, "seeds": len(subset_all), "converged_seeds": len(subset),
                   "converged_fraction": float(len(subset) / len(subset_all))}
            for key in keys:
                values = subset[key].to_numpy(dtype=float)
                row[f"{key}_seed_range"] = float(np.max(values) - np.min(values)) if values.size else np.nan
                row[f"{key}_seed_std"] = float(np.std(values)) if values.size else np.nan
            spread_rows.append(row)
        spread = pd.DataFrame(spread_rows).sort_values("origin")
        spread.to_csv(output / f"{group.lower()}_f0_seed_per_window_spread.csv", index=False)
        best = data.loc[data["variant"] == "grid_best",
                        ["origin", "F_0_seed", "R_0_seed", "F_0", "R_0", "qp_loss", "fit_mse", "state_iterations", "state_converged"]]
        best.to_csv(output / f"{group.lower()}_f0_selected.csv", index=False)

        fig, axes = plt.subplots(len(keys), 1, figsize=(12, 24), sharex=True)
        for ax, key in zip(axes, keys):
            x = summary["F_0_grid"].to_numpy(dtype=float)
            y = summary[f"{key}_mean"].to_numpy(dtype=float)
            sd = summary[f"{key}_std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label="Across-window mean (converged runs)")
            ax.fill_between(x, y - sd, y + sd, alpha=0.2, label="Across-window SD")
            if key == "F_0": ax.plot(x, x, linestyle="--", linewidth=1, label="Initial guess")
            ax.set_ylabel("final " + key); ax.grid(True, alpha=0.2); ax.legend()
        axes[-1].set_xlabel("Initial F_0 guess supplied to alternating QP")
        fig.suptitle(f"{group}: alternating QP sensitivity to initial F0 seed")
        fig.tight_layout(); fig.savefig(output / f"{group.lower()}_f0_parameter_sensitivity.png", dpi=180); plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        for ax, key in zip(axes, ("r_F", "r_U", "r_R")):
            ax.plot(spread["origin"], spread[f"{key}_seed_range"], label=f"{key} range across converged seeds")
            ax.set_ylabel(key + " seed range"); ax.grid(True, alpha=0.2); ax.legend()
        axes[-1].set_xlabel("Fit origin")
        fig.suptitle(f"{group}: within-window sensitivity to sequential-fit initial guesses")
        fig.tight_layout(); fig.savefig(output / f"{group.lower()}_f0_seed_within_window_spread.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(best["origin"], best["F_0_seed"], label="Best seed by final QP loss")
        ax.plot(best["origin"], best["F_0"], label="Corresponding optimized F_0", alpha=0.8)
        ax.axhline(0.0, linestyle="--")
        ax.set_ylim(-1.1, 1.1); ax.set_title(f"{group}: selected initial guess versus optimized F_0")
        ax.set_xlabel("Origin"); ax.set_ylabel("F_0"); ax.grid(True, alpha=0.2); ax.legend()
        fig.tight_layout(); fig.savefig(output / f"{group.lower()}_f0_selected_time.png", dpi=180); plt.close(fig)

def main():
    parser = ArgumentParser(description="Hourly 24-hour moving alternating-QP WM4 fit: drift-intercept test and sequential-fit seed sensitivity.")
    parser.add_argument("--start", default="2022-02-28")
    parser.add_argument("--end", default="2022-08-22 23:00")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--f0-steps", type=int, default=21)
    parser.add_argument("--sensitivity-stride", type=int, default=1, help="Evaluate the full alternating QP F0-seed sweep every N fitting origins; default every origin.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--hac-lag", type=int, default=24)
    parser.add_argument("--equivalence-margin", type=float)
    parser.add_argument("--output", default=str(OUTPUT / "qp"))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.f0_steps < 3 or args.f0_steps % 2 != 1: raise ValueError("f0-steps must be odd and at least 3 so the grid includes 0.")
    if not 0.0 < args.alpha < 1.0 or args.hac_lag < 0: raise ValueError("Invalid alpha or HAC lag.")
    if args.sensitivity_stride < 1 or args.max_state_iter < 1 or args.state_tol <= 0.0: raise ValueError("Invalid alternating fitting settings.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, failures = [], []
    for group in GROUPS:
        rows, missed = fit_windows(group, args.start, args.end, args.stride, args.steps_per_hour,
                                  np.linspace(-1.0, 1.0, args.f0_steps), args.max_origins, args.fail_fast,
                                  max_state_iter=args.max_state_iter, state_tol=args.state_tol,
                                  sensitivity_stride=args.sensitivity_stride)
        all_rows.extend(rows); failures.extend(missed)
        if rows: save_table(rows, output / f"{group.lower()}_fit_parameters.csv")
    save_table(all_rows, output / "fit_parameters.csv")
    save_table(failures, output / "skipped.csv")
    if not all_rows: raise RuntimeError("No QP windows successfully fitted.")
    frame = pd.DataFrame(all_rows)
    for group in GROUPS:
        data = frame.loc[(frame["group"] == group) & (frame["variant"] != "f0_grid")]
        if not data.empty:
            target = output / group.lower()
            target.mkdir(parents=True, exist_ok=True)
            plot_parameter_time(data, target, group)
    summary = save_r0_analysis(frame, output, args.alpha, args.hac_lag, args.equivalence_margin)
    save_sensitivity(frame, output)
    print(f"QP results: {output}; fitted rows={len(frame)}, failed windows={len(failures)}")
    if not summary.empty: print(summary.to_string(index=False))


if __name__ == "__main__": main()

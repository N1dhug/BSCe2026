from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.method.moving_horizon import OUTPUT, GROUPS, HOURS, baseline_before, group_frame, metrics, save_table, window_data
from src.params.wm_4 import optimized_qp_params


DEFAULT_QP_VARIANTS = ("zero", "observed", "zero_fixed", "observed_fixed", "observed_intercept", "grid_best", "zero_intercept")
DEFAULT_NUMERICAL_VARIANTS = ("zero", "zero_intercept")


def load_fits(qp_path, numerical_path, qp_variants, numerical_variants, start=None, end=None, max_origins=0):
    files = (("qp", Path(qp_path), qp_variants), ("numerical", Path(numerical_path), numerical_variants))
    frames = []
    for fitter, path, variants in files:
        if not path.exists(): raise FileNotFoundError(f"Missing fitted parameter results: {path}")
        frame = pd.read_csv(path, parse_dates=["origin"])
        mandatory = {"group", "origin", "variant", "r_F", "r_U", "r_R", "r_0", "success"}
        if not mandatory.issubset(frame.columns): raise ValueError(f"{path} is missing {sorted(mandatory - set(frame.columns))}.")
        frame = frame.loc[frame["variant"].isin(variants) & (frame["success"].astype(str).str.lower() == "true")].copy()
        frame["fitter"] = fitter
        frames.append(frame)
    all_fits = pd.concat(frames, ignore_index=True)
    if start is not None: all_fits = all_fits.loc[all_fits["origin"] >= pd.Timestamp(start)]
    if end is not None: all_fits = all_fits.loc[all_fits["origin"] <= pd.Timestamp(end)]
    if max_origins:
        selected = sorted(all_fits["origin"].unique())[:max_origins]
        all_fits = all_fits.loc[all_fits["origin"].isin(selected)]
    return all_fits.sort_values(["group", "origin", "fitter", "variant"]).reset_index(drop=True)


def evaluate(fits, steps_per_hour, fail_fast=False):
    results, hourly_rows, failed = [], [], []
    frames, baselines, window_cache = {}, {}, {}
    for group in GROUPS:
        if group in fits["group"].values:
            frames[group] = group_frame(group)
            baselines[group] = {}

    for count, row in enumerate(fits.itertuples(index=False), start=1):
        origin, group = pd.Timestamp(row.origin), row.group
        try:
            frame = frames[group]
            baseline = baseline_before(frame, origin, baselines[group])
            cache_key = (group, origin)
            if cache_key not in window_cache:
                window_cache.clear()
                window_cache[cache_key] = window_data(frame, origin, baseline, steps_per_hour)
            model, data, _, observed = window_cache[cache_key]
            oracle = optimized_qp_params(model).optimize_prediction_initial_state(
                data, np.array([row.r_F, row.r_U, row.r_R], dtype=float),
                f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0), r_0=float(row.r_0),
            )
            indices = np.arange(HOURS, dtype=int) * int(steps_per_hour)
            predicted = oracle["deterministic_state"]["I"][indices]
            baseline_I = np.asarray(data["O"][indices], dtype=float)
            score = metrics(observed, predicted)
            base = metrics(observed, baseline_I)
            init = oracle["init_cond"]
            results.append({
                "group": group, "origin": origin, "fitter": row.fitter, "variant": row.variant,
                "oracle_F_0": float(init[0]), "oracle_R_0": float(init[1]),
                "mse": score["mse"], "rmse": score["rmse"], "mae": score["mae"], "bias": score["bias"],
                "baseline_mse": base["mse"], "baseline_rmse": base["rmse"],
                "r_0": float(row.r_0), "r_F": float(row.r_F), "r_U": float(row.r_U), "r_R": float(row.r_R),
            })
            for h in range(HOURS):
                hourly_rows.append({
                    "group": group, "origin": origin, "fitter": row.fitter, "variant": row.variant,
                    "horizon": h + 1, "time": origin + pd.Timedelta(hours=h),
                    "observed_I": observed[h], "predicted_I": predicted[h],
                    "baseline_I": baseline_I[h], "error": predicted[h] - observed[h],
                })
            if count % 200 == 0: print(f"[oracle] evaluated {count}/{len(fits)} fitted windows")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            failed.append({"group": group, "origin": origin, "fitter": row.fitter, "variant": row.variant, "error": str(exc)})
            print(f"[oracle] {group} {origin} {row.fitter}/{row.variant} skipped: {exc}")
    if not results: raise RuntimeError("No fitted windows could be evaluated with an oracle initial state.")
    return pd.DataFrame(results), pd.DataFrame(hourly_rows), failed


def paired_primary(daily, hourly, qp_variant="zero"):
    a = daily.loc[(daily["fitter"] == "qp") & (daily["variant"] == qp_variant)]
    b = daily.loc[(daily["fitter"] == "numerical") & (daily["variant"] == "zero")]
    pairs = a.merge(b, on=["group", "origin"], suffixes=("_qp", "_numerical"))
    if pairs.empty: raise ValueError(f"No matching QP/{qp_variant} and numerical/zero windows.")
    pairs["mse_gain_numerical"] = pairs["mse_qp"] - pairs["mse_numerical"]
    pairs["rmse_gain_numerical"] = pairs["rmse_qp"] - pairs["rmse_numerical"]
    keys = pairs[["group", "origin"]].drop_duplicates()
    common = hourly.merge(keys, on=["group", "origin"], how="inner")
    return pairs, common


def summary_tables(daily, hourly):
    summary, horizon = [], []
    for (group, fitter, variant), scores in daily.groupby(["group", "fitter", "variant"]):
        samples = hourly.loc[
            (hourly["group"] == group) & (hourly["fitter"] == fitter) & (hourly["variant"] == variant)
        ]
        q = np.quantile(scores["rmse"], [0.2, 0.4, 0.5, 0.6, 0.8, 0.9])
        summary.append({
            "group": group, "fitter": fitter, "variant": variant,
            "windows": len(scores), "observations": len(samples),
            "aggregate_rmse": float(np.sqrt(np.mean(samples["error"].to_numpy(dtype=float) ** 2))),
            "mean_window_rmse": float(scores["rmse"].mean()), "median_window_rmse": float(q[2]),
            "q20": float(q[0]), "q40": float(q[1]), "q60": float(q[3]),
            "q80": float(q[4]), "q90": float(q[5]),
            "mean_mae": float(scores["mae"].mean()), "mean_bias": float(scores["bias"].mean()),
            "aggregate_baseline_rmse": float(np.sqrt(np.mean((samples["baseline_I"] - samples["observed_I"]) ** 2))),
        })
        for h, values in samples.groupby("horizon"):
            horizon.append({
                "group": group, "fitter": fitter, "variant": variant, "horizon": int(h),
                "rmse": float(np.sqrt(np.mean(values["error"] ** 2))), "mae": float(np.mean(np.abs(values["error"]))),
                "bias": float(np.mean(values["error"])), "samples": len(values),
            })
    return pd.DataFrame(summary), pd.DataFrame(horizon)


def save_plots(paired, common, summary, horizon, daily, hourly, output):
    for group in GROUPS:
        scores = paired.loc[paired["group"] == group].sort_values("origin")
        if scores.empty: continue
        group_summary = summary.loc[summary["group"] == group]
        subdir = output / group.lower(); subdir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(scores["origin"], scores["rmse_qp"], label="QP + oracle")
        ax.plot(scores["origin"], scores["rmse_numerical"], label="Numerical + oracle", alpha=0.75)
        ax.set_title(f"{group}: matched 24-hour oracle RMSE by prediction origin")
        ax.set_ylabel("24-hour RMSE"); ax.set_xlabel("Prediction origin")
        ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(subdir / "oracle_rmse_time.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([scores["rmse_qp"], scores["rmse_numerical"]], showfliers=False)
        ax.set_xticks([1, 2], ["QP", "Numerical"])
        ax.set_title(f"{group}: matched oracle RMSE distributions")
        ax.set_ylabel("24-hour RMSE"); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(subdir / "oracle_rmse_distribution.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(4)
        ax.plot(x, np.quantile(scores["rmse_qp"], [0.2, 0.4, 0.6, 0.8]), marker="o", label="QP")
        ax.plot(x, np.quantile(scores["rmse_numerical"], [0.2, 0.4, 0.6, 0.8]), marker="o", label="Numerical")
        ax.set_xticks(x, ["20%", "40%", "60%", "80%"])
        ax.set_ylabel("24-hour RMSE"); ax.set_title(f"{group}: matched window RMSE quintiles")
        ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(subdir / "oracle_rmse_quintiles.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        for fitter in ("qp", "numerical"):
            values = horizon.loc[(horizon["group"] == group) & (horizon["fitter"] == fitter) & (horizon["variant"] == "zero")]
            ax.plot(values["horizon"], values["rmse"], marker=".", label=fitter)
        ax.set_title(f"{group}: oracle RMSE across 24-hour forecast horizon")
        ax.set_xlabel("Forecast hour"); ax.set_ylabel("RMSE"); ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(subdir / "oracle_rmse_horizon.png", dpi=180); plt.close(fig)

        example = scores.iloc[len(scores) // 2]["origin"]
        samples = common.loc[(common["group"] == group) & (common["origin"] == example) & (common["variant"] == "zero")]
        fig, ax = plt.subplots(figsize=(12, 5))
        for fitter, subset in samples.groupby("fitter"):
            ax.plot(subset["time"], subset["predicted_I"], label=f"{fitter} oracle")
        reference = samples.loc[samples["fitter"] == "qp"]
        ax.scatter(reference["time"], reference["observed_I"], label="Observed I", s=15)
        ax.plot(reference["time"], reference["baseline_I"], linestyle="--", label="Baseline O")
        ax.set_title(f"{group}: representative oracle prediction {pd.Timestamp(example)}")
        ax.set_ylabel("Normalized electricity input"); ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(subdir / "oracle_example.png", dpi=180); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        for fitter in ("qp", "numerical"):
            values = daily.loc[(daily["group"] == group) & (daily["fitter"] == fitter) & (daily["variant"] == "zero")].sort_values("origin")
            axes[0].plot(values["origin"], values["oracle_F_0"], label=fitter)
            axes[1].plot(values["origin"], values["oracle_R_0"], label=fitter)
        axes[0].set_ylabel("Oracle F_0"); axes[1].set_ylabel("Oracle R_0")
        axes[1].set_xlabel("Prediction origin")
        for ax in axes: ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(subdir / "oracle_initial_states.png", dpi=180); plt.close(fig)

        group_summary.to_csv(subdir / "all_variant_summary.csv", index=False)


def main():
    parser = ArgumentParser(description="Compare oracle-state 24-hour WM4 top-end predictions using saved QP and numerical hourly moving fits.")
    parser.add_argument("--qp", default=str(OUTPUT / "qp" / "fit_parameters.csv"))
    parser.add_argument("--numerical", default=str(OUTPUT / "numerical" / "fit_parameters.csv"))
    parser.add_argument("--qp-variants", default=",".join(DEFAULT_QP_VARIANTS))
    parser.add_argument("--numerical-variants", default=",".join(DEFAULT_NUMERICAL_VARIANTS))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--output", default=str(OUTPUT / "oracle"))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    fits = load_fits(args.qp, args.numerical, args.qp_variants.split(","), args.numerical_variants.split(","),
                     start=args.start, end=args.end, max_origins=args.max_origins)
    if fits.empty: raise RuntimeError("No successful saved fitting windows found for the selected variants.")
    daily, hourly, failures = evaluate(fits, args.steps_per_hour, args.fail_fast)
    summary, horizon = summary_tables(daily, hourly)
    paired, common = paired_primary(daily, hourly, qp_variant="zero")
    fixed_paired, fixed_common = paired_primary(daily, hourly, qp_variant="zero_fixed")
    common_summary, common_horizon = summary_tables(
        daily.merge(paired[["group", "origin"]].drop_duplicates(), on=["group", "origin"], how="inner"),
        common,
    )
    daily.to_csv(output / "oracle_window_metrics.csv", index=False)
    hourly.to_csv(output / "oracle_hourly_predictions.csv", index=False)
    summary.to_csv(output / "oracle_summary.csv", index=False)
    horizon.to_csv(output / "oracle_horizon.csv", index=False)
    paired.to_csv(output / "paired_qp_numerical.csv", index=False)
    common_summary.to_csv(output / "matched_oracle_summary.csv", index=False)
    common_horizon.to_csv(output / "matched_oracle_horizon.csv", index=False)
    fixed_paired.to_csv(output / "paired_fixed_qp_numerical.csv", index=False)
    fixed_summary, fixed_horizon = summary_tables(
        daily.merge(fixed_paired[["group", "origin"]].drop_duplicates(), on=["group", "origin"], how="inner"),
        fixed_common,
    )
    fixed_summary.to_csv(output / "matched_fixed_oracle_summary.csv", index=False)
    fixed_horizon.to_csv(output / "matched_fixed_oracle_horizon.csv", index=False)
    save_table(failures, output / "skipped.csv")
    save_plots(paired, common, common_summary, common_horizon, daily, common, output)
    print(f"Oracle comparison: {output}; evaluated windows={len(daily)}, skipped={len(failures)}")
    print("Matched-origin sequential-QP versus numerical oracle (aggregate RMSE is pooled):")
    print(common_summary.loc[(common_summary["variant"] == "zero")].to_string(index=False))
    print("Matched-origin fixed-state-QP versus numerical oracle (same historical initial state):")
    print(fixed_summary.loc[(fixed_summary["variant"].isin(("zero_fixed", "zero")))].to_string(index=False))


if __name__ == "__main__": main()

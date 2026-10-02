from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t


ROOT = Path(__file__).resolve().parents[4]
PREDICTION = ROOT / "files" / "output" / "wm_4" / "prediction"
WARMUP = PREDICTION / "all_days_top_end"
OPTIMIZED = PREDICTION / "all_days_optimized_top_end"
BRUTE_FORCE = PREDICTION / "all_days_brute_force_top_end"
OUTPUT = PREDICTION / "r0_null_analysis"
METHODS = ("warmup", "optimized", "brute_force")


def load_r0(name, root):
    path = Path(root) / "daily_metrics.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing {path}. Run the {name} top-end predictor first.")
    frame = pd.read_csv(path, parse_dates=["day"])
    if "R_0" not in frame.columns: raise RuntimeError(f"{path} does not contain R_0.")
    frame = frame[["day", "R_0"]].dropna().sort_values("day").drop_duplicates("day", keep="last").reset_index(drop=True)
    if len(frame) < 3: raise RuntimeError(f"{name} requires at least 3 R_0 observations.")
    frame["method"] = name
    return frame


def align_common_days(frames):
    common = set(frames[METHODS[0]]["day"])
    for method in METHODS[1:]: common &= set(frames[method]["day"])
    if len(common) < 3: raise RuntimeError("The top-end fitters have fewer than 3 common R_0 prediction days.")
    days = sorted(common)
    return {method: frame.loc[frame["day"].isin(days)].sort_values("day").reset_index(drop=True) for method, frame in frames.items()}


def hac_standard_error(values, lag):
    x = np.asarray(values, dtype=float)
    n = x.size
    lag = min(max(0, int(lag)), n - 1)
    centered = x - np.mean(x)
    gamma0 = float(centered @ centered) / n
    long_run_variance = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma = float(centered[k:] @ centered[:-k]) / n
        long_run_variance += 2.0 * weight * gamma
    long_run_variance = max(0.0, long_run_variance)
    return float(np.sqrt(long_run_variance / n)), lag


def tost(mean, se, df, margin):
    if not np.isfinite(se) or se <= 0.0: return np.nan, np.nan, np.nan, False
    lower_t = (mean + margin) / se
    upper_t = (mean - margin) / se
    p_lower = float(t.sf(lower_t, df))
    p_upper = float(t.cdf(upper_t, df))
    return float(lower_t), float(upper_t), max(p_lower, p_upper), None


def analyze_method(name, frame, alpha, hac_lag, equivalence_margin=None):
    values = frame["R_0"].to_numpy(dtype=float)
    n = values.size
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    median = float(np.median(values))
    sem = std / np.sqrt(n)
    df = n - 1
    critical_t = float(t.ppf(1.0 - alpha / 2.0, df))
    ci_low, ci_high = mean - critical_t * sem, mean + critical_t * sem
    t_stat = mean / sem if sem > 0.0 else np.nan
    t_p = float(2.0 * t.sf(abs(t_stat), df)) if np.isfinite(t_stat) else np.nan

    hac_se, used_lag = hac_standard_error(values, hac_lag)
    hac_z = mean / hac_se if hac_se > 0.0 else np.nan
    hac_p = float(2.0 * norm.sf(abs(hac_z))) if np.isfinite(hac_z) else np.nan
    critical_z = float(norm.ppf(1.0 - alpha / 2.0))
    hac_ci_low, hac_ci_high = mean - critical_z * hac_se, mean + critical_z * hac_se
    cohen_d = mean / std if std > 0.0 else np.nan

    row = {
        "method": name, "n": n, "R0_mean": mean, "R0_std": std, "R0_median": median,
        "R0_min": float(np.min(values)), "R0_max": float(np.max(values)),
        "alpha": alpha,
        "iid_sem": sem, "iid_t_stat": t_stat, "iid_p_value": t_p,
        "iid_ci_low": ci_low, "iid_ci_high": ci_high,
        "iid_reject_H0_mean_zero": bool(t_p < alpha) if np.isfinite(t_p) else False,
        "hac_lag": used_lag, "hac_sem": hac_se, "hac_z_stat": hac_z, "hac_p_value": hac_p,
        "hac_ci_low": hac_ci_low, "hac_ci_high": hac_ci_high,
        "hac_reject_H0_mean_zero": bool(hac_p < alpha) if np.isfinite(hac_p) else False,
        "cohen_d": cohen_d,
    }

    if equivalence_margin is not None:
        margin = float(equivalence_margin)
        if not np.isfinite(margin) or margin <= 0.0: raise ValueError("equivalence_margin must be a positive finite value.")
        lower_t, upper_t, tost_p, _ = tost(mean, sem, df, margin)
        hac_lower_z = (mean + margin) / hac_se if hac_se > 0.0 else np.nan
        hac_upper_z = (mean - margin) / hac_se if hac_se > 0.0 else np.nan
        hac_p_lower = float(norm.sf(hac_lower_z)) if np.isfinite(hac_lower_z) else np.nan
        hac_p_upper = float(norm.cdf(hac_upper_z)) if np.isfinite(hac_upper_z) else np.nan
        hac_tost_p = max(hac_p_lower, hac_p_upper) if np.isfinite(hac_p_lower) and np.isfinite(hac_p_upper) else np.nan
        row.update({
            "equivalence_margin": margin,
            "iid_tost_lower_stat": lower_t, "iid_tost_upper_stat": upper_t, "iid_tost_p_value": tost_p,
            "iid_equivalent_to_zero": bool(tost_p < alpha) if np.isfinite(tost_p) else False,
            "hac_tost_lower_z": hac_lower_z, "hac_tost_upper_z": hac_upper_z, "hac_tost_p_value": hac_tost_p,
            "hac_equivalent_to_zero": bool(hac_tost_p < alpha) if np.isfinite(hac_tost_p) else False,
        })
    return row


def save_plots(frames, summary, output):
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    for method, frame in frames.items(): ax.plot(frame["day"], frame["R_0"], label=method)
    ax.axhline(0.0, linestyle="--", label="H0: R0 mean = 0")
    ax.set_xlabel("Prediction day"); ax.set_ylabel("Oracle R0")
    ax.set_title("WM4 oracle prediction-state R0"); ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "r0_time_series.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(summary))
    mean = summary["R0_mean"].to_numpy(dtype=float)
    low = summary["hac_ci_low"].to_numpy(dtype=float)
    high = summary["hac_ci_high"].to_numpy(dtype=float)
    ax.errorbar(x, mean, yerr=np.vstack((mean - low, high - mean)), fmt="o", capsize=5)
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(x, summary["method"]); ax.set_ylabel("R0 mean with HAC confidence interval")
    ax.set_title("H0: E[R0] = 0"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "r0_mean_hac_confidence_intervals.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(summary, alpha, equivalence_margin):
    print("H0: E[R0] = 0")
    print(f"alpha={alpha:.4f}; HAC/Newey-West test is the primary test because R0 is a daily time series.")
    print()
    columns = ["method", "n", "R0_mean", "R0_std", "iid_t_stat", "iid_p_value", "hac_z_stat", "hac_p_value", "hac_ci_low", "hac_ci_high", "hac_reject_H0_mean_zero"]
    print(summary[columns].to_string(index=False))
    if equivalence_margin is not None:
        print()
        print(f"equivalence test: H1 is |E[R0]| < {equivalence_margin:g}")
        columns = ["method", "iid_tost_p_value", "iid_equivalent_to_zero", "hac_tost_p_value", "hac_equivalent_to_zero"]
        print(summary[columns].to_string(index=False))


def main():
    parser = ArgumentParser(description="Test the WM4 oracle initial-rate null hypothesis H0: E[R0] = 0.")
    parser.add_argument("--warmup", default=str(WARMUP))
    parser.add_argument("--optimized", default=str(OPTIMIZED))
    parser.add_argument("--brute-force", default=str(BRUTE_FORCE))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--hac-lag", type=int, default=7, help="Newey-West lag for serial dependence; default 7 days.")
    parser.add_argument("--equivalence-margin", type=float, default=None, help="Optional practical-equivalence half-width around R0=0 for TOST.")
    args = parser.parse_args()

    if not 0.0 < args.alpha < 1.0: raise ValueError("alpha must lie in (0, 1).")
    if args.hac_lag < 0: raise ValueError("hac_lag must be non-negative.")

    frames = align_common_days({
        "warmup": load_r0("warmup", args.warmup),
        "optimized": load_r0("optimized", args.optimized),
        "brute_force": load_r0("brute_force", args.brute_force),
    })
    summary = pd.DataFrame([
        analyze_method(method, frames[method], args.alpha, args.hac_lag, args.equivalence_margin)
        for method in METHODS
    ])
    values = pd.concat([frame.assign(method=method) for method, frame in frames.items()], ignore_index=True)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "r0_null_summary.csv", index=False)
    values.to_csv(output / "r0_values.csv", index=False)
    save_plots(frames, summary, output)
    print_summary(summary, args.alpha, args.equivalence_margin)


if __name__ == "__main__": main()

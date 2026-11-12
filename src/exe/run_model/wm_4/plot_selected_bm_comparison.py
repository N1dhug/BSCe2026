"""Compare the saved BM1 beta-mixture and WM4 ARM 14-day prediction samples.

No parameter estimation, oracle optimization, or model simulation is performed.
Requires hourly_predictions.csv from both selected_bm_dates_* executables.

From repository root:
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week 1 --show
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week 2
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week both
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.selected_bm_dates_arm import K, reference_dates


ROOT = Path(__file__).resolve().parents[4]
BM_INPUT = ROOT/"files"/"output"/"bm_1"/"selected_bm_dates_beta"/"hourly_predictions.csv"
WM_INPUT = ROOT/"files"/"output"/"wm_4"/"selected_bm_dates_arm"/"hourly_predictions.csv"
OUTPUT = ROOT/"files"/"output"/"model_comparison"/"selected_bm_dates"


def load_predictions(path, model, mean_col, lower_col="lower_95", upper_col="upper_95"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{model} saved predictions not found: {path}\n"
                                "Run the matching selected_bm_dates_* estimation executable first.")
    frame = pd.read_csv(path)
    needed = {"day", "hour", "observed", mean_col, lower_col, upper_col}
    missing = needed.difference(frame.columns)
    if missing: raise ValueError(f"{model}: missing saved CSV columns: {sorted(missing)}")
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"], errors="raise").dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="raise").astype(int)
    if "baseline" in frame:
        frame = frame.loc[frame["baseline"]=="weekly_temperature"].copy()
    if frame.duplicated(["day", "hour"]).any():
        raise ValueError(f"{model}: duplicate (day,hour) entries in {path}.")
    for col in ("observed", mean_col, lower_col, upper_col):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{model}: nonfinite {col} values.")
    if (frame[lower_col]>frame[upper_col]).any():
        raise ValueError(f"{model}: reversed 95% uncertainty bounds.")
    return frame.set_index(["day", "hour"]).sort_index()


def comparison(bm_path=BM_INPUT, wm_path=WM_INPUT, week=1):
    if week not in (1, 2): raise ValueError("week must be 1 or 2.")
    dates, _ = reference_dates()
    selected = dates[(week-1)*7:week*7]
    bm = load_predictions(bm_path, "BM1", "beta_expected_demand")
    wm = load_predictions(wm_path, "WM4", "predicted")
    keys = pd.MultiIndex.from_product([selected, range(K)], names=["day", "hour"])
    missing_bm = keys.difference(bm.index)
    missing_wm = keys.difference(wm.index)
    if len(missing_bm) or len(missing_wm):
        raise ValueError(f"Saved predictions are missing requested week {week} hourly rows: "
                         f"BM1={list(missing_bm[:5])}, WM4={list(missing_wm[:5])}.")
    b, w = bm.loc[keys], wm.loc[keys]
    if not np.allclose(b["observed"], w["observed"], atol=1e-8, rtol=1e-8):
        difference = float(np.max(np.abs(b["observed"]-w["observed"])))
        raise ValueError(f"BM1 and WM4 observed demands differ (max {difference:.6g}). "
                         "Check the sample dates and normalization.")
    return selected, b, w


def plot_week(week=1, bm_path=BM_INPUT, wm_path=WM_INPUT, output=OUTPUT, show=False):
    dates, bm, wm = comparison(bm_path, wm_path, week)
    hours = np.arange(K)
    fig, axes = plt.subplots(7, 1, figsize=(13, 20), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, day in zip(axes, dates):
        b, w = bm.loc[day].sort_index(), wm.loc[day].sort_index()
        ax.fill_between(hours, b["lower_95"].to_numpy(), b["upper_95"].to_numpy(),
                        color="tab:orange", alpha=0.16)
        ax.fill_between(hours, w["lower_95"].to_numpy(), w["upper_95"].to_numpy(),
                        color="tab:blue", alpha=0.13)
        ax.plot(hours, b["beta_expected_demand"].to_numpy(), color="tab:orange", lw=1.9)
        ax.plot(hours, w["predicted"].to_numpy(), color="tab:blue", lw=1.9)
        ax.plot(hours, b["observed"].to_numpy(), color="black", lw=1.6,
                marker="o", markersize=2.8)
        ax.set_title(f"{day:%A, %d %B %Y}", loc="left", fontsize=11)
        ax.set_ylabel("Normalized demand")
        ax.grid(alpha=0.23)
        ax.set_xlim(0, K-1)
    axes[-1].set_xticks([0, 4, 8, 12, 16, 20, 23])
    axes[-1].set_xlabel("Hour of day")
    legend = [plt.Line2D([], [], color="black", marker="o", markersize=3,
                         label="Observed demand"),
              plt.Line2D([], [], color="tab:orange", lw=2, label="BM1 mean"),
              Patch(facecolor="tab:orange", alpha=0.16, label="BM1 central 95%"),
              plt.Line2D([], [], color="tab:blue", lw=2, label="WM4 mean"),
              Patch(facecolor="tab:blue", alpha=0.13, label="WM4 Gaussian 95%")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5, fontsize=10)
    fig.suptitle(f"BM1 vs WM4 | Monday–Sunday | sample week {week}\n"
                 "ALL · weekly temperature · N=6 historical weekdays · 24-hour oracle-conditioned predictions",
                 fontsize=13)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output/f"bm1_wm4_week_{week}_95pct.png"
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    observed = bm["observed"].to_numpy(dtype=float)
    bm_rmse = float(np.sqrt(np.mean((bm["beta_expected_demand"].to_numpy()-observed)**2)))
    wm_rmse = float(np.sqrt(np.mean((wm["predicted"].to_numpy()-observed)**2)))
    bm_coverage = float(np.mean((observed>=bm["lower_95"].to_numpy())&
                                (observed<=bm["upper_95"].to_numpy()))*100.)
    wm_coverage = float(np.mean((observed>=wm["lower_95"].to_numpy())&
                                (observed<=wm["upper_95"].to_numpy()))*100.)
    print(f"Week {week}: BM1 RMSE={bm_rmse:.6f}, WM4 RMSE={wm_rmse:.6f}; "
          f"BM1 95% coverage={bm_coverage:.2f}%, WM4 95% coverage={wm_coverage:.2f}%")
    print(f"Saved: {destination}")
    return destination


def main():
    parser = ArgumentParser(description="Plot seven weekday comparisons using saved BM1 and WM4 hourly predictions.")
    parser.add_argument("--week", choices=("1", "2", "both"), default="1",
                        help="First or second Monday–Sunday sample from the 14 fitted dates.")
    parser.add_argument("--bm-input", type=Path, default=BM_INPUT)
    parser.add_argument("--wm-input", type=Path, default=WM_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for week in (1, 2) if args.week=="both" else (int(args.week),):
        plot_week(week, args.bm_input, args.wm_input, args.output, args.show)


if __name__=="__main__": main()

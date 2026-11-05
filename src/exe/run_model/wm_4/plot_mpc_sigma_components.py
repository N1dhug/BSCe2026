"""Decompose initial-state and Wiener variance in six-hour WM4 MPC uncertainty quintiles.

Reuses the matched 7..28-day diffusion sweep and the original MPC forecast
archive. Neither the deterministic WM4 model nor diffusion QPs are refitted.
Each window creates its OWN ranked Q1..Q5 origins, as in the original quintile
sweep. All windows are evaluated on the same matched MPC origin cohort.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_mpc_sigma_days import (
    HORIZON, OUTPUT as SWEEP_OUTPUT, SOURCE, read_mpc_archive,
)


OUTPUT = SWEEP_OUTPUT / "variance_components"
INITIAL_TERMS = ("initial_F_component_variance",
    "initial_R_component_variance", "initial_FR_component_variance")


def load_matched_fits(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No matched diffusion sweep at {path}. Run sweep_mpc_sigma_days first.")
    data = pd.read_csv(path, parse_dates=["origin"])
    cols = {"origin", "history_days", "diffusion_q",
        *(f"variance_h{h}" for h in range(1, HORIZON + 1))}
    if missing := cols - set(data):
        raise ValueError(f"Matched sweep missing columns: {sorted(missing)}.")
    if data.duplicated(["origin", "history_days"]).any():
        raise ValueError("Duplicate diffusion fit origin/window pairs.")
    if (data["diffusion_q"] < 0).any():
        raise ValueError("A fitted diffusion variance coefficient cannot be negative.")
    if not np.isfinite(data[list(cols - {"origin"})].to_numpy(dtype=float)).all():
        raise ValueError("Matched diffusion fits contain nonfinite fields.")
    count = data.groupby("history_days")["origin"].nunique()
    if count.nunique() != 1:
        raise ValueError("Diffusion fits must have the same number of MPC origins per window.")
    expected_origins = set(data.loc[data["history_days"] == data["history_days"].min(), "origin"])
    for _, part in data.groupby("history_days"):
        if set(part["origin"]) != expected_origins:
            raise ValueError("Diffusion fitting windows must use identical matched MPC origins.")
    return data


def _initial_details(frame, archive):
    """Optional FF, RR and signed FR variance contributions already in the MPC CSV."""
    if not set(INITIAL_TERMS) <= set(frame):
        return None
    detail = {}
    wanted = np.arange(1, HORIZON + 1)
    for origin, group in frame.groupby("origin", sort=True):
        if origin not in archive: continue
        group = group.sort_values("lead_hours")
        if not np.array_equal(group["lead_hours"].to_numpy(), wanted): continue
        values = group[list(INITIAL_TERMS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite initial covariance contribution at {origin}.")
        if not np.allclose(values.sum(axis=1), archive[origin][:, 1],
            atol=1e-8, rtol=1e-6):
            raise ValueError(f"F0/R0/cross contributions do not reproduce initial variance at {origin}.")
        detail[origin] = values
    return detail


def summarize_components(fits, archive, quantiles=5, initial_terms=None):
    """One point per matched MPC origin/window, plus quintile mean decomposition.

    Each quantile is ranked by average TOTAL predicted variance across h1..h6.
    Fractions are ratios of mean variance contributions, not mean percentages.
    The FR term may be negative; it must not be clipped or stacked.
    """
    quantiles = int(quantiles)
    if quantiles < 2: raise ValueError("quantiles must be >= 2.")
    origin_rows, quantile_rows = [], []
    variance_cols = [f"variance_h{h}" for h in range(1, HORIZON + 1)]
    for days, part in fits.groupby("history_days", sort=True):
        part = part.sort_values("origin").reset_index(drop=True)
        if len(part) < quantiles: raise ValueError("Insufficient origins for requested quantiles.")
        origins = part["origin"].tolist()
        if missing := set(origins) - set(archive):
            raise ValueError(f"Missing {len(missing)} MPC origins from forecast archive.")
        errors = np.stack([archive[o][:, 0] for o in origins])
        initial = np.stack([archive[o][:, 1] for o in origins])
        kernel = np.stack([archive[o][:, 2] for o in origins])
        wiener = part["diffusion_q"].to_numpy(dtype=float)[:, None] * kernel
        predicted = part[variance_cols].to_numpy(dtype=float)
        if not np.allclose(predicted, initial + wiener, atol=1e-9, rtol=1e-6):
            raise ValueError(f"Diffusion sweep variance decomposition failed for {days} days.")
        qrank = pd.Series(np.mean(predicted, axis=1)).rank(method="first")
        labels = pd.qcut(qrank, quantiles, labels=False).astype(int).to_numpy() + 1
        mse = np.mean(errors ** 2, axis=1)
        v0, vw, vt = initial.mean(axis=1), wiener.mean(axis=1), predicted.mean(axis=1)
        fields = {"initial_F_component_variance": None, "initial_R_component_variance": None,
            "initial_FR_component_variance": None}
        if initial_terms is not None:
            if missing := set(origins) - set(initial_terms):
                raise ValueError(f"Missing initial-state covariance terms for {len(missing)} MPC origins.")
            values = np.stack([initial_terms[o] for o in origins]).mean(axis=1)
            for j, field in enumerate(INITIAL_TERMS): fields[field] = values[:, j]
        for i, origin in enumerate(origins):
            row = {"origin": origin, "history_days": int(days),
                "uncertainty_quantile": int(labels[i]), "observed_mse": float(mse[i]),
                "initial_variance": float(v0[i]), "wiener_variance": float(vw[i]),
                "predicted_variance": float(vt[i]), "diffusion_q": float(part.iloc[i]["diffusion_q"])}
            if initial_terms is not None:
                row.update({col: float(values[i, j]) for j, col in enumerate(INITIAL_TERMS)})
            origin_rows.append(row)
        for quintile in range(1, quantiles + 1):
            keep = labels == quintile
            real, a, b, total = (float(np.mean(x[keep])) for x in (mse, v0, vw, vt))
            row = {"history_days": int(days), "uncertainty_quantile": int(quintile),
                "n_origins": int(np.count_nonzero(keep)),
                "observed_mse": real, "predicted_variance": total,
                "initial_variance": a, "wiener_variance": b,
                "initial_share_pct": 100.0 * a / total if total > 0 else np.nan,
                "wiener_share_pct": 100.0 * b / total if total > 0 else np.nan,
                "initial_to_observed_mse": a / real if real > 0 else np.nan,
                "wiener_to_observed_mse": b / real if real > 0 else np.nan,
                "predicted_to_observed_mse": total / real if real > 0 else np.nan,
                "mean_sigma": float(np.mean(np.sqrt(part.loc[keep, "diffusion_q"])))}
            if initial_terms is not None:
                row.update({col: float(np.mean(values[keep, j]))
                    for j, col in enumerate(INITIAL_TERMS)})
            quantile_rows.append(row)
    return pd.DataFrame(origin_rows), pd.DataFrame(quantile_rows)


def plot_components(details, summary, output, reference_days=24):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    days = np.sort(summary["history_days"].unique())
    if reference_days not in days:
        raise ValueError(f"Reference days {reference_days} unavailable; choose from {days.tolist()}.")
    reference = summary.loc[summary["history_days"] == reference_days].sort_values("uncertainty_quantile")
    q = reference["uncertainty_quantile"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(q, reference["initial_variance"], label="Initial-state P0")
    axes[0].bar(q, reference["wiener_variance"], bottom=reference["initial_variance"],
        label="Wiener diffusion")
    axes[0].plot(q, reference["observed_mse"], "ko--", label="Observed six-hour MSE")
    axes[0].set(xlabel="Predicted-uncertainty quantile", ylabel="Six-hour mean variance / MSE",
        title=f"{reference_days}-day fit: predicted components vs realized error")
    axes[0].legend()
    axes[1].bar(q, reference["initial_share_pct"], label="Initial-state P0")
    axes[1].bar(q, reference["wiener_share_pct"], bottom=reference["initial_share_pct"],
        label="Wiener diffusion")
    axes[1].set(xlabel="Predicted-uncertainty quantile", ylabel="% of predicted variance",
        title=f"{reference_days}-day fit: attribution of total predicted variance", ylim=(0, 103))
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(q)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output / "reference_variance_components_by_quantile.png", dpi=170,
        bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    specs = [
        ("initial_share_pct", "Initial-state share of predicted variance (%)", None),
        ("initial_to_observed_mse", "Initial-state variance / observed MSE", 1.),
        ("wiener_to_observed_mse", "Wiener variance / observed MSE", 1.),
    ]
    for ax, (column, ylabel, benchmark) in zip(axes, specs):
        for quantile, part in summary.groupby("uncertainty_quantile", sort=True):
            part = part.sort_values("history_days")
            ax.plot(part["history_days"], part[column], "o-", label=f"Q{quantile}")
        if benchmark is not None: ax.axhline(benchmark, color="black", ls="--", alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(ncol=5)
    axes[-1].set_xlabel("Historical diffusion fitting days")
    fig.suptitle("ARM (Affine Revised Model) | initial-state vs Wiener attribution by uncertainty quantile")
    fig.tight_layout()
    fig.savefig(output / "variance_components_across_backlogs.png", dpi=170,
        bbox_inches="tight")
    plt.close(fig)

    # One point per MPC origin, not one point per overlapping forecast lead.
    # The same axis limits permit comparisons across the entire history sweep.
    nrows = int(np.ceil(len(days) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(18, 3.8 * nrows), squeeze=False)
    upper = float(max(details["predicted_variance"].max(), details["observed_mse"].max()) * 1.05)
    last = None
    for ax, n in zip(axes.flat, days):
        subset = details.loc[details["history_days"] == n]
        last = ax.scatter(subset["predicted_variance"], subset["observed_mse"],
            c=100. * subset["initial_variance"] / np.maximum(subset["predicted_variance"], 1e-12),
            s=10, cmap="viridis", vmin=0, vmax=100, alpha=0.7)
        ax.plot([0, upper], [0, upper], "k--", lw=0.8)
        ax.set(xlim=(0, upper), ylim=(0, upper), title=f"{n}-day diffusion window")
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(days):]: ax.axis("off")
    fig.supxlabel("Predicted six-hour mean variance")
    fig.supylabel("Observed six-hour mean squared error")
    fig.suptitle("ARM (Affine Revised Model) | same MPC origins in every panel; color is initial-state variance share")
    fig.subplots_adjust(right=0.9, top=0.95, bottom=0.07, hspace=0.38, wspace=0.28)
    bar = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.55, pad=0.02)
    bar.set_label("Initial-state share of predicted variance (%)")
    fig.savefig(output / "variance_component_scatter_all_backlogs.png", dpi=155,
        bbox_inches="tight")
    plt.close(fig)

    if set(INITIAL_TERMS) <= set(reference):
        fig, ax = plt.subplots(figsize=(9, 5))
        offsets = [-0.26, 0., 0.26]
        for col, offset, label in zip(INITIAL_TERMS, offsets,
            ("F0 marginal variance", "R0 marginal variance", "F0-R0 covariance term")):
            ax.bar(q + offset, reference[col], width=0.24, label=label)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(xlabel="Predicted-uncertainty quantile",
            ylabel="Contribution to six-hour mean predicted variance",
            title=f"{reference_days}-day fit: initial-state covariance terms")
        ax.set_xticks(q)
        ax.legend()
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(output / "reference_initial_covariance_terms.png", dpi=170,
            bbox_inches="tight")
        plt.close(fig)
    return output


def analyze(fits=SWEEP_OUTPUT / "matched_fits.csv", forecasts=SOURCE,
    output=OUTPUT, quantiles=5, reference_days=24):
    matched = load_matched_fits(fits)
    archive, frame = read_mpc_archive(forecasts, horizon=HORIZON, fit_hours=24)
    components = _initial_details(frame, archive)
    detail, table = summarize_components(matched, archive, quantiles=quantiles,
        initial_terms=components)
    output = plot_components(detail, table, output, reference_days=reference_days)
    detail.to_csv(output / "origin_variance_components.csv", index=False)
    table.to_csv(output / "variance_components_by_quantile.csv", index=False)
    shown = table.loc[table["history_days"] == reference_days]
    print(f"Matched MPC origins: {detail['origin'].nunique()} | diffusion windows: "
        f"{table['history_days'].nunique()} | six-hour forecast leads")
    print(shown[["uncertainty_quantile", "n_origins", "observed_mse",
        "initial_variance", "wiener_variance", "predicted_variance",
        "initial_share_pct", "predicted_to_observed_mse"]].to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Saved P0/Wiener attribution CSVs and plots at {output}")
    return {"origins": detail, "quantiles": table, "output": output}


def main():
    parser = ArgumentParser(description="Decompose P0 and Wiener variance in every quintile of the 7..28 day diffusion sweep; no WM4 refits.")
    parser.add_argument("--fits", type=Path, default=SWEEP_OUTPUT / "matched_fits.csv")
    parser.add_argument("--forecasts", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--reference-days", type=int, default=24)
    args = parser.parse_args()
    return analyze(args.fits, args.forecasts, args.output, args.quantiles,
        args.reference_days)


if __name__ == "__main__": main()

"""WM4 matched MPC variance ablation: remove initial-state covariance propagation.

One saved 24h-parameter WM4 MPC archive and one already computed rolling
diffusion sweep suffice: deterministic forecasts, realized errors, origin
cohort and process kernels are identical across variants. No WM4 rerun.

Variants:
 - full_p0: original V_h = B_h + q K_h.
 - no_p0_fixed_q: remove B_h from predictions, retain the ORIGINAL q.
 - no_p0_refit_q: remove B_h from predictions AND from the historical
   six-hour sigma QP. This is the consistent no-initial-covariance model.

Each variant is scored both on fixed FULL-P0 quintiles (same exact origins
in each bin) and on its OWN predicted-variance quintiles (conditional
calibration as in the previous plots).
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_mpc_sigma_components import load_matched_fits
from src.exe.run_model.wm_4.sweep_mpc_sigma_days import (
    HORIZON, OUTPUT as SWEEP_OUTPUT, SOURCE, fit_window_variances, read_mpc_archive,
)


OUTPUT = SWEEP_OUTPUT / "p0_ablation"
VARIANTS = ("full_p0", "no_p0_fixed_q", "no_p0_refit_q")
VARIANT_LABELS = {
    "full_p0": "Full P0 (original)",
    "no_p0_fixed_q": "No P0 (same sigma)",
    "no_p0_refit_q": "No P0 (refit sigma)",
}


def _rank_quantiles(values, quantiles):
    """Equal-number quintiles using timestamp-sorted origin tie breaks."""
    ranking = pd.Series(values).rank(method="first")
    return (pd.qcut(ranking, int(quantiles), labels=False).astype(int)
        .to_numpy() + 1)


def _no_p0_fit_q(archive, origin, days):
    """Recompute causal six-lead Wiener QP with zero historical initial variance."""
    history = []
    for offset in range(1, int(days) + 1):
        prev = origin - pd.Timedelta(days=offset)
        data = archive.get(prev)
        if data is None or prev + pd.Timedelta(hours=HORIZON) > origin:
            raise ValueError(f"Incomplete {days}-day causal MPC archive for origin {origin}.")
        history.append(data)
    values = np.asarray(history, dtype=float).copy()
    values[:, :, 1] = 0.
    _, q, _, _, _ = fit_window_variances(values, min_days=days, max_days=days)
    return float(q[0])


def ablate(fits, archive, history_days=24, quantiles=5):
    """Build origin- and lead-level matched ablations for one historical window."""
    history_days, quantiles = int(history_days), int(quantiles)
    subset = fits.loc[fits["history_days"] == history_days].sort_values("origin").reset_index(drop=True)
    if len(subset) < quantiles:
        raise ValueError(f"Need {quantiles} complete matched origins for {history_days}d; got {len(subset)}.")
    if subset["origin"].duplicated().any():
        raise ValueError("Each matched MPC origin must occur once for the selected history_days.")
    origins = subset["origin"].tolist()
    if not set(origins) <= set(archive):
        raise ValueError("Some matched MPC origins are missing from the forecast archive.")

    errors = np.stack([archive[o][:, 0] for o in origins])
    initial = np.stack([archive[o][:, 1] for o in origins])
    kernels = np.stack([archive[o][:, 2] for o in origins])
    q_full = subset["diffusion_q"].to_numpy(dtype=float)
    q_refit = np.asarray([_no_p0_fit_q(archive, origin, history_days)
        for origin in origins], dtype=float)
    full = subset[[f"variance_h{h}" for h in range(1, HORIZON + 1)]].to_numpy(dtype=float)
    fixed = q_full[:, None] * kernels
    refit = q_refit[:, None] * kernels
    if not np.allclose(full, initial + fixed, atol=1e-9, rtol=1e-6):
        raise ValueError("Original diffusion-sweep variance is not B + q K.")
    if np.any(refit < -1e-12) or np.any(fixed < -1e-12):
        raise ValueError("Negative propagated Wiener variance.")
    predictions = (full, fixed, refit)
    qs = (q_full, q_full, q_refit)
    reference_labels = _rank_quantiles(full.mean(axis=1), quantiles)
    origin_rows, lead_rows = [], []
    for name, predicted, q in zip(VARIANTS, predictions, qs):
        own_labels = _rank_quantiles(predicted.mean(axis=1), quantiles)
        for i, origin in enumerate(origins):
            origin_rows.append({"origin": origin, "variant": name, "history_days": history_days,
                "original_quantile": int(reference_labels[i]),
                "own_quantile": int(own_labels[i]), "diffusion_q": float(q[i]),
                "diffusion_sigma": float(np.sqrt(q[i])),
                "observed_mse": float(np.mean(errors[i] ** 2)),
                "predicted_variance": float(np.mean(predicted[i])),
                "original_initial_variance": float(np.mean(initial[i])),
                "wiener_variance": float(np.mean(q[i] * kernels[i]))})
            for lead in range(HORIZON):
                lead_rows.append({"origin": origin, "variant": name,
                    "history_days": history_days, "lead_hours": lead + 1,
                    "original_quantile": int(reference_labels[i]),
                    "own_quantile": int(own_labels[i]),
                    "forecast_error": float(errors[i, lead]),
                    "observed_squared_error": float(errors[i, lead] ** 2),
                    "predicted_variance": float(predicted[i, lead]),
                    "original_initial_variance": float(initial[i, lead]),
                    "wiener_variance": float(q[i] * kernels[i, lead])})
    return pd.DataFrame(origin_rows), pd.DataFrame(lead_rows)


def _metrics(rows):
    e = rows["forecast_error"].to_numpy(dtype=float)
    v = rows["predicted_variance"].to_numpy(dtype=float)
    v_safe = np.maximum(v, 1e-12)
    mse, mean_v = float(np.mean(e ** 2)), float(np.mean(v))
    z = np.sqrt(np.maximum(v, 0.))
    return {"n_origins": int(rows["origin"].nunique()), "n_predictions": int(len(e)),
        "observed_mse": mse, "predicted_variance": mean_v,
        "observed_rmse": float(np.sqrt(mse)),
        "predicted_rms_std": float(np.sqrt(mean_v)),
        "predicted_to_observed_mse": mean_v / mse if mse > 0 else np.nan,
        "coverage_68_pct": 100. * float(np.mean(np.abs(e) <= z)),
        "coverage_95_pct": 100. * float(np.mean(np.abs(e) <= 1.96 * z)),
        "gaussian_nll": 0.5 * float(np.mean(
            np.log(2. * np.pi * v_safe) + e ** 2 / v_safe)),
        "initial_variance": float(np.mean(rows["original_initial_variance"])),
        "wiener_variance": float(np.mean(rows["wiener_variance"]))}


def summarize(lead_data):
    """Score aggregate, exact leads, and quintiles with fixed and reranked cohorts."""
    rows = []
    for name, part in lead_data.groupby("variant", sort=False):
        rows.append({"variant": name, "binning": "overall", "uncertainty_quantile": 0,
            "lead_hours": 0, **_metrics(part)})
        for h, data in part.groupby("lead_hours", sort=True):
            rows.append({"variant": name, "binning": "by_lead", "uncertainty_quantile": 0,
                "lead_hours": int(h), **_metrics(data)})
        for mode, field in (("fixed", "original_quantile"), ("own", "own_quantile")):
            for q, data in part.groupby(field, sort=True):
                rows.append({"variant": name, "binning": mode,
                    "uncertainty_quantile": int(q), "lead_hours": 0, **_metrics(data)})
                for lead, group in data.groupby("lead_hours", sort=True):
                    rows.append({"variant": name, "binning": f"{mode}_lead",
                        "uncertainty_quantile": int(q),
                        "lead_hours": int(lead), **_metrics(group)})
    return pd.DataFrame(rows)


def plot_comparison(origins, metrics, output, history_days):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, mode in zip(axes[0], ("fixed", "own")):
        ref = metrics.loc[(metrics["variant"] == "full_p0")
            & (metrics["binning"] == mode)].sort_values("uncertainty_quantile")
        ax.plot(ref["uncertainty_quantile"], ref["observed_mse"], "ko--",
            label="Observed MSE" if mode == "fixed" else "Observed MSE (original bins only)")
        for name in VARIANTS:
            frame = metrics.loc[(metrics["variant"] == name)
                & (metrics["binning"] == mode)].sort_values("uncertainty_quantile")
            ax.plot(frame["uncertainty_quantile"], frame["predicted_variance"], "o-",
                label=VARIANT_LABELS[name])
            if mode == "own":
                # Each variant's quintiles contain different origins.
                ax.plot(frame["uncertainty_quantile"], frame["observed_mse"],
                    ":", alpha=0.7)
        ax.set(title=f"{mode.capitalize()} Q1–Q5: mean predicted variance vs realized MSE",
            xlabel="Predicted uncertainty quantile", ylabel="Mean six-hour variance / MSE")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax, (column, target) in zip(axes[1],
        (("predicted_to_observed_mse", 1.), ("coverage_95_pct", 95.))):
        ax.axhline(target, color="black", ls="--")
        for name in VARIANTS:
            frame = metrics.loc[(metrics["variant"] == name)
                & (metrics["binning"] == "own")].sort_values("uncertainty_quantile")
            ax.plot(frame["uncertainty_quantile"], frame[column], "o-",
                label=VARIANT_LABELS[name])
        ax.set(xlabel="Variant-specific predicted-uncertainty quantile",
            ylabel="Predicted / observed MSE" if column.startswith("predicted_")
                else "95% Gaussian interval coverage (%)",
            title="Conditional variance calibration" if column.startswith("predicted_")
                else "Conditional interval coverage")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"WM4 | {history_days}-day / 6h Wiener QP | P0 propagation ablation\n"
        f"{origins['origin'].nunique()} matched MPC origins; unchanged deterministic forecasts")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "p0_ablation_quantiles.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    high = 1.05 * float(max(origins["predicted_variance"].max(), origins["observed_mse"].max()))
    for ax, name in zip(axes, VARIANTS):
        part = origins.loc[origins["variant"] == name]
        ax.scatter(part["predicted_variance"], part["observed_mse"], s=13, alpha=0.55)
        ax.plot([0, high], [0, high], "k--", lw=1)
        ax.set(xlim=(0, high), ylim=(0, high), title=VARIANT_LABELS[name],
            xlabel="Predicted six-hour mean variance")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Realized six-hour mean squared error")
    fig.suptitle("WM4 | matched-origin variance-vs-realized-error scatter")
    fig.tight_layout()
    fig.savefig(output / "p0_ablation_scatter.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def analyze(fits=SWEEP_OUTPUT / "matched_fits.csv", forecasts=SOURCE, output=OUTPUT,
    history_days=24, quantiles=5):
    matched = load_matched_fits(fits)
    archive, _ = read_mpc_archive(forecasts, horizon=HORIZON, fit_hours=24)
    origins, forecasts_by_lead = ablate(matched, archive, history_days, quantiles)
    score = summarize(forecasts_by_lead)
    output = plot_comparison(origins, score, output, history_days)
    origins.to_csv(output / "p0_ablation_origins.csv", index=False)
    forecasts_by_lead.to_csv(output / "p0_ablation_forecasts.csv", index=False)
    score.to_csv(output / "p0_ablation_scores.csv", index=False)
    overall = score.loc[score["binning"] == "overall"]
    own = score.loc[score["binning"] == "own"]
    print(f"WM4 P0 ablation: {history_days}-day Wiener fit, "
        f"{origins['origin'].nunique()} identical six-hour MPC origins")
    print(overall[["variant", "predicted_to_observed_mse", "coverage_68_pct",
        "coverage_95_pct", "gaussian_nll"]].to_string(index=False,
        float_format=lambda x: f"{x:.5f}"))
    print("Variance calibration by own uncertainty quintile:")
    print(own[["variant", "uncertainty_quantile", "n_origins",
        "predicted_to_observed_mse", "coverage_95_pct"]].to_string(index=False,
        float_format=lambda x: f"{x:.4f}"))
    print(f"Saved paired quintiles, scatter, per-lead scores and CSVs at {output}")
    return {"origins": origins, "forecasts": forecasts_by_lead, "scores": score,
        "output": output}


def main():
    parser = ArgumentParser(description="Ablate all P0 propagation with unchanged or refitted six-hour Wiener coefficient; compare matched quintiles.")
    parser.add_argument("--fits", type=Path, default=SWEEP_OUTPUT / "matched_fits.csv")
    parser.add_argument("--forecasts", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--history-days", type=int, default=24)
    parser.add_argument("--quantiles", type=int, default=5)
    args = parser.parse_args()
    return analyze(args.fits, args.forecasts, args.output, args.history_days,
        args.quantiles)


if __name__ == "__main__": main()

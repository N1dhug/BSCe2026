"""One-panel BM1 beta-mixture demand plot across all historical and oracle windows.

Use saved results without refitting:
    python -m src.exe.run_model.bm_1.plot_beta_moving_horizon --N 5 --k 24 --origin "2022-08-22 06:00" --tag ALL
or:
    python -m src.exe.run_model.bm_1.plot_beta_moving_horizon --run-dir files/output/bm_1/methodology/20220822_0600_N5_k24_ALL

The x-axis concatenates separate weekly windows; it is NOT continuous calendar
time. Historical curves are unconditional/open-loop beta-mixture propagation
from each historically fitted X0; the final forecast is conditioned on the
oracle-fitted target X0. Shaded bands include process and observation noise.

Older historical_fit.csv exports lack uncertainty columns. In that case, the
script reconstructs only the historical beta-mixture distribution using saved
parameters (no parameter refitting), checks the old means, and caches the
reconstructed bounds for subsequent plotting.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.fit_beta_moving_horizon import OUTPUT, BASELINES
from src.method.bm_1_beta import bm_1_beta
from src.validation.exe.run_model.bm_1.fit_beta_moving_horizon import bm_1_beta_moving_horizon_validation as checks


BANDS = ("lower_68", "upper_68", "lower_95", "upper_95")
COLORS = {"weekly_average":"#2878b5", "weekly_temperature":"#d47726"}


def _read_csv(path, columns):
    if not path.is_file(): raise FileNotFoundError(f"Missing BM1 output: {path}")
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing: raise ValueError(f"{path} is missing columns {sorted(missing)}.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].isna().any() or frame["timestamp"].dt.tz is not None:
        raise ValueError(f"Invalid timestamps in {path}.")
    frame["lead_hour"] = pd.to_numeric(frame["lead_hour"], errors="raise").astype(int)
    return frame


def _load(root, name):
    directory = root / name
    result_path = directory / "result.json"
    if not result_path.is_file(): raise FileNotFoundError(f"No completed BM1 fit for {name}: {result_path}")
    with result_path.open("r", encoding="utf-8") as handle: report = json.load(handle)
    history = _read_csv(directory / "historical_fit.csv",
                        ("historical_origin", "timestamp", "lead_hour", "observed", "baseline", "beta_mean"))
    forecast = _read_csv(directory / "oracle_prediction.csv",
                         ("timestamp", "lead_hour", "observed", "baseline", "beta_mean", *BANDS))
    history["historical_origin"] = pd.to_datetime(history["historical_origin"], errors="raise")
    history = history.sort_values(["historical_origin", "lead_hour"]).reset_index(drop=True)
    forecast = forecast.sort_values("lead_hour").reset_index(drop=True)
    N, k = int(report["N"]), int(report["k"])
    dates = pd.DatetimeIndex(pd.to_datetime(report["historical_origins"]))
    origin = checks.origin(report["origin"])
    if N < 1 or k < 2 or len(dates) != N or len(history) != N*k or len(forecast) != k:
        raise ValueError(f"Inconsistent historical/prediction lengths for {name}.")
    if list(pd.unique(history["historical_origin"])) != list(dates):
        raise ValueError(f"Historical origins differ from the saved fit for {name}.")
    for i, start in enumerate((*dates, origin)):
        window = history.iloc[i*k:(i+1)*k] if i < N else forecast
        expected_time = pd.date_range(start, periods=k, freq="h")
        if not np.array_equal(window["lead_hour"].to_numpy(), np.arange(k)) or not np.array_equal(
                window["timestamp"].to_numpy(dtype="datetime64[ns]"), expected_time.to_numpy()):
            raise ValueError(f"Historical or prediction hours are incomplete or out of order for {name}.")
    for field in ("observed", "baseline", "beta_mean"):
        if not np.isfinite(pd.to_numeric(history[field], errors="coerce")).all() or not np.isfinite(
                pd.to_numeric(forecast[field], errors="coerce")).all():
            raise ValueError(f"Nonfinite {field} in saved data for {name}.")
    return {"name":name, "report":report, "history":history, "forecast":forecast,
            "N":N, "k":k, "origin":origin, "dates":dates, "directory":directory}


def _have_bands(frame):
    if any(key not in frame.columns for key in BANDS): return False
    try: values = frame[list(BANDS)].to_numpy(dtype=float)
    except (ValueError, TypeError): return False
    if not np.isfinite(values).all(): return False
    lower_68, upper_68, lower_95, upper_95 = values.T
    if np.any(lower_68 > upper_68) or np.any(lower_95 > upper_95):
        raise ValueError("Saved uncertainty interval has lower bound above upper bound.")
    if np.any(lower_95 > lower_68 + 1e-9) or np.any(upper_95 < upper_68 - 1e-9):
        raise ValueError("Saved 68% and 95% uncertainty bounds are not nested.")
    return True


def _historical_prices(group, *, root, tag, cache):
    """Older historical CSVs contain no price U: recover without refitting BM1."""
    if "price" in group["history"].columns:
        u = pd.to_numeric(group["history"]["price"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(u).all() and np.all((u >= 0.0) & (u <= 1.0)): return u
    if "frame" not in cache:
        from src.exe.run_model.wm_3.fit_weekly import aligned_data
        from src.exe.run_model.bm_1.fit_beta_moving_horizon import rescale_past_only
        stamp = group["origin"]
        print("[BM1 beta plot] Older historical export: recovering prices from input workbook.", flush=True)
        raw, _ = aligned_data(None if tag == "ALL" else tag, price_domain=(0.0, 1.0), price_year=None)
        cache["frame"], _ = rescale_past_only(checks.frame(raw), stamp)
    times = pd.DatetimeIndex(group["history"]["timestamp"])
    recovered = cache["frame"].reindex(times)
    if recovered[["I", "U"]].isna().any().any():
        raise ValueError("Cannot recover historical demand/price to plot old BM1 results.")
    saved_y = group["history"]["observed"].to_numpy(dtype=float)
    if not np.allclose(recovered["I"].to_numpy(dtype=float), saved_y, atol=1e-6, rtol=0.0):
        raise ValueError("Input workbook no longer matches saved normalized historical observations.")
    return recovered["U"].to_numpy(dtype=float)


def _restore_historical_bands(group, *, root, tag, cache):
    history = group["history"]
    if _have_bands(history): return
    print(f"[BM1 beta plot/{group['name']}] Reconstructing missing historical 68%/95% bands "
          "(reusing saved parameters, no MLE).", flush=True)
    started = perf_counter()
    params = dict(group["report"]["stochastic_parameters"])
    states = np.asarray(group["report"]["historical_stochastic_states"], dtype=float)
    if states.shape != (group["N"],): raise ValueError("Missing historical fitted initial states.")
    sigma_y = float(params["sigma_y"])
    # Historical open propagation was saved at the same resolution used by
    # the oracle-prediction solver (24, 12, 48, 0.125).
    resolution = group["report"].get("resolutions", {}).get("prediction", {})
    solver = bm_1_beta(components=int(resolution.get("components", 24)),
        quadrature=int(resolution.get("quadrature", 12)),
        reconstruction_quadrature=int(resolution.get("reconstruction_quadrature", 48)),
        max_step=float(resolution.get("max_step", 0.125)))
    prices = _historical_prices(group, root=root, tag=tag, cache=cache)
    for field in BANDS: history[field] = np.nan
    history["price"] = prices
    k = group["k"]
    for i, state in enumerate(states):
        selection = slice(i*k, (i+1)*k)
        block = history.iloc[selection]
        signals = {"B":block["baseline"].to_numpy(dtype=float),
                   "U":block["price"].to_numpy(dtype=float)}
        result = solver.simulate(signals, params, float(state))
        old_mean = block["beta_mean"].to_numpy(dtype=float)
        max_deviation = float(np.max(np.abs(result["mean"] - old_mean)))
        if max_deviation > 2e-4:
            raise ValueError(f"Reconstructed historical mean differs from saved prediction "
                             f"for {group['name']}, interval {i+1}: {max_deviation:.6g}. "
                             "Check the original numerical method version before plotting bands.")
        bounds = solver.intervals(result, sigma_y)
        for field in BANDS: history.loc[history.index[selection], field] = bounds[field]
    if not _have_bands(history): raise RuntimeError("Historical uncertainty reconstruction was incomplete.")
    # Cache numerical output so replotting does not invoke the stochastic solver.
    history.to_csv(group["directory"] / "historical_fit.csv", index=False)
    print(f"[BM1 beta plot/{group['name']}] Historical bands reconstructed and cached "
          f"in {perf_counter()-started:.2f}s.", flush=True)


def _equal_observed(first, other):
    for section in ("history", "forecast"):
        a, b = first[section], other[section]
        if not np.array_equal(a["timestamp"].to_numpy(), b["timestamp"].to_numpy()) or not np.allclose(
                a["observed"].to_numpy(dtype=float), b["observed"].to_numpy(dtype=float),
                rtol=0.0, atol=1e-8):
            raise ValueError(f"The two baseline results have different observed {section} data.")


def plot_saved(*, run_dir, baseline="both", tag=None, show=False, dpi=190, reconstruct=True):
    """Draw exactly one Axes containing N concatenated historical windows and one oracle window."""
    root = Path(run_dir)
    if not root.is_dir(): raise FileNotFoundError(f"BM1 methodology output folder not found: {root}")
    if baseline not in (*BASELINES, "both"): raise ValueError(f"Unknown baseline: {baseline}")
    names = list(BASELINES) if baseline == "both" else [baseline]
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 40: raise ValueError("dpi must be an integer >=40.")
    summary_path = root / "run_summary.json"
    summary = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle: summary = json.load(handle)
    if tag is None: tag = summary.get("tag", root.name.rsplit("_", 1)[-1])
    if tag not in ("ALL", "MED", "UDEN"): raise ValueError("Specify --tag ALL, MED, or UDEN for historical uncertainty reconstruction.")
    groups = [_load(root, name) for name in names]
    reference = groups[0]
    for group in groups[1:]:
        if group["N"] != reference["N"] or group["k"] != reference["k"] or group["origin"] != reference["origin"]:
            raise ValueError("Baselines must share exactly the same origin, N and k.")
        _equal_observed(reference, group)
    cache = {}
    for group in groups:
        if not _have_bands(group["history"]):
            if not reconstruct:
                raise ValueError(f"{group['name']} historical_fit.csv lacks uncertainty bands; omit --no-reconstruct.")
            _restore_historical_bands(group, root=root, tag=tag, cache=cache)
        if not _have_bands(group["forecast"]):
            raise ValueError(f"{group['name']} oracle_prediction.csv lacks uncertainty bands.")
    N, k = reference["N"], reference["k"]
    gap = max(3, int(np.ceil(k / 10)))
    stride = k + gap
    fig, ax = plt.subplots(figsize=(max(15.0, (N+1)*2.75), 6.8))
    handles = [Line2D([0], [0], color="#171717", linewidth=2.1, marker="o",
                      markersize=3.0, label="Observed demand")]
    for name in names:
        handles.append(Line2D([0], [0], color=COLORS[name], linewidth=2.0,
                              label=name.replace("_", " ").title() + " — beta mean"))
    handles.extend([Patch(facecolor="0.5", alpha=0.19, label="Central 68% demand interval"),
                    Patch(facecolor="0.5", alpha=0.08, label="Central 95% demand interval")])
    ax.axvspan(N*stride-gap/2, N*stride+k-0.5, color="#c6d7c6", alpha=0.17, zorder=0)
    for j in range(N+1):
        x = j*stride + np.arange(k)
        observed = (reference["history"].iloc[j*k:(j+1)*k]["observed"].to_numpy(dtype=float)
                    if j < N else reference["forecast"]["observed"].to_numpy(dtype=float))
        for group in groups:
            frame = group["history"].iloc[j*k:(j+1)*k] if j < N else group["forecast"]
            color = COLORS[group["name"]]
            ax.fill_between(x, frame["lower_95"].to_numpy(dtype=float),
                            frame["upper_95"].to_numpy(dtype=float),
                            color=color, alpha=0.09, linewidth=0, zorder=1)
            ax.fill_between(x, frame["lower_68"].to_numpy(dtype=float),
                            frame["upper_68"].to_numpy(dtype=float),
                            color=color, alpha=0.22, linewidth=0, zorder=2)
            ax.plot(x, frame["beta_mean"].to_numpy(dtype=float),
                    color=color, linewidth=1.9,
                    linestyle="--" if j == N else "-", zorder=4)
        ax.plot(x, observed, color="#171717", linewidth=1.9, marker="o",
                markersize=2.4, zorder=5)
        if j > 0: ax.axvline(j*stride-gap/2, color="0.7", linewidth=0.85, linestyle=":", zorder=0)
    ticks = [i*stride+(k-1)/2 for i in range(N+1)]
    labels = [f"Historical {i+1}\n{date:%d %b %Y %H:%M}" for i, date in enumerate(reference["dates"])]
    labels.append(f"Oracle prediction\n{reference['origin']:%d %b %Y %H:%M}")
    ax.set_xticks(ticks, labels)
    ax.set_xlim(-0.8, N*stride+k-0.2)
    ax.set_xlabel("Fitting / prediction interval (hours concatenated; weekly gaps omitted)")
    ax.set_ylabel("Normalized aggregate demand")
    ax.set_title(f"BM1 beta-mixture: {N} historical fits and one oracle prediction | k={k} h | {tag}")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=min(len(handles), 3), framealpha=0.94, fontsize=9)
    fig.tight_layout()
    filename = "demand_fit_all_intervals.png" if baseline == "both" else f"demand_fit_all_intervals_{baseline}.png"
    destination = root / filename
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    print(f"[BM1 beta plot] Single-panel figure: {destination}", flush=True)
    return {"plot":destination, "N":N, "k":k, "origin":reference["origin"], "baselines":names,
            "historical_windows":N, "prediction_windows":1, "axes":1}


def main():
    parser = ArgumentParser(description="Plot all saved BM1 historical fits and the oracle prediction in ONE figure.")
    parser.add_argument("--run-dir", type=Path, help="Existing methodology output directory; bypass N/k/origin folder calculation.")
    parser.add_argument("--N", type=int, default=5, help="Historical weekly backlog when --run-dir is omitted.")
    parser.add_argument("--k", type=int, default=24, help="Historical/prediction hours when --run-dir is omitted.")
    parser.add_argument("--origin", default="2022-08-22 06:00", help="Prediction origin, date and hour.")
    parser.add_argument("--tag", choices=("ALL", "MED", "UDEN"), default=None)
    parser.add_argument("--baseline", choices=("both", *BASELINES), default="both")
    parser.add_argument("--no-reconstruct", action="store_true", help="Fail if old historical CSV lacks uncertainty bands.")
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    origin = checks.origin(args.origin)
    tag = args.tag if args.tag is not None else "ALL"
    run_dir = args.run_dir if args.run_dir is not None else OUTPUT / f"{origin:%Y%m%d_%H%M}_N{args.N}_k{args.k}_{tag}"
    return plot_saved(run_dir=run_dir, baseline=args.baseline,
                      tag=args.tag, show=args.show, dpi=args.dpi,
                      reconstruct=not args.no_reconstruct)


if __name__ == "__main__": main()

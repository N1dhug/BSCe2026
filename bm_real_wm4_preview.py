#!/usr/bin/env python3
"""BM-only real-data F4 preview. Never imports or runs WM4 dynamics.

Run from BSCe2026: python -B path/to/bm_real_wm4_preview.py --start auto
The default uses MED row means and original DK1 prices from the same workbooks
as WM4. It creates no synthetic measurements. See README.md for qualifications.
This exploratory coarse FV bank is not certified F4-H/I/J estimation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve_banded
from scipy.special import betainc, logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REFERENCE_COMMIT = "9415d8c881e52e0a024cd8effa904a9e44dcabf5"
KNOTS = np.linspace(0., 1., 5)
CURVES = (
    (1., .5, 0., -.5, -1.),
    (1., .8, .4, -.2, -1.),
    (1., .2, -.4, -.8, -1.),
)


@dataclass(frozen=True)
class Parameters:
    rho: float
    Xi: float
    sigma_x: float
    f_values: tuple
    g_values: tuple
    k: float = 2.


def demand(x, u, b, p):
    f = np.interp(x, KNOTS, p.f_values)
    g = np.interp(u, KNOTS, p.g_values)
    delta = np.tanh(.5 * p.k * (f + g))
    return b + p.Xi * delta * np.where(delta >= 0., 1. - b, b)


class FiniteVolume:
    """Conservative upwind/product-diffusion FV, reflecting zero boundary flux."""
    def __init__(self, cells, steps_per_hour):
        self.edges = np.linspace(0., 1., cells + 1)
        self.x = .5 * (self.edges[:-1] + self.edges[1:])
        self.dx = 1. / cells
        self.h = 1. / (24. * steps_per_hour)  # Model time: days.
        self.initial = np.diff(betainc(20., 20., self.edges))

    def rates(self, u, b, p):
        a = p.rho * (demand(self.edges[1:-1], u, b, p) - b)
        diffusion = .5 * (p.sigma_x * self.x * (1. - self.x)) ** 2
        right = np.maximum(a, 0.) / self.dx + diffusion[:-1] / self.dx ** 2
        left = np.maximum(-a, 0.) / self.dx + diffusion[1:] / self.dx ** 2
        return right, left

    def matrix(self, rates):
        right, left = rates
        ab = np.zeros((3, len(self.x)))
        ab[1] = 1.
        ab[1, :-1] += self.h * right
        ab[1, 1:] += self.h * left
        ab[0, 1:] = -self.h * left
        ab[2, :-1] = -self.h * right
        return ab

    @staticmethod
    def generator(m, rates):
        right, left = rates
        flux = right * m[:-1] - left * m[1:]
        out = np.zeros_like(m)
        out[:-1] -= flux
        out[1:] += flux
        return out

    @staticmethod
    def step(matrix, m):
        return solve_banded((1, 1), matrix, m, check_finite=False)


def probability_check(m):
    error = float(np.max(np.abs(m.sum(axis=-1) - 1.)))
    smallest = float(m.min())
    if not np.all(np.isfinite(m)) or error > 1e-9 or smallest < -1e-13:
        raise FloatingPointError(f"Invalid probability: mass error={error}, min={smallest}")
    return error


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_real_data(repo, input_path, price_path, tag):
    # The existing consumption importer exposes original readings separately
    # from its imputed frame. Only original observed_data is used here.
    sys.path.insert(0, str(repo))
    from src.tools.import_data.import_input import input_data
    source = input_data(input_path)
    columns = [c for c in source.measurement_columns
               if source.measurement_tag(c).upper() == tag.upper()]
    if not columns:
        raise ValueError(f"No measurement columns with tag {tag!r}.")
    values = source.observed_data.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    I = values.mean(axis=1).rename("I_raw").sort_index()  # Exactly WM4's MED aggregate.
    counts = values.notna().sum(axis=1).rename("available_MED_columns")
    if I.index.has_duplicates or I.index.tz is not None:
        raise ValueError("Consumption timestamps must be unique local wall times, as in WM4.")
    if not (I.index == I.index.floor("h")).all():
        raise ValueError("Consumption timestamps are not on the hourly WM4 grid.")

    # Read ORIGINAL prices, bypassing the project importer's future interpolation.
    # On complete January/February hours this is exactly WM4's DK1 DKK series.
    data = pd.read_excel(price_path, engine="openpyxl")
    data.columns = data.columns.astype(str).str.strip()
    data = data.loc[data["PriceArea"].astype(str).str.upper().eq("DK1")].copy()
    stamps = pd.to_datetime(data["HourUTC"], utc=True, errors="raise")
    index = pd.DatetimeIndex(stamps).tz_convert("Europe/Copenhagen").tz_localize(None)
    U = pd.Series(pd.to_numeric(data["SpotPriceDKK"], errors="coerce").to_numpy(),
                  index=index, name="U_raw").groupby(level=0).mean().sort_index()
    if I.dropna().empty or U.dropna().empty:
        raise ValueError("Real consumption or DK1 prices are empty.")
    return I, U, counts, columns


def hourly_index(start, hours):
    return pd.date_range(start, periods=hours, freq="h")


def require_complete(series, index, name):
    out = series.reindex(index)
    bad = ~np.isfinite(out.to_numpy(dtype=float))
    if bad.any():
        raise ValueError(f"Missing original {name} at {list(index[bad][:3])}; no imputation.")
    return out.to_numpy(dtype=float)


def five_week_baseline(history, targets, origin):
    """Five previous calendar weekday/hour observations, all strictly pre-origin."""
    sources, result = [], []
    for stamp in targets:
        dates = [stamp - pd.Timedelta(weeks=w) for w in range(1, 6)]
        if any(t >= origin for t in dates):
            raise ValueError("Baseline attempted to use an observation at/after forecast origin.")
        result.append(float(np.mean(require_complete(history, pd.DatetimeIndex(dates), "baseline history"))))
        sources.append([t.isoformat() for t in dates])
    return np.array(result), sources


def make_windows(I, U, start, fit_days, warmup_hours):
    # This preview rejects DST-transition windows rather than hiding time shifts.
    train_hours = fit_days * 24 + warmup_hours
    train_index = hourly_index(start - pd.Timedelta(hours=train_hours), train_hours)
    future_index = hourly_index(start, 168)
    all_index = train_index.append(future_index)
    aware = all_index.tz_localize("Europe/Copenhagen", ambiguous="raise", nonexistent="raise")
    if not np.all(np.diff(aware.asi8) == 3600 * 10**9):
        raise ValueError("This exploratory preview supports weeks with 168 elapsed hours; DST window rejected.")
    history = I.loc[I.index < start]  # Enforced before constructing either baseline.
    train_y = require_complete(history, train_index, "training consumption")
    # Training baseline is causal at each reading; forecasting baseline is frozen at start.
    train_b, train_sources = [], []
    for t in train_index:
        b, sources = five_week_baseline(history, pd.DatetimeIndex([t]), t)
        train_b.append(b[0]); train_sources.extend(sources)
    future_b, future_sources = five_week_baseline(history, future_index, start)
    train_u = require_complete(U, train_index, "training DK1 price")
    future_u = require_complete(U, future_index, "target DK1 price")
    # Scoring availability is checked; these readings NEVER reach calibration/propagation.
    require_complete(I, future_index, "target consumption for scoring")
    return dict(start=start, train_index=train_index, future_index=future_index,
                train_y_raw=train_y, train_b_raw=np.array(train_b), train_u_raw=train_u,
                future_b_raw=future_b, future_u_raw=future_u,
                train_baseline_sources=train_sources, future_baseline_sources=future_sources)


def choose_week(I, U, args):
    if args.start != "auto":
        start = pd.Timestamp(args.start)
        if start.tzinfo is not None or start != start.normalize() or start.weekday() != 0:
            raise ValueError("--start must be a Monday date YYYY-MM-DD (forecast week, not fitting week).")
        return make_windows(I, U, start, args.fit_days, args.warmup_hours), []
    earliest = max(I.dropna().index.min(), U.dropna().index.min(), pd.Timestamp("2022-01-01"))
    earliest += pd.Timedelta(weeks=5, days=args.fit_days, hours=args.warmup_hours)
    candidate = earliest.normalize()
    if candidate < earliest: candidate += pd.Timedelta(days=1)
    candidate += pd.Timedelta(days=(-candidate.weekday()) % 7)
    last = min(I.dropna().index.max() - pd.Timedelta(hours=167),
               U.dropna().index.max() - pd.Timedelta(hours=167),
               pd.Timestamp(args.search_until) - pd.Timedelta(days=7))
    skipped = []
    while candidate <= last:
        try:
            window = make_windows(I, U, candidate, args.fit_days, args.warmup_hours)
            return window, skipped
        except ValueError as exc:
            skipped.append({"start": str(candidate), "reason": str(exc)})
        candidate += pd.Timedelta(weeks=1)
    raise ValueError(f"No complete eligible week before {args.search_until}. "
                     f"First candidate={earliest}; skipped={skipped[:2]}. "
                     "Add original history or explicitly extend --search-until.")


def freeze_scales(I, U, start, scale_path):
    if scale_path:
        info = json.loads(Path(scale_path).read_text(encoding="utf-8"))
        keys = ("consumption_min", "consumption_max", "price_min", "price_max")
        scales = {key: float(info[key]) for key in keys}
        source = "explicit frozen scales supplied by caller; no range estimated in preview"
    else:
        past_i = I.loc[(I.index >= pd.Timestamp("2022-01-01")) & (I.index < start)].dropna()
        past_u = U.loc[(U.index >= pd.Timestamp("2022-01-01")) & (U.index < start)].dropna()
        scales = dict(consumption_min=float(past_i.min()), consumption_max=float(past_i.max()),
                      price_min=min(0., float(past_u.min())), price_max=max(0., float(past_u.max())))
        source = "Jan 1 to forecast origin, exclusive; training only; price range includes zero by fixed rule"
    for low, high in (("consumption_min", "consumption_max"), ("price_min", "price_max")):
        if not np.isfinite([scales[low], scales[high]]).all() or scales[high] <= scales[low]:
            raise ValueError(f"Invalid normalization: {low}/{high}.")
    return scales, source


def scale(values, scales, kind):
    return (np.asarray(values) - scales[kind + "_min"]) / (scales[kind + "_max"] - scales[kind + "_min"])


def parameter_bank(fg_mode):
    curves = CURVES if fg_mode == "piecewise" else CURVES[:1]
    return [Parameters(float(r), float(q), float(s), f, g)
            for r in (.5, 1., 2.) for q in (0., .5, .9) for s in (.2, .6)
            for f in curves for g in curves]


def calibrate(fv, y, b, u, parameters, sigma_y, steps, warmup):
    masses = np.tile(fv.initial, (len(parameters), 1))
    log_weights = np.full(len(parameters), -np.log(len(parameters)))
    history, max_error = [], 0.
    begin = perf_counter()
    for i, (yi, bi, ui) in enumerate(zip(y, b, u)):
        # Readings at hh:00, right-continuous hourly price. Assimilate at timestamp,
        # then propagate to the next timestamp using only this hour's inputs.
        d = np.array([demand(fv.x, ui, bi, p) for p in parameters])
        if i >= warmup:
            log_m = np.full_like(masses, -np.inf)
            positive = masses > 0.
            log_m[positive] = np.log(masses[positive])
            log_like = -.5 * ((yi - d) / sigma_y) ** 2 - np.log(sigma_y * np.sqrt(2. * np.pi))
            scores = logsumexp(log_m + log_like, axis=1)
            if not np.isfinite(scores).all(): raise FloatingPointError("Observation likelihood underflow.")
            masses = np.exp(log_m + log_like - scores[:, None])
            log_weights += scores
            log_weights -= logsumexp(log_weights)
        history.append(np.exp(log_weights))
        matrices = [fv.matrix(fv.rates(ui, bi, p)) for p in parameters]
        for _ in range(steps):
            masses = np.array([fv.step(ab, m) for ab, m in zip(matrices, masses)])
        max_error = max(max_error, probability_check(masses))
        if (i + 1) % 24 == 0:
            elapsed = perf_counter() - begin
            remaining = elapsed * (len(y) - i - 1) / (i + 1)
            print(f"BM calibration {i+1}/{len(y)} hours; elapsed {elapsed:.1f}s; "
                  f"ETA {remaining:.1f}s; estimated calibration total {elapsed+remaining:.1f}s", flush=True)
    return masses, np.exp(log_weights), np.array(history), max_error


def quantiles(values, weights):
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    return values[order[np.minimum(np.searchsorted(cdf, (.05, .95)), len(order)-1)]]


def forecast(fv, initial, weights, b, u, parameters, steps):
    """No observation argument by design. Forecast weights are frozen."""
    masses = initial.copy()
    records, marginal, max_error = [], [], 0.
    begin = perf_counter()
    for hour, (bi, ui) in enumerate(zip(b, u)):
        d = np.array([demand(fv.x, ui, bi, p) for p in parameters])
        rates = [fv.rates(ui, bi, p) for p in parameters]
        matrices = [fv.matrix(r) for r in rates]
        for substep in range(steps):
            joint = weights[:, None] * masses
            mean_d = float(np.sum(joint * d))
            mean_x = float(np.sum(joint * fv.x))
            qlo, qhi = quantiles(d.ravel(), joint.ravel())
            q_m = np.array([fv.generator(m, r) for m, r in zip(masses, rates)])
            r_dot = float(np.sum(weights[:, None] * q_m * d))
            records.append((hour + substep/steps, mean_d, qlo, qhi, mean_x, mean_d - bi, r_dot, bi, ui))
            marginal.append(joint.sum(axis=0))
            masses = np.array([fv.step(ab, m) for ab, m in zip(matrices, masses)])
            max_error = max(max_error, probability_check(masses))
        if (hour + 1) % 24 == 0:
            elapsed = perf_counter() - begin
            remaining = elapsed * (len(b) - hour - 1) / (hour + 1)
            print(f"BM forecast {hour+1}/{len(b)} hours; elapsed {elapsed:.1f}s; "
                  f"ETA {remaining:.1f}s; estimated forecast total {elapsed+remaining:.1f}s", flush=True)
    frame = pd.DataFrame(records, columns=("elapsed_hours", "predicted_I", "latent_q05", "latent_q95",
                                          "X", "R", "R_dot", "baseline_O", "U"))
    return frame, np.array(marginal), masses, max_error


def deviation_metrics(observed, predicted):
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return dict(mse=float(np.mean(residual**2)), rmse=float(np.sqrt(np.mean(residual**2))),
                mae=float(np.mean(np.abs(residual))), bias=float(np.mean(residual)))


def score_hourly(hourly):
    records = []
    periods = [("whole_week", hourly)] + [(str(day.date()), group) for day, group in hourly.groupby(hourly.index.normalize())]
    for period, frame in periods:
        for units, observed, predictions in (
            ("normalized", "observed_I", (("BM", "predicted_I"), ("baseline", "baseline_O"))),
            ("original_workbook_units", "observed_I_raw", (("BM", "predicted_I_raw"), ("baseline", "baseline_O_raw"))),
        ):
            for model, predicted in predictions:
                records.append(dict(period=period, model=model, units=units, n=len(frame),
                                    **deviation_metrics(frame[observed], frame[predicted])))
    return pd.DataFrame(records)


def save_plots(out, fine, hourly, fv, marginal, parameters, weights, start):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(hourly.index, hourly.observed_I, label="Observed hourly MED input I", color="black", lw=1.)
    axes[0].scatter(hourly.index, hourly.observed_I, s=8, color="black")
    axes[0].plot(fine.index, fine.predicted_I, label="BM prediction E[D(X,U,B)]")
    axes[0].fill_between(fine.index, fine.latent_q05, fine.latent_q95, alpha=.15, label="Exploratory 90% latent band")
    axes[0].plot(fine.index, fine.baseline_O, label="B: previous five weekday/hour mean", alpha=.8)
    axes[0].plot(fine.index, fine.U, label="DK1 price, signed WM4-style coordinate", alpha=.7)
    axes[0].set_ylabel("Normalized input / price")
    axes[1].plot(fine.index, fine.U, label="Known DK1 price (retrospective input)")
    axes[1].plot(fine.index, fine.R_dot, label="BM within-hour flexibility change dE[D-B]/dt")
    axes[1].set_ylabel("Price / change per day")
    axes[2].plot(fine.index, fine.X, label="BM charge state E[X], range [0,1]")
    axes[2].plot(fine.index, fine.R, label="BM flexibility E[D]-B")
    axes[2].set_ylabel("BM state / flexibility")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    axes[-1].set_xlabel("Europe/Copenhagen local time")
    fig.suptitle(f"BM-only real-data F4 preview | {start:%Y-%m-%d} to {start + pd.Timedelta(days=7):%Y-%m-%d} (exclusive)")
    fig.tight_layout()
    fig.savefig(out / "weekly_prediction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(hourly.index, hourly.observed_I_raw, label="Observed MED mean")
    ax.plot(fine.index, fine.predicted_I_raw, label="BM prediction")
    ax.plot(fine.index, fine.baseline_O_raw, label="Five-week baseline", alpha=.8)
    ax.set(ylabel="Original workbook units (unit not asserted)", xlabel="Local time",
           title="Demand in the original data units: use for cross-model comparison")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "demand_original_units.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].imshow((marginal/fv.dx).T, origin="lower", aspect="auto", extent=(0, 7, 0, 1))
    axes[0].set(xlabel="Days since forecast origin", ylabel="BM charge X", title="Forecast state density")
    fig.colorbar(im, ax=axes[0], label="Density")
    grid = np.linspace(0., 1., 201)
    for ax, field in zip(axes[1:], ("f_values", "g_values")):
        curves = np.array([np.interp(grid, KNOTS, getattr(p, field)) for p in parameters])
        ax.plot(grid, weights @ curves, label="Posterior mean shared curve")
        ax.plot(grid, 1. - 2.*grid, "--", alpha=.7, label="Linear reference")
        ax.set(xlabel="X" if field.startswith("f") else "BM price coordinate u", ylabel=field[0],
               title="One global f(X)" if field.startswith("f") else "One global g(u)")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "density_and_response_curves.png", dpi=180); plt.close(fig)


def run(args):
    begin = perf_counter()
    repo = Path(args.repo).resolve()
    if not (repo / "src/tools/import_data/import_input.py").is_file():
        raise ValueError("--repo must point to your BSCe2026 repository; run from its root or supply --repo.")
    input_path = Path(args.input).resolve() if args.input else repo / "files/input/data/input.xlsx"
    price_path = Path(args.prices).resolve() if args.prices else repo / "files/input/data/prices.xlsx"
    print("BM ONLY. Loading original MED consumption and DK1 prices; no WM4 execution.", flush=True)
    I, U, counts, columns = load_real_data(repo, input_path, price_path, args.tag)
    window, skipped = choose_week(I, U, args)
    start = window["start"]
    print(f"Forecast: {start} -> {start + pd.Timedelta(days=7)} (exclusive).", flush=True)
    print(f"Fit: {window['train_index'][args.warmup_hours]} -> {start}; warmup={args.warmup_hours}h.", flush=True)
    scales, scale_source = freeze_scales(I, U, start, args.scales_json)
    train_y = scale(window["train_y_raw"], scales, "consumption")
    train_b = scale(window["train_b_raw"], scales, "consumption")
    future_b = scale(window["future_b_raw"], scales, "consumption")
    train_u = scale(window["train_u_raw"], scales, "price")
    future_u = scale(window["future_u_raw"], scales, "price")
    # f/g are defined only on [0,1]. No implicit extension, clipping, or rescaling
    # with future prices is allowed. Out-of-range future consumption is scored as-is.
    for name, arr in (("training baseline", train_b), ("forecast baseline", future_b),
                      ("training price", train_u), ("forecast price", future_u)):
        if not np.isfinite(arr).all() or np.any(arr < -1e-12) or np.any(arr > 1. + 1e-12):
            raise ValueError(f"{name} exceeds the frozen [0,1] range (min={arr.min():.5g}, max={arr.max():.5g}). "
                             "Provide explicit wider shared ranges with --scales-json; no future-based repair was applied.")
    fv = FiniteVolume(args.cells, args.steps_per_hour)
    parameters = parameter_bank(args.fg)
    print(f"{len(parameters)} joint parameter/curve hypotheses; shared f/g across all days.", flush=True)
    masses, weights, weight_history, train_error = calibrate(
        fv, train_y, train_b, train_u, parameters, args.sigma_y, args.steps_per_hour, args.warmup_hours)
    origin_masses = masses.copy()
    fine, marginal, final_masses, forecast_error = forecast(
        fv, origin_masses, weights, future_b, future_u, parameters, args.steps_per_hour)
    fine.index = start + pd.to_timedelta(fine.elapsed_hours, unit="h")
    fine.index.name = "timestamp"
    fine["U_bm"] = fine["U"]
    fine["U"] = 2.*fine.U_bm - 1.  # Signed coordinate for WM4-style price plots.
    span = scales["consumption_max"] - scales["consumption_min"]
    for name in ("predicted_I", "baseline_O", "latent_q05", "latent_q95"):
        fine[name + "_raw"] = scales["consumption_min"] + span * fine[name]
    hourly = fine.iloc[::args.steps_per_hour].copy()
    hourly["observed_I_raw"] = require_complete(I, window["future_index"], "scoring consumption")
    hourly["observed_I"] = scale(hourly.observed_I_raw, scales, "consumption")
    hourly["price_DKK_per_MWh"] = window["future_u_raw"]
    hourly["available_MED_columns"] = counts.reindex(hourly.index).to_numpy()
    metrics = score_hourly(hourly)
    # Fresh run directory prevents an interrupted run from looking like a previous success.
    root = Path(args.output).resolve() if args.output else repo / "files/output/bm_1/preview_wm4"
    out = root / (f"{start:%Y%m%d}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    fine.to_csv(out / "fine_predictions.csv")
    hourly.to_csv(out / "hourly_predictions.csv")
    metrics.loc[metrics.period != "whole_week"].to_csv(out / "daily_metrics.csv", index=False)
    metrics.loc[metrics.period == "whole_week"].to_csv(out / "weekly_metrics.csv", index=False)
    pd.DataFrame({"timestamp": window["train_index"], "observed_I_raw": window["train_y_raw"],
                  "baseline_B_raw": window["train_b_raw"], "price_DKK_per_MWh": window["train_u_raw"]}).to_csv(out / "training_inputs.csv", index=False)
    (out / "shared_scales.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
    np.savez_compressed(out / "posterior_and_density.npz", x=fv.x, initial_masses=origin_masses,
                        final_masses=final_masses, weights=weights, weight_history=weight_history,
                        forecast_marginal_masses=marginal)
    save_plots(out, fine, hourly, fv, marginal, parameters, weights, start)
    try:
        git_revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): git_revision = None
    metadata = dict(
        status="exploratory_BM_only_F4_preview", wm4_executed=False, synthetic_measurements=False,
        reference_commit=REFERENCE_COMMIT, local_repository_commit=git_revision,
        forecast_start=start.isoformat(), forecast_end_exclusive=(start+pd.Timedelta(days=7)).isoformat(),
        fit_start=window["train_index"][args.warmup_hours].isoformat(), warmup_hours=args.warmup_hours,
        future_measurements_used_by_model=0, observation_timestamps="hour starts; right-continuous price",
        observation_likelihood="exploratory Gaussian hourly working likelihood; point/interval contract unverified",
        sigma_y=args.sigma_y, time_unit="day", measured_unit="original workbook unit; no power/energy assertion",
        consumption_tag=args.tag, aggregation="arithmetic mean of available original MED columns, same as WM4",
        measurement_columns=columns, scales=scales, scale_source=scale_source,
        wm4_scaling_note="WM4 currently uses Jan-Oct I and full-2022 price ranges. Default preview uses pre-origin ranges. Compare raw units or supply identical frozen scales to both models.",
        inputs=dict(consumption=str(input_path), consumption_sha256=file_hash(input_path),
                    prices=str(price_path), prices_sha256=file_hash(price_path)),
        solver=dict(cells=args.cells, steps_per_hour=args.steps_per_hour, max_mass_error=max(train_error, forecast_error)),
        parameters=[dict(**asdict(p), posterior_weight=float(w)) for p, w in zip(parameters, weights)],
        fg_mode=args.fg, curve_knots=KNOTS.tolist(), curve_policy="one shared joint bank; no day-specific refits; weights frozen at origin",
        baseline="five previous same-weekday/hour arithmetic mean", future_baseline_sources=window["future_baseline_sources"],
        auto_skipped_weeks=skipped, elapsed_seconds=perf_counter()-begin,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                     pandas=pd.__version__, matplotlib=matplotlib.__version__, system=platform.platform()),
        limitations=["Coarse surrogate, not completed F4-H/I/J or G validation.",
                     "Small constrained curve/parameter bank; intervals are exploratory latent bands, not calibrated observation intervals.",
                     "Known target-week actual spot prices: retrospective conditional forecast, not advance price forecasting.",
                     "No weather-aware WM4 O regression: B uses the F4 five-week historical reference.",
                     "BM X and BM R_dot are not WM4's F and R_dot; derivative excludes hourly input jumps."],
    )
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    report = metrics.loc[(metrics.period=="whole_week") & (metrics.units=="normalized")]
    print(report[["model", "mse", "rmse", "mae", "bias"]].to_string(index=False), flush=True)
    print(f"WM4 was not run. Results: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--input", help="Optional original input.xlsx override; same project sheet structure.")
    parser.add_argument("--prices", help="Optional original prices.xlsx override.")
    parser.add_argument("--tag", default="MED")
    parser.add_argument("--start", default="auto", help="Forecast Monday YYYY-MM-DD, or first complete eligible week.")
    parser.add_argument("--search-until", default="2022-03-01", help="Exclusive auto-search boundary; restricts inspection to early data.")
    parser.add_argument("--fit-days", type=int, default=7)
    parser.add_argument("--warmup-hours", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--fg", choices=("piecewise", "linear"), default="piecewise")
    parser.add_argument("--sigma-y", type=float, default=.06, help="Declared exploratory observation noise, normalized units; not estimated from target week.")
    parser.add_argument("--scales-json", help="Explicit fixed min/max consumption and price ranges, shared with comparator.")
    parser.add_argument("--output", help="Output parent; each run creates a new timestamped subdirectory.")
    args = parser.parse_args()
    if args.fit_days < 1 or args.warmup_hours < 0 or args.steps_per_hour < 2 or args.cells < 16:
        parser.error("Need fit-days>=1, warmup-hours>=0, steps-per-hour>=2, cells>=16.")
    if not np.isfinite(args.sigma_y) or args.sigma_y <= 0.: parser.error("sigma-y must be finite and positive.")
    try: run(args)
    except (ValueError, FileNotFoundError, FloatingPointError, ImportError) as exc:
        print(f"BM preview stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

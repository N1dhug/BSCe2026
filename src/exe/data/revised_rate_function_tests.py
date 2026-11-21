"""Statistical specification tests for the reconstructed revised-model rate derivative.

Two allowed baselines; no statistical-output baseline and no WM4 parameter fit.
Run: python -m src.exe.data.revised_rate_function_tests --group ALL --backlog 4
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.method.moving_horizon import group_frame
from src.method.moving_horizon import historical_weekdays
from src.tools.baseline import weekly_average, weekly_temperature
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT/"files"/"output"/"data"/"revised_rate_functions"
BASELINES = ("weekly_average", "weekly_temperature")
BASELINE_CLASSES = {"weekly_average": weekly_average, "weekly_temperature": weekly_temperature}
LABELS = {"weekly_average": "Weekly average", "weekly_temperature": "Temperature-adjusted weekly"}
FUNCTIONS = {
    "rate_quadratic": ("r2",),
    "rate_cubic": ("r3",),
    "rate_signed_quadratic": ("r_signed",),
    "rate_saturating": ("r_tanh",),
    "price_quadratic": ("u2",),
    "price_cubic": ("u3",),
    "price_signed_quadratic": ("u_signed",),
    "price_saturating": ("u_tanh",),
    "rate_polynomial": ("r2", "r3"),
    "price_polynomial": ("u2", "u3"),
    "joint_polynomial": ("r2", "r3", "u2", "u3"),
}
TERMS = ("C", "R", "U")


def reconstruct_day(day, frame, baseline, difference="forward"):
    """One complete measured day, at most 23 hourly transitions; no sub-hour pseudo-data."""
    index = pd.date_range(day, periods=24, freq="h")
    observed = frame.reindex(index)[["I", "Temp", "U"]]
    if not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError(f"Incomplete target-day I/Temp/U at {pd.Timestamp(day):%Y-%m-%d}.")
    demand = observed["I"].to_numpy(dtype=float)
    temperature = observed["Temp"].to_numpy(dtype=float)
    price = observed["U"].to_numpy(dtype=float)
    output = np.asarray(baseline.predict(index, temperature), dtype=float)
    if output.shape != (24,) or not np.isfinite(output).all():
        raise ValueError("Baseline must predict 24 finite hourly values.")
    rate = demand-output
    cumulative = np.r_[0.0, np.cumsum(rate[:-1])]  # C_j = sum_{l<j} R_l*Delta t; Delta t=1 hour
    if difference == "forward":
        hours = np.arange(23, dtype=int)
        derivative = rate[1:]-rate[:-1]
    elif difference == "central":
        hours = np.arange(1, 23, dtype=int)
        derivative = (rate[2:]-rate[:-2])/2.0
    else: raise ValueError("difference must be 'forward' or 'central'.")
    return pd.DataFrame({
        "day": pd.Timestamp(day).normalize(), "hour": hours, "timestamp": index[hours],
        "D": demand[hours], "B": output[hours], "R": rate[hours],
        "C": cumulative[hours], "U": price[hours], "R_dot": derivative,
    })


def collect(frame, *, backlog=4, start_day=None, end_day=None, max_days=0, difference="forward",
            progress_every=20):
    """Matched target days for both baselines; baseline fits strictly precede each day."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must be a datetime-indexed DataFrame.")
    if not {"I", "U", "Temp"} <= set(frame): raise ValueError("frame needs I, U, Temp.")
    if frame.index.has_duplicates or frame.index.hasnans or frame.index.tz is not None:
        raise ValueError("Data timestamps must be unique, timezone-naive, and finite.")
    if backlog < 2 or backlog > 14: raise ValueError("backlog must be in 2..14.")
    if max_days < 0 or progress_every < 1: raise ValueError("max_days >= 0 and progress_every >= 1 required.")
    frame = frame[["I", "U", "Temp"]].sort_index().astype(float)
    earliest = frame.index.min().normalize()
    latest = frame.index.max().normalize()
    start = max(earliest+pd.Timedelta(weeks=backlog), pd.Timestamp(start_day).normalize()) if start_day else earliest+pd.Timedelta(weeks=backlog)
    end = min(latest, pd.Timestamp(end_day).normalize()) if end_day else latest
    rows, skipped, successful = [], [], 0
    for i, day in enumerate(pd.date_range(start, end, freq="D"), 1):
        try:
            past = historical_weekdays(day, backlog)
            history, _ = impute_hourly_days(frame, days=past, columns=("I", "U", "Temp"), require_days=True)
            today = []
            for name in BASELINES:
                baseline = BASELINE_CLASSES[name](min_samples=48).fit(history, day)
                part = reconstruct_day(day, frame, baseline, difference=difference)
                part["baseline"] = name
                today.append(part)
            rows.extend(today)
            successful += 1
            if max_days and successful >= max_days: break
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            skipped.append({"day": day, "reason": str(exc)})
        if i % progress_every == 0:
            print(f"[rate test] {day:%Y-%m-%d}: {successful} matched days, {len(skipped)} skipped", flush=True)
    if not rows: raise RuntimeError("No valid matched days. Inspect dates and historical input coverage.")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(skipped, columns=("day", "reason"))


def _within_day(values, days):
    x = np.asarray(values, dtype=float)
    vector = x.ndim == 1
    if vector: x = x[:, None]
    codes, unique = pd.factorize(np.asarray(days), sort=True)
    counts = np.bincount(codes, minlength=len(unique))
    sums = np.zeros((len(unique), x.shape[1]), dtype=float)
    np.add.at(sums, codes, x)
    centered = x-sums[codes]/counts[codes, None]
    return centered[:, 0] if vector else centered


def _features(frame, scale_r, scale_u):
    """Fixed candidate functions on shared scales estimated from TRAIN only."""
    r = frame["R"].to_numpy(dtype=float)/scale_r
    u = frame["U"].to_numpy(dtype=float)/scale_u
    base = frame[list(TERMS)].to_numpy(dtype=float)
    extra = {
        "r2": r**2, "r3": r**3, "r_signed": r*np.abs(r), "r_tanh": np.tanh(2*r),
        "u2": u**2, "u3": u**3, "u_signed": u*np.abs(u), "u_tanh": np.tanh(2*u),
    }
    return base, extra


def _hac_day_covariance(x, error, days, max_lags):
    """Day-score Bartlett HAC: all intraday correlation, plus up to L calendar-day lags."""
    inv = np.linalg.pinv(x.T@x)
    unique, codes = np.unique(pd.to_datetime(days).normalize().values, return_inverse=True)
    scores = np.zeros((len(unique), x.shape[1]), dtype=float)
    np.add.at(scores, codes, x*error[:, None])
    calendar = pd.date_range(pd.Timestamp(unique[0]), pd.Timestamp(unique[-1]), freq="D")
    aligned = np.zeros((len(calendar), x.shape[1]), dtype=float)
    positions = calendar.get_indexer(pd.DatetimeIndex(unique))
    aligned[positions] = scores
    meat = aligned.T@aligned
    for lag in range(1, min(int(max_lags), len(calendar)-1)+1):
        cross = aligned[lag:].T@aligned[:-lag]
        meat += (1.0-lag/(int(max_lags)+1.0))*(cross+cross.T)
    return 0.5*(inv@meat@inv + (inv@meat@inv).T)


def _fit(x, y, dates, extra_count, hac_days):
    """OLS after removing one intercept per target day, with HAC-Wald inference."""
    x = _within_day(x, dates)
    y = _within_day(y, dates)
    scales = np.std(x, axis=0)
    if np.any(scales < 1e-12):
        raise ValueError("Non-identifiable feature: zero within-day variation.")
    scaled = x/scales
    if np.linalg.matrix_rank(scaled) < scaled.shape[1]:
        raise ValueError("Non-identifiable feature: rank-deficient design.")
    beta_scaled = np.linalg.lstsq(scaled, y, rcond=None)[0]
    beta = beta_scaled/scales
    error = y-x@beta
    covariance_scaled = _hac_day_covariance(scaled, error, dates, hac_days)
    covariance = covariance_scaled/(scales[:, None]*scales[None, :])
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if extra_count:
        # Robust, asymptotic chi-square test H0: all added coefficients = 0.
        block = covariance[-extra_count:, -extra_count:]
        if (np.linalg.matrix_rank(block) < extra_count or
                not np.isfinite(block).all() or np.min(np.linalg.eigvalsh(block)) <= 0):
            raise ValueError("HAC covariance cannot identify the nonlinear Wald test.")
        selected = beta[-extra_count:]
        statistic = float(selected@np.linalg.solve(block, selected))
        pvalue = float(chi2.sf(max(0.0, statistic), extra_count))
    else: statistic, pvalue = np.nan, np.nan
    return {"beta": beta, "standard_error": standard_error, "statistic": statistic,
            "p": pvalue, "train_rmse": float(np.sqrt(np.mean(error**2)))}


def _holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    ordered = valid[np.argsort(p[valid])]
    maximum = 0.0
    for rank, idx in enumerate(ordered):
        maximum = max(maximum, (len(ordered)-rank)*p[idx])
        out[idx] = min(1.0, maximum)
    return out


def analyze(observations, *, train_fraction=.75, hac_days=14, alpha=.05, gain_threshold_pct=1.):
    """Compare added fixed nonlinear functions to affine drift on future held-out days.

    Held-out metric is conditional on a *retrospectively removed daily intercept*,
    not a deployable forecast; F0 is unidentified by the reconstructed derivative.
    """
    if not 0.5 <= train_fraction < 1.0: raise ValueError("train_fraction must be in [0.5,1).")
    if hac_days < 0 or not 0 < alpha < 1 or gain_threshold_pct < 0:
        raise ValueError("Require hac_days >= 0, 0 < alpha < 1 and gain_threshold_pct >= 0.")
    for col in ("day", "baseline", "C", "R", "U", "R_dot"):
        if col not in observations: raise ValueError(f"Missing observations column: {col}.")
    tables, coefficients, diagnostics, splits = [], [], [], {}
    for name in BASELINES:
        frame = observations.loc[observations["baseline"] == name].sort_values(["day", "hour"]).copy()
        days = pd.DatetimeIndex(pd.to_datetime(frame["day"]).dt.normalize().unique()).sort_values()
        n_train = int(np.floor(len(days)*train_fraction))
        if n_train < 10 or len(days)-n_train < 3:
            raise ValueError("At least 10 training days and 3 held-out days per baseline are required.")
        train = frame.loc[frame["day"].isin(days[:n_train])]
        test = frame.loc[frame["day"].isin(days[n_train:])]
        if train.empty or test.empty: raise ValueError("Empty chronological train/test split.")
        scale_r = max(float(np.quantile(np.abs(train["R"]), .90)), 1e-5)
        scale_u = max(float(np.quantile(np.abs(train["U"]), .90)), 1e-5)
        base_train, ext_train = _features(train, scale_r, scale_u)
        base_test, ext_test = _features(test, scale_r, scale_u)
        y_train = train["R_dot"].to_numpy(dtype=float)
        y_test = test["R_dot"].to_numpy(dtype=float)
        dates_train, dates_test = train["day"].to_numpy(), test["day"].to_numpy()
        y_test_centered = _within_day(y_test, dates_test)
        baseline_fit = _fit(base_train, y_train, dates_train, 0, hac_days)
        baseline_error = y_test_centered-_within_day(base_test, dates_test)@baseline_fit["beta"]
        baseline_rmse = float(np.sqrt(np.mean(baseline_error**2)))
        if baseline_rmse < 1e-12: raise ValueError("Affine drift exactly fits held-out derivative; comparison undefined.")
        diagnostics.append(pd.DataFrame({
            "day": test["day"].to_numpy(), "hour": test["hour"].to_numpy(),
            "baseline": name, "R": test["R"].to_numpy(dtype=float),
            "U": test["U"].to_numpy(dtype=float), "affine_residual": baseline_error,
        }))
        splits[name] = {"train_days": n_train, "test_days": len(days)-n_train,
                        "train_start": str(days[0].date()), "train_end": str(days[n_train-1].date()),
                        "test_start": str(days[n_train].date()), "test_end": str(days[-1].date()),
                        "scale_R": scale_r, "scale_U": scale_u,
                        "train_transitions": len(train), "test_transitions": len(test)}
        for label, added in [("affine", ())]+list(FUNCTIONS.items()):
            columns = list(TERMS)+list(added)
            x_train = np.column_stack([base_train]+[ext_train[term][:, None] for term in added])
            x_test = np.column_stack([base_test]+[ext_test[term][:, None] for term in added])
            try:
                fit = baseline_fit if not added else _fit(x_train, y_train, dates_train, len(added), hac_days)
                residual = y_test_centered-_within_day(x_test, dates_test)@fit["beta"]
                test_rmse = float(np.sqrt(np.mean(residual**2)))
                gain = float(100*(baseline_rmse-test_rmse)/baseline_rmse)
                pvalue = fit["p"]
                status = "ok"
                for j, term in enumerate(columns):
                    se = float(fit["standard_error"][j])
                    z = float(fit["beta"][j]/se) if se > 1e-15 else np.nan
                    coefficients.append({
                        "baseline": name, "candidate": label, "term": term,
                        "estimate": float(fit["beta"][j]), "hac_se": se,
                        "z": z, "p_normal": float(2*norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                    })
            except ValueError as exc:
                fit = None; test_rmse = np.nan; gain = np.nan; pvalue = np.nan
                status = str(exc)
            tables.append({
                "baseline": name, "candidate": label, "added_terms": ",".join(added),
                "added_df": len(added), "wald_chi2": fit["statistic"] if fit else np.nan,
                "p_hac": pvalue, "train_rmse": fit["train_rmse"] if fit else np.nan,
                "heldout_rmse": test_rmse, "heldout_gain_pct": gain,
                "affine_heldout_rmse": baseline_rmse, "status": status,
                "train_days": n_train, "heldout_days": len(days)-n_train,
            })
    result = pd.DataFrame(tables)
    result["p_holm"] = np.nan
    for name in BASELINES:
        select = (result["baseline"] == name)&(result["candidate"] != "affine")
        result.loc[select, "p_holm"] = _holm_adjust(result.loc[select, "p_hac"])
    result["significant_holm"] = result["p_holm"] < alpha
    result["material_heldout_gain"] = result["heldout_gain_pct"] >= gain_threshold_pct
    result["both_criteria"] = result["significant_holm"] & result["material_heldout_gain"]
    return result, pd.DataFrame(coefficients), pd.concat(diagnostics, ignore_index=True), splits


def _plot_gains(tests, output, threshold=1.):
    names = list(FUNCTIONS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, baseline in zip(axes, BASELINES):
        data = tests.loc[tests["baseline"] == baseline].set_index("candidate").reindex(names)
        values = data["heldout_gain_pct"].to_numpy(dtype=float)
        bars = ax.barh(range(len(names)), np.nan_to_num(values), color="C0", alpha=.75)
        for i, bar in enumerate(bars):
            if bool(data["both_criteria"].iloc[i]): bar.set_edgecolor("black"); bar.set_linewidth(2)
        ax.axvline(0, color="black", lw=.8)
        ax.axvline(threshold, color="0.45", lw=1, ls="--", label=f"{threshold:g}% threshold")
        ax.set_title(LABELS[baseline]); ax.set_xlabel("Held-out derivative RMSE reduction (%)")
        ax.grid(axis="x", alpha=.2); ax.legend(loc="lower right")
        ax.set_yticks(range(len(names)), [v.replace("_", " ") for v in names])
        ax.invert_yaxis()
    fig.suptitle("Nonlinear terms versus affine drift (edge = HAC-Holm significant and material)")
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_residual(diagnostics, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, baseline in enumerate(BASELINES):
        data = diagnostics.loc[diagnostics["baseline"] == baseline]
        for col, variable in enumerate(("R", "U")):
            ax = axes[row, col]
            x = data[variable].to_numpy(dtype=float)
            residual = data["affine_residual"].to_numpy(dtype=float)
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 17)))
            if len(edges) >= 3:
                bins = np.digitize(x, edges[1:-1], right=True)
                for b in range(len(edges)-1):
                    sel = bins == b
                    if sel.sum() >= 3:
                        ax.scatter(np.median(x[sel]), np.mean(residual[sel]),
                                   s=25 + .1*sel.sum())
            ax.axhline(0, color="black", ls="--", lw=.8)
            ax.set_title(f"{LABELS[baseline]} | {variable}")
            ax.set_xlabel(f"Reconstructed {variable}"); ax.set_ylabel("Mean held-out affine residual")
            ax.grid(alpha=.2)
    fig.suptitle("Within-day affine derivative residuals by state and price (quantile bins)")
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)


def run(*, group="ALL", backlog=4, start_day=None, end_day=None, max_days=0,
        difference="forward", train_fraction=.75, hac_days=14, alpha=.05,
        gain_threshold_pct=1., output=None, progress_every=20):
    group = group.upper()
    if group not in ("ALL", "MED"): raise ValueError("group must be ALL or MED.")
    if difference not in ("forward", "central"): raise ValueError("difference must be forward or central.")
    dest = Path(output) if output else OUTPUT/group.lower()/f"backlog_{backlog:02d}_{difference}"
    dest.mkdir(parents=True, exist_ok=True)
    frame = group_frame(group)
    observations, skipped = collect(frame, backlog=backlog, start_day=start_day, end_day=end_day,
                                    max_days=max_days, difference=difference, progress_every=progress_every)
    result, coefficients, residuals, splits = analyze(
        observations, train_fraction=train_fraction, hac_days=hac_days,
        alpha=alpha, gain_threshold_pct=gain_threshold_pct)
    observations.to_csv(dest/"reconstructed_rate_derivative.csv", index=False)
    skipped.to_csv(dest/"skipped_days.csv", index=False)
    result.to_csv(dest/"function_tests.csv", index=False)
    coefficients.to_csv(dest/"function_coefficients.csv", index=False)
    residuals.to_csv(dest/"affine_heldout_residuals.csv", index=False)
    _plot_gains(result, dest/"nonlinear_heldout_improvements.png", threshold=gain_threshold_pct)
    _plot_residual(residuals, dest/"affine_rate_residual_diagnostics.png")
    report = {
        "group": group, "backlog_N": int(backlog), "difference": difference,
        "training_fraction": float(train_fraction), "hac_lags_calendar_days": int(hac_days),
        "alpha": float(alpha), "minimum_material_rmse_gain_pct": float(gain_threshold_pct),
        "matched_observed_days": int(observations["day"].nunique()),
        "skipped_days": len(skipped), "chronological_splits": splits,
        "baseline_names": list(BASELINES), "features": {k: list(v) for k,v in FUNCTIONS.items()},
        "interpretation": (
            "Conditional association of fixed nonlinear response functions with reconstructed "
            "hourly rate changes, controlling for cumulative flexibility, current rate and price, "
            "with day-specific intercepts. Wald chi-square p-values use approximate day-score "
            "HAC inference; Holm correction is within each baseline. The holdout metric removes "
            "an intercept using the complete held-out day, and is NOT an operational forecast. "
            "Observation differencing and baseline fitting introduce generated-regressor error; "
            "non-rejection does not prove affine drift. No nonlinear constrained dynamical model "
            "was fitted. The baseline was estimated only from earlier corresponding weekdays."
        ),
    }
    (dest/"run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRevised rate-function study | {group} | matched days={report['matched_observed_days']}")
    for name in BASELINES:
        print(f"\n{LABELS[name]} | train={splits[name]['train_days']} days "
              f"| held-out={splits[name]['test_days']} days")
        excerpt = result.loc[result["baseline"] == name,
                             ["candidate", "p_hac", "p_holm", "heldout_rmse", "heldout_gain_pct", "both_criteria", "status"]]
        print(excerpt.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print(f"\nSaved CSV, figures and JSON: {dest}")
    print("CAUTION: A significant nonlinear feature is not evidence that it improves full dynamical forecasting.")
    return result, coefficients, observations, report


def main():
    parser = ArgumentParser(description="Test nonlinear rate/price functions against affine revised drift.")
    parser.add_argument("--group", choices=("ALL", "MED"), default="ALL")
    parser.add_argument("--backlog", type=int, default=4, help="Strictly preceding matching weekdays N (2..14).")
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--max-days", type=int, default=0, help="Cap matched observed target days; 0=all.")
    parser.add_argument("--difference", choices=("forward", "central"), default="forward")
    parser.add_argument("--train-fraction", type=float, default=.75)
    parser.add_argument("--hac-days", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--gain-threshold-pct", type=float, default=1.)
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    return run(group=args.group, backlog=args.backlog, start_day=args.start_day, end_day=args.end_day,
               max_days=args.max_days, difference=args.difference,
               train_fraction=args.train_fraction, hac_days=args.hac_days,
               alpha=args.alpha, gain_threshold_pct=args.gain_threshold_pct,
               output=args.output, progress_every=args.progress_every)


if __name__ == "__main__": main()

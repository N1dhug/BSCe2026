"""ARM diffusion calibration from saved hourly (origin, baseline, N, k) drift fits.

No dynamical parameter or initial-state optimization is repeated. Each stochastic
fit uses precisely the N historical k-hour trajectories of its saved drift fit.
Only lead hours 1,...,k-1 enter the variance fit and all evaluation scores;
lead hour 0 is a deterministic initialization and is never scored.

Historical evaluation is in-sample. Target evaluation is ORACLE-CONDITIONED:
the deterministic target initial state was fitted using target observations.
This evaluates conditional uncertainty calibration, NOT operational forecasting.

Outputs are streamed to disk and pooled by (baseline, N, k). The pooled
Gaussian similarity uses a high-resolution binned KS approximation to avoid
holding hundreds of millions of residuals in memory.
"""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader, DictWriter
from contextlib import nullcontext
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.exe.run_model.wm_4.plot_moving_horizon_quality import (
    BASELINES, bounds, common_origins,
    configuration, historical_errors, saved_baseline, selected,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon import OUTPUT as SWEEP_OUTPUT
from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"moving_horizon_diffusion"
HIST_BINS = np.linspace(-8., 8., 4097)
Z_SCORE = 1.959963984540054
NLL_VARIANCE_FLOOR = 1e-12  # Likelihood evaluation only; never changes fitted variances.
SCORE_FIELDS = ("day", "baseline", "training_weekdays", "horizon_hours")
PARAMETER_FIELDS = (
    "day", "baseline", "training_weekdays", "horizon_hours",
    "observation_variance", "observation_sigma", "process_variance_rate",
    "process_sigma", "process_identifiable", "historical_trajectories",
    "historical_observations", "historical_observed_rmse",
    "historical_expected_rmse", "historical_coverage_95", "historical_interval_nll",
)
TARGET_FIELDS = (
    "day", "baseline", "training_weekdays", "horizon_hours",
    "observations", "oracle_observed_rmse", "oracle_expected_rmse",
    "oracle_coverage_95", "oracle_bias", "oracle_interval_nll",
)
TITLES = {"weekly_average":"Weekly average", "weekly_temperature":"Weekly temperature",
          "statistical_output":"Statistical output"}


def variance_kernel(theta_F, theta_R, steps_per_hour, horizon):
    """Exact unit process variance at each observed hour under ARM Euler-Maruyama.

    The impulse response sums dt*(e_R^T A^i e_R)^2. This is equivalent to
    wm_4.process_variance_kernel(...).sum(axis=1) at hourly sample indices,
    without building a (k*s+1)^2 kernel matrix.
    """
    steps_per_hour, horizon = int(steps_per_hour), int(horizon)
    if steps_per_hour < 2 or not 1 <= horizon <= 24:
        raise ValueError("steps_per_hour must be >=2 and horizon in [1,24].")
    dt = 1./(24*steps_per_hour)
    f, r, value = 0., 1., 0.
    kernel = np.empty(horizon, dtype=float)
    for hour in range(horizon):
        kernel[hour] = value
        if hour == horizon-1: break
        for _ in range(steps_per_hour):
            value += dt*r*r
            f, r = f+dt*r, r+dt*(theta_F*f+theta_R*r)
    if not np.isfinite(kernel).all() or np.min(kernel) < -1e-12:
        raise ValueError("Nonfinite or negative Euler-Maruyama variance kernel.")
    return np.maximum(kernel, 0.)


def fit_variances(errors, kernel):
    """Analytic nonnegative LS for E[error^2 | K]=tau^2+q*K.

    Checks the unconstrained nonnegative solution and both active faces.
    Equivalent to NNLS with columns [1,K], avoiding millions of solver calls.
    """
    errors = np.asarray(errors, dtype=float).reshape(-1)
    K = np.asarray(kernel, dtype=float).reshape(-1)
    if errors.size != K.size or not errors.size or not np.isfinite(errors).all():
        raise ValueError("Error and kernel must contain matching finite observations.")
    if not np.isfinite(K).all() or np.min(K) < 0:
        raise ValueError("Variance kernel must be finite and nonnegative.")
    y = errors*errors
    mean_y = float(np.mean(y))
    if np.ptp(K) < 1e-12: return mean_y, 0., False
    mean_K = float(np.mean(K))
    mean_K2 = float(np.mean(K*K))
    mean_Ky = float(np.mean(K*y))
    covariance_K = mean_K2-mean_K**2
    if covariance_K <= 0: return mean_y, 0., False
    q_ols = (mean_Ky-mean_K*mean_y)/covariance_K
    tau_ols = mean_y-q_ols*mean_K
    # All candidate faces of the two-variable nonnegative quadratic program.
    candidates = [(mean_y, 0.), (0., max(0., mean_Ky/mean_K2))]
    if tau_ols >= 0. and q_ols >= 0.: candidates.append((tau_ols, q_ols))
    tau2, q = min(candidates, key=lambda x: float(np.mean((y-x[0]-x[1]*K)**2)))
    return float(tau2), float(q), True


def row_key(row):
    return (pd.Timestamp(row["day"]), str(row["baseline"]),
            int(row["training_weekdays"]), int(row["horizon_hours"]))


def oracle_errors(frame, row, score, steps_per_hour):
    """Reconstruct the saved target oracle trajectory without fitting anything."""
    if row_key(row) != row_key(score):
        raise ValueError("Saved fit and oracle-score rows are misaligned.")
    origin = pd.Timestamp(row["day"])
    k = int(row["horizon_hours"])
    index = pd.date_range(origin, periods=k, freq="h")
    source = frame.reindex(index)[["I", "U", "Temp"]]
    if source.isna().any().any():
        raise ValueError(f"Target observations must be complete at {origin}, k={k}.")
    observed = source["I"].to_numpy(dtype=float)
    U = source["U"].to_numpy(dtype=float)
    baseline = saved_baseline(row)
    O = baseline.predict(index, source["Temp"].to_numpy(dtype=float))
    F, R = float(score["oracle_F0"]), float(score["oracle_R0"])
    theta_F, theta_U, theta_R = float(row["theta_F"]), float(row["theta_U"]), float(row["theta_R"])
    dt = 1./(24*int(steps_per_hour))
    error = np.empty(k, dtype=float)
    for hour in range(k):
        error[hour] = observed[hour]-(O[hour]+R)
        if hour == k-1: break
        for _ in range(int(steps_per_hour)):
            dR = theta_F*F+theta_U*U[hour]+theta_R*R
            F, R = F+dt*R, R+dt*dR
    if not np.isfinite(error).all(): raise ValueError("Nonfinite oracle target error.")
    # Protect against subtle differences in model, baseline, price or time units.
    saved_mse = float(score["mse"])
    if not np.isclose(float(np.mean(error*error)), saved_mse, atol=5e-9, rtol=1e-5):
        raise ValueError(f"Oracle replay differs from saved prediction at {origin}, "
                         f"N={row['training_weekdays']}, k={k}; "
                         "source data or saved results may have changed.")
    return error


def scored_hours(values, horizon):
    """Flatten lead hours 1,...,k-1 from complete k-hour trajectories."""
    k = int(horizon)
    data = np.asarray(values, dtype=float).reshape(-1)
    if k < 2 or not len(data) or len(data) % k:
        raise ValueError("Scoring requires complete k-hour intervals with k >= 2.")
    return data.reshape(-1, k)[:, 1:].reshape(-1)


def interval_nll(errors, variance, horizon):
    """Sum marginal Gaussian NLL on lead hours 1,...,k-1 only.

    Each input contains complete k-hour intervals (including initialization
    hour 0); this routine excludes hour 0 from EVERY interval. Likelihoods
    use a variance floor only for genuinely degenerate fitted variances.
    No Kalman conditioning occurs between observations.
    """
    e = scored_hours(errors, horizon)
    v = scored_hours(variance, horizon)
    if len(e) != len(v) or not np.isfinite(e).all() or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError("NLL requires aligned, finite residuals and nonnegative variances.")
    safe_v = np.maximum(v, NLL_VARIANCE_FLOOR)
    return float(np.sum(0.5*(np.log(2.*np.pi*safe_v)+e*e/safe_v)))


def coverage(errors, variance):
    e = np.asarray(errors, dtype=float)
    v = np.asarray(variance, dtype=float)
    return float(np.mean(np.abs(e) <= Z_SCORE*np.sqrt(np.maximum(v, 0.))))


def pooled_record(k):
    return {"k":int(k), "origins":0, "count":0,
            "observed_square":0., "expected_variance":0., "signed":0., "interval_nll_sum":0.,
            "covered":0, "hourly_observed":np.zeros(k-1), "hourly_expected":np.zeros(k-1),
            "hourly_count":np.zeros(k-1, dtype=np.int64),
            "tau2_sum":0., "q_sum":0., "tau_sum":0., "sigma_sum":0.,
            "identifiable":0, "histogram":np.zeros(len(HIST_BINS)-1, dtype=np.int64),
            "standardized":0, "zero_variance_errors":0}


def update_pool(pool, key, errors, variance, tau2, q, identifiable, historical_N):
    """Pool only scored lead hours 1,...,k-1, keeping each origin as N intervals."""
    full_e = np.asarray(errors, dtype=float).reshape(-1)
    full_v = np.asarray(variance, dtype=float).reshape(-1)
    k = int(key[2])
    if k < 2 or len(full_e) != len(full_v) or len(full_e) != historical_N*k:
        raise ValueError("Each configuration must have complete N*k historical or k target errors with k >= 2.")
    if not np.isfinite(full_e).all() or not np.isfinite(full_v).all() or np.min(full_v) < -1e-12:
        raise ValueError("Invalid predicted variance or residual.")
    e = scored_hours(full_e, k)
    v = scored_hours(full_v, k)
    item = pool.setdefault(key, pooled_record(k))
    item["origins"] += 1
    item["count"] += len(e)
    item["observed_square"] += float(np.dot(e, e))
    item["expected_variance"] += float(v.sum())
    item["signed"] += float(e.sum())
    item["interval_nll_sum"] += interval_nll(full_e, full_v, k)
    item["covered"] += int(np.count_nonzero(np.abs(e) <= Z_SCORE*np.sqrt(np.maximum(v, 0.))))
    item["hourly_observed"] += np.square(e.reshape(historical_N, k-1)).sum(axis=0)
    item["hourly_expected"] += v.reshape(historical_N, k-1).sum(axis=0)
    item["hourly_count"] += historical_N
    item["tau2_sum"] += tau2
    item["q_sum"] += q
    item["tau_sum"] += float(np.sqrt(tau2))
    item["sigma_sum"] += float(np.sqrt(q))
    item["identifiable"] += int(identifiable)
    good = v > 1e-14
    if np.any(good):
        z = e[good]/np.sqrt(v[good])
        item["histogram"] += np.histogram(np.clip(z, HIST_BINS[0], HIST_BINS[-1]),
                                            bins=HIST_BINS)[0].astype(np.int64)
        item["standardized"] += int(np.count_nonzero(good))
    item["zero_variance_errors"] += int(np.count_nonzero(~good & (e != 0.)))


def ks_similarity(histogram, count):
    """Approximate 1-KS via fixed-width empirical CDF bins (width 0.00391).

    The Gaussian CDF changes by at most ~0.00156 within any histogram bin,
    bounding this discretization's effect on the distance estimate.
    """
    if count < 3: return np.nan
    cumulative = np.cumsum(histogram, dtype=np.int64)/int(count)
    distance = float(np.max(np.abs(cumulative-ndtr(HIST_BINS[1:]))))
    return float(np.clip(1.-distance, 0., 1.))


def grid_summary(pool):
    rows = []
    for (baseline, n, k), x in sorted(pool.items()):
        count, origins = x["count"], x["origins"]
        hourly_counts = x["hourly_count"]
        if count < 1 or not np.all(hourly_counts > 0): raise ValueError("Incomplete pooled hourly residuals.")
        observed = float(np.sqrt(x["observed_square"]/count))
        expected = float(np.sqrt(x["expected_variance"]/count))
        observed_hourly = np.sqrt(x["hourly_observed"]/hourly_counts)
        expected_hourly = np.sqrt(x["hourly_expected"]/hourly_counts)
        similarity = ks_similarity(x["histogram"], x["standardized"])
        rows.append({
            "baseline":baseline, "training_weekdays":n, "horizon_hours":k,
            "origins":origins, "observations":count,
            "observed_rmse":observed, "expected_rmse":expected,
            "rmse_absolute_error":abs(expected-observed),
            "mean_interval_nll":x["interval_nll_sum"]/(count//(k-1)),
            "mean_hourly_rmse_absolute_error":float(np.mean(np.abs(expected_hourly-observed_hourly))),
            "bias_observed_minus_predicted":x["signed"]/count,
            "coverage_95":x["covered"]/count,
            "ks_similarity":similarity,
            "ks_distance":1.-similarity if np.isfinite(similarity) else np.nan,
            "standardized_samples":x["standardized"],
            "excluded_zero_variance_nonzero_errors":x["zero_variance_errors"],
            "observation_variance_mean":x["tau2_sum"]/origins,
            "process_variance_rate_mean":x["q_sum"]/origins,
            "observation_sigma_mean":x["tau_sum"]/origins,
            "process_sigma_mean":x["sigma_sum"]/origins,
            "process_identifiable_pct":100*x["identifiable"]/origins,
        })
    return pd.DataFrame(rows)


def plot_grid(frame, filename, eval_name, names, ns, ks):
    if frame.empty: raise ValueError(f"No diffusion {eval_name} results available.")
    metrics = (
        ("mean_interval_nll", "Mean interval NLL (lead 1..k-1)", "viridis", None, None),
        ("ks_similarity", "Gaussian similarity (1 - KS; lead 1..k-1)", "viridis", 0., 1.),
        ("coverage_95", "Empirical 95% coverage (lead 1..k-1)", "viridis", 0., 1.),
    )
    fig, axes = plt.subplots(len(metrics), len(names), figsize=(max(10, 7.4*len(names)), 12),
                             squeeze=False, constrained_layout=True)
    for i, (metric, title, cmap, forced_min, forced_max) in enumerate(metrics):
        values = frame[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lo = forced_min if forced_min is not None else (float(finite.min()) if finite.size else 0.)
        hi = forced_max if forced_max is not None else (float(finite.max()) if finite.size else 1.)
        if hi <= lo: hi = lo+1e-12
        for j, name in enumerate(names):
            ax = axes[i, j]
            part = frame.loc[frame["baseline"] == name]
            grid = part.pivot(index="training_weekdays", columns="horizon_hours", values=metric)
            vals = grid.reindex(index=ns, columns=ks).to_numpy(dtype=float)
            im = ax.imshow(np.ma.masked_invalid(vals), origin="lower", aspect="auto",
                           interpolation="nearest", vmin=lo, vmax=hi, cmap=cmap)
            ax.set_title(f"{title} | {TITLES[name]}")
            ax.set_xticks(np.arange(len(ks)), labels=ks, fontsize=8)
            ax.set_yticks(np.arange(len(ns)), labels=ns, fontsize=8)
            ax.set_xlabel("Historical fitting and evaluation horizon k (hours)")
            if j == 0: ax.set_ylabel("N preceding matching weekdays")
            fig.colorbar(im, ax=ax, shrink=0.84, pad=0.012,
                         label=metric.replace("_", " "))
    fig.suptitle(f"ARM diffusion: {eval_name} calibration | per-(origin,N,k) variance fit",
                 fontsize=13)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_parameters(frame, filename, names, ns, ks):
    if frame.empty: return
    metrics = (("process_sigma_mean", "Mean process noise sigma"),
               ("observation_sigma_mean", "Mean observation noise tau"))
    fig, axes = plt.subplots(2, len(names), figsize=(max(10, 7.4*len(names)), 8),
                             squeeze=False, constrained_layout=True)
    for i, (field, title) in enumerate(metrics):
        maximum = float(np.nanmax(frame[field].to_numpy(dtype=float)))
        maximum = max(1e-12, maximum)
        for j, name in enumerate(names):
            part = frame.loc[frame["baseline"] == name]
            grid = part.pivot(index="training_weekdays", columns="horizon_hours", values=field)
            values = grid.reindex(index=ns, columns=ks).to_numpy(dtype=float)
            im = axes[i, j].imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto",
                                    interpolation="nearest", vmin=0., vmax=maximum, cmap="viridis")
            axes[i, j].set_title(f"{title} | {TITLES[name]}")
            axes[i, j].set_xticks(np.arange(len(ks)), labels=ks, fontsize=8)
            axes[i, j].set_yticks(np.arange(len(ns)), labels=ns, fontsize=8)
            axes[i, j].set_xlabel("Historical fitting horizon k (hours)")
            if j == 0: axes[i, j].set_ylabel("N preceding matching weekdays")
            fig.colorbar(im, ax=axes[i, j], shrink=0.83)
    fig.suptitle("ARM diffusion | historically estimated noise amplitudes")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=175, bbox_inches="tight")
    plt.close(fig)


def iter_saved(source, include_scores):
    """Read fit and N state records sequentially; check keys before using them."""
    path = Path(source)
    fits, states, scores = (path/"parameter_estimates.csv",
                            path/"historical_initial_states.csv",
                            path/"oracle_horizon_scores.csv")
    for required in (fits, states):
        if not required.is_file(): raise FileNotFoundError(required)
    if include_scores and not scores.is_file(): raise FileNotFoundError(scores)
    with fits.open(newline="", encoding="utf-8-sig") as f_fit, \
         states.open(newline="", encoding="utf-8-sig") as f_states:
        fit_reader, state_reader = DictReader(f_fit), iter(DictReader(f_states))
        score_file = scores.open(newline="", encoding="utf-8-sig") if include_scores else None
        try:
            score_reader = iter(DictReader(score_file)) if score_file else None
            for fit in fit_reader:
                n = int(fit["training_weekdays"])
                key = row_key(fit)
                history = []
                for _ in range(n):
                    row = next(state_reader, None)
                    if row is None: raise ValueError(f"Truncated historical states for {key}.")
                    if row_key(row) != key:
                        raise ValueError(f"Saved drift and initial-state rows are not aligned at {key}.")
                    history.append(row)
                if fit.get("training_days"):
                    expected = tuple(pd.Timestamp(value) for value in fit["training_days"].split(";"))
                    actual = tuple(pd.Timestamp(row["historical_day"]) for row in history)
                    if actual != expected:
                        raise ValueError(f"Saved historical initial states differ from original corresponding intervals at {key}.")
                score = next(score_reader, None) if score_reader else None
                if score_reader and (score is None or row_key(score) != key):
                    raise ValueError(f"Saved drift and oracle-score rows are not aligned at {key}.")
                yield fit, history, score
            if next(state_reader, None) is not None or (
                    score_reader and next(score_reader, None) is not None):
                raise ValueError("Saved CSV files have inconsistent row counts.")
        finally:
            if score_file is not None: score_file.close()


def run(args):
    source = Path(args.source) if args.source else SWEEP_OUTPUT/args.group.lower()
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()
    config = configuration(source)
    if config.get("group") != args.group:
        raise ValueError(f"Saved group {config.get('group')} differs from --group {args.group}.")
    selection = bounds(config, args)
    names, ns, ks, first, until = selection
    if max(ks) < 2:
        raise ValueError("k=1 has no post-initialization observations; select --max-horizon >= 2.")
    if 1 in ks:
        print("[ARM diffusion] excluding k=1: no scored lead hour after initialization.", flush=True)
        ks = tuple(k for k in ks if k >= 2)
        selection = (names, ns, ks, first, until)
    matched = common_origins(source, selection) if args.cohort == "matched" else None
    if matched is not None:
        print(f"[ARM diffusion] matched hourly origins={len(matched):,}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    in_path = output/"diffusion_interval_parameters.csv"
    target_path = output/"diffusion_oracle_interval_scores.csv"
    needs_target = args.evaluation == "both"
    historical_pool, oracle_pool = {}, {}
    raw = group_frame(args.group)
    history_frame, _ = impute_hourly_days(raw, columns=("I", "U", "Temp"))
    steps = int(config["steps_per_hour"])
    scanned, processed = 0, 0
    mode = "w"
    with in_path.open(mode, encoding="utf-8", newline="") as f_in:
        writer_in = DictWriter(f_in, fieldnames=PARAMETER_FIELDS)
        writer_in.writeheader()
        with (target_path.open(mode, encoding="utf-8", newline="") if needs_target else nullcontext(None)) as f_target:
            writer_target = DictWriter(f_target, fieldnames=TARGET_FIELDS) if needs_target else None
            if writer_target: writer_target.writeheader()
            for row, states, score in iter_saved(source, include_scores=needs_target):
                scanned += 1
                if not selected(row, *selection): continue
                origin = pd.Timestamp(row["day"])
                if matched is not None and origin not in matched: continue
                n, k = int(row["training_weekdays"]), int(row["horizon_hours"])
                key = (row["baseline"], n, k)
                # Shared replay utility reports model-observed; diffusion uses observed-model.
                e = -historical_errors(history_frame, row, states, steps)
                K = variance_kernel(float(row["theta_F"]), float(row["theta_R"]), steps, k)
                K_hist = np.tile(K, n)
                tau2, q, identifiable = fit_variances(scored_hours(e, k), scored_hours(K_hist, k))
                v_hist = tau2+q*K_hist
                update_pool(historical_pool, key, e, v_hist, tau2, q, identifiable, n)
                writer_in.writerow({
                    "day":row["day"], "baseline":key[0], "training_weekdays":n,
                    "horizon_hours":k, "observation_variance":tau2,
                    "observation_sigma":float(np.sqrt(tau2)),
                    "process_variance_rate":q, "process_sigma":float(np.sqrt(q)),
                    "process_identifiable":identifiable, "historical_trajectories":n,
                    "historical_observations":n*(k-1),
                    "historical_observed_rmse":float(np.sqrt(np.mean(scored_hours(e, k)**2))),
                    "historical_expected_rmse":float(np.sqrt(np.mean(scored_hours(v_hist, k)))),
                    "historical_coverage_95":coverage(scored_hours(e, k), scored_hours(v_hist, k)),
                    "historical_interval_nll":interval_nll(e, v_hist, k)/n,
                })
                if needs_target:
                    target = oracle_errors(raw, row, score, steps)
                    v_target = tau2+q*K
                    update_pool(oracle_pool, key, target, v_target, tau2, q, identifiable, 1)
                    writer_target.writerow({
                        "day":row["day"], "baseline":key[0], "training_weekdays":n,
                        "horizon_hours":k, "observations":k-1,
                        "oracle_observed_rmse":float(np.sqrt(np.mean(target[1:]**2))),
                        "oracle_expected_rmse":float(np.sqrt(np.mean(v_target[1:]))),
                        "oracle_coverage_95":coverage(target[1:], v_target[1:]),
                        "oracle_bias":float(np.mean(target[1:])),
                        "oracle_interval_nll":interval_nll(target, v_target, k),
                    })
                processed += 1
                if args.max_configurations and processed >= args.max_configurations: break
                if processed % args.progress_every == 0:
                    f_in.flush()
                    if needs_target: f_target.flush()
                    print(f"[ARM diffusion] processed={processed:,}; scanned={scanned:,}; drift QP refits=0", flush=True)
    if not processed: raise RuntimeError("No eligible historical drift configurations found.")
    historical_summary = grid_summary(historical_pool)
    historical_summary.to_csv(output/"diffusion_historical_summary.csv", index=False)
    plot_grid(historical_summary, output/"diffusion_historical_heatmaps.png",
              "historical in-sample", names, ns, ks)
    plot_parameters(historical_summary, output/"diffusion_parameter_heatmaps.png", names, ns, ks)
    if needs_target:
        target_summary = grid_summary(oracle_pool)
        target_summary.to_csv(output/"diffusion_oracle_summary.csv", index=False)
        plot_grid(target_summary, output/"diffusion_oracle_heatmaps.png",
                  "oracle-conditioned target", names, ns, ks)
    info = {"source":str(source), "group":args.group, "evaluation":args.evaluation,
            "cohort":args.cohort, "matched_origins":len(matched) if matched is not None else None,
            "selected_configurations":processed, "scanned_configs":scanned,
            "steps_per_hour":steps, "selection_baselines":names,
            "N_range":[min(ns), max(ns)], "k_range":[min(ks), max(ks)],
            "distributions":"historical=in_sample; target=oracle_conditioned_not_operational",
            "ks":"binned_4096_edges_-8_8_approx_ks_distance_error_below_0.0016",
            "process_diffusion":"dR=sigma*dW; sigma2=q; observation_noise_variance=tau2",
            "scored_lead_hours":"1,...,k-1 (k-1 hourly scores per interval); hour 0 excluded from variance fitting, NLL, KS, coverage, and other residual summaries",
            "nll":"unfiltered marginal Gaussian; sum across scored k-1 lead hours, mean over intervals; variance floor 1e-12 for likelihood only",
            "drift_parameter_refits":0, "fit_method":"analytic_exact_two_variable_nonnegative_least_squares"}
    (output/"diffusion_configuration.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"[ARM diffusion] complete={processed:,}; scanned={scanned:,}; drift QP refits=0", flush=True)
    print(f"[ARM diffusion] saved={output}", flush=True)
    return historical_summary


def main():
    parser = ArgumentParser(description="Fit ARM constant Brownian and observation variances on saved hourly N-k historical fits; optional oracle-conditioned target evaluation.")
    parser.add_argument("--group", choices=("ALL", "MED"), default="ALL")
    parser.add_argument("--source", default=None, help="Saved original drift sweep output directory.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--evaluation", choices=("historical", "both"), default="both")
    parser.add_argument("--cohort", choices=("all", "matched"), default="all",
                        help="all: each cell's eligible origins; matched: only origins with every selected baseline/N/k.")
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=None)
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--min-weekdays", type=int, default=None)
    parser.add_argument("--max-weekdays", type=int, default=None)
    parser.add_argument("--min-horizon", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--max-configurations", type=int, default=0,
                        help="Only for small test runs; 0 processes all saved configurations.")
    args = parser.parse_args()
    if args.progress_every < 1 or args.max_configurations < 0:
        parser.error("progress-every must be positive and max-configurations nonnegative.")
    return run(args)


if __name__ == "__main__": main()

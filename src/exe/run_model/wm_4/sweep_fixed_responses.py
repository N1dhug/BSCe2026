"""Compare the 20 ordered fixed g(R)/h(U) pairs over 2-10 weekdays, 6-24h.

Fixed bases preserve a convex joint rate-derivative QP. The oracle F0
problem is nonlinear and is solved separately for every forecast horizon.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.exe.run_model.wm_4.moving_horizon_common import HOURS, group_frame, window_data
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    BASELINE_NAMES, baseline_from_history, RANKING_METRIC,
)
from src.model.wm_5 import wm_5, response_function
from src.params.wm_5 import joint_qp_params
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"fixed_response_sweep"
VERSION = "fixed_g_h_joint_qp_oracle_v1"
QUANTILES = (20, 40, 50, 60, 80)


def fixed_responses():
    k_exp, k_log = 2.0, 4.0
    exp_den, log_den = np.expm1(k_exp), np.log1p(k_log)

    exponential = response_function(
        "exponential",
        lambda x: np.sign(x)*np.expm1(k_exp*abs(x))/exp_den,
        lambda x: k_exp*np.exp(k_exp*abs(x))/exp_den,
        (float(k_exp/exp_den), float(k_exp*np.exp(k_exp)/exp_den)),
    )
    logarithmic = response_function(
        "logarithmic",
        lambda x: np.sign(x)*np.log1p(k_log*abs(x))/log_den,
        lambda x: k_log/((1.0+k_log*abs(x))*log_den),
        (float(k_log/((1.0+k_log)*log_den)), float(k_log/log_den)),
    )
    functions = [exponential, logarithmic]
    for name, b in (("polynomial_steep", 0.15),
                    ("polynomial_very_steep", 0.35),
                    ("polynomial_extremely_steep", 0.49)):
        a = 1.0+b
        functions.append(response_function(
            name,
            lambda x, a=a, b=b: a*x-b*x**3,
            lambda x, a=a, b=b: a-3.0*b*x*x,
            (1.0-2.0*b, a),
        ))
    return {function.name: function for function in functions}


def ordered_response_pairs():
    functions = fixed_responses()
    return [(g, h) for g in functions for h in functions if g != h]



def prep(frame, day, n, baseline_name, steps_per_hour):
    history = historical_weekdays(day, n)
    training, imputation = impute_hourly_days(
        frame, days=history, columns=("I", "U", "Temp"), require_days=True)
    baseline = baseline_from_history(training, history, baseline_name)
    historical = [window_data(training, past, baseline, steps_per_hour)[1] for past in history]
    _, future, _, observed = window_data(frame, day, baseline, steps_per_hour)
    idx = np.arange(HOURS, dtype=int)*steps_per_hour
    temps = training.reindex(pd.DatetimeIndex(history))["Temp"].to_numpy(dtype=float)
    origins = pd.DatetimeIndex(history)
    historical_R0 = np.array([dayfit["I"][0] for dayfit in historical]) - baseline.predict(origins, temps)
    R0 = float(np.median(historical_R0))
    if not -1.0 <= R0 <= 1.0: raise ValueError("Historical forecast initial R0 is outside [-1,1].")
    return {
        "history": history, "historical": historical, "future": future, "observed": observed,
        "baseline": baseline, "baseline_name": baseline_name, "imputation": imputation,
        "R0": R0, "indices": idx, "steps_per_hour": steps_per_hour,
        "base_hourly": np.asarray(future["O"])[idx],
        "price_hourly": np.asarray(future["U"])[idx],
    }


def simulate_batch(f0_grid, r0, p, g, h, price, baseline, steps_per_hour):
    """Exactly reproduce WM5/WM4 explicit Euler, batched over candidate F0."""
    seeds = np.asarray(f0_grid, dtype=float).reshape(-1)
    F, R = seeds.copy(), np.full(seeds.shape, float(r0), dtype=float)
    dt = 1.0/(HOURS*steps_per_hour)
    n = len(price)
    if n != HOURS*steps_per_hour+1 or len(baseline) != n:
        raise ValueError("Expected complete 24-hour fine-grid price and baseline signals.")
    hourly = np.empty((len(seeds), HOURS), dtype=float)
    for k in range(HOURS*steps_per_hour):
        if k % steps_per_hour == 0:
            hourly[:, k//steps_per_hour] = baseline[k]+R
        drift = p[0]*F + p[1]*g(R) + p[2]*h(price[k])
        next_F = F+dt*R
        R = R+dt*drift
        F = next_F
    if not np.isfinite(hourly).all() or not np.isfinite(F).all() or not np.isfinite(R).all():
        raise ValueError("Fixed-response dynamics diverged or became nonfinite.")
    return hourly


def oracle_horizons(observed, prepared, p, g, h, first_horizon, last_horizon, grid_points=17):
    """Bounded scalar oracle: global grid bracket + local bounded minimization.

    Unlike the linear WM4 model, nonlinear g makes I(F0) nonlinear.
    No global minimizer is claimed; record coarse-grid and refined cost.
    """
    seeds = np.linspace(-1.0, 1.0, grid_points)
    future, sph = prepared["future"], prepared["steps_per_hour"]
    candidates = simulate_batch(seeds, prepared["R0"], p, g.value, h.value,
                                future["U"], future["O"], sph)
    costs = np.cumsum((candidates-np.asarray(observed)[None, :])**2, axis=1)
    results = []
    for horizon in range(first_horizon, last_horizon+1):
        i = int(np.argmin(costs[:, horizon-1]))
        lower, upper = seeds[max(0, i-1)], seeds[min(grid_points-1, i+1)]
        best_f0, best_pred = float(seeds[i]), candidates[i, :horizon]
        best_sse = float(costs[i, horizon-1])
        def objective(f0):
            predicted = simulate_batch([f0], prepared["R0"], p, g.value, h.value,
                                       future["U"], future["O"], sph)[0, :horizon]
            return float(np.sum((predicted-observed[:horizon])**2))
        if lower < upper:
            opt = minimize_scalar(objective, bounds=(float(lower), float(upper)),
                                  method="bounded", options={"xatol": 2e-5, "maxiter": 45})
            if opt.success and opt.fun < best_sse:
                best_f0, best_sse = float(opt.x), float(opt.fun)
        predictions = simulate_batch([best_f0], prepared["R0"], p, g.value, h.value,
                                     future["U"], future["O"], sph)[0, :horizon]
        residual = predictions-observed[:horizon]
        baseline_error = prepared["base_hourly"][:horizon]-observed[:horizon]
        results.append({
            "horizon_hours": horizon, "oracle_F0": best_f0,
            "mse": float(np.mean(residual**2)), "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))), "bias": float(np.mean(residual)),
            "baseline_mse": float(np.mean(baseline_error**2)),
            "baseline_rmse": float(np.sqrt(np.mean(baseline_error**2))),
            "coarse_grid_sse": float(costs[i, horizon-1]),
            "oracle_solver": f"bounded_grid_{grid_points}_plus_local",
        })
    return results


def evaluate(prepared, day, n, g, h, first_horizon, last_horizon, grid_points=17):
    hist, future = prepared["historical"], prepared["future"]
    baseline = lambda hour: float(np.interp(hour, future["Time"], future["O"]))
    model = wm_5((0.0, 1.0), len(future["U"]), baseline, (g,), (h,))
    started = perf_counter()
    result = joint_qp_params(model).fit_many(
        hist, f_bounds=(-1.0, 1.0), stabilize=True, enforce_path=True)
    p = result["coefficients"]                # [theta_F, theta_R, theta_U]
    # The oracle is nonlinear in F0, even though the joint historical QP is convex.
    horizons = oracle_horizons(prepared["observed"], prepared, p, g, h,
                              first_horizon, last_horizon, grid_points)
    timing = perf_counter()-started
    base = {
        "day": day, "baseline": prepared["baseline_name"], "training_weekdays": n,
        "g_name": g.name, "h_name": h.name, "pair": f"g={g.name} | h={h.name}",
        "training_days": ";".join(f"{d:%Y-%m-%d}" for d in prepared["history"]),
        "theta_F": float(p[0]), "theta_R": float(p[1]), "theta_U": float(p[2]),
        "damping_slope_at_zero": float(p[1]*g.derivative(0.0)),
        "damping_ratio_at_zero": float(-p[1]*g.derivative(0.0)/(2*np.sqrt(-p[0]))),
        "qp_loss": float(result["qp_loss"]), "qp_design_rank": result["design_rank"],
        "qp_design_columns": result["design_columns"],
        "transition_count": result["transition_count"], "fit_and_oracle_seconds": timing,
        "historical_R0": prepared["R0"], "imputed_training_hours": prepared["imputation"]["imputed_hourly_slots"],
        "baseline_coefficients": json.dumps(prepared["baseline"].coef.tolist()),
        "baseline_metadata": json.dumps(prepared["baseline"].metadata, default=str),
        "historical_F0_mean": float(np.mean(result["F0"])),
        "historical_F0_std": float(np.std(result["F0"])),
    }
    fit_row = base.copy()
    scores = [{**{name: base[name] for name in ("day","baseline","training_weekdays",
                                               "g_name","h_name","pair","theta_F","theta_R",
                                               "theta_U","historical_R0")},
               **row} for row in horizons]
    states = [{"day": day, "baseline": prepared["baseline_name"], "training_weekdays": n,
               "g_name": g.name, "h_name": h.name, "historical_day": past,
               "historical_F0": float(f0), "historical_R0": float(R[0])}
              for past, f0, data in zip(prepared["history"], result["F0"], hist)
              for R in [data["I"]-data["O"]]]
    return fit_row, scores, states


def append_csv(path, rows):
    if rows: pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.is_file())


def load_csv(path, date_cols=("day",)):
    if not path.exists(): return pd.DataFrame()
    data = pd.read_csv(path)
    for name in date_cols:
        if name in data: data[name] = pd.to_datetime(data[name])
    return data


def aggregate(scores):
    rows = []
    for (g, h, n, horizon), grp in scores.groupby(
            ["g_name", "h_name", "training_weekdays", "horizon_hours"]):
        rmse = grp["rmse"].to_numpy(dtype=float)
        record = {"g_name": g, "h_name": h, "pair": f"g={g} | h={h}",
                  "training_weekdays": int(n), "horizon_hours": int(horizon),
                  "days": len(grp), "aggregate_rmse": float(np.sqrt(np.mean(grp["mse"]))),
                  "baseline_rmse": float(np.sqrt(np.mean(grp["baseline_mse"]))),
                  "mean_daily_rmse": float(np.mean(rmse)),
                  "median_daily_rmse": float(np.median(rmse))}
        for q in QUANTILES: record[f"rmse_q{q}"] = float(np.percentile(rmse, q))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["aggregate_rmse", "g_name", "h_name", "training_weekdays", "horizon_hours"]).reset_index(drop=True)


def matched_dates(scores, pairs, n_values, h_values):
    # Each date must contain all N, H for all 20 ordered response pairs.
    required = len(pairs)*len(n_values)*len(h_values)
    distinct = scores.drop_duplicates(["day", "g_name", "h_name", "training_weekdays", "horizon_hours"])
    eligible = distinct.groupby("day").size()
    return pd.DatetimeIndex(eligible.index[eligible == required]).sort_values()


def rank(scores, pairs, n_values, h_values):
    common = matched_dates(scores, pairs, n_values, h_values)
    basis = "strictly_matched" if len(common) else "unmatched_fallback"
    cohort = scores.loc[scores["day"].isin(common)] if len(common) else scores
    all_splits = aggregate(cohort)
    # Rank PAIRS by their own best weekday/horizon combination using pooled absolute error.
    winners = all_splits.sort_values("aggregate_rmse").drop_duplicates(
        ["g_name","h_name"], keep="first").reset_index(drop=True)
    winners.insert(0, "rank", np.arange(1, len(winners)+1))
    winners.insert(1, "ranking_basis", basis)
    return all_splits, winners, common, basis


def save_plots(all_splits, winners, output):
    fig, ax = plt.subplots(figsize=(15, 8))
    shown = winners.sort_values("aggregate_rmse").iloc[::-1]
    ax.barh(shown["pair"], shown["aggregate_rmse"])
    ax.set_title("20 ordered nonlinear response pairs: best pooled oracle RMSE")
    ax.set_xlabel("Lowest pooled RMSE over N=2–10 and H=6–24")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(output/"response_pair_ranking.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    catalog = fixed_responses()
    nodes = np.linspace(-1.0, 1.0, 301)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, func in catalog.items(): ax.plot(nodes, func.value(nodes), label=name)
    ax.set_xlabel("x in [-1,1]"); ax.set_ylabel("Fixed response f(x)")
    ax.set_title("Five admissible monotone normalized response functions")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"response_functions.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    top = winners.head(min(5, len(winners)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for row in top.itertuples(index=False):
        data = all_splits.loc[(all_splits["g_name"] == row.g_name) &
                              (all_splits["h_name"] == row.h_name)]
        by_h = data.sort_values("aggregate_rmse").drop_duplicates("horizon_hours").sort_values("horizon_hours")
        ax.plot(by_h["horizon_hours"], by_h["aggregate_rmse"],
                label=f"g={row.g_name}, h={row.h_name}", marker=".")
    ax.set_xlabel("Forecast horizon H (hours)"); ax.set_ylabel("Best pooled RMSE across N")
    ax.set_title("Five strongest response pairs: horizon sensitivity")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"top_five_horizon_curves.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(19, 13), sharex=True, sharey=True)
    shared_min = float(all_splits["aggregate_rmse"].min())
    shared_max = float(all_splits["aggregate_rmse"].max())
    color_image = None
    for ax, (g, h) in zip(axes.flat, ordered_response_pairs()):
        group = all_splits.loc[(all_splits["g_name"] == g) & (all_splits["h_name"] == h)]
        table = group.pivot(index="training_weekdays", columns="horizon_hours", values="aggregate_rmse")
        if not table.empty:
            color_image = ax.imshow(table, aspect="auto", origin="lower",
                      vmin=shared_min, vmax=shared_max, interpolation="nearest",
                      extent=[min(table.columns)-.5, max(table.columns)+.5,
                              min(table.index)-.5, max(table.index)+.5])
        ax.set_title(f"g={g}\nh={h}", fontsize=8)
    if color_image is not None: fig.colorbar(color_image, ax=axes.ravel().tolist(), label="Pooled RMSE", shrink=.8)
    fig.supxlabel("Prediction horizon H"); fig.supylabel("Historical corresponding weekdays N")
    fig.suptitle("Aggregate oracle RMSE by N, H and ordered response pair")
    fig.savefig(output/"twenty_response_heatmaps.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def report(output, args):
    scores = load_csv(output/"horizon_scores.csv")
    fits = load_csv(output/"parameter_fits.csv")
    if scores.empty: raise RuntimeError("No successful fits or scores saved.")
    scores = scores.drop_duplicates(["day","g_name","h_name","training_weekdays","horizon_hours"], keep="last")
    fits = fits.drop_duplicates(["day","g_name","h_name","training_weekdays"], keep="last")
    scores.to_csv(output/"horizon_scores.csv", index=False)
    fits.to_csv(output/"parameter_fits.csv", index=False)
    pairs = ordered_response_pairs()
    n_values = tuple(range(args.min_weekdays,args.max_weekdays+1))
    h_values = tuple(range(args.min_horizon,args.max_horizon+1))
    all_splits, winners, common, basis = rank(scores, pairs, n_values, h_values)
    all_splits.to_csv(output/"all_weekday_horizon_splits.csv", index=False)
    winners.to_csv(output/"response_pair_ranking.csv", index=False)
    if len(winners): save_plots(all_splits, winners, output)
    print("\nWM4/WM5 fixed nonlinear response g(R) and h(U) comparison")
    print(f"Successful parameter fits: {len(fits)}, scores: {len(scores)}, matched days: {len(common)}")
    if basis != "strictly_matched":
        print("WARNING: No days have all pairs/N/H available; ranking uses unequal cohorts.")
    if len(winners) != len(pairs):
        print(f"WARNING: Only {len(winners)} of {len(pairs)} response pairs were successfully fitted.")
    print("\nALL ORDERED PAIRS RANKED BY LOWEST ABSOLUTE AGGREGATE ORACLE RMSE:")
    print(winners[["rank","g_name","h_name","training_weekdays","horizon_hours",
                   "days","aggregate_rmse","baseline_rmse","median_daily_rmse"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved full parameterizations, horizons, fit states and plots: {output}")
    print("NOTE: H-specific oracle F0 uses target-day demand and is NOT a deployable forecast.")
    return winners


def run(args):
    if not 2 <= args.min_weekdays <= args.max_weekdays <= 10: raise ValueError("Weekdays must be within [2,10].")
    if not 6 <= args.min_horizon <= args.max_horizon <= 24: raise ValueError("Horizon must lie within [6,24].")
    if args.steps_per_hour < 2 or args.max_days < 0 or args.grid_points < 5 or args.progress_every < 1:
        raise ValueError("Invalid numerical settings.")
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()/args.baseline
    output.mkdir(parents=True, exist_ok=True)
    config = {"version": VERSION, "group": args.group, "baseline": args.baseline,
              "min_weekdays": args.min_weekdays, "max_weekdays": args.max_weekdays,
              "min_horizon": args.min_horizon, "max_horizon": args.max_horizon,
              "steps_per_hour": args.steps_per_hour, "grid_points": args.grid_points,
              "start_day": args.start_day, "end_day": args.end_day}
    file = output/"configuration.json"
    funcs = fixed_responses()
    specifications = {
        "exponential": "sign(x)*expm1(2*abs(x))/expm1(2)",
        "logarithmic": "sign(x)*log1p(4*abs(x))/log1p(4)",
        "polynomial_steep": "1.15*x-0.15*x^3",
        "polynomial_very_steep": "1.35*x-0.35*x^3",
        "polynomial_extremely_steep": "1.49*x-0.49*x^3",
    }
    pd.DataFrame([{"name": name, "formula": specifications[name],
                   "derivative_min": func.derivative_bounds[0],
                   "derivative_max": func.derivative_bounds[1],
                   "center_slope": float(func.derivative(0.0)),
                   "value_at_minus_one": float(func.value(-1.0)),
                   "value_at_zero": float(func.value(0.0)),
                   "value_at_plus_one": float(func.value(1.0))}
                  for name, func in funcs.items()]).to_csv(
                      output/"response_library.csv", index=False)
    if args.report_only:
        if not file.is_file() or json.loads(file.read_text(encoding="utf-8")) != config:
            raise ValueError("Report settings do not match saved experiment configuration.")
        return report(output, args)
    if args.resume:
        if not file.is_file() or json.loads(file.read_text(encoding="utf-8")) != config:
            raise ValueError("Cannot resume with differing saved settings.")
        existing = load_csv(output/"parameter_fits.csv")
        previous_scores = load_csv(output/"horizon_scores.csv")
        if existing.empty or previous_scores.empty: done = set()
        else:
            counts = previous_scores.groupby(["day","g_name","h_name","training_weekdays"])["horizon_hours"].nunique()
            done = {(d,g,h,int(n)) for d,g,h,n in
                    zip(existing["day"],existing["g_name"],existing["h_name"],existing["training_weekdays"])
                    if counts.get((d,g,h,n),0) == args.max_horizon-args.min_horizon+1}
    else:
        for filename in ("parameter_fits.csv", "horizon_scores.csv",
                         "historical_states.csv", "skipped_fits.csv"):
            (output/filename).unlink(missing_ok=True)
        file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        done = set()

    frame = group_frame(args.group)
    first = max(frame.index.min().normalize(), pd.Timestamp("2022-01-01")) + pd.Timedelta(weeks=args.max_weekdays)
    last = min(frame.index.max().normalize(), pd.Timestamp("2022-10-31"))
    if args.start_day: first = max(first, pd.Timestamp(args.start_day))
    if args.end_day: last = min(last, pd.Timestamp(args.end_day))
    days = pd.date_range(first,last,freq="D")
    if args.max_days: days=days[:args.max_days]
    if days.empty: raise ValueError("No days remain after historical weekday warm-up.")
    pairs = ordered_response_pairs()
    catalog = fixed_responses()
    attempts = successes = 0
    for day in days:
        for n in range(args.min_weekdays,args.max_weekdays+1):
            try:
                prepared = prep(frame,day,n,args.baseline,args.steps_per_hour)
            except (ValueError,RuntimeError) as exc:
                for g_name,h_name in pairs:
                    if (day,g_name,h_name,n) not in done:
                        append_csv(output/"skipped_fits.csv",[{"day":day,"training_weekdays":n,
                            "g_name":g_name,"h_name":h_name,"error":str(exc)}])
                continue
            for g_name,h_name in pairs:
                if (day,g_name,h_name,n) in done: continue
                attempts += 1
                try:
                    fit, scores, states = evaluate(prepared,day,n,catalog[g_name],catalog[h_name],
                                                   args.min_horizon,args.max_horizon,args.grid_points)
                    append_csv(output/"parameter_fits.csv",[fit])
                    append_csv(output/"horizon_scores.csv",scores)
                    append_csv(output/"historical_states.csv",states)
                    successes += 1
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    if args.fail_fast: raise
                    append_csv(output/"skipped_fits.csv",[{"day":day,"training_weekdays":n,
                        "g_name":g_name,"h_name":h_name,"error":str(exc)}])
                if attempts % args.progress_every == 0:
                    print(f"[fixed responses] {day:%Y-%m-%d} N={n}: {successes}/{attempts} fits "
                          f"(plus {len(done)} previously completed)",flush=True)
    return report(output,args)


def main():
    parser = ArgumentParser(description="20 fixed ordered g/h pairs, 2–10 matching weekdays, 6–24h oracle horizons.")
    parser.add_argument("--group", choices=("MED","ALL"), default="MED")
    parser.add_argument("--baseline", choices=BASELINE_NAMES, default="weekly_temperature")
    parser.add_argument("--min-weekdays", type=int, default=2)
    parser.add_argument("--max-weekdays", type=int, default=10)
    parser.add_argument("--min-horizon", type=int, default=6)
    parser.add_argument("--max-horizon", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=17)
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()

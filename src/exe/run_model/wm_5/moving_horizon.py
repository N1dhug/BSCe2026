from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.method.moving_horizon import GROUPS, HOURS, baseline_before, group_frame, metrics, origins, save_table, window_data
from src.model.wm_5 import response_library, wm_5
from src.params.wm_4 import optimized_qp_params as wm4_optimized_qp
from src.params.wm_5 import qp_params as wm5_qp_params
from src.validation.exe.run_model.wm_5.moving_horizon import moving_horizon_validation


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_5" / "moving_horizon"
CASES = {
    "linear_linear": (("linear",), ("linear",)),
    "poly3_linear": (("linear", "cubic"), ("linear",)),
    "poly5_linear": (("linear", "quintic"), ("linear",)),
    "exp_linear": (("linear", "exponential"), ("linear",)),
    "log_linear": (("linear", "logarithmic"), ("linear",)),
    "linear_poly3": (("linear",), ("linear", "cubic")),
    "linear_exp": (("linear",), ("linear", "exponential")),
    "linear_log": (("linear",), ("linear", "logarithmic")),
    "poly3_poly3": (("linear", "cubic"), ("linear", "cubic")),
    "exp_exp": (("linear", "exponential"), ("linear", "exponential")),
    "log_log": (("linear", "logarithmic"), ("linear", "logarithmic")),
    "poly3_exp": (("linear", "cubic"), ("linear", "exponential")),
}
WM4_NAME = "wm4_linear"


class wm4_fixed_rate_qp(wm4_optimized_qp):
    """WM4 alternating parameter-QP and scalar F0-QP with observed R0 held fixed."""

    def _optimize_initial_state(self, data, r, seed, f_bounds=(-1.0, 1.0), r_bounds=(-np.inf, np.inf)):
        R0 = float(data["I"][0] - data["O"][0])
        if not np.isclose(float(seed[1]), R0, atol=1e-10):
            raise ValueError("The fair WM4 comparator must keep R0=Iobs[0]-O[0].")
        obs = self._observed_state(data, [0.0, R0])
        mask = obs["transition_mask"]
        residual = obs["y"][mask] - obs["X"][mask] @ r
        rFdt = self.model.dt * float(r[0])
        if abs(rFdt) < 1e-14: raise RuntimeError("WM4 flexibility coefficient is too close to zero.")
        f0 = float(np.clip(np.mean(residual) / rFdt, *f_bounds))
        return np.array([f0, R0])


def model_variant(reference, case, library):
    r_names, u_names = CASES[case]
    return wm_5(reference.T, reference.t_steps, reference.O,
                [library[name] for name in r_names], [library[name] for name in u_names])


def fit_window(reference, data, cases, library, max_state_iter, state_tol, fail_fast):
    """Identical observed R0, baseline, time grid and F0-only alternating QP for WM4 and WM5."""
    R0 = float(data["I"][0] - data["O"][0])
    fits, failures = {}, []
    try:
        # WM4 uses its existing structural parameter QP, but its state block
        # is the same scalar exact F0 QP as WM5, with R0 observed and fixed.
        fit = wm4_fixed_rate_qp(reference).fit(
            data, [0.0, R0], fit_initial_state=True,
            f_bounds=(-1.0, 1.0), r_bounds=(R0, R0),
            max_state_iter=max_state_iter, state_tol=state_tol,
        )
        history = fit.get("state_optimization_history", [])
        fit["state_converged"] = bool(history and
            history[-1]["max_state_change"] <= state_tol and
            history[-1]["loss_change"] <= state_tol)
        fits[WM4_NAME] = (reference, fit)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        if fail_fast: raise
        failures.append({"variant": WM4_NAME, "stage": "fit", "error": str(exc)})

    for name in cases:
        try:
            model = model_variant(reference, name, library)
            fit = wm5_qp_params(model).fit(
                data, initial_conditions=[0.0, R0], max_state_iter=max_state_iter,
                state_tol=state_tol,
            )
            fits[name] = (model, fit)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            failures.append({"variant": name, "stage": "fit", "error": str(exc)})
    return fits, failures


def evaluate_window(group, origin, name, model, fit, train_observed, forecast_data,
                    forecast_observed, steps_per_hour, oracle_starts, oracle_nfev):
    idx = np.arange(HOURS, dtype=int) * steps_per_hour
    fitting = metrics(train_observed, fit["deterministic_state"]["I"][idx])
    started = perf_counter()
    if name == WM4_NAME:
        oracle = wm4_optimized_qp(model).optimize_prediction_initial_state(
            forecast_data, fit["r"], f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0))
    else:
        oracle = wm5_qp_params(model).optimize_prediction_initial_state(
            forecast_data, fit["coefficients"], starts=oracle_starts, max_nfev=oracle_nfev)
    oracle_time = perf_counter() - started
    prediction = oracle["deterministic_state"]["I"][idx]
    baseline = np.asarray(forecast_data["O"][idx], dtype=float)
    score = metrics(forecast_observed, prediction)
    baseline_score = metrics(forecast_observed, baseline)
    errors = prediction - forecast_observed
    seed, initial = np.asarray(fit["initial_condition_seed"]), np.asarray(fit["init_cond"])
    info = {
        "group": group, "origin": origin, "variant": name,
        "fit_mse": fitting["mse"], "fit_rmse": fitting["rmse"],
        "fit_bias": fitting["bias"], "fit_mae": fitting["mae"],
        "qp_loss": float(fit["qp_loss"]), "fit_time": float(fit.get("fit_time", fit.get("qp_fit_time", np.nan))),
        "qp_solve_time": float(fit["qp_solve_time"]),
        "state_iterations": int(fit["state_iterations"]),
        "state_converged": bool(fit["state_converged"]),
        "F0_seed": float(seed[0]), "F0_fit": float(initial[0]),
        "R0_fit": float(initial[1]),
        "oracle_F0": float(oracle["init_cond"][0]), "oracle_R0": float(oracle["init_cond"][1]),
        "oracle_time": oracle_time, "oracle_rmse": score["rmse"],
        "oracle_mse": score["mse"], "oracle_bias": score["bias"], "oracle_mae": score["mae"],
        "baseline_rmse": baseline_score["rmse"], "baseline_mse": baseline_score["mse"],
    }
    coefficients = fit["r"] if name == WM4_NAME else fit["coefficients"]
    names = ("r_F", "r_U", "r_R") if name == WM4_NAME else fit["coefficient_names"]
    for label, coefficient in zip(names, coefficients): info["coefficient_" + label] = float(coefficient)
    horizon = {"group": group, "origin": origin, "variant": name}
    for hour, error in enumerate(errors, start=1): horizon[f"error_{hour:02d}"] = float(error)
    return info, horizon


def summaries(frame, errors):
    if frame.empty: return pd.DataFrame(), pd.DataFrame()
    rows, horizon = [], []
    for (group, variant), subset in frame.groupby(["group", "variant"]):
        matched_error = errors.loc[(errors["group"] == group) & (errors["variant"] == variant)]
        row = {"group": group, "variant": variant, "windows": len(subset),
               "aggregate_rmse": float(np.sqrt(subset["oracle_mse"].mean())),
               "mean_window_rmse": float(subset["oracle_rmse"].mean()),
               "median_window_rmse": float(subset["oracle_rmse"].median()),
               "baseline_rmse": float(np.sqrt(subset["baseline_mse"].mean())),
               "mean_fit_rmse": float(subset["fit_rmse"].mean()),
               "mean_qp_loss": float(subset["qp_loss"].mean()),
               "mean_fit_time": float(subset["fit_time"].mean()),
               "mean_oracle_time": float(subset["oracle_time"].mean()),
               "converged_pct": float(100 * subset["state_converged"].mean()),
               "mean_bias": float(subset["oracle_bias"].mean())}
        for quantile in (0.2, 0.4, 0.6, 0.8, 0.9):
            row[f"q{int(100*quantile)}"] = float(subset["oracle_rmse"].quantile(quantile))
        rows.append(row)
        for hour in range(1, HOURS+1):
            values = matched_error[f"error_{hour:02d}"].to_numpy(dtype=float)
            horizon.append({"group": group, "variant": variant, "horizon": hour,
                            "rmse": float(np.sqrt(np.mean(values ** 2))),
                            "mae": float(np.mean(np.abs(values))),
                            "bias": float(np.mean(values)), "windows": len(values)})
    return pd.DataFrame(rows).sort_values(["group", "aggregate_rmse"]), pd.DataFrame(horizon)


def save_plots(summary, horizon, output, *, max_lines=6):
    for group in GROUPS:
        scored = summary.loc[summary["group"] == group].sort_values("aggregate_rmse")
        if scored.empty: continue
        variants = list(scored["variant"].head(max_lines))
        if WM4_NAME in scored["variant"].values and WM4_NAME not in variants: variants[-1] = WM4_NAME
        subset = horizon.loc[(horizon["group"] == group) & (horizon["variant"].isin(variants))]
        dest = output / group.lower(); dest.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, values in subset.groupby("variant"):
            ax.plot(values["horizon"], values["rmse"], label=name)
        ax.set_title(f"{group}: 24-hour oracle prediction RMSE (common forecast origins)")
        ax.set_xlabel("Forecast hour"); ax.set_ylabel("RMSE"); ax.grid(True, alpha=0.25); ax.legend()
        fig.tight_layout(); fig.savefig(dest / "oracle_rmse_by_horizon.png", dpi=180); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        selected = scored.sort_values("mean_fit_rmse")
        x = np.arange(len(selected))
        axes[0].bar(x, selected["mean_fit_rmse"]); axes[0].set_ylabel("Mean training RMSE")
        axes[1].bar(x, selected["mean_fit_time"]); axes[1].set_ylabel("Mean fitting time (s)")
        axes[1].set_xticks(x, selected["variant"], rotation=60, ha="right")
        fig.suptitle(f"{group}: WM4 versus WM5 fitting accuracy and runtime")
        fig.tight_layout(); fig.savefig(dest / "fit_rmse_runtime.png", dpi=180); plt.close(fig)


def run(args):
    moving_horizon_validation.arguments(args, CASES)
    library = response_library()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    cases = list(CASES) if args.variants.lower() == "all" else [x.strip() for x in args.variants.split(",")]
    fit_rows, horizon_rows, failures = [], [], []
    timeline = origins(args.start, args.end, args.stride)
    if args.max_origins: timeline = timeline[:args.max_origins]

    for group in args.groups.split(","):
        group = group.strip().upper()
        frame, baselines = group_frame(group), {}
        for i, origin in enumerate(timeline, start=1):
            try:
                baseline = baseline_before(frame, origin, baselines)
                reference, data, _, historical = window_data(
                    frame, origin - pd.Timedelta(hours=HOURS), baseline, args.steps_per_hour)
                _, future_data, _, future = window_data(frame, origin, baseline, args.steps_per_hour)
                fits, errors = fit_window(reference, data, cases, library, args.max_state_iter,
                                          args.state_tol, args.fail_fast)
                for error in errors: failures.append({"group": group, "origin": origin, **error})
                for name, (model, fit) in fits.items():
                    try:
                        score, hourly = evaluate_window(
                            group, origin, name, model, fit, historical, future_data, future,
                            args.steps_per_hour, args.oracle_starts, args.oracle_nfev)
                        fit_rows.append(score); horizon_rows.append(hourly)
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                        if args.fail_fast: raise
                        failures.append({"group": group, "origin": origin, "variant": name,
                                         "stage": "oracle", "error": str(exc)})
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                if args.fail_fast: raise
                failures.append({"group": group, "origin": origin, "variant": "*",
                                 "stage": "window", "error": str(exc)})
            if i % 24 == 0: print(f"[wm5 horizon] {group}: {i}/{len(timeline)} origins")

    save_table(fit_rows, output / "fit_parameters.csv")
    save_table(horizon_rows, output / "oracle_horizon_errors.csv")
    save_table(failures, output / "skipped.csv")
    if not fit_rows: raise RuntimeError("No fitting windows completed WM5/WM4 oracle evaluation.")
    frame, errors = pd.DataFrame(fit_rows), pd.DataFrame(horizon_rows)
    summary, horizon = summaries(frame, errors)
    summary.to_csv(output / "oracle_summary_all_available.csv", index=False)
    horizon.to_csv(output / "oracle_horizon_all_available.csv", index=False)

    # Strict matched cohort: each group-origin must contain WM4 and *all*
    # requested WM5 variants, so baseline and sample sizes agree exactly.
    expected = 1 + len(cases)
    available = frame.groupby(["group", "origin"])["variant"].nunique()
    matched_keys = available.loc[available == expected].index
    common = frame.set_index(["group", "origin"]).loc[
        frame.set_index(["group", "origin"]).index.isin(matched_keys)].reset_index()
    common_errors = errors.set_index(["group", "origin"]).loc[
        errors.set_index(["group", "origin"]).index.isin(matched_keys)].reset_index()
    if common.empty: raise RuntimeError("No common forecast origins across all selected WM4/WM5 variants.")
    common_summary, common_horizon = summaries(common, common_errors)
    common.to_csv(output / "matched_oracle_window_metrics.csv", index=False)
    common_summary.to_csv(output / "matched_oracle_summary.csv", index=False)
    common_horizon.to_csv(output / "matched_oracle_horizon.csv", index=False)
    save_plots(common_summary, common_horizon, output)
    print(f"WM5 results: {output}; matched origins={len(matched_keys)}; failures={len(failures)}")
    print(common_summary.to_string(index=False))
    return common_summary


def main():
    parser = ArgumentParser(description="Fair hourly moving 24h WM4-vs-WM5 nonlinear rate/price response experiment.")
    parser.add_argument("--start", default="2022-02-28")
    parser.add_argument("--end", default="2022-08-22 23:00")
    parser.add_argument("--groups", default="ALL,MED")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--variants", default="all", help="Comma-separated keys from CASES, or 'all'.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--oracle-starts", type=int, default=2)
    parser.add_argument("--oracle-nfev", type=int, default=50)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__": main()

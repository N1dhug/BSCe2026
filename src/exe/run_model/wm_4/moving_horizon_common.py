from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data
from src.model.wm_4 import wm_4
from src.tools.baseline import output_callable, statistical_output


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "moving_horizon"
GROUPS = ("ALL", "MED")
HOURS = 24


def group_frame(group):
    if group not in GROUPS: raise ValueError(f"group must be one of {GROUPS}.")
    frame, _ = aligned_wm4_data(None if group == "ALL" else "MED")
    return frame


def origins(start, end, stride=1):
    start, end, stride = pd.Timestamp(start), pd.Timestamp(end), int(stride)
    if end < start or stride < 1: raise ValueError("end must be >= start and stride must be positive.")
    return pd.date_range(start, end, freq=f"{stride}h")


def baseline_before(frame, origin, cache):
    day = pd.Timestamp(origin).normalize()
    if day not in cache: cache[day] = statistical_output().fit(frame, day)
    return cache[day]


def window_data(frame, start, baseline, steps_per_hour):
    start, steps_per_hour = pd.Timestamp(start), int(steps_per_hour)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")
    index = pd.date_range(start, periods=HOURS, freq="h")
    source = frame.reindex(index)[["I", "U", "Temp"]]
    if source.isna().any().any(): raise ValueError(f"Missing consumption, price or temperature in {start} -> {index[-1]}.")

    n = HOURS * steps_per_hour + 1
    elapsed = np.arange(n, dtype=float) / steps_per_hour
    fine_index = start + pd.to_timedelta(elapsed, unit="h")
    hourly_I = source["I"].to_numpy(dtype=float)
    hourly_U = source["U"].to_numpy(dtype=float)
    hourly_T = source["Temp"].to_numpy(dtype=float)
    # Do not use the unknown next-hour consumption or exogenous signals.
    I = np.interp(elapsed, np.arange(HOURS + 1), np.r_[hourly_I, hourly_I[-1]])
    Temp = np.interp(elapsed, np.arange(HOURS + 1), np.r_[hourly_T, hourly_T[-1]])
    U = np.r_[hourly_U, hourly_U[-1]][np.minimum(np.floor(elapsed + 1e-10).astype(int), HOURS)]
    O = baseline.predict(fine_index, Temp)
    obs = np.zeros(n, dtype=bool)
    obs[np.arange(HOURS) * steps_per_hour] = True
    transitions = np.zeros(n - 1, dtype=bool)
    transitions[:(HOURS - 1) * steps_per_hour] = True
    data = {
        "I": I, "U": U, "PE": U.copy(), "Time": elapsed,
        "O": O, "observation_mask": obs, "transition_mask": transitions,
    }
    model = wm_4((0.0, 1.0), n, output_callable(fine_index, O))
    return model, data, fine_index, hourly_I


def metrics(observed, predicted):
    error = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    if error.size < 1 or not np.all(np.isfinite(error)): raise ValueError("Metrics require finite observations.")
    return {
        "mse": float(np.mean(error ** 2)), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
    }


def fit_row(group, origin, variant, fit, observed, steps_per_hour, f0_grid=np.nan):
    state = fit["deterministic_state"]
    prediction = np.asarray(state["I"])[np.arange(HOURS) * int(steps_per_hour)]
    quality = metrics(observed, prediction)
    r = np.asarray(fit["r"], dtype=float)
    init = np.asarray(fit["init_cond"], dtype=float)
    seed = np.asarray(fit.get("initial_condition_seed", init), dtype=float)
    history = fit.get("state_optimization_history", [])
    final_iteration = history[-1] if history else None
    state_converged = (bool(final_iteration is not None and final_iteration["max_state_change"] <= fit.get("state_tol", 1e-8) and final_iteration["loss_change"] <= fit.get("state_tol", 1e-8))
                       if fit.get("fit_initial_state", False) else np.nan)
    return {
        "group": group, "origin": pd.Timestamp(origin), "variant": variant,
        "method": fit["fit_method"], "success": bool(fit.get("success", True)),
        "r_0": float(fit.get("r_0", 0.0)), "r_F": float(r[0]), "r_U": float(r[1]), "r_R": float(r[2]),
        "q_F": float(fit["q"][0]), "q_U": float(fit["q"][1]), "kappa_R": float(fit["kappa_R"]),
        "F_0": float(init[0]), "R_0": float(init[1]),
        "F_0_seed": float(seed[0]), "R_0_seed": float(seed[1]),
        "F_0_grid": float(f0_grid),
        "fit_initial_state": bool(fit.get("fit_initial_state", False)),
        "state_iterations": int(fit.get("state_iterations", 0)),
        "state_converged": state_converged,
        "fit_intercept": bool(fit.get("fit_intercept", False)),
        "fit_mse": quality["mse"], "fit_rmse": quality["rmse"],
        "fit_mae": quality["mae"], "fit_bias": quality["bias"],
        "qp_loss": float(fit.get("qp_loss", np.nan)),
        "fit_time": float(fit.get("fit_time", fit.get("qp_fit_time", np.nan))),
        "qp_solve_time": float(fit.get("qp_solve_time", np.nan)),
        "qp_seed_time": float(fit.get("qp_seed_time", np.nan)),
        "numerical_solve_time": float(fit.get("numerical_solve_time", np.nan)),
        "iterations": int(fit.get("iterations", 0)), "evaluations": int(fit.get("evaluations", 0)),
        "message": str(fit.get("message", "")),
    }


def save_table(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_parameter_time(frame, output, title):
    import matplotlib.pyplot as plt
    for parameter in ("r_0", "r_F", "r_U", "r_R"):
        fig, ax = plt.subplots(figsize=(15, 5))
        for variant, group in frame.groupby("variant"):
            ax.plot(group["origin"], group[parameter], label=variant, linewidth=1.0)
        ax.axhline(0.0, linestyle="--", alpha=0.4)
        ax.set_title(f"{title}: {parameter}"); ax.set_xlabel("Origin")
        ax.set_ylabel(parameter); ax.grid(True, alpha=0.2); ax.legend()
        fig.tight_layout(); fig.savefig(output / f"{parameter}_time.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5))
    for variant, group in frame.groupby("variant"):
        ax.plot(group["origin"], group["fit_rmse"], label=variant, linewidth=1.0)
    ax.set_title(f"{title}: free-run in-window RMSE"); ax.set_xlabel("Origin")
    ax.set_ylabel("RMSE"); ax.grid(True, alpha=0.2); ax.legend()
    fig.tight_layout(); fig.savefig(output / "fit_rmse_time.png", dpi=180); plt.close(fig)

"""Compare saved IM (BM1 beta) and ARM (WM4) on seven matched weekdays.

For every day plot three side-by-side panels:
    IM E[f(X_t)] and g(U_t) | observed/IM/ARM demand with 95% intervals
    | ARM theta_F*F_t, theta_R*R_t and theta_U*g(U_t).

The ARM contributions are read directly from saved hourly predictions. IM
contributions are reconstructed from its previously fitted parameters and
oracle initial state by *propagating the beta mixture*, without fitting or
optimizing anything. IM f(X_t) is shown as its predictive expectation because
X_t is stochastic. The ARM implementation has g(U_t)=U_t (called h(U) in the
WM4 executable); IM and ARM prices use their own original normalizations.

From the repository root:
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week 1 --show
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week 2
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week both
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.selected_bm_dates_arm import K, reference_dates
from src.method.bm_1_beta import bm_1_beta
from src.params.bm_1_beta import bm_1_beta_params


ROOT = Path(__file__).resolve().parents[4]
IM_INPUT = ROOT/"files"/"output"/"bm_1"/"selected_bm_dates_beta"/"hourly_predictions.csv"
ARM_INPUT = ROOT/"files"/"output"/"wm_4"/"selected_bm_dates_arm"/"hourly_predictions.csv"
OUTPUT = ROOT/"files"/"output"/"model_comparison"/"selected_bm_dates"
IM_PARAMETERS = ("alpha", "beta", "k", "Xi", "C", "sigma_x", "sigma_y", "oracle_X0")
ARM_CONTRIBUTIONS = ("theta_F_times_F", "theta_R_times_R", "theta_U_times_h_U")
ORACLE_DEFAULTS = {"components":24, "quadrature":12, "reconstruction_quadrature":48, "max_step":0.125}


def load_predictions(path, model, mean_col, extra=()):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{model} saved predictions not found: {path}\n"
                                "Run the corresponding selected_bm_dates_* estimation executable first.")
    frame = pd.read_csv(path)
    needed = {"day", "hour", "observed", mean_col, "lower_95", "upper_95", *extra}
    missing = needed.difference(frame.columns)
    if missing: raise ValueError(f"{model}: missing saved CSV columns: {sorted(missing)}")
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"], errors="raise").dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="raise")
    if not np.isfinite(frame["hour"]).all() or (frame["hour"]%1. != 0.).any():
        raise ValueError(f"{model}: invalid hourly indices.")
    frame["hour"] = frame["hour"].astype(int)
    if "baseline" in frame:
        frame = frame.loc[frame["baseline"]=="weekly_temperature"].copy()
    if frame.duplicated(["day", "hour"]).any():
        raise ValueError(f"{model}: duplicate (day,hour) entries in {path}.")
    for col in ("observed", mean_col, "lower_95", "upper_95", *extra):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{model}: nonfinite {col} values.")
    if (frame["lower_95"]>frame["upper_95"]).any():
        raise ValueError(f"{model}: reversed 95% uncertainty bounds.")
    return frame.set_index(["day", "hour"]).sort_index()


def comparison(im_path=IM_INPUT, arm_path=ARM_INPUT, week=1):
    if week not in (1, 2): raise ValueError("week must be 1 or 2.")
    dates, _ = reference_dates()
    selected = dates[(week-1)*7:week*7]
    im = load_predictions(im_path, "IM", "beta_expected_demand",
                          extra=("weekly_temperature_baseline", "price_U_0_1"))
    arm = load_predictions(arm_path, "ARM", "predicted", extra=ARM_CONTRIBUTIONS)
    keys = pd.MultiIndex.from_product([selected, range(K)], names=["day", "hour"])
    missing_im, missing_arm = keys.difference(im.index), keys.difference(arm.index)
    if len(missing_im) or len(missing_arm):
        raise ValueError(f"Saved predictions missing requested week {week} hourly rows: "
                         f"IM={list(missing_im[:5])}, ARM={list(missing_arm[:5])}.")
    i, a = im.loc[keys], arm.loc[keys]
    if not np.allclose(i["observed"], a["observed"], atol=1e-8, rtol=1e-8):
        difference = float(np.max(np.abs(i["observed"]-a["observed"])))
        raise ValueError(f"IM and ARM observed demands differ (max {difference:.6g}). "
                         "Check sample dates and normalization.")
    return selected, i, a


def load_im_fits(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IM saved parameter table not found: {path}. "
                                "The plotting executable requires the already fitted parameters.")
    frame = pd.read_csv(path)
    needed = {"day", *IM_PARAMETERS}
    missing = needed.difference(frame.columns)
    if missing: raise ValueError(f"IM parameters: missing columns {sorted(missing)} in {path}.")
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"], errors="raise").dt.normalize()
    if "baseline" in frame:
        frame = frame.loc[frame["baseline"]=="weekly_temperature"].copy()
    if frame["day"].duplicated().any(): raise ValueError("IM parameter table has duplicate day entries.")
    for name in IM_PARAMETERS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(f"IM parameters contain nonfinite {name}.")
    return frame.set_index("day")


def im_resolution(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IM saved configuration not found: {path}. "
                                "It is required to reproduce the original stochastic propagation settings.")
    config = json.loads(path.read_text(encoding="utf-8"))
    settings = config.get("oracle_resolution")
    if not isinstance(settings, dict):
        raise ValueError(f"IM configuration is missing oracle_resolution: {path}")
    resolution = {key:settings.get(key, value) for key, value in ORACLE_DEFAULTS.items()}
    return resolution


def im_responses(day, hourly, fit, resolution):
    """Reproduce open-loop IM density, then calculate E[f(X_t)] and g(U_t)."""
    b = hourly["weekly_temperature_baseline"].to_numpy(dtype=float)
    u = hourly["price_U_0_1"].to_numpy(dtype=float)
    if b.shape!=(K,) or u.shape!=(K,) or np.any((b<0.)|(b>1.)) or np.any((u<0.)|(u>1.)):
        raise ValueError(f"IM input signals are not K={K} normalized hourly values on {day:%Y-%m-%d}.")
    p = {key:float(fit[key]) for key in bm_1_beta_params.NAMES}
    p.update({key:float(fit[key]) for key in ("sigma_x","sigma_y")})
    x0 = float(fit["oracle_X0"])
    solver = bm_1_beta(int(resolution["components"]), int(resolution["quadrature"]),
                       int(resolution["reconstruction_quadrature"]), float(resolution["max_step"]))
    forecast = solver.simulate({"B":b, "U":u}, p, x0)
    expected = hourly["beta_expected_demand"].to_numpy(dtype=float)
    mismatch = float(np.max(np.abs(forecast["mean"]-expected)))
    if mismatch>1e-3:
        raise ValueError(f"IM recomputed predictive mean differs from saved forecast by {mismatch:.5g} "
                         f"on {day:%Y-%m-%d}; verify oracle resolution and saved parameters.")
    alpha, beta = p["alpha"], p["beta"]
    def response(value, shape):
        return np.tanh(0.5*shape*(0.5-np.asarray(value,dtype=float)))/np.tanh(shape/4.)
    f = np.empty(K,dtype=float)
    for h, mix in enumerate(forecast["states"]):
        states, weights = solver.support(mix)
        f[h] = float(weights@response(states, alpha))
    g = response(u, beta)
    return f, g


def plot_week(week=1, im_path=IM_INPUT, arm_path=ARM_INPUT, output=OUTPUT, show=False,
              im_params=None, im_config=None):
    days, im, arm = comparison(im_path, arm_path, week)
    im_params = Path(im_params) if im_params is not None else Path(im_path).parent/"per_day_metrics.csv"
    im_config = Path(im_config) if im_config is not None else Path(im_path).parent/"configuration.json"
    fits, resolution = load_im_fits(im_params), im_resolution(im_config)
    missing = [str(day.date()) for day in days if day not in fits.index]
    if missing: raise ValueError(f"IM fitted parameters missing days: {missing}")

    hours = np.arange(K)
    fig, axes = plt.subplots(7, 3, figsize=(25, 19), sharex="col",
                             gridspec_kw={"width_ratios":[1.0,1.7,1.25]})
    for row, day in enumerate(days):
        left, middle, right = axes[row]
        i, a = im.loc[day].sort_index(), arm.loc[day].sort_index()
        f, g = im_responses(day, i, fits.loc[day], resolution)
        left.plot(hours, f, color="tab:purple", lw=1.8, label=r"$E[f(X_t)]$")
        left.plot(hours, g, color="tab:green", lw=1.8, label=r"$g(U_t)$")
        left.axhline(0, color="0.35", alpha=0.5, lw=0.8)
        left.set_ylim(-1.05,1.05)

        middle.fill_between(hours, i["lower_95"].to_numpy(), i["upper_95"].to_numpy(),
                            color="tab:orange", alpha=0.16)
        middle.fill_between(hours, a["lower_95"].to_numpy(), a["upper_95"].to_numpy(),
                            color="tab:blue", alpha=0.13)
        middle.plot(hours, i["beta_expected_demand"].to_numpy(), color="tab:orange", lw=1.9)
        middle.plot(hours, a["predicted"].to_numpy(), color="tab:blue", lw=1.9)
        middle.plot(hours, i["observed"].to_numpy(), color="black", lw=1.6,
                    marker="o", markersize=2.8)
        middle.set_title(f"{day:%A %d %B %Y}", fontsize=10, pad=4)

        for key, color, label in (
            ("theta_F_times_F","tab:purple",r"$\theta_F F_t$"),
            ("theta_R_times_R","tab:green",r"$\theta_R R_t$"),
            ("theta_U_times_h_U","tab:red",r"$\theta_U g(U_t)$"),
        ):
            right.plot(hours, a[key].to_numpy(), color=color, lw=1.8, label=label)
        right.axhline(0, color="0.35", alpha=0.5, lw=0.8)

        for ax in (left, middle, right):
            ax.grid(alpha=0.2)
            ax.set_xlim(0, K-1)
            ax.tick_params(axis="both",labelsize=8)
        left.set_ylabel("Response", fontsize=9)
        middle.set_ylabel("Normalized demand", fontsize=9)
        right.set_ylabel("Rate contribution", fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks([0,4,8,12,16,20,23])
        ax.set_xlabel("Hour of day")

    fig.text(0.173,0.945,"IM responses",ha="center",va="top",fontweight="bold",fontsize=13)
    fig.text(0.505,0.945,"IM vs ARM demand fit",ha="center",va="top",fontweight="bold",fontsize=13)
    fig.text(0.835,0.945,"ARM drift contributions",ha="center",va="top",fontweight="bold",fontsize=13)

    im_handles = [Line2D([],[],color="tab:purple",lw=1.8,label=r"$E[f(X_t)]$"),
                  Line2D([],[],color="tab:green",lw=1.8,label=r"$g(U_t)$")]
    fit_handles = [Line2D([],[],color="black",lw=1.6,marker="o",markersize=3,label="Observed"),
                   Line2D([],[],color="tab:orange",lw=1.9,label="IM mean"),
                   Patch(color="tab:orange",alpha=0.16,label="IM 95%"),
                   Line2D([],[],color="tab:blue",lw=1.9,label="ARM mean"),
                   Patch(color="tab:blue",alpha=0.13,label="ARM 95%")]
    arm_handles = [Line2D([],[],color=color,lw=1.8,label=label) for color,label in (
                   ("tab:purple",r"$\theta_F F_t$"),("tab:green",r"$\theta_R R_t$"),
                   ("tab:red",r"$\theta_U g(U_t)$"))]
    fig.legend(handles=im_handles,loc="upper center",bbox_to_anchor=(0.173,0.925),
               ncol=2,fontsize=10,frameon=False)
    fig.legend(handles=fit_handles,loc="upper center",bbox_to_anchor=(0.505,0.925),
               ncol=5,fontsize=9,frameon=False)
    fig.legend(handles=arm_handles,loc="upper center",bbox_to_anchor=(0.835,0.925),
               ncol=3,fontsize=10,frameon=False)
    fig.suptitle(f"IM vs ARM | seven corresponding weekdays | sample week {week}\n"
                 "ALL · weekly temperature · N=6 historical weekdays · 24-hour oracle-conditioned predictions",
                 y=0.997,fontsize=15)
    fig.subplots_adjust(top=0.88,bottom=0.05,hspace=0.44,wspace=0.24)

    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    destination = output/f"im_arm_week_{week}_responses_and_demand_95pct.png"
    fig.savefig(destination,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

    observed = im["observed"].to_numpy(dtype=float)
    im_rmse = float(np.sqrt(np.mean((im["beta_expected_demand"].to_numpy()-observed)**2)))
    arm_rmse = float(np.sqrt(np.mean((arm["predicted"].to_numpy()-observed)**2)))
    im_coverage = 100.*float(np.mean((observed>=im["lower_95"].to_numpy())&
                                       (observed<=im["upper_95"].to_numpy())))
    arm_coverage = 100.*float(np.mean((observed>=arm["lower_95"].to_numpy())&
                                        (observed<=arm["upper_95"].to_numpy())))
    print(f"Week {week}: IM RMSE={im_rmse:.6f}, ARM RMSE={arm_rmse:.6f}; "
          f"IM 95% coverage={im_coverage:.2f}%, ARM 95% coverage={arm_coverage:.2f}%")
    print(f"Saved: {destination}")
    return destination


def main():
    parser = ArgumentParser(description="Plot IM responses, model demand fit, and ARM rate contributions for seven weekdays.")
    parser.add_argument("--week",choices=("1","2","both"),default="1",
                        help="First or second Monday–Sunday sample from the 14 fitted dates.")
    parser.add_argument("--im-input","--bm-input",dest="im_input",type=Path,default=IM_INPUT)
    parser.add_argument("--arm-input","--wm-input",dest="arm_input",type=Path,default=ARM_INPUT)
    parser.add_argument("--im-params",type=Path,default=None,
                        help="IM previously estimated per_day_metrics.csv (defaults beside IM hourly CSV).")
    parser.add_argument("--im-config",type=Path,default=None,
                        help="IM saved configuration.json containing oracle beta-mixture resolution.")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--show",action="store_true")
    args = parser.parse_args()
    for week in (1,2) if args.week=="both" else (int(args.week),):
        plot_week(week,args.im_input,args.arm_input,args.output,args.show,args.im_params,args.im_config)


if __name__=="__main__": main()

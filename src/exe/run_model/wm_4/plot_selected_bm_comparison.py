"""Compare seven IM and ARM prediction days and their response components.

Four figures are saved per requested Monday-Sunday week:
  1. Seven rows x two columns: observed demand, baseline, model mean and 95%
     prediction interval for IM (left) and ARM (right).
  2. Seven rows x two columns: instantaneous IM E[f(X_t)], g(U_t) (left)
     and numerically integrated ARM rate-drift terms (right).
  3. Seven vertically stacked IM response panels.
  4. Seven vertically stacked integrated ARM response panels.

IM state-response means are reconstructed from the saved fitted beta-mixture
parameters and oracle initial state, without parameter refitting. ARM rate
contributions are integrated using its original fine-step forward-Euler
dynamics, the saved oracle initial state, fitted theta coefficients and hourly
piecewise-constant prices. Time in the ARM ODE is measured in days, not hours.
Each ARM integrated term contributes to R(t)-R(0), not to R(t) itself.

From repository root:
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week 1 --show
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week 2
    python -m src.exe.run_model.wm_4.plot_selected_bm_comparison --week both
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
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
ARM_PARAMETERS = ("theta_F", "theta_R", "theta_U", "oracle_F0", "oracle_R0")
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
    arm = load_predictions(arm_path, "ARM", "predicted", extra=("baseline_predicted", "price_response_h_U", "oracle_F", "oracle_R", *ARM_CONTRIBUTIONS))
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


def load_arm_fits(path):
    """Load saved ARM fitted drift parameters and oracle initial states."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ARM saved parameter table not found: {path}. "
                                "Run the ARM selected_bm_dates estimation executable first.")
    frame = pd.read_csv(path)
    needed = {"day", *ARM_PARAMETERS}
    missing = needed.difference(frame.columns)
    if missing: raise ValueError(f"ARM parameters: missing columns {sorted(missing)} in {path}.")
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"], errors="raise").dt.normalize()
    if "baseline" in frame:
        frame = frame.loc[frame["baseline"]=="weekly_temperature"].copy()
    if frame["day"].duplicated().any(): raise ValueError("ARM parameter table has duplicate day entries.")
    for key in ARM_PARAMETERS:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
        if not np.isfinite(frame[key].to_numpy(dtype=float)).all():
            raise ValueError(f"ARM parameters contain nonfinite {key}.")
    return frame.set_index("day")


def arm_resolution(path):
    """Read fine Euler resolution used to create the saved ARM predictions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ARM saved configuration not found: {path}.")
    config = json.loads(path.read_text(encoding="utf-8"))
    value = config.get("steps_per_hour")
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"ARM configuration has invalid steps_per_hour: {value!r}.")
    return value


def arm_integrated_responses(day, hourly, fit, steps_per_hour):
    """Integrate each ARM rate derivative contribution with the saved Euler dynamics.

    ARM equations are dF/dt=R, dR/dt=theta_F F + theta_R R + theta_U U,
    where t is *normalized days*. A single hourly interval is 1/24 day.
    Integrals start from zero at the prediction origin and add up to R(t)-R0.
    """
    u = hourly["price_response_h_U"].to_numpy(dtype=float)
    f_saved = hourly["oracle_F"].to_numpy(dtype=float)
    r_saved = hourly["oracle_R"].to_numpy(dtype=float)
    theta_f, theta_r, theta_u = (float(fit[name]) for name in ("theta_F","theta_R","theta_U"))
    f, r = float(fit["oracle_F0"]), float(fit["oracle_R0"])
    dt = 1./(24.*steps_per_hour)
    integrated = np.zeros((K,3), dtype=float)
    total = np.zeros(3, dtype=float)
    for hour in range(K):
        # Save the contributions and compare reconstructed states at the same
        # observation times as the original ARM estimation executable.
        integrated[hour] = total
        if not np.allclose((f,r), (f_saved[hour],r_saved[hour]), rtol=2e-7, atol=2e-8):
            raise ValueError(f"ARM reconstruction differs from saved oracle state on "
                             f"{day:%Y-%m-%d} at hour {hour}; check step resolution.")
        if hour==K-1: break
        for _ in range(steps_per_hour):
            components = np.array([theta_f*f, theta_r*r, theta_u*u[hour]])
            total += dt*components
            # Original WM4 solver uses simultaneous forward-Euler state updates.
            f, r = f+dt*r, r+dt*float(components.sum())
    difference = r_saved-r_saved[0]
    if not np.allclose(integrated.sum(axis=1),difference,rtol=2e-7,atol=2e-8):
        raise ValueError(f"ARM numerical response integrals do not reconstruct "
                         f"R(t)-R(0) on {day:%Y-%m-%d}.")
    return integrated


def plot_week(week=1, im_path=IM_INPUT, arm_path=ARM_INPUT, output=OUTPUT, show=False,
              im_params=None, im_config=None, arm_params=None, arm_config=None):
    days, im, arm = comparison(im_path, arm_path, week)
    im_params = Path(im_params) if im_params is not None else Path(im_path).parent/"per_day_metrics.csv"
    im_config = Path(im_config) if im_config is not None else Path(im_path).parent/"configuration.json"
    fits, resolution = load_im_fits(im_params), im_resolution(im_config)
    missing = [str(day.date()) for day in days if day not in fits.index]
    if missing: raise ValueError(f"IM fitted parameters missing days: {missing}")

    hours = np.arange(K)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Main figure: IM on the left and ARM on the right. Both columns have
    # the same normalized-demand scale, but use their own saved baselines.
    fig, axes = plt.subplots(7, 2, figsize=(20, 20), sharex=True, sharey=True)
    for row, day in enumerate(days):
        for col, (model, data, baseline, mean, color) in enumerate((
            ("IM", im.loc[day].sort_index(), "weekly_temperature_baseline",
             "beta_expected_demand", "tab:orange"),
            ("ARM", arm.loc[day].sort_index(), "baseline_predicted",
             "predicted", "tab:blue"),
        )):
            ax = axes[row, col]
            ax.fill_between(hours, data["lower_95"].to_numpy(), data["upper_95"].to_numpy(),
                            color=color, alpha=0.18, label="Central 95% interval")
            ax.plot(hours, data["observed"].to_numpy(), color="black", lw=1.6,
                    marker="o", markersize=2.5, label="Observed demand")
            ax.plot(hours, data[baseline].to_numpy(), color="tab:green", lw=1.5,
                    ls="--", label="Weekly-temperature baseline")
            ax.plot(hours, data[mean].to_numpy(), color=color, lw=2.,
                    label=f"{model} expected demand")
            ax.set_title(f"{day:%A %d %B %Y} | {model}", fontsize=10, loc="left")
            ax.set_xlim(0, K-1)
            ax.set_ylabel("Normalized demand" if col==0 else "")
            ax.grid(alpha=0.22)
            ax.tick_params(axis="both", labelsize=9)
    for ax in axes[-1]:
        ax.set_xticks([0,4,8,12,16,20,23])
        ax.set_xlabel("Hour of day")
    for ax in axes[0]:
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.suptitle(f"IM (left) vs ARM (right) | 7-day demand fit | sample week {week}\n"
                 "ALL · weekly temperature · N=6 prior weekdays · oracle-conditioned 24-hour predictions",
                 fontsize=14, y=0.996)
    fig.subplots_adjust(top=0.94, bottom=0.05, hspace=0.43, wspace=0.13)
    demand_path = output/f"im_arm_week_{week}_demand_95pct.png"
    fig.savefig(demand_path, dpi=190, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

    # Function responses: IM is instantaneous, while the ARM terms are
    # individually integrated to give contributions to the *rate* R(t).
    # They are shown side by side, one weekday per row, in a second figure.
    arm_params = Path(arm_params) if arm_params is not None else Path(arm_path).parent/"per_day_metrics.csv"
    arm_config = Path(arm_config) if arm_config is not None else Path(arm_path).parent/"configuration.json"
    arm_fits, steps_per_hour = load_arm_fits(arm_params), arm_resolution(arm_config)
    missing_arm = [str(day.date()) for day in days if day not in arm_fits.index]
    if missing_arm: raise ValueError(f"ARM fitted parameters missing days: {missing_arm}")

    response_fig, response_axes = plt.subplots(7,2,figsize=(20,20),sharex="col")
    response_data = []
    for row, day in enumerate(days):
        i, a = im.loc[day].sort_index(), arm.loc[day].sort_index()
        f, g = im_responses(day, i, fits.loc[day], resolution)
        cumulative = arm_integrated_responses(day, a, arm_fits.loc[day], steps_per_hour)
        response_data.append((day,f,g,cumulative))
        left, right = response_axes[row]

        left.plot(hours, f, color="tab:purple", lw=2., label=r"$\mathbb{E}[f(X_t)]$")
        left.plot(hours, g, color="tab:green", lw=2., label=r"$g(U_t)$")
        left.set_ylim(-1.05,1.05)

        right.plot(hours,cumulative[:,0],color="tab:purple",lw=2.,
                   label=r"$\int_0^t \theta_F F_s\,ds$")
        right.plot(hours,cumulative[:,1],color="tab:green",lw=2.,
                   label=r"$\int_0^t \theta_R R_s\,ds$")
        right.plot(hours,cumulative[:,2],color="tab:red",lw=2.,
                   label=r"$\int_0^t \theta_U g(U_s)\,ds$")
        right.plot(hours,cumulative.sum(axis=1),color="0.35",lw=1.3,ls="--",
                   label=r"$R_t-R_0$")

        for ax, name, ylabel in ((left, "IM", "Direct response"),
                                 (right, "ARM", r"Integrated response ($\Delta R$)")):
            ax.axhline(0, color="0.4",lw=0.8,alpha=0.6)
            ax.set_title(f"{day:%A %d %B %Y} | {name}", loc="left", fontsize=10)
            ax.set_xlim(0,K-1)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
            ax.tick_params(axis="both", labelsize=9)

    for ax in response_axes[-1]:
        ax.set_xticks([0,4,8,12,16,20,23])
        ax.set_xlabel("Hour of day")
    for ax in response_axes[0]:
        ax.legend(loc="best",fontsize=8.3,framealpha=0.92)
    response_fig.suptitle(
        f"IM direct responses (left) vs ARM integrated rate contributions (right) | week {week}\n"
        "ARM time is in days; cumulative terms sum to R(t)-R(0); ARM g(U)=U",
        y=0.996,fontsize=14)
    response_fig.subplots_adjust(top=0.94,bottom=0.05,hspace=0.43,wspace=0.18)
    response_path = output/f"im_arm_week_{week}_responses_7x2.png"
    response_fig.savefig(response_path,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(response_fig)

    # Keep the model-specific response figures as well as the new side-by-side
    # layout. Reuse the computed arrays without rerunning IM propagation.
    fig_im, axes_im = plt.subplots(7,1,figsize=(12,19),sharex=True,sharey=True)
    fig_arm, axes_arm = plt.subplots(7,1,figsize=(12,19),sharex=True)
    for row, (day,f,g,cumulative) in enumerate(response_data):
        ax_i, ax_a = axes_im[row], axes_arm[row]
        ax_i.plot(hours,f,color="tab:purple",lw=1.9,label=r"$\mathbb{E}[f(X_t)]$")
        ax_i.plot(hours,g,color="tab:green",lw=1.9,label=r"$g(U_t)$")
        ax_i.set_ylim(-1.05,1.05)
        for col,color,label in (
            (0,"tab:purple",r"$\int_0^t \theta_F F_s\,ds$"),
            (1,"tab:green",r"$\int_0^t \theta_R R_s\,ds$"),
            (2,"tab:red",r"$\int_0^t \theta_U g(U_s)\,ds$"),
        ):
            ax_a.plot(hours,cumulative[:,col],color=color,lw=1.9,label=label)
        ax_a.plot(hours,cumulative.sum(axis=1),color="0.35",lw=1.3,ls="--",
                  label=r"$R_t-R_0$")
        for ax, ylabel in ((ax_i,"Direct response"),(ax_a,"Integrated response")):
            ax.set_xlim(0,K-1)
            ax.set_title(f"{day:%A %d %B %Y}",loc="left",fontsize=10)
            ax.set_ylabel(ylabel)
            ax.axhline(0,color="0.4",lw=0.8,alpha=0.6)
            ax.grid(alpha=0.22)
    for fig,axes1,title in (
        (fig_im,axes_im,"IM direct responses"),
        (fig_arm,axes_arm,"ARM numerically integrated rate contributions"),
    ):
        axes1[-1].set_xticks([0,4,8,12,16,20,23])
        axes1[-1].set_xlabel("Hour of day")
        handles,labels=axes1[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,0.952),
                   ncol=len(handles),fontsize=10,frameon=False)
        fig.suptitle(f"{title} | sample week {week}",y=0.996,fontsize=14)
        fig.subplots_adjust(top=0.915,bottom=0.05,hspace=0.43)
    im_path_out = output/f"im_week_{week}_function_contributions.png"
    arm_path_out = output/f"arm_week_{week}_function_contributions.png"
    fig_im.savefig(im_path_out,dpi=190,bbox_inches="tight")
    fig_arm.savefig(arm_path_out,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig_im)
    plt.close(fig_arm)

    observed = im["observed"].to_numpy(dtype=float)
    im_rmse = float(np.sqrt(np.mean((im["beta_expected_demand"].to_numpy()-observed)**2)))
    arm_rmse = float(np.sqrt(np.mean((arm["predicted"].to_numpy()-observed)**2)))
    im_coverage = 100.*float(np.mean((observed>=im["lower_95"].to_numpy())&
                                       (observed<=im["upper_95"].to_numpy())))
    arm_coverage = 100.*float(np.mean((observed>=arm["lower_95"].to_numpy())&
                                        (observed<=arm["upper_95"].to_numpy())))
    print(f"Week {week}: IM RMSE={im_rmse:.6f}, ARM RMSE={arm_rmse:.6f}; "
          f"IM 95% coverage={im_coverage:.2f}%, ARM 95% coverage={arm_coverage:.2f}%")
    for name, path in (("Demand",demand_path),("IM/ARM responses",response_path),
                       ("IM contributions",im_path_out),("ARM contributions",arm_path_out)):
        print(f"Saved {name}: {path}")
    return {"demand":demand_path, "responses":response_path,
            "im_contributions":im_path_out, "arm_contributions":arm_path_out}


def main():
    parser = ArgumentParser(description="Plot seven-day 7x2 IM/ARM demand fits and 7x2 direct/integrated function responses.")
    parser.add_argument("--week",choices=("1","2","both"),default="1",
                        help="First or second Monday–Sunday sample from the 14 fitted dates.")
    parser.add_argument("--im-input","--bm-input",dest="im_input",type=Path,default=IM_INPUT)
    parser.add_argument("--arm-input","--wm-input",dest="arm_input",type=Path,default=ARM_INPUT)
    parser.add_argument("--im-params",type=Path,default=None,
                        help="IM previously estimated per_day_metrics.csv (defaults beside IM hourly CSV).")
    parser.add_argument("--im-config",type=Path,default=None,
                        help="IM saved configuration.json containing oracle beta-mixture resolution.")
    parser.add_argument("--arm-params",type=Path,default=None,
                        help="ARM saved per_day_metrics.csv containing theta and oracle F0/R0.")
    parser.add_argument("--arm-config",type=Path,default=None,
                        help="ARM saved configuration.json containing steps_per_hour.")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--show",action="store_true")
    args = parser.parse_args()
    for week in (1,2) if args.week=="both" else (int(args.week),):
        plot_week(week,args.im_input,args.arm_input,args.output,args.show,args.im_params,args.im_config,
                  args.arm_params,args.arm_config)


if __name__=="__main__": main()

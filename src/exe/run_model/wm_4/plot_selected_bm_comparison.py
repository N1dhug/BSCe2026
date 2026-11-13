"""Compare seven IM and ARM prediction days and their response components.

Five figures are saved per requested Monday-Sunday week:
  1. Seven rows x two columns: IM demand (left), ARM demand (right),
     with central 68% and 95% prediction intervals. The terminal reports
     both empirical coverage levels and the fitted parameters/oracle states
     for each date plus the weekly average coverage.
  2. Seven rows x two columns: instantaneous IM E[f(X_t)], g(U_t) (left)
     and numerically integrated ARM rate-drift terms (right).
  3. Seven vertically stacked IM response panels.
  4. Seven vertically stacked integrated ARM response panels.
  5. Seven rows x two columns of observed vs model-implied cumulative
     hourly RMSE curves, IM left and ARM right. Default matches the
     project convention: sum of hourly RMSE (absolute errors for one day)
     and sum of hourly predicted standard deviations.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
ARM_PARAMETERS = ("theta_F", "theta_R", "theta_U", "oracle_F0", "oracle_R0", "observation_tau", "process_sigma")
ORACLE_DEFAULTS = {"components":24, "quadrature":12, "reconstruction_quadrature":48, "max_step":0.125}


def load_predictions(path, model, mean_col, extra=()):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{model} saved predictions not found: {path}\n"
                                "Run the corresponding selected_bm_dates_* estimation executable first.")
    frame = pd.read_csv(path)
    needed = {"day", "hour", "observed", mean_col, "lower_68", "upper_68", "lower_95", "upper_95", *extra}
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
    for col in ("observed", mean_col, "lower_68", "upper_68", "lower_95", "upper_95", *extra):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{model}: nonfinite {col} values.")
    if (frame["lower_95"]>frame["upper_95"]).any() or (frame["lower_68"]>frame["upper_68"]).any():
        raise ValueError(f"{model}: reversed predictive uncertainty bounds.")
    if ((frame["lower_95"]>frame["lower_68"]+1e-9) |
        (frame["upper_68"]>frame["upper_95"]+1e-9)).any():
        raise ValueError(f"{model}: 68% intervals must lie within the 95% intervals.")
    return frame.set_index(["day", "hour"]).sort_index()


def comparison(im_path=IM_INPUT, arm_path=ARM_INPUT, week=1):
    if week not in (1, 2): raise ValueError("week must be 1 or 2.")
    dates, _ = reference_dates()
    selected = dates[(week-1)*7:week*7]
    im = load_predictions(im_path, "IM", "beta_expected_demand",
                          extra=("weekly_temperature_baseline", "price_U_0_1", "predicted_demand_variance"))
    arm = load_predictions(arm_path, "ARM", "predicted", extra=("baseline_predicted", "predicted_variance", "price_response_h_U", "oracle_F", "oracle_R", *ARM_CONTRIBUTIONS))
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


def cumulative_rmse(observed, predicted, variance, mode="sum"):
    """Per-day empirical and model-implied error accumulation over lead hours.

    'sum' follows the existing selected-14-day cumulative-hourly-RMSE
    convention: sum_j sqrt(error_j^2) vs sum_j sqrt(predicted_var_j).
    A single target-day observation exists at each hour, so empirical
    hourly RMSE is its absolute prediction error.

    'running' is conventional cumulative RMSE over the first h samples:
      sqrt(sum_j error_j^2 / h) vs sqrt(sum_j predicted_var_j / h).
    Model-implied variance is assumed to represent total predictive
    uncertainty, including the fitted observation-noise component.
    """
    observed = np.asarray(observed,dtype=float)
    predicted = np.asarray(predicted,dtype=float)
    variance = np.asarray(variance,dtype=float)
    if observed.shape!=(K,) or predicted.shape!=(K,) or variance.shape!=(K,):
        raise ValueError(f"Cumulative RMSE requires three length-{K} hourly vectors.")
    if not np.isfinite(np.r_[observed,predicted,variance]).all() or (variance<-1e-12).any():
        raise ValueError("Invalid cumulative-RMSE observations, predictions or predictive variances.")
    variance = np.maximum(variance,0.)
    sq_error = (predicted-observed)**2
    if mode=="sum":
        return np.cumsum(np.sqrt(sq_error)), np.cumsum(np.sqrt(variance))
    if mode=="running":
        divisor = np.arange(1,K+1,dtype=float)
        return np.sqrt(np.cumsum(sq_error)/divisor), np.sqrt(np.cumsum(variance)/divisor)
    raise ValueError("Cumulative RMSE mode must be 'sum' or 'running'.")


def _raise_default_axis_fonts(fig, tick_size=None):
    """Increase inherited default label sizes without overriding explicit ticks."""
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        if tick_size is not None: ax.tick_params(axis="both",labelsize=tick_size)


def plot_cumulative_rmse(days, im, arm, output, week, show=False, mode="sum"):
    """Seven rows x IM/ARM columns comparing realized and expected RMSE."""
    fig, axes = plt.subplots(7,2,figsize=(24,28),sharex=True,sharey="row")
    hours = np.arange(1,K+1)
    table = []
    for row,day in enumerate(days):
        for col,(name,part,mean,variance,color) in enumerate((
            ("IM",im.loc[day].sort_index(),"beta_expected_demand",
             "predicted_demand_variance","tab:orange"),
            ("ARM",arm.loc[day].sort_index(),"predicted",
             "predicted_variance","tab:blue"),
        )):
            obs = part["observed"].to_numpy(dtype=float)
            pred = part[mean].to_numpy(dtype=float)
            var = part[variance].to_numpy(dtype=float)
            actual, expected = cumulative_rmse(obs,pred,var,mode)
            axis = axes[row,col]
            axis.plot(hours,actual,color="black",lw=1.9,marker="o",markersize=2.5,
                      label="Observed cumulative RMSE")
            axis.plot(hours,expected,color=color,lw=2.,
                      label=f"{name} predicted cumulative RMSE")
            axis.fill_between(hours,actual,expected,color=color,alpha=0.10)
            axis.set_title(f"{day:%A %d %B %Y} | {name}",loc="left",fontsize=14)
            axis.set_ylabel("Cumulative RMSE" if mode=="running" else "Sum of hourly RMSE")
            axis.set_xlim(1,K)
            axis.set_ylim(bottom=0.)
            axis.grid(alpha=0.23)
            axis.tick_params(axis="both",labelsize=13)
            for lead,(actual_value,expected_value) in enumerate(zip(actual,expected),1):
                table.append({"day":str(day.date()),"model":name,
                              "lead_hour":lead,"mode":mode,
                              "observed_cumulative_rmse":float(actual_value),
                              "predicted_cumulative_rmse":float(expected_value)})
    for ax in axes[-1]:
        ax.set_xticks([1,4,8,12,16,20,24])
        ax.set_xlabel("Prediction lead hour")
    for ax in axes[0]:
        ax.legend(loc="upper left",fontsize=12.5,framealpha=0.9)
    note = ("Sum over lead hours of observed |prediction - demand| versus model-implied "
            "predicted standard deviation" if mode=="sum" else
            "Running RMSE over the first h hourly observations versus "
            "sqrt(mean predicted variance)")
    fig.suptitle(f"IM (left) vs ARM (right) | cumulative RMSE by weekday | week {week}\n"
                 +note,fontsize=17,y=0.995)
    fig.subplots_adjust(top=0.942,bottom=0.05,hspace=0.56,wspace=0.13)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    suffix="" if mode=="sum" else "_running"
    image_path=output/f"im_arm_week_{week}_cumulative_rmse_7x2{suffix}.png"
    csv_path=output/f"im_arm_week_{week}_cumulative_rmse_7x2{suffix}.csv"
    _raise_default_axis_fonts(fig)
    fig.savefig(image_path,dpi=190,bbox_inches="tight")
    pd.DataFrame(table).to_csv(csv_path,index=False)
    if show: plt.show()
    plt.close(fig)
    print(f"Saved cumulative RMSE figure: {image_path}")
    print(f"Saved cumulative RMSE hourly data: {csv_path}")
    return image_path, csv_path


def empirical_coverage(data, interval):
    """Number and fraction of target observations within the central interval."""
    if interval not in (68,95): raise ValueError("Expected a central 68% or 95% interval.")
    obs = data["observed"].to_numpy(dtype=float)
    low = data[f"lower_{interval}"].to_numpy(dtype=float)
    high = data[f"upper_{interval}"].to_numpy(dtype=float)
    count = int(np.count_nonzero((obs>=low)&(obs<=high)))
    return count, 100.*count/len(obs)


def report_week_coverage(days, im, arm, im_fits, arm_fits, output, week):
    """Print all saved fitted parameters, oracle states and empirical interval coverage.

    Coverage is counted over each day's 24 observed target hours. Because
    each target day has the same number of hours, the mean of the seven daily
    coverage percentages equals the pooled 168-hour empirical coverage.
    """
    specs = (
        ("IM",im,im_fits,IM_PARAMETERS),
        ("ARM",arm,arm_fits,ARM_PARAMETERS),
    )
    rows=[]
    print(f"\n{'='*78}\nWEEK {week} | EMPIRICAL PREDICTIVE INTERVAL COVERAGE AND FITTED PARAMETERS")
    print(f"Seven prediction days | {K} hourly observations per day | central 68% and 95%")
    for day in days:
        print(f"\n{day:%A %d %B %Y}")
        for model,frame,fits,parameters in specs:
            sample=frame.loc[day].sort_index()
            if len(sample)!=K: raise ValueError(f"{model}: expected {K} observations for {day}.")
            fitted=fits.loc[day]
            for name in parameters:
                value=float(fitted[name])
                if not np.isfinite(value):
                    raise ValueError(f"{model}: invalid fitted parameter {name} on {day}.")
            parameter_text=", ".join(f"{name}={float(fitted[name]):.6g}" for name in parameters)
            observed=[]
            for nominal in (68,95):
                count,percent=empirical_coverage(sample,nominal)
                rows.append({
                    "day":str(day.date()),"model":model,
                    "nominal_coverage_pct":nominal,
                    "covered_hours":count,"hours":K,
                    "empirical_coverage_pct":percent,
                    "coverage_minus_nominal_pp":percent-nominal,
                })
                observed.append(f"{nominal}%: {percent:5.1f}% ({count:2d}/{K})")
            print(f"  {model}: {' | '.join(observed)}")
            print(f"    Parameters and oracle initial state: {parameter_text}")

    detail=pd.DataFrame(rows)
    summary=[]
    print(f"\nWEEK {week} AVERAGE EMPIRICAL COVERAGE ({len(days)} days, {len(days)*K} hours)")
    for model,_,_,_ in specs:
        for nominal in (68,95):
            group=detail.loc[(detail["model"]==model)&
                             (detail["nominal_coverage_pct"]==nominal)]
            if len(group)!=len(days): raise ValueError("Coverage summary is missing prediction dates.")
            count=int(group["covered_hours"].sum())
            total=int(group["hours"].sum())
            pooled=100.*count/total
            average=float(group["empirical_coverage_pct"].mean())
            if not np.isclose(average,pooled,rtol=0.,atol=1e-9):
                raise RuntimeError("Week coverage average and pooled coverage disagree.")
            print(f"  {model} {nominal}% interval: mean daily={average:5.1f}%, "
                  f"pooled={pooled:5.1f}% ({count}/{total}), "
                  f"nominal difference={pooled-nominal:+.1f} pp")
            summary.append({
                "model":model,"nominal_coverage_pct":nominal,
                "days":len(days),"covered_hours":count,"hours":total,
                "mean_daily_coverage_pct":average,
                "pooled_coverage_pct":pooled,
                "coverage_minus_nominal_pp":pooled-nominal,
            })
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    detail_path=output/f"im_arm_week_{week}_empirical_coverage.csv"
    summary_path=output/f"im_arm_week_{week}_coverage_summary.csv"
    detail.to_csv(detail_path,index=False)
    pd.DataFrame(summary).to_csv(summary_path,index=False)
    print(f"Saved per-day coverage: {detail_path}")
    print(f"Saved weekly average: {summary_path}\n{'='*78}")
    return detail_path,summary_path


def plot_demand_week(days, im, arm, output, week, show=False):
    """Plot the seven observed demand trajectories and nested predictive intervals.

    Only IM and ARM demand panels are shown, seven weekdays by two columns.
    The 68% interval is drawn over the wider 95% interval, with model-specific
    color shading. Empirical coverage and fitted parameters are instead
    printed to the terminal by report_week_coverage().
    """
    with plt.rc_context({"font.family":"serif",
                         "font.serif":["Times New Roman","DejaVu Serif"],
                         "font.size":15,"axes.titlesize":16,
                         "axes.labelsize":15,"xtick.labelsize":14,"ytick.labelsize":14}):
        fig,axes=plt.subplots(7,2,figsize=(23,25),sharex=True,sharey=True)
        hours=np.arange(K)
        specs=(
            ("IM",im,"weekly_temperature_baseline","beta_expected_demand","tab:orange"),
            ("ARM",arm,"baseline_predicted","predicted","tab:blue"),
        )
        values=[]
        for row,day in enumerate(days):
            for col,(model,frame,baseline,mean,color) in enumerate(specs):
                data=frame.loc[day].sort_index()
                if len(data)!=K: raise ValueError(f"{model}: expected {K} hours for {day}.")
                ax=axes[row,col]
                ax.fill_between(hours,data["lower_95"].to_numpy(),data["upper_95"].to_numpy(),
                                color=color,alpha=.12)
                ax.fill_between(hours,data["lower_68"].to_numpy(),data["upper_68"].to_numpy(),
                                color=color,alpha=.29)
                ax.plot(hours,data["observed"].to_numpy(),color="black",lw=1.6,
                        marker="o",markersize=3.)
                ax.plot(hours,data[baseline].to_numpy(),color="tab:green",ls="--",lw=1.5)
                ax.plot(hours,data[mean].to_numpy(),color=color,lw=2.)
                ax.set_title(f"{day:%A %d %B %Y} | {model}",loc="left",pad=8)
                ax.set_xlim(0,K-1)
                ax.set_xticks([0,4,8,12,16,20,23])
                if col==0: ax.set_ylabel("Normalized demand")
                if row==6: ax.set_xlabel("Hour of day")
                ax.grid(alpha=.22)
                values.append(data[["observed",baseline,mean,"lower_95","upper_95"]].to_numpy(dtype=float))

        # All IM and ARM demand panels use the same vertical scale.
        minimum=min(float(np.min(v)) for v in values)
        maximum=max(float(np.max(v)) for v in values)
        pad=max(.02,.04*(maximum-minimum))
        axes[0,0].set_ylim(minimum-pad,maximum+pad)
        fig.subplots_adjust(left=.075,right=.985,top=.89,bottom=.055,
                            hspace=.48,wspace=.14)
        for col,(model,_,_,_,color) in enumerate(specs):
            handles=[
                Line2D([],[],color="black",lw=1.6,marker="o",markersize=3,label="Observed demand"),
                Line2D([],[],color="tab:green",ls="--",lw=1.5,label="Baseline"),
                Line2D([],[],color=color,lw=2.,label=f"{model} expected demand"),
                Patch(facecolor=color,alpha=.29,label="Central 68% interval"),
                Patch(facecolor=color,alpha=.12,label="Central 95% interval"),
            ]
            box=axes[0,col].get_position()
            fig.legend(handles=handles,loc="upper center",
                       bbox_to_anchor=((box.x0+box.x1)/2,.957),
                       ncol=3,fontsize=13,frameon=False)
        fig.suptitle(
            f"IM (left) vs ARM (right) | 7-day demand prediction | sample week {week}\n"
            "ALL · weekly-temperature baseline · N=6 corresponding weekdays · 24-hour oracle predictions",
            fontsize=19,y=.995)
        output=Path(output)
        output.mkdir(parents=True,exist_ok=True)
        image_path=output/f"im_arm_week_{week}_demand_95pct.png"
        pdf_path=output/f"im_arm_week_{week}_demand_95pct.pdf"
        fig.savefig(image_path,dpi=190,bbox_inches="tight")
        fig.savefig(pdf_path,bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
    print(f"Saved demand comparison (68% and 95% intervals): {image_path}, {pdf_path}")
    return image_path,pdf_path


def plot_week(week=1, im_path=IM_INPUT, arm_path=ARM_INPUT, output=OUTPUT, show=False,
              im_params=None, im_config=None, arm_params=None, arm_config=None,
              cumulative_mode="sum"):
    days, im, arm = comparison(im_path, arm_path, week)
    im_params = Path(im_params) if im_params is not None else Path(im_path).parent/"per_day_metrics.csv"
    im_config = Path(im_config) if im_config is not None else Path(im_path).parent/"configuration.json"
    fits, resolution = load_im_fits(im_params), im_resolution(im_config)
    missing = [str(day.date()) for day in days if day not in fits.index]
    if missing: raise ValueError(f"IM fitted parameters missing days: {missing}")

    hours = np.arange(K)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arm_params = Path(arm_params) if arm_params is not None else Path(arm_path).parent/"per_day_metrics.csv"
    arm_config = Path(arm_config) if arm_config is not None else Path(arm_path).parent/"configuration.json"
    arm_fits, steps_per_hour = load_arm_fits(arm_params), arm_resolution(arm_config)
    missing_arm = [str(day.date()) for day in days if day not in arm_fits.index]
    if missing_arm: raise ValueError(f"ARM fitted parameters missing days: {missing_arm}")
    cumulative_path, cumulative_csv = plot_cumulative_rmse(
        days, im, arm, output, week, show, mode=cumulative_mode)

    # Keep the figure focused on demand. Print saved parameters and all
    # empirical 68%/95% coverage statistics to the terminal instead.
    coverage_csv, coverage_summary_csv = report_week_coverage(
        days, im, arm, fits, arm_fits, output, week)
    demand_path, demand_pdf = plot_demand_week(days, im, arm, output, week, show)

    # Function responses: IM is instantaneous, while the ARM terms are
    # individually integrated to give contributions to the *rate* R(t).
    # They are shown side by side, one weekday per row, in a second figure.
    response_fig, response_axes = plt.subplots(7,2,figsize=(24,28),sharex="col")
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
            ax.set_title(f"{day:%A %d %B %Y} | {name}", loc="left", fontsize=14)
            ax.set_xlim(0,K-1)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
            ax.tick_params(axis="both", labelsize=13)

    for ax in response_axes[-1]:
        ax.set_xticks([0,4,8,12,16,20,23])
        ax.set_xlabel("Hour of day")
    for ax in response_axes[0]:
        ax.legend(loc="best",fontsize=12.3,framealpha=0.92)
    response_fig.suptitle(
        f"IM direct responses (left) vs ARM integrated rate contributions (right) | week {week}\n"
        "ARM time is in days; cumulative terms sum to R(t)-R(0); ARM g(U)=U",
        y=0.996,fontsize=18)
    response_fig.subplots_adjust(top=0.94,bottom=0.05,hspace=0.56,wspace=0.18)
    response_path = output/f"im_arm_week_{week}_responses_7x2.png"
    _raise_default_axis_fonts(response_fig)
    response_fig.savefig(response_path,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(response_fig)

    # Keep the model-specific response figures as well as the new side-by-side
    # layout. Reuse the computed arrays without rerunning IM propagation.
    fig_im, axes_im = plt.subplots(7,1,figsize=(16,28),sharex=True,sharey=True)
    fig_arm, axes_arm = plt.subplots(7,1,figsize=(16,28),sharex=True)
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
            ax.set_title(f"{day:%A %d %B %Y}",loc="left",fontsize=14)
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
                   ncol=len(handles),fontsize=14,frameon=False)
        fig.suptitle(f"{title} | sample week {week}",y=0.996,fontsize=18)
        fig.subplots_adjust(top=0.915,bottom=0.05,hspace=0.56)
    im_path_out = output/f"im_week_{week}_function_contributions.png"
    arm_path_out = output/f"arm_week_{week}_function_contributions.png"
    _raise_default_axis_fonts(fig_im, tick_size=14)
    _raise_default_axis_fonts(fig_arm, tick_size=14)
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
    return {"demand":demand_path, "demand_pdf":demand_pdf,
            "coverage_csv":coverage_csv, "coverage_summary_csv":coverage_summary_csv,
            "responses":response_path,
            "im_contributions":im_path_out, "arm_contributions":arm_path_out,
            "cumulative_rmse":cumulative_path, "cumulative_rmse_csv":cumulative_csv}


def main():
    parser = ArgumentParser(description="Plot seven-day IM/ARM demand curves with 68%/95% bands and report coverage/parameters in the terminal.")
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
    parser.add_argument("--cumulative-mode",choices=("sum","running"),default="sum",
                        help="sum: cumulative sum of hourly RMSE (existing convention); "
                             "running: conventional prefix RMSE over hours 1..h.")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--show",action="store_true")
    args = parser.parse_args()
    for week in (1,2) if args.week=="both" else (int(args.week),):
        plot_week(week,args.im_input,args.arm_input,args.output,args.show,args.im_params,args.im_config,
                  args.arm_params,args.arm_config,args.cumulative_mode)


if __name__=="__main__": main()

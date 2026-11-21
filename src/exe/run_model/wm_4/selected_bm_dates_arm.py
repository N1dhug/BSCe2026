"""ARM evaluation on seven corresponding non-holiday weekdays, N=6, k=24, origins 00:00.

Using only the temperature-adjusted weekly baseline, fit historical ARM drift by joint QP, fit
tau^2 and sigma^2 on the six free-running historical trajectories, then
optimize only [F0,R0] on the 24-hour target interval (oracle conditioning).

Report pooled target RMSE, MAE, signed bias, Gaussian similarity, variance/
RMSE discrepancy, empirical 95% coverage, and mean complete computation time.
Data loading and historical-day imputation are excluded from computation time.

Outputs: seven-row target-day demand and drift-contribution panels, a pooled
cumulative hourly-RMSE comparison, timing averages and CSV data and metrics.
Use --week 1 (default) for 15-21 August 2022, or --week 2 for 22-28 August.

From repository root:
    python -m src.exe.run_model.wm_4.selected_bm_dates_arm --show

IMPORTANT: oracle states use target observations, so this is NOT operational
forecast skill; the stochastic parameters use historical data only.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from src.method.arm_variance import fit_variances, variance_kernel
from src.method.moving_horizon import group_frame, metrics
from src.method.moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    baseline_from_history, historical_calendar_days, interval_window,
)
from src.params.wm_4 import joint_qp_params
from src.method.sample_timing import timing_report
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"selected_bm_dates_arm"
N, K = 6, 24
BASELINES = ("weekly_temperature",)
LABELS = {"weekly_temperature":"Weekly temperature"}
COLORS = {"weekly_temperature":"tab:orange"}
DATES = (
    ("2022-08-15", "2022-07-04", "2022-08-08"),
    ("2022-08-16", "2022-07-05", "2022-08-09"),
    ("2022-08-17", "2022-07-06", "2022-08-10"),
    ("2022-08-18", "2022-07-07", "2022-08-11"),
    ("2022-08-19", "2022-07-08", "2022-08-12"),
    ("2022-08-20", "2022-07-09", "2022-08-13"),
    ("2022-08-21", "2022-07-10", "2022-08-14"),
    ("2022-08-22", "2022-07-11", "2022-08-15"),
    ("2022-08-23", "2022-07-12", "2022-08-16"),
    ("2022-08-24", "2022-07-13", "2022-08-17"),
    ("2022-08-25", "2022-07-14", "2022-08-18"),
    ("2022-08-26", "2022-07-15", "2022-08-19"),
    ("2022-08-27", "2022-07-16", "2022-08-20"),
    ("2022-08-28", "2022-07-17", "2022-08-21"),
)
Z68, Z95 = float(norm.ppf(0.84)), float(norm.ppf(0.975))
EXPECTED_ABS_NORMAL = float(np.sqrt(2./np.pi))


def reference_dates():
    """Check two normal weeks, two of each weekday, and their six-week histories."""
    dates, histories = [], []
    for day, first, last in DATES:
        target = pd.Timestamp(day)
        history = historical_weekdays(target, N)
        if target.hour != 0 or history[0] != pd.Timestamp(first) or history[-1] != pd.Timestamp(last):
            raise ValueError(f"Selected-date history mismatch at {day}.")
        dates.append(target)
        histories.append(history)
    if len(dates)!=14 or any(sum(date.weekday()==weekday for date in dates)!=2
                             for weekday in range(7)):
        raise ValueError("Reference selection must contain exactly two of every weekday.")
    holiday_dates = {pd.Timestamp(value) for value in (
        "2022-04-14","2022-04-15","2022-04-17","2022-04-18",
        "2022-05-13","2022-05-26","2022-06-05","2022-06-06",
    )}
    if any(date in holiday_dates for date in dates):
        raise ValueError("Selected prediction date is a Danish 2022 public holiday.")
    return dates, histories


def reference_week(week=1):
    """Return exactly one Monday-Sunday sample with its six historical weekdays."""
    if week not in (1,2): raise ValueError("Sample week must be 1 or 2.")
    dates,histories=reference_dates()
    sl=slice(7*(week-1),7*week)
    selected,previous=dates[sl],histories[sl]
    if len(selected)!=7 or [day.weekday() for day in selected]!=list(range(7)):
        raise ValueError("Selected ARM sample must cover Monday through Sunday exactly once.")
    return selected,previous


def fit_origin(frame, training, day, history, baseline_name, steps):
    """One complete independently fitted historical + oracle target estimate."""
    start = perf_counter()
    baseline = baseline_from_history(training, history, baseline_name, K)
    historical = [interval_window(training, previous, baseline, steps, K)[1]
                  for previous in history]
    model, forecast, _, observed = interval_window(frame, day, baseline, steps, K)
    sampled = np.arange(K, dtype=int)*steps
    preparation_seconds = perf_counter()-start

    drift_start = perf_counter()
    fit = joint_qp_params(model).fit_many(historical)
    theta = np.asarray(fit["r"], dtype=float)
    drift_time = perf_counter()-drift_start
    historical_observed = np.vstack([data["I"][sampled] for data in historical])
    historical_predictions = np.vstack([state["I"][sampled]
                                        for state in fit["deterministic_states"]])
    historical_error = historical_observed-historical_predictions

    diffusion_start = perf_counter()
    K_unit = variance_kernel(float(theta[0]), float(theta[2]), steps, K)
    tau2, sigma2, identifiable = fit_variances(historical_error.reshape(-1),
                                               np.tile(K_unit, N))
    variance = tau2+sigma2*K_unit
    if np.any(variance < -1e-12) or not np.isfinite(variance).all():
        raise ValueError(f"Invalid ARM stochastic variance at {day} ({baseline_name}).")
    variance = np.maximum(variance, 0.)
    diffusion_time = perf_counter()-diffusion_start

    oracle_start = perf_counter()
    zero_state = model._solve_deterministic([0., 0.], forecast["U"], forecast["PE"],
                                            forecast["Time"], forecast["O"], theta)
    base = np.asarray(zero_state["I"], dtype=float)[sampled]
    S = np.asarray(model.sensitivity({"r":theta})["input"], dtype=float)[sampled]
    init = np.asarray(model._least_squares_qp(
        S, np.asarray(observed,dtype=float)-base,
        lb=np.array([-1.,-1.]), ub=np.array([1.,1.]), ridge=model._FIT_RIDGE),dtype=float)
    predicted = base+S@init
    # Evaluate the SAME oracle state trajectory as the demand forecast. ARM
    # uses g(R)=R and h(U)=U, so the three drift contributions are additive.
    oracle_state = model._solve_deterministic(
        init, forecast["U"], forecast["PE"], forecast["Time"], forecast["O"], theta)
    oracle_F = np.asarray(oracle_state["F"],dtype=float)[sampled]
    oracle_R = np.asarray(oracle_state["R"],dtype=float)[sampled]
    oracle_U = np.asarray(oracle_state["U"],dtype=float)[sampled]
    effect_F = theta[0]*oracle_F
    effect_R = theta[2]*oracle_R
    effect_U = theta[1]*oracle_U
    total_drift = effect_F+effect_R+effect_U
    if not np.allclose(np.asarray(oracle_state["I"])[sampled],predicted,rtol=1e-7,atol=1e-9):
        raise RuntimeError(f"Oracle trajectory and affine initial-state prediction disagree at {day}.")
    if not np.allclose(np.asarray(oracle_state["R_dot"])[sampled],total_drift,rtol=1e-9,atol=1e-10):
        raise RuntimeError(f"ARM rate contributions do not sum to the predicted rate derivative at {day}.")
    oracle_time = perf_counter()-oracle_start
    total_time = perf_counter()-start

    if not np.isfinite(np.r_[observed,predicted]).all():
        raise ValueError(f"Nonfinite predictions for {day} ({baseline_name}).")
    signed_error = predicted-np.asarray(observed,dtype=float)
    sigma = np.sqrt(variance)
    return {
        "day":day, "baseline":baseline_name, "observed":np.asarray(observed,dtype=float),
        "predicted":predicted, "baseline_predicted":forecast["O"][sampled],
        "error":signed_error, "variance":variance, "std":sigma, "theta":theta,
        "oracle_F":oracle_F, "oracle_R":oracle_R, "oracle_U":oracle_U,
        "effect_F":effect_F, "effect_R":effect_R, "effect_U":effect_U,
        "total_drift":total_drift,
        "initial":init, "tau2":float(tau2), "sigma2":float(sigma2),
        "process_identifiable":bool(identifiable),
        "demand_metrics":metrics(observed,predicted),
        "historical_inputs":historical, "historical_states":fit["deterministic_states"],
        "historical_days":history,
        "historical_rmse":float(np.sqrt(np.mean(historical_error**2))),
        "baseline_demand_rmse":float(np.sqrt(np.mean(
            (forecast["O"][sampled]-observed)**2))),
        "baseline_seconds":preparation_seconds,
        "total_compute_seconds":total_time, "drift_seconds":drift_time,
        "diffusion_seconds":diffusion_time, "oracle_seconds":oracle_time,
    }


def pooled_metrics(items):
    """Pooled across seven target days, each with K hourly observations."""
    error = np.stack([item["error"] for item in items])
    variance = np.stack([item["variance"] for item in items])
    observed_hourly_rmse = np.sqrt(np.mean(error**2,axis=0))
    predicted_hourly_rmse = np.sqrt(np.mean(variance,axis=0))
    observed_hourly_mae = np.mean(np.abs(error),axis=0)
    predicted_hourly_mae = EXPECTED_ABS_NORMAL*np.mean(np.sqrt(variance),axis=0)

    valid = np.isfinite(variance)&(variance>1e-14)
    scaled = error[valid]/np.sqrt(variance[valid])
    ks = float(kstest(scaled, "norm").statistic) if scaled.size >= 3 else float("nan")
    covered = np.abs(error) <= Z95*np.sqrt(variance)
    observed_rmse = float(np.sqrt(np.mean(error**2)))
    predicted_rmse = float(np.sqrt(np.mean(variance)))
    observed_mae = float(np.mean(np.abs(error)))
    predicted_mae = float(np.mean(EXPECTED_ABS_NORMAL*np.sqrt(variance)))
    timing = np.asarray([item["total_compute_seconds"] for item in items])
    return {
        "baseline":items[0]["baseline"], "historical_weekdays":N,"horizon_hours":K,
        "prediction_days":len(items), "observations":int(error.size),
        "rmse":observed_rmse, "mae":observed_mae,
        "bias_predicted_minus_observed":float(np.mean(error)),
        "model_implied_rmse":predicted_rmse,
        "pooled_rmse_absolute_discrepancy":abs(observed_rmse-predicted_rmse),
        "mean_hourly_rmse_absolute_discrepancy":float(np.mean(
            np.abs(observed_hourly_rmse-predicted_hourly_rmse))),
        "sum_hourly_rmse_absolute_discrepancy":float(np.sum(
            np.abs(observed_hourly_rmse-predicted_hourly_rmse))),
        "model_implied_gaussian_mae":predicted_mae,
        "pooled_mae_absolute_discrepancy":abs(observed_mae-predicted_mae),
        "mean_hourly_mae_absolute_discrepancy":float(np.mean(
            np.abs(observed_hourly_mae-predicted_hourly_mae))),
        "gaussian_similarity_1_minus_ks":1.-ks if np.isfinite(ks) else float("nan"),
        "ks_distance":ks, "standardized_residual_count":int(scaled.size),
        "empirical_coverage_95_percent":float(100.*np.mean(covered)),
        "mean_compute_seconds":float(np.mean(timing)),
        "std_compute_seconds":float(np.std(timing)),
        "total_compute_seconds":float(np.sum(timing)),
        "mean_observation_tau":float(np.mean([np.sqrt(item["tau2"]) for item in items])),
        "mean_process_sigma":float(np.mean([np.sqrt(item["sigma2"]) for item in items])),
        "process_identifiable_percent":float(100.*np.mean(
            [item["process_identifiable"] for item in items])),
    }


def make_tables(results, output):
    hourly, daily, pooled, curves = [], [], [], []
    for baseline in BASELINES:
        items = results[baseline]
        pooled.append(pooled_metrics(items))
        for sample, item in enumerate(items, 1):
            metric = item["demand_metrics"]
            daily.append({
                "sample":sample, "day":str(item["day"]), "baseline":baseline,
                "historical_weekdays":N, "horizon_hours":K, **metric,
                "historical_rmse":item["historical_rmse"],
                "baseline_demand_rmse":item["baseline_demand_rmse"],
                "oracle_F0":float(item["initial"][0]),"oracle_R0":float(item["initial"][1]),
                "theta_F":float(item["theta"][0]),"theta_U":float(item["theta"][1]),
                "theta_R":float(item["theta"][2]),
                "observation_tau":float(np.sqrt(item["tau2"])),
                "process_sigma":float(np.sqrt(item["sigma2"])),
                "process_identifiable":item["process_identifiable"],
                "complete_compute_seconds":item["total_compute_seconds"],
                "baseline_fit_and_windows_seconds":item["baseline_seconds"],
                "drift_fit_seconds":item["drift_seconds"],
                "diffusion_fit_seconds":item["diffusion_seconds"],
                "oracle_optimization_seconds":item["oracle_seconds"],
            })
            for hour, (obs, pred, base, err, var, std) in enumerate(zip(
                    item["observed"],item["predicted"],item["baseline_predicted"],
                    item["error"],item["variance"],item["std"])):
                hourly.append({
                    "sample":sample, "day":str(item["day"]), "baseline":baseline,
                    "hour":hour, "observed":obs, "predicted":pred,
                    "baseline_predicted":base, "error_predicted_minus_observed":err,
                    "predicted_variance":var, "predicted_std":std,
                    "oracle_F":float(item["oracle_F"][hour]),
                    "oracle_R":float(item["oracle_R"][hour]),
                    "price_response_h_U":float(item["oracle_U"][hour]),
                    "theta_F_times_F":float(item["effect_F"][hour]),
                    "theta_R_times_R":float(item["effect_R"][hour]),
                    "theta_U_times_h_U":float(item["effect_U"][hour]),
                    "predicted_R_dot":float(item["total_drift"][hour]),
                    "lower_68":pred-Z68*std,"upper_68":pred+Z68*std,
                    "lower_95":pred-Z95*std,"upper_95":pred+Z95*std,
                    "covered_95":abs(err)<=Z95*std,
                })
        err = np.stack([item["error"] for item in items])
        variances = np.stack([item["variance"] for item in items])
        obs_rmse = np.sqrt(np.mean(err**2,axis=0))
        pred_rmse = np.sqrt(np.mean(variances,axis=0))
        for hour in range(K):
            curves.append({
                "baseline":baseline, "lead_hour":hour+1,
                "empirical_pooled_hourly_rmse":obs_rmse[hour],
                "model_implied_pooled_hourly_rmse":pred_rmse[hour],
                "empirical_cumulative_hourly_rmse":np.sum(obs_rmse[:hour+1]),
                "predicted_cumulative_hourly_rmse":np.sum(pred_rmse[:hour+1]),
                "cumulative_sum_abs_hourly_rmse_discrepancy":np.sum(
                    np.abs(obs_rmse[:hour+1]-pred_rmse[:hour+1])),
            })
    hourly, daily, pooled, curves = map(pd.DataFrame,(hourly,daily,pooled,curves))
    hourly.to_csv(output/"hourly_predictions.csv",index=False)
    daily.to_csv(output/"per_day_metrics.csv",index=False)
    pooled.to_csv(output/"pooled_metrics.csv",index=False)
    curves.to_csv(output/"pooled_cumulative_errors.csv",index=False)
    return hourly,daily,pooled,curves


def sample_parameter_legend(item):
    """Display the per-day fitted ARM parameters in each subplot legend."""
    theta = np.asarray(item["theta"], dtype=float).reshape(-1)
    if theta.size != 3: raise ValueError("WM4 sample legend needs [theta_F, theta_U, theta_R].")
    tau, sigma = np.sqrt(float(item["tau2"])), np.sqrt(float(item["sigma2"]))
    return (
        rf"$\theta_F={theta[0]:+.4g}$  $\theta_U={theta[1]:+.4g}$" "\n"
        rf"$\theta_R={theta[2]:+.4g}$" "\n"
        rf"$\tau={tau:.4g}$  $\sigma={sigma:.4g}$"
    )


def save_historical_predictions(results,output,steps):
    """Persist ARM fitted free-running trajectories from the joint historical QP.

    Historical central 95% marginal bands use the same fitted tau^2 and
    sigma^2 variance kernel as the target day. Postprocessing is outside
    the measured parameter-estimation runtime.
    """
    records=[]
    for name in BASELINES:
        for item in results[name]:
            dates=item["historical_days"]
            data=item["historical_inputs"]
            states=item["historical_states"]
            if not len(dates)==len(data)==len(states)==N:
                raise ValueError("ARM historical states do not match the six fitting intervals.")
            kernel=variance_kernel(float(item["theta"][0]),float(item["theta"][2]),steps,K)
            var=np.maximum(0.,float(item["tau2"])+float(item["sigma2"])*kernel)
            if not np.isfinite(var).all() or len(var)!=K:
                raise ValueError("Invalid ARM historical variance kernel.")
            std=np.sqrt(var)
            sampled=np.arange(K,dtype=int)*steps
            for slot,(past,d,state) in enumerate(zip(dates,data,states),1):
                observed=np.asarray(d["I"],dtype=float)[sampled]
                baseline=np.asarray(d["O"],dtype=float)[sampled]
                expected=np.asarray(state["I"],dtype=float)[sampled]
                if any(len(v)!=K or not np.isfinite(v).all()
                       for v in (observed,baseline,expected)):
                    raise ValueError(f"Invalid fitted ARM historical trajectory {past}.")
                for hour in range(K):
                    records.append({
                        "target_day":str(item["day"].date()),
                        "historical_day":str(past.date()),
                        "historical_number":slot,"hour":hour,
                        "observed":float(observed[hour]),
                        "baseline_predicted":float(baseline[hour]),
                        "fitted_demand":float(expected[hour]),
                        "lower_95":float(expected[hour]-Z95*std[hour]),
                        "upper_95":float(expected[hour]+Z95*std[hour]),
                        "fit_type":"historical joint QP; free-running",
                    })
    path=output/"historical_fitted_predictions.csv"
    pd.DataFrame(records).to_csv(path,index=False)
    print(f"Saved ARM historical fitted curves: {path}",flush=True)


def plot_days(results,path,show=False):
    """Seven vertically stacked oracle-conditioned weekday demand forecasts."""
    days=results[BASELINES[0]]
    if len(days)!=7: raise ValueError("ARM weekday plot requires exactly seven prediction dates.")
    fig,axes=plt.subplots(7,1,figsize=(13,19),sharex=True,sharey=True)
    h=np.arange(K)
    for index,ax in enumerate(axes):
        item=days[index]
        ax.plot(h,item["observed"],color="black",lw=1.7,marker="o",markersize=2.5,
                label="Observed demand")
        for baseline in BASELINES:
            row=results[baseline][index]
            mean,std,color=row["predicted"],row["std"],COLORS[baseline]
            ax.fill_between(h,mean-Z95*std,mean+Z95*std,color=color,alpha=0.10,
                            label="95% predicted interval")
            ax.fill_between(h,mean-Z68*std,mean+Z68*std,color=color,alpha=0.15,
                            label="68% predicted interval")
            ax.plot(h,mean,color=color,lw=1.9,label=LABELS[baseline])
        ax.set_title(f"{index+1:02d} | {item['day']:%A %d %B %Y}",loc="left",fontsize=10)
        ax.set_xlim(0,K-1)
        ax.set_xticks([0,4,8,12,16,20,23])
        ax.set_ylabel("Normalized demand")
        ax.grid(alpha=0.23)
        if index==0:
            ax.legend(title=sample_parameter_legend(item),loc="upper right",
                      fontsize=8,title_fontsize=8,framealpha=0.88)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("ARM | 7 oracle-conditioned weekdays | weekly temperature | ALL | N=6, k=24",
                 fontsize=14,y=0.997)
    fig.tight_layout(rect=(0,0,1,0.975))
    fig.savefig(path,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def plot_rate_contributions(results,path,show=False):
    """ARM deterministic drift terms along each saved oracle target trajectory."""
    days=results[BASELINES[0]]
    if len(days)!=7: raise ValueError("ARM drift plot requires exactly seven prediction dates.")
    fig,axes=plt.subplots(7,1,figsize=(13,19),sharex=True,sharey=True)
    h=np.arange(K)
    components=(
        ("effect_F",r"$\theta_F F_t$","tab:blue"),
        ("effect_R",r"$\theta_R R_t$","tab:green"),
        ("effect_U",r"$\theta_U h(U_t)$","tab:orange"),
    )
    for index,ax in enumerate(axes):
        item=days[index]
        for key,label,color in components:
            ax.plot(h,item[key],label=label,color=color,lw=1.9)
        ax.plot(h,item["total_drift"],label=r"Total $\dot{R}_t$",
                color="0.2",ls="--",lw=1.3)
        ax.axhline(0.,color="0.45",lw=0.8)
        ax.set_title(f"{index+1:02d} | {item['day']:%A %d %B %Y}",loc="left",fontsize=10)
        ax.set_xlim(0,K-1)
        ax.set_xticks([0,4,8,12,16,20,23])
        ax.set_ylabel(r"Contribution to $\dot{R}$")
        ax.grid(alpha=0.23)
        if index==0:
            ax.legend(title=sample_parameter_legend(item),loc="upper right",
                      fontsize=8,title_fontsize=8,framealpha=0.88)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("ARM drift responses | 7 oracle-conditioned weekdays | weekly temperature",
                 fontsize=14,y=0.997)
    fig.tight_layout(rect=(0,0,1,0.975))
    fig.savefig(path,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def plot_cumulative(curves,path,show=False):
    """Cumulative sums of pooled hourly RMSE, not RMSE of accumulated energy."""
    baseline = BASELINES[0]
    part = curves.loc[curves["baseline"]==baseline].sort_values("lead_hour")
    h = part["lead_hour"].to_numpy(dtype=int)
    empirical = part["empirical_cumulative_hourly_rmse"].to_numpy(dtype=float)
    predicted = part["predicted_cumulative_hourly_rmse"].to_numpy(dtype=float)
    color = COLORS[baseline]

    fig,ax=plt.subplots(figsize=(10,5.6))
    ax.plot(h,empirical,color="black",lw=2.2,marker="o",ms=3,
            label="Empirical cumulative hourly RMSE")
    ax.plot(h,predicted,color=color,lw=2.2,ls="--",marker="s",ms=3,
            label="Model-predicted cumulative hourly RMSE")
    ax.fill_between(h,empirical,predicted,color=color,alpha=0.11)
    ax.set_title("ARM | Weekly temperature | 7-day pooled empirical vs predicted error accumulation")
    ax.set_xlabel("Prediction lead hour")
    ax.set_ylabel("Cumulative sum of pooled hourly RMSE (normalized demand)")
    ax.set_xlim(1,K)
    ax.set_ylim(bottom=0.)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper left",fontsize=9)
    fig.tight_layout()
    fig.savefig(path,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def run(args):
    if args.steps_per_hour<2: raise ValueError("steps-per-hour must be at least 2.")
    dates,histories=reference_week(args.week)
    if args.week==2 and args.output==OUTPUT: args.output=OUTPUT/"week_2"
    # Historical fitting dates span July and August, so neither the
    # June 6-22 nor September/October full-day data gaps enter the N=6 backlog.
    # Disk reading, alignment, availability checking and hourly imputation
    # are deliberately outside the complete predictive computation timer.
    frame=group_frame("ALL")
    needed=pd.DatetimeIndex([d for history in histories
                             for d in historical_calendar_days(history,K)]).unique().sort_values()
    # Fail with a direct distinction between missing historical days (which
    # cannot be imputed) and incomplete target days (which must be measured).
    signals=["I","U","Temp"]
    missing_history=[]
    for day in needed:
        block=frame.reindex(pd.date_range(day,periods=24,freq="h"))[signals]
        if block.notna().sum().eq(0).any(): missing_history.append(day.strftime("%Y-%m-%d"))
    incomplete_targets=[]
    for day in dates:
        block=frame.reindex(pd.date_range(day,periods=K,freq="h"))[signals]
        if block.isna().any().any(): incomplete_targets.append(day.strftime("%Y-%m-%d"))
    if missing_history or incomplete_targets:
        raise ValueError(
            "The selected August reference dates do not have complete required data. "
            f"Missing historical full days: {', '.join(missing_history) or 'none'}; "
            f"incomplete prediction days: {', '.join(incomplete_targets) or 'none'}. "
            "Choose other dates or correct the source data rather than imputing full days."
        )
    training,imputed=impute_hourly_days(frame,days=needed,
                                        columns=("I","U","Temp"),require_days=True)
    results={name:[] for name in BASELINES}
    args.output.mkdir(parents=True,exist_ok=True)

    for sample,(day,history) in enumerate(zip(dates,histories),1):
        for name in BASELINES:
            item=fit_origin(frame,training,day,history,name,args.steps_per_hour)
            results[name].append(item)
            metric=item["demand_metrics"]
            print(f"[{sample:02d}/{len(dates)}] {day:%Y-%m-%d} | {LABELS[name]:18s} | "
                  f"RMSE={metric['rmse']:.6f} | MAE={metric['mae']:.6f} | "
                  f"bias={metric['bias']:+.6f} | compute={item['total_compute_seconds']:.3f}s",
                  flush=True)

    hourly,daily,pooled,curves=make_tables(results,args.output)
    save_historical_predictions(results,args.output,args.steps_per_hour)
    timing_report(daily,"ARM",args.output,(
        ("Deterministic drift QP","drift_fit_seconds"),
        ("Variance fit","diffusion_fit_seconds"),
        ("Oracle initial-state QP","oracle_optimization_seconds"),
        ("Complete process","complete_compute_seconds"),
    ),show=args.show)
    plot_days(results,args.output/"selected_7_days_7x1.png",args.show)
    plot_rate_contributions(results,args.output/"selected_7_days_rate_contributions_7x1.png",args.show)
    plot_cumulative(curves,args.output/"pooled_cumulative_error.png",args.show)
    config={
        "group":"ALL","N":N,"k":K,"sample_week":args.week,"prediction_days":len(dates),
        "origin_hour":0,"steps_per_hour":args.steps_per_hour,
        "dates":[f"{day:%Y-%m-%d}" for day in dates],
        "baselines":list(BASELINES),
        "input":"six corresponding prior weekdays (July-August) for each non-holiday August prediction date",
        "deterministic_fit":"historical joint convex QP",
        "diffusion_fit":"historical free-running squared residuals; joint nonnegative variance matching",
        "target_initial_state":"oracle optimized from target observations",
        "time_definition":"complete weekly-temperature estimation and prediction, excludes initial ALL data reading and imputation",
        "cumulative_plot":"sum over lead hours of sqrt(mean over 7 dates of squared hourly errors); predicted uses sqrt(mean model-implied variance)",
        "rate_contributions":"theta_F*F + theta_R*R + theta_U*h(U) = predicted dR/dt; g(R)=R, h(U)=U; evaluated on oracle trajectory",
        "signed_bias":"predicted minus observed",
        "empirical_coverage":"pointwise 95 percent Gaussian intervals",
        "historical_imputed_hourly_slots":int(imputed["imputed_hourly_slots"]),
    }
    (args.output/"configuration.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print(f"\nPOOLED TARGET PERFORMANCE ({len(dates)} non-holiday dates x 24 hours, weekly temperature baseline)")
    for row in pooled.itertuples(index=False):
        print(f"\n{LABELS[row.baseline]} | N=6 k=24 | {row.observations} observations")
        print(f"  Demand RMSE={row.rmse:.8f} | MAE={row.mae:.8f} | "
              f"bias(pred-obs)={row.bias_predicted_minus_observed:+.8f}")
        print(f"  Diffusion observed RMSE={row.rmse:.8f} | model-implied RMSE={row.model_implied_rmse:.8f}")
        print(f"  Pooled |RMSE discrepancy|={row.pooled_rmse_absolute_discrepancy:.8f} | "
              f"mean hourly |RMSE discrepancy|={row.mean_hourly_rmse_absolute_discrepancy:.8f}")
        print(f"  Observed MAE={row.mae:.8f} | Gaussian expected MAE={row.model_implied_gaussian_mae:.8f} | "
              f"pooled |MAE discrepancy|={row.pooled_mae_absolute_discrepancy:.8f}")
        print(f"  Gaussian similarity (1-KS)={row.gaussian_similarity_1_minus_ks:.6f} | "
              f"empirical 95% coverage={row.empirical_coverage_95_percent:.2f}%")
        print(f"  Mean COMPLETE compute time per origin={row.mean_compute_seconds:.3f}s "
              f"(excludes data read-in and historical imputation)")
    print(f"\nMean complete compute time per selected day (weekly temperature only)="
          f"{pooled.iloc[0]['mean_compute_seconds']:.3f}s")
    print(f"\nSaved rate-contribution plot: {args.output/'selected_7_days_rate_contributions_7x1.png'}")
    print(f"Saved results to {args.output}")
    return pooled


def main():
    parser=ArgumentParser(description="Fit ARM on seven non-holiday weekdays with historical drift QP and variance matching (ALL, N=6, k=24).")
    parser.add_argument("--week",type=int,choices=(1,2),default=1,
                        help="Fit only Aug 15-21 (week 1, default) or Aug 22-28 (week 2), seven days each.")
    parser.add_argument("--steps-per-hour",type=int,default=60)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--show",action="store_true")
    return run(parser.parse_args())


if __name__=="__main__": main()

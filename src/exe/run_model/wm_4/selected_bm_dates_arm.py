"""ARM evaluation on the 12 BM1 reference dates, N=6, k=24, origins 00:00.

Using only the temperature-adjusted weekly baseline, fit historical ARM drift by joint QP, fit
tau^2 and sigma^2 on the six free-running historical trajectories, then
optimize only [F0,R0] on the 24-hour target interval (oracle conditioning).

Report pooled target RMSE, MAE, signed bias, Gaussian similarity, variance/
RMSE discrepancy, empirical 95% coverage, and mean complete computation time.
Data loading and historical-day imputation are excluded from computation time.

Outputs: a 3x4 weekly-temperature target-day panel, a separate pooled cumulative hourly-RMSE
comparison, and hourly, per-day, pooled, and cumulative-metric CSV files.

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

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import fit_variances, variance_kernel
from src.exe.run_model.wm_4.moving_horizon_common import group_frame, metrics
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    baseline_from_history, historical_calendar_days, interval_window,
)
from src.params.wm_4 import joint_qp_params
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"selected_bm_dates_arm"
N, K = 6, 24
BASELINES = ("weekly_temperature",)
LABELS = {"weekly_temperature":"Weekly temperature"}
COLORS = {"weekly_temperature":"tab:orange"}
DATES = (
    ("2022-04-11", "2022-02-28", "2022-04-04"),
    ("2022-04-19", "2022-03-08", "2022-04-12"),
    ("2022-04-27", "2022-03-16", "2022-04-20"),
    ("2022-05-05", "2022-03-24", "2022-04-28"),
    ("2022-05-13", "2022-04-01", "2022-05-06"),
    ("2022-05-21", "2022-04-09", "2022-05-14"),
    ("2022-05-29", "2022-04-17", "2022-05-22"),
    ("2022-05-30", "2022-04-18", "2022-05-23"),
    ("2022-06-07", "2022-04-26", "2022-05-31"),
    ("2022-08-10", "2022-06-29", "2022-08-03"),
    ("2022-08-18", "2022-07-07", "2022-08-11"),
    ("2022-08-26", "2022-07-15", "2022-08-19"),
)
Z68, Z95 = float(norm.ppf(0.84)), float(norm.ppf(0.975))
EXPECTED_ABS_NORMAL = float(np.sqrt(2./np.pi))


def reference_dates():
    """Check the six corresponding weekdays against the exact BM1 date table."""
    dates, histories = [], []
    for day, first, last in DATES:
        target = pd.Timestamp(day)
        history = historical_weekdays(target, N)
        if target.hour != 0 or history[0] != pd.Timestamp(first) or history[-1] != pd.Timestamp(last):
            raise ValueError(f"BM1 reference-date mismatch at {day}.")
        dates.append(target)
        histories.append(history)
    return dates, histories


def fit_origin(frame, training, day, history, baseline_name, steps):
    """One complete independently fitted historical + oracle target estimate."""
    start = perf_counter()
    baseline = baseline_from_history(training, history, baseline_name, K)
    historical = [interval_window(training, previous, baseline, steps, K)[1]
                  for previous in history]
    model, forecast, _, observed = interval_window(frame, day, baseline, steps, K)
    sampled = np.arange(K, dtype=int)*steps

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
        "initial":init, "tau2":float(tau2), "sigma2":float(sigma2),
        "process_identifiable":bool(identifiable),
        "demand_metrics":metrics(observed,predicted),
        "historical_rmse":float(np.sqrt(np.mean(historical_error**2))),
        "baseline_demand_rmse":float(np.sqrt(np.mean(
            (forecast["O"][sampled]-observed)**2))),
        "total_compute_seconds":total_time, "drift_seconds":drift_time,
        "diffusion_seconds":diffusion_time, "oracle_seconds":oracle_time,
    }


def pooled_metrics(items):
    """Pooled across exactly 12*24 target observations for each baseline."""
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


def plot_days(results, path, show=False):
    """Exactly 3 rows x 4 columns, one subplot per reference prediction date."""
    fig, axes = plt.subplots(3,4,figsize=(20,12),sharex=True,sharey=True)
    hours = np.arange(K)
    for index,axis in enumerate(axes.flat):
        target = results[BASELINES[0]][index]["day"]
        axis.plot(hours,results[BASELINES[0]][index]["observed"],color="black",
                  lw=1.9,marker="o",markersize=2.2,label="Observed")
        for baseline in BASELINES:
            result = results[baseline][index]
            mean,std,color=result["predicted"],result["std"],COLORS[baseline]
            axis.fill_between(hours,mean-Z95*std,mean+Z95*std,color=color,alpha=0.085)
            axis.fill_between(hours,mean-Z68*std,mean+Z68*std,color=color,alpha=0.15)
            axis.plot(hours,mean,color=color,lw=1.7,label=LABELS[baseline])
        axis.set_title(f"{index+1:02d} | {target:%a %d %b %Y}",fontsize=10)
        axis.set_xlim(0,K-1)
        axis.set_xticks([0,4,8,12,16,20,23])
        axis.grid(alpha=0.23)
        if index//4 == 2: axis.set_xlabel("Hour of day")
        if index%4 == 0: axis.set_ylabel("Normalized demand")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.extend([Patch(color="gray",alpha=0.25,label="Central 68% interval"),
                    Patch(color="gray",alpha=0.12,label="Central 95% interval")])
    fig.legend(handles,[*labels,"Central 68% interval","Central 95% interval"],
               loc="upper center",bbox_to_anchor=(0.5,0.962),ncol=4,fontsize=10)
    fig.suptitle("ARM: 12 selected oracle-conditioned prediction days | Weekly temperature | ALL | N=6, k=24 h",
                 fontsize=15,y=0.995)
    fig.tight_layout(rect=(0,0,1,0.92))
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
    ax.set_title("ARM | Weekly temperature | 12-day pooled empirical vs predicted error accumulation")
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
    dates,histories=reference_dates()
    # Disk reading, alignment, and historical hourly imputation are outside timers.
    frame=group_frame("ALL")
    needed=pd.DatetimeIndex([d for history in histories
                             for d in historical_calendar_days(history,K)]).unique().sort_values()
    training,imputed=impute_hourly_days(frame,days=needed,
                                        columns=("I","U","Temp"),require_days=True)
    results={name:[] for name in BASELINES}
    args.output.mkdir(parents=True,exist_ok=True)

    for sample,(day,history) in enumerate(zip(dates,histories),1):
        for name in BASELINES:
            item=fit_origin(frame,training,day,history,name,args.steps_per_hour)
            results[name].append(item)
            metric=item["demand_metrics"]
            print(f"[{sample:02d}/12] {day:%Y-%m-%d} | {LABELS[name]:18s} | "
                  f"RMSE={metric['rmse']:.6f} | MAE={metric['mae']:.6f} | "
                  f"bias={metric['bias']:+.6f} | compute={item['total_compute_seconds']:.3f}s",
                  flush=True)

    hourly,daily,pooled,curves=make_tables(results,args.output)
    plot_days(results,args.output/"selected_12_days_3x4.png",args.show)
    plot_cumulative(curves,args.output/"pooled_cumulative_error.png",args.show)
    config={
        "group":"ALL","N":N,"k":K,"origin_hour":0,"steps_per_hour":args.steps_per_hour,
        "dates":[f"{day:%Y-%m-%d}" for day in dates],
        "baselines":list(BASELINES),
        "input":"historical N corresponding weekdays as defined in BM1 reference date table",
        "deterministic_fit":"historical joint convex QP",
        "diffusion_fit":"historical free-running squared residuals; joint nonnegative variance matching",
        "target_initial_state":"oracle optimized from target observations",
        "time_definition":"complete weekly-temperature estimation and prediction, excludes initial ALL data reading and imputation",
        "cumulative_plot":"sum over lead hours of sqrt(mean over 12 dates of squared hourly errors); predicted uses sqrt(mean model-implied variance)",
        "signed_bias":"predicted minus observed",
        "empirical_coverage":"pointwise 95 percent Gaussian intervals",
        "historical_imputed_hourly_slots":int(imputed["imputed_hourly_slots"]),
    }
    (args.output/"configuration.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("\nPOOLED TARGET PERFORMANCE (12 dates x 24 hours, weekly temperature baseline)")
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
    print(f"\nSaved results to {args.output}")
    return pooled


def main():
    parser=ArgumentParser(description="Fit ARM using only the weekly-temperature baseline and historical diffusion on 12 BM1 dates (ALL, N=6, k=24, origin 00:00).")
    parser.add_argument("--steps-per-hour",type=int,default=60)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--show",action="store_true")
    return run(parser.parse_args())


if __name__=="__main__": main()

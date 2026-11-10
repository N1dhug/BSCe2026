"""BM1 beta-mixture Fokker-Planck MLE on the same 14 ALL dates as selected_bm_dates_arm.

N=6 matching historical weekdays, k=24 hourly observations, origin at 00:00,
weekly_temperature baseline only, and the same historical imputation.

BM1 price U must lie in [0,1]; map the ARM [-1,1] price U to (U+1)/2.
Demand I retains EXACTLY the ARM season-normalized scale: no extra
per-origin demand normalization and no clipping of evaluation data.

For every date: fit baseline, deterministic BM1, stochastic joint likelihood
over six historical trajectories, then retrospectively optimize target X0
and solve the Fokker-Planck equation using a beta-mixture approximation.
The reported predictive RMSE uses the open-loop distribution's expected demand.

Target "pooled marginal NLL" sums -log p(I_j | historical fit, oracle X0)
from each OPEN-LOOP predicted beta-mixture demand density convolved with
normal observation noise. This is NOT the joint time-series NLL: marginal
errors are dependent. The separate sequential/filtered joint NLL uses the
normal BM1 observation-updating rule, also with the oracle X0.

Oracle X0 is selected against target observations, so this evaluates an
oracle-conditioned fit, NOT an operational forecast. Computation times
include fitting, Fokker-Planck propagation, oracle search and target scoring,
but exclude data loading, imputation, CSV writes, and figure rendering.

Run from repository root:
    python -m src.exe.run_model.bm_1.selected_bm_dates_beta --show
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from src.exe.run_model.wm_4.moving_horizon_common import group_frame, metrics
from src.exe.run_model.wm_4.selected_bm_dates_arm import (
    BASELINES, K, N, reference_dates,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    baseline_from_history, historical_calendar_days,
)
from src.method.bm_1_beta import bm_1_beta
from src.params.bm_1_beta import bm_1_beta_params
from src.tools.impute_hourly import impute_hourly_days
from src.validation.exe.run_model.bm_1.fit_beta_moving_horizon import (
    bm_1_beta_moving_horizon_validation as checks,
)


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"bm_1"/"selected_bm_dates_beta"
BASELINE = "weekly_temperature"
Z95 = float(norm.ppf(0.975))


def inputs():
    """Read ALL once and enforce the exact ARM N,k dates, scale and imputation."""
    if tuple(BASELINES)!=(BASELINE,) or N!=6 or K!=24:
        raise ValueError("Expected ARM reference experiment: weekly_temperature, N=6, k=24.")
    dates,histories=reference_dates()
    if len(dates)!=14: raise ValueError("Expected 14 prediction dates, two per weekday.")
    raw=group_frame("ALL").copy()
    raw["U"]=0.5*(raw["U"]+1.0)  # BM1 price domain; ARM I normalization unchanged.
    needed=pd.DatetimeIndex([day for history in histories
                             for day in historical_calendar_days(history,K)]).unique().sort_values()
    missing_hist=[]
    for day in needed:
        section=raw.reindex(pd.date_range(day,periods=24,freq="h"))[["I","U","Temp"]]
        if section.notna().sum().eq(0).any(): missing_hist.append(str(day.date()))
    missing_target=[]
    for date in dates:
        section=raw.reindex(pd.date_range(date,periods=K,freq="h"))[["I","U","Temp"]]
        if section.isna().any().any(): missing_target.append(str(date.date()))
    if missing_hist or missing_target:
        raise ValueError("Incomplete BM1/ARM common data: "
                         f"missing historical day(s)={missing_hist}; incomplete target day(s)={missing_target}.")
    training,imputation=impute_hourly_days(raw,days=needed,columns=("I","U","Temp"),
                                           require_days=True)
    for item in (training,raw.reindex(pd.DatetimeIndex([date+pd.Timedelta(hours=h)
                                                        for date in dates for h in range(K)]))):
        for field in ("I","U"):
            values=item[field].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values<-1e-12).any() or (values>1.+1e-12).any():
                raise ValueError(f"BM1 requires {field} in [0,1] on all selected dates.")
    return dates,histories,raw,training,imputation


def open_loop_density(observed, forecast, observation_sd):
    """Compute marginal demand variance and negative log density from beta mixtures.

    Forecast supports/weights exclude sigma_y; observation noise is added
    analytically as a Gaussian convolution at every hour.
    """
    observed=np.asarray(observed,dtype=float)
    mean=np.asarray(forecast["mean"],dtype=float)
    if observed.shape!=(K,) or mean.shape!=(K,):
        raise ValueError("Target observations and beta means must have 24 hourly samples.")
    if not np.isfinite(observed).all() or observation_sd<=0.:
        raise ValueError("BM1 likelihood needs finite demand and positive sigma_y.")
    variance=np.empty(K,dtype=float)
    nll=np.empty(K,dtype=float)
    for hour,(support,mass) in enumerate(zip(forecast["supports"],forecast["weights"])):
        x=np.asarray(support,dtype=float).reshape(-1)
        w=np.asarray(mass,dtype=float).reshape(-1)
        if len(x)!=len(w) or not len(x) or (w<0.).any() or not np.isfinite(np.r_[x,w]).all():
            raise ValueError(f"Invalid beta-mixture demand distribution at hour {hour}.")
        w=w/w.sum()
        actual_mean=float(w@x)
        if not np.isclose(actual_mean,mean[hour],atol=1e-8,rtol=1e-7):
            raise ValueError(f"BM1 beta-mixture moments inconsistent at hour {hour}.")
        variance[hour]=float(w@((x-actual_mean)**2)+observation_sd**2)
        valid=w>0.
        logpdf=norm.logpdf(observed[hour],loc=x[valid],scale=observation_sd)
        nll[hour]=float(-logsumexp(np.log(w[valid])+logpdf))
    if not np.isfinite(np.r_[variance,nll]).all() or (variance<=0.).any():
        raise ValueError("Invalid BM1 predicted mixture variance or target density.")
    return variance,nll


def fit_day(raw,training,date,history,args):
    """Exact historical BM1 deterministic + stochastic MLE and oracle forecast."""
    t0=perf_counter()
    base=baseline_from_history(training,history,BASELINE,K)
    historical=[checks.window(training,day,K,base) for day in history]
    target=checks.window(raw,date,K,base)
    prepared_seconds=perf_counter()-t0

    estimator=bm_1_beta_params()
    start=perf_counter()
    deterministic=estimator.fit_deterministic(
        historical,initial_state=args.x0,cycles=args.deterministic_cycles,
        max_nfev=args.max_nfev,state_maxiter=args.state_maxiter,substeps=args.substeps)
    deterministic_seconds=perf_counter()-start

    start=perf_counter()
    fitted=estimator.fit_mle(
        historical,deterministic,components=args.fit_components,
        quadrature=args.fit_quadrature,reconstruction_quadrature=args.fit_reconstruction_quadrature,
        max_step=args.fit_max_step,fixed_steps_per_hour=args.fit_steps_per_hour,
        maxiter=args.maxiter,maxfun=args.maxfun,
        nll_ratio_threshold=args.nll_ratio_threshold,progress=args.progress)
    mle_seconds=perf_counter()-start
    if not fitted["converged"]:
        msg=f"BM1 MLE not converged at {date:%Y-%m-%d}: {fitted['message']}"
        if args.strict: raise RuntimeError(msg)
        print("[BM1 beta] WARNING: "+msg,flush=True)

    start=perf_counter()
    oracle=estimator.oracle(
        target,fitted,components=args.oracle_components,quadrature=args.oracle_quadrature,
        reconstruction_quadrature=args.oracle_reconstruction_quadrature,
        max_step=args.oracle_max_step,maxiter=args.oracle_maxiter)
    oracle_seconds=perf_counter()-start

    start=perf_counter()
    forecast=oracle["prediction"]
    expected=np.asarray(forecast["mean"],dtype=float)
    variance,marginal_nll=open_loop_density(target["I"],forecast,float(fitted["sigma_y"]))
    # Sequential likelihood is computed using the same oracle X0 and
    # observation-updating filter as the historical stochastic MLE.
    prediction_solver=bm_1_beta(
        args.oracle_components,args.oracle_quadrature,
        args.oracle_reconstruction_quadrature,args.oracle_max_step)
    sequential_nll=float(prediction_solver.simulate(
        target,fitted,float(oracle["initial_state"]),observations=target["I"],
        nll_only=True)["nll"])
    scoring_seconds=perf_counter()-start
    total_seconds=perf_counter()-t0
    if not np.isfinite(sequential_nll):
        raise ValueError(f"Invalid BM1 target sequential NLL at {date}.")

    error=expected-target["I"]
    demand=metrics(target["I"],expected)
    return {
        "date":date,"history":history,"target":target,
        "expected":expected,"variance":variance,"std":np.sqrt(variance),
        "forecast":forecast,"error":error,"marginal_nll":marginal_nll,
        "sequential_nll":sequential_nll,"demand":demand,
        "oracle_x0":float(oracle["initial_state"]),
        "historical_mle_nll":float(fitted["nll"]),
        "fitted":fitted,"deterministic":deterministic,
        "mle_converged":bool(fitted["converged"]),
        "mle_stop_reason":str(fitted["stop_reason"]),
        "mle_evaluations":int(fitted["evaluations"]),
        "baseline_seconds":prepared_seconds,"deterministic_seconds":deterministic_seconds,
        "mle_seconds":mle_seconds,"oracle_seconds":oracle_seconds,
        "scoring_seconds":scoring_seconds,"total_compute_seconds":total_seconds,
    }


def summarize(rows):
    """Pool 14x24 fixed dates, never average the daily RMSE values."""
    e=np.stack([row["error"] for row in rows])
    variance=np.stack([row["variance"] for row in rows])
    likelihood=np.stack([row["marginal_nll"] for row in rows])
    observed_rmse_by_hour=np.sqrt(np.mean(e**2,axis=0))
    predicted_rmse_by_hour=np.sqrt(np.mean(variance,axis=0))
    nll_total=float(np.sum(likelihood))
    covered=np.stack([(row["target"]["I"]>=row["forecast"]["lower_95"]) &
                      (row["target"]["I"]<=row["forecast"]["upper_95"]) for row in rows])
    total_time=np.array([row["total_compute_seconds"] for row in rows],dtype=float)
    return {
        "model":"bm_1_beta_mixture","baseline":BASELINE,"N":N,"k":K,
        "prediction_days":len(rows),"observations":int(e.size),
        "expected_value_rmse":float(np.sqrt(np.mean(e**2))),
        "expected_value_mae":float(np.mean(np.abs(e))),
        "expected_value_bias":float(np.mean(e)),
        "predicted_marginal_rmse":float(np.sqrt(np.mean(variance))),
        "pooled_rmse_absolute_discrepancy":float(abs(np.sqrt(np.mean(e**2))-
                                                      np.sqrt(np.mean(variance)))),
        "mean_hourly_rmse_absolute_discrepancy":float(np.mean(
            np.abs(observed_rmse_by_hour-predicted_rmse_by_hour))),
        "pooled_marginal_nll":nll_total,
        "mean_marginal_nll_per_observation":float(np.mean(likelihood)),
        "pooled_sequential_target_nll":float(np.sum([r["sequential_nll"] for r in rows])),
        "pooled_historical_mle_nll":float(np.sum([r["historical_mle_nll"] for r in rows])),
        "empirical_95pct_coverage":float(100.*np.mean(covered)),
        "mean_complete_compute_seconds":float(np.mean(total_time)),
        "total_complete_compute_seconds":float(np.sum(total_time)),
        "mean_mle_seconds":float(np.mean([r["mle_seconds"] for r in rows])),
        "mle_converged_days":int(sum(r["mle_converged"] for r in rows)),
    }


def save_tables(rows,pooled,output):
    hourly,daily,cumulative=[],[],[]
    e=np.stack([row["error"] for row in rows])
    variance=np.stack([row["variance"] for row in rows])
    observed_hourly=np.sqrt(np.mean(e**2,axis=0))
    predicted_hourly=np.sqrt(np.mean(variance,axis=0))
    for sample,item in enumerate(rows,1):
        data=item["target"]
        f=item["forecast"]
        p=item["fitted"]
        daily.append({
            "sample":sample,"day":str(item["date"].date()),"baseline":BASELINE,
            "rmse":item["demand"]["rmse"],"mae":item["demand"]["mae"],
            "bias":item["demand"]["bias"],"oracle_X0":item["oracle_x0"],
            "target_marginal_nll":float(np.sum(item["marginal_nll"])),
            "target_marginal_nll_per_observation":float(np.mean(item["marginal_nll"])),
            "target_sequential_nll":item["sequential_nll"],
            "historical_mle_nll":item["historical_mle_nll"],
            "mle_converged":item["mle_converged"],
            "mle_stop_reason":item["mle_stop_reason"],
            "mle_evaluations":item["mle_evaluations"],
            "baseline_fit_and_windows_seconds":item["baseline_seconds"],
            "deterministic_fit_seconds":item["deterministic_seconds"],
            "stochastic_mle_seconds":item["mle_seconds"],
            "oracle_search_and_prediction_seconds":item["oracle_seconds"],
            "target_nll_evaluation_seconds":item["scoring_seconds"],
            "total_compute_seconds":item["total_compute_seconds"],
            **{key:float(p[key]) for key in (*bm_1_beta_params.NAMES,"sigma_x","sigma_y")},
        })
        for hour in range(K):
            y=float(data["I"][hour])
            hourly.append({
                "sample":sample,"day":str(item["date"].date()),"baseline":BASELINE,
                "hour":hour,"timestamp":str(data["timestamps"][hour]),
                "observed":y,"weekly_temperature_baseline":float(data["B"][hour]),
                "price_U_0_1":float(data["U"][hour]),
                "beta_expected_demand":float(item["expected"][hour]),
                "predicted_demand_std":float(item["std"][hour]),
                "predicted_demand_variance":float(item["variance"][hour]),
                "residual_predicted_minus_observed":float(item["error"][hour]),
                "target_marginal_nll":float(item["marginal_nll"][hour]),
                "lower_68":float(f["lower_68"][hour]),
                "upper_68":float(f["upper_68"][hour]),
                "lower_95":float(f["lower_95"][hour]),
                "upper_95":float(f["upper_95"][hour]),
            })
    for hour in range(K):
        cumulative.append({
            "lead_hour":hour+1,
            "empirical_hourly_rmse":float(observed_hourly[hour]),
            "predicted_hourly_rmse":float(predicted_hourly[hour]),
            "empirical_cumulative_hourly_rmse":float(np.sum(observed_hourly[:hour+1])),
            "predicted_cumulative_hourly_rmse":float(np.sum(predicted_hourly[:hour+1])),
            "cumulative_abs_hourly_rmse_discrepancy":float(np.sum(
                np.abs(observed_hourly[:hour+1]-predicted_hourly[:hour+1]))),
        })
    pd.DataFrame(hourly).to_csv(output/"hourly_predictions.csv",index=False)
    pd.DataFrame(daily).to_csv(output/"per_day_metrics.csv",index=False)
    pd.DataFrame([pooled]).to_csv(output/"pooled_metrics.csv",index=False)
    pd.DataFrame(cumulative).to_csv(output/"pooled_cumulative_errors.csv",index=False)
    return pd.DataFrame(cumulative)


def plot_days(rows,output,show=False):
    fig,axes=plt.subplots(2,7,figsize=(25,9),sharex=True,sharey=True)
    h=np.arange(K)
    for i,ax in enumerate(axes.flat):
        item=rows[i]
        observed=item["target"]["I"]
        pred=item["expected"]
        f=item["forecast"]
        ax.plot(h,observed,color="black",lw=1.7,marker="o",markersize=2.4,
                label="Observed demand")
        ax.plot(h,pred,color="tab:orange",lw=1.9,label="Beta-mixture expected demand")
        ax.fill_between(h,f["lower_95"],f["upper_95"],color="tab:orange",alpha=0.09,
                        label="95% predicted interval" if i==0 else None)
        ax.fill_between(h,f["lower_68"],f["upper_68"],color="tab:orange",alpha=0.18,
                        label="68% predicted interval" if i==0 else None)
        ax.set_title(f"{i+1:02d} | {item['date']:%a %d %b %Y}",fontsize=10)
        ax.set_xlim(0,K-1)
        ax.set_xticks([0,4,8,12,16,20,23])
        ax.grid(alpha=0.22)
        if i//7==1: ax.set_xlabel("Hour of day")
        if i%7==0: ax.set_ylabel("Normalized demand")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,0.965),
               ncol=4,fontsize=10)
    fig.suptitle("BM1 Fokker-Planck beta-mixture | 14 oracle-conditioned days | "
                 "ALL, weekly temperature, N=6, k=24",fontsize=15,y=0.995)
    fig.tight_layout(rect=(0,0,1,0.92))
    fig.savefig(output/"bm1_14_days_2x7.png",dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def plot_cumulative(cumulative,output,show=False):
    h=cumulative["lead_hour"].to_numpy(dtype=int)
    actual=cumulative["empirical_cumulative_hourly_rmse"].to_numpy(dtype=float)
    expected=cumulative["predicted_cumulative_hourly_rmse"].to_numpy(dtype=float)
    fig,ax=plt.subplots(figsize=(10,5.6))
    ax.plot(h,actual,color="black",lw=2.2,marker="o",ms=3,
            label="Observed cumulative hourly RMSE")
    ax.plot(h,expected,color="tab:orange",ls="--",lw=2.2,marker="s",ms=3,
            label="Beta-mixture predicted cumulative hourly RMSE")
    ax.fill_between(h,actual,expected,color="tab:orange",alpha=0.10)
    ax.set_title("BM1 beta-mixture | 14-day pooled marginal uncertainty vs empirical error")
    ax.set_xlabel("Prediction lead hour")
    ax.set_ylabel("Cumulative sum of hourly RMSE (normalized demand)")
    ax.set_xlim(1,K)
    ax.set_ylim(bottom=0.)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output/"bm1_pooled_cumulative_rmse.png",dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def run(args):
    if args.fit_components<1 or args.fit_steps_per_hour<1:
        raise ValueError("Fokker-Planck beta-mixture fitting resolution must be positive.")
    if args.strict and args.maxiter<1: raise ValueError("Invalid MLE maximum iterations.")
    dates,histories,raw,training,imputation=inputs()  # OUTSIDE timing.
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(date,history) in enumerate(zip(dates,histories),1):
        print(f"[BM1 beta {i:02d}/{len(dates)}] {date:%Y-%m-%d} | "
              f"historical N={N} | k={K} | start",flush=True)
        item=fit_day(raw,training,date,history,args)
        rows.append(item)
        print(f"[BM1 beta {i:02d}/{len(dates)}] RMSE={item['demand']['rmse']:.7f} | "
              f"oracle X0={item['oracle_x0']:.5f} | "
              f"target marginal NLL={item['marginal_nll'].sum():.5f} | "
              f"MLE evaluations={item['mle_evaluations']} | "
              f"complete compute={item['total_compute_seconds']:.3f}s",flush=True)

    summary=summarize(rows)
    cumulative=save_tables(rows,summary,args.output)
    plot_days(rows,args.output,args.show)
    plot_cumulative(cumulative,args.output,args.show)
    configuration={
        "group":"ALL","baseline":BASELINE,"N":N,"k":K,
        "dates":[str(date.date()) for date in dates],
        "historical_imputation":"same within-day I/Temp interpolation and piecewise constant price as ARM",
        "imputed_hourly_slots":int(imputation["imputed_hourly_slots"]),
        "demand_scale":"original ALL season-normalized [0,1], exactly the ARM demand scale; not rescaled per origin",
        "price_scale":"ARM price [-1,1] mapped to BM1 [0,1] by (U+1)/2",
        "fit":"BM1 alternating deterministic fit then joint beta-mixture stochastic MLE on N historical intervals",
        "prediction":"oracle fitted X0; open-loop beta-mixture Fokker-Planck propagation, not operational",
        "primary_nll":"sum of negative log open-loop marginal target densities, Gaussian observation-convolved beta mixtures",
        "secondary_nll":"sequential target likelihood with hourly observation updates (also oracle-conditioned)",
        "timing":"baseline preparation + deterministic + MLE + oracle optimization/propagation + NLL, excluding input read, imputation and output",
        "mle_resolution":{"components":args.fit_components,
                           "fixed_steps_per_hour":args.fit_steps_per_hour,
                           "quadrature":args.fit_quadrature,
                           "reconstruction_quadrature":args.fit_reconstruction_quadrature},
        "oracle_resolution":{"components":args.oracle_components,
                             "quadrature":args.oracle_quadrature,
                             "reconstruction_quadrature":args.oracle_reconstruction_quadrature,
                             "max_step":args.oracle_max_step},
    }
    (args.output/"configuration.json").write_text(json.dumps(configuration,indent=2),encoding="utf-8")
    print("\nBM1 FOKKER-PLANCK BETA-MIXTURE | SAME 14 ARM DATES, N=6, k=24, ALL")
    print(f"Expected-demand pooled RMSE={summary['expected_value_rmse']:.8f} | "
          f"MAE={summary['expected_value_mae']:.8f} | "
          f"bias(pred-obs)={summary['expected_value_bias']:+.8f}")
    print(f"Total pooled open-loop marginal target NLL={summary['pooled_marginal_nll']:.8f} | "
          f"mean marginal NLL/observation={summary['mean_marginal_nll_per_observation']:.8f}")
    print(f"Total pooled filtered sequential target NLL={summary['pooled_sequential_target_nll']:.8f}")
    print(f"Model-implied RMSE={summary['predicted_marginal_rmse']:.8f} | "
          f"mean hourly RMSE discrepancy={summary['mean_hourly_rmse_absolute_discrepancy']:.8f} | "
          f"empirical 95% coverage={summary['empirical_95pct_coverage']:.2f}%")
    print(f"Mean complete compute time per date={summary['mean_complete_compute_seconds']:.3f}s | "
          f"sum across dates={summary['total_complete_compute_seconds']:.3f}s | "
          f"mean MLE time={summary['mean_mle_seconds']:.3f}s | "
          f"MLE converged={summary['mle_converged_days']}/{len(rows)}")
    print(f"Outputs: {args.output}")
    return summary


def main():
    parser=ArgumentParser(description="Evaluate BM1 beta-mixture stochastic MLE on exact ARM 14-day N=6 k=24 weekly-temperature cohort.")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--x0",type=float,default=0.5)
    parser.add_argument("--deterministic-cycles",type=int,default=4)
    parser.add_argument("--max-nfev",type=int,default=120)
    parser.add_argument("--state-maxiter",type=int,default=60)
    parser.add_argument("--substeps",type=int,default=2)
    parser.add_argument("--maxiter",type=int,default=45)
    parser.add_argument("--maxfun",type=int,default=700)
    parser.add_argument("--fit-components",type=int,default=8)
    parser.add_argument("--fit-quadrature",type=int,default=8)
    parser.add_argument("--fit-reconstruction-quadrature",type=int,default=48)
    parser.add_argument("--fit-max-step",type=float,default=0.25)
    parser.add_argument("--fit-steps-per-hour",type=int,default=1)
    parser.add_argument("--nll-ratio-threshold",type=float,default=0.99)
    parser.add_argument("--oracle-components",type=int,default=24)
    parser.add_argument("--oracle-quadrature",type=int,default=12)
    parser.add_argument("--oracle-reconstruction-quadrature",type=int,default=48)
    parser.add_argument("--oracle-max-step",type=float,default=0.125)
    parser.add_argument("--oracle-maxiter",type=int,default=70)
    parser.add_argument("--strict",action="store_true",help="Reject an unconverged historical stochastic MLE.")
    parser.add_argument("--progress",action="store_true",help="Print historical MLE optimization progress.")
    parser.add_argument("--show",action="store_true")
    return run(parser.parse_args())


if __name__=="__main__": main()

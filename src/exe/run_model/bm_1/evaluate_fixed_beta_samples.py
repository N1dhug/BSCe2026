"""Twelve-date BM1 fixed-beta evaluation: historical fit and oracle target.

Run from the repository root:
    python -m src.exe.run_model.bm_1.evaluate_fixed_beta_samples --output files/output/bm_1/fixed_beta_samples

Each of the 12 dates uses N=6 matching weekday intervals, k=24 hours and
the weekly-temperature baseline. The seven BM1 stochastic parameters and six
independent initial beta distributions are fitted to historical observations
only; the target uses a separate full-simplex oracle initial density.

Historical NLL / expected-demand RMSE / coverage use sequential one-step-ahead
densities conditioned on *earlier observations in the same historical day*.
Target NLL / RMSE / coverage use OPEN-LOOP marginal distributions from the
oracle-conditioned target initial density; no target state filtering occurs.
The combined scores concatenate these two clearly distinguished contracts.
The target observations do enter the oracle initial density (not a real forecast).

Output: daily_scores.csv, pooled_scores.csv, target_hourly.csv,
historical_hourly.csv, summary.json, best_worst_nll.png, stage_timings.png.
The initial-guess stage estimates seed initial densities from historical
first-hour observations; no legacy deterministic initial-model MLE is run.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.method.moving_horizon import group_frame
from src.exe.run_model.wm_4.sweep_weekday_horizon import baseline_from_history, historical_calendar_days
from src.tools.impute_hourly import impute_hourly_days
from src.validation.exe.run_model.bm_1.fit_beta_moving_horizon import bm_1_beta_moving_horizon_validation as checks
from src.model.bm_1 import bm_1


N, K, BASELINE = 6, 24, "weekly_temperature"
DATES = (
    "2022-04-11", "2022-04-19", "2022-04-27", "2022-05-05",
    "2022-05-13", "2022-05-21", "2022-05-29", "2022-05-30",
    "2022-06-07", "2022-08-10", "2022-08-18", "2022-08-26",
)
ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "bm_1" / "fixed_beta_samples"
RED, BLACK, LIGHT = "#c72e35", "#232323", "#f2b7b9"


def json_value(value):
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, pd.Timestamp): return str(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def historical_origins(day):
    day = pd.Timestamp(day)
    return [day - pd.Timedelta(weeks=week) for week in range(N, 0, -1)]


def prepare_all(days, tag):
    """Load input once, verify all required calendar days, impute history only."""
    raw = group_frame(tag).copy()
    raw["U"] = .5 * (raw["U"] + 1.)
    histories = {day: historical_origins(day) for day in days}
    needed = pd.DatetimeIndex([date for history in histories.values()
                               for date in historical_calendar_days(history,K)]).unique().sort_values()
    missing = []
    for date in needed:
        window = raw.reindex(pd.date_range(date, periods=K, freq="h"))[["I", "U", "Temp"]]
        if window.notna().sum().eq(0).any(): missing.append(str(date.date()))
    for day in days:
        target = raw.reindex(pd.date_range(day, periods=K, freq="h"))[["I", "U", "Temp"]]
        if target.isna().any().any(): raise ValueError(f"Target {day.date()} has missing observations.")
    if missing: raise ValueError(f"Entire historical days are unavailable: {sorted(set(missing))}.")
    history_data, imputation = impute_hourly_days(raw, days=needed, columns=("I","U","Temp"),
                                                   require_days=True)
    for data in (history_data, raw.loc[raw.index.isin(pd.DatetimeIndex(
            [day + pd.Timedelta(hours=i) for day in days for i in range(K)]))]):
        for name in ("I", "U"):
            values = data[name].to_numpy(dtype=float)
            if np.any(~np.isfinite(values)) or np.any((values < 0.) | (values > 1.)):
                raise ValueError(f"BM1 {name} input is not finite and normalized to [0,1].")
    return raw, history_data, histories, imputation


def interval_score(observed, simulation, lower, upper):
    """Sufficient pooled statistics; NLL can legitimately be negative."""
    observed = np.asarray(observed, dtype=float)
    prediction = np.asarray(simulation["mean"], dtype=float)
    logpdf = np.asarray(simulation["loglik"], dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if not (observed.shape == prediction.shape == logpdf.shape == lower.shape == upper.shape):
        raise ValueError("Scoring requires the same number of observations, means, and quantiles.")
    if not np.all(np.isfinite(np.r_[observed,prediction,logpdf,lower,upper])):
        raise ValueError("Scoring encountered nonfinite observations, means or intervals.")
    if np.any(lower > upper): raise ValueError("Observation interval boundaries are reversed.")
    errors = prediction-observed
    covered = (observed >= lower) & (observed <= upper)
    return {"observations":int(observed.size), "nll_total":float(-np.sum(logpdf)),
            "sse":float(np.dot(errors,errors)), "covered":int(np.sum(covered)),
            "rmse":float(np.sqrt(np.mean(errors**2))),
            "nll_mean":float(-np.mean(logpdf)),
            "coverage_95_pct":float(100.*np.mean(covered))}


def pooled(scores):
    """Pool hours before RMSE, mean NLL and coverage calculations."""
    count = int(sum(item["observations"] for item in scores))
    if count <= 0: raise ValueError("Cannot pool an empty set of scored intervals.")
    total_nll = float(sum(item["nll_total"] for item in scores))
    sse = float(sum(item["sse"] for item in scores))
    covered = int(sum(item["covered"] for item in scores))
    return {"observations":count, "nll_total":total_nll,
            "nll_mean":total_nll/count, "rmse":float(np.sqrt(sse/count)),
            "coverage_95_pct":100.*covered/count,
            "covered":covered, "sse":sse}


def one_date(day, history, raw, training, args):
    """One complete timed fit and oracle; exclude global data loading and plotting."""
    started = perf_counter()
    baseline = baseline_from_history(training, history, BASELINE, K)
    hist = [checks.window(training, date, K, baseline) for date in history]
    target = checks.window(raw, day, K, baseline)
    prepared_seconds = perf_counter() - started

    # The solver calls the existing BM1 response and demand functions.
    model = bm_1(lambda t:.5, lambda t:.5, (0.,float(K)), K+1)
    estimator = model.fixed_beta_estimator(args.degree,args.quadrature,args.steps_per_hour)

    t = perf_counter()
    seed = estimator.initial_guess(hist)
    initial_guess_seconds = perf_counter()-t

    t = perf_counter()
    fitted = estimator.fit(hist,maxiter=args.maxiter,maxfun=args.maxfun,
                           seed=seed["parameters"],
                           initial_distributions=seed["initial_distributions"],
                           progress=not args.quiet)
    stochastic_seconds = perf_counter()-t
    if not fitted["converged"]:
        print(f"[BM1 fixed beta] WARNING: {day.date()} MLE has not converged: {fitted['message']}",flush=True)
        if args.strict: raise RuntimeError(f"Stochastic MLE failed at {day.date()}: {fitted['message']}")

    t = perf_counter()
    oracle = estimator.oracle(target,fitted,maxiter=args.oracle_maxiter)
    oracle_seconds = perf_counter()-t
    if not oracle["oracle_converged"]:
        print(f"[BM1 fixed beta] WARNING: {day.date()} oracle optimizer: {oracle['oracle_message']}",flush=True)
        if args.strict: raise RuntimeError(f"Oracle optimizer failed at {day.date()}: {oracle['oracle_message']}")

    t = perf_counter()
    historical_scores, historical_rows = [], []
    for date, data, initial in zip(history,hist,fitted["initial_distributions"]):
        forecast = estimator.solver.simulate(data,fitted,initial,observations=True)
        lower, upper = estimator.solver.intervals(forecast,fitted["sigma_y"])
        score = interval_score(data["I"],forecast,lower,upper)
        historical_scores.append(score)
        for hour in range(K):
            historical_rows.append({
                "target_day":str(day.date()), "historical_day":str(date.date()), "hour":hour,
                "observed":float(data["I"][hour]),"expected_demand":float(forecast["mean"][hour]),
                "lower_95":float(lower[hour]),"upper_95":float(upper[hour]),
                "conditional_hourly_nll":float(-forecast["loglik"][hour]),
            })
    historical = pooled(historical_scores)
    if not np.isclose(historical["nll_total"],fitted["nll"],atol=1e-5,rtol=1e-9):
        raise RuntimeError("Historical reconstructed NLL disagrees with the optimizer objective.")

    target_forecast = oracle["prediction"]
    target_score = interval_score(target["I"],target_forecast,
                                  oracle["lower_95"],oracle["upper_95"])
    complete = pooled([historical,target_score])
    target_rows = [{
        "day":str(day.date()),"hour":hour,"timestamp":str(target["timestamps"][hour]),
        "observed":float(target["I"][hour]),"baseline":float(target["B"][hour]),
        "price":float(target["U"][hour]),"expected_demand":float(target_forecast["mean"][hour]),
        "lower_95":float(oracle["lower_95"][hour]),"upper_95":float(oracle["upper_95"][hour]),
        "marginal_hourly_nll":float(-target_forecast["loglik"][hour]),
    } for hour in range(K)]
    scoring_seconds = perf_counter()-t
    total_seconds = perf_counter()-started
    timings = {"input_baseline":float(prepared_seconds),
               "initial_guess":float(initial_guess_seconds),
               "stochastic_mle":float(stochastic_seconds),
               "oracle_optimization":float(oracle_seconds),
               "historical_scoring":float(scoring_seconds),
               "complete":float(total_seconds)}
    row = {"day":str(day.date()),"history_nll_total":historical["nll_total"],
           "history_nll_mean":historical["nll_mean"],
           "history_rmse":historical["rmse"],
           "history_coverage_95_pct":historical["coverage_95_pct"],
           "target_nll_total":target_score["nll_total"],
           "target_nll_mean":target_score["nll_mean"],
           "target_rmse":target_score["rmse"],
           "target_coverage_95_pct":target_score["coverage_95_pct"],
           "combined_nll_total":complete["nll_total"],
           "combined_nll_mean":complete["nll_mean"],
           "combined_rmse":complete["rmse"],
           "combined_coverage_95_pct":complete["coverage_95_pct"],
           **{f"seconds_{key}":value for key,value in timings.items()},
           "mle_converged":bool(fitted["converged"]),
           "oracle_converged":bool(oracle["oracle_converged"]),
           "mle_evaluations":int(fitted["evaluations"]),
           "invalid_evaluations":int(fitted["invalid_evaluations"])}
    print(f"[BM1 fixed beta] {day:%Y-%m-%d} | history NLL {historical['nll_mean']:.5f}/h "
          f"({historical['nll_total']:.4f} total) | target NLL {target_score['nll_mean']:.5f}/h "
          f"({target_score['nll_total']:.4f} total) | target RMSE {target_score['rmse']:.5f} "
          f"| target C95 {target_score['coverage_95_pct']:.1f}% | runtime {total_seconds:.1f}s",flush=True)
    return {"row":row,"history_score":historical,"target_score":target_score,
            "complete_score":complete,"target_rows":target_rows,"history_rows":historical_rows,
            "oracle":oracle,"fitted":fitted,"history_days":[str(d.date()) for d in history],
            "timings":timings}


def plot_best_worst(results, output, show=False):
    ordered = sorted(results,key=lambda item:(item["target_score"]["nll_mean"],item["row"]["day"]))
    if len(ordered)<6: raise ValueError("Ranking plot requires at least six completed days.")
    selected = [ordered[:3],list(reversed(ordered[-3:]))]
    fig, axes = plt.subplots(3,2,figsize=(15.2,11.1),sharex=True,sharey=True)
    for column, items in enumerate(selected):
        for row, case in enumerate(items):
            ax=axes[row,column]
            d=case["row"]["day"]
            prediction=case["oracle"]["prediction"]["mean"]
            observed=np.array([item["observed"] for item in case["target_rows"]])
            lower=case["oracle"]["lower_95"]
            upper=case["oracle"]["upper_95"]
            h=np.arange(K)
            ax.fill_between(h,lower,upper,color=LIGHT,alpha=.42,
                            label="Central 95% observation interval" if row==0 and column==0 else None)
            ax.plot(h,observed,color=BLACK,marker="o",markersize=2.8,lw=1.4,
                    label="Observed demand" if row==0 and column==0 else None)
            ax.plot(h,prediction,color=RED,lw=2.2,
                    label="BM1 expected demand" if row==0 and column==0 else None)
            stamp=pd.Timestamp(d)
            ax.text(0.,1.045,stamp.strftime("%a %d %b %Y"),
                    transform=ax.transAxes,fontsize=10,va="bottom")
            ax.text(1.,1.045,f"Mean NLL = {case['target_score']['nll_mean']:.5f}",
                    transform=ax.transAxes,fontsize=9,ha="right",va="bottom",color="#505050")
            ax.set_xlim(0,K-1)
            ax.set_xticks([0,4,8,12,16,20,23])
            ax.grid(alpha=.16)
            if column==0: ax.set_ylabel("Normalized demand")
            if row==2: ax.set_xlabel("Hour of day")
    fig.suptitle("IM (BM1) | 24-hour oracle predictions\n"
                 f"ALL | weekly temperature | N={N} historical days | ranked by mean marginal NLL",
                 y=.998,fontsize=16)
    fig.text(.255,.910,"3 best days (lowest NLL)",ha="center",fontsize=11)
    fig.text(.755,.910,"3 worst days (highest NLL)",ha="center",fontsize=11)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.943),ncol=3,frameon=False)
    fig.text(.06,.016,"Ranking uses target open-loop marginal NLL. Intervals include observation noise; "
             "target initial densities are retrospectively optimized.",fontsize=8.5,color="#565656")
    fig.subplots_adjust(top=.862,bottom=.075,left=.065,right=.985,hspace=.40,wspace=.11)
    path=output/"best_worst_nll.png"
    fig.savefig(path,dpi=180,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    return [item["row"]["day"] for item in selected[0]],[item["row"]["day"] for item in selected[1]]


def plot_timings(results, output, show=False):
    keys=["initial_guess","stochastic_mle","oracle_optimization","complete"]
    labels=["Historical initial guess","Stochastic MLE","Oracle initial density + forecast","Complete process"]
    vals=[float(np.mean([row["timings"][key] for row in results])) for key in keys]
    fig, ax=plt.subplots(figsize=(11.8,5.4))
    places=np.arange(len(vals))
    ax.barh(places,vals,color=[RED,RED,RED,"#737373"],alpha=.9,height=.68)
    ax.set_yticks(places,labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0,max(vals)*1.31 if max(vals)>0 else 1.)
    ax.set_xlabel("Mean time (s)")
    ax.set_title(f"IM | Mean computation cost by stage | {len(results)} selected target days",fontsize=15,pad=20)
    for i,value in enumerate(vals):
        ax.text(value+max(vals)*.014,i,f"{value:.4f} s",va="center",fontsize=11)
    ax.xaxis.grid(True,alpha=.19)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path=output/"stage_timings.png"
    fig.savefig(path,dpi=180,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    return dict(zip(keys,vals))


def run(days=DATES, *, tag="ALL",degree=6,quadrature=48,steps_per_hour=2,
        maxiter=12,maxfun=400,oracle_maxiter=35,quiet=False,strict=False,
        continue_on_error=False,show=False,output=OUTPUT):
    if not days: raise ValueError("Select at least one target day.")
    days = [pd.Timestamp(day) for day in days]
    if len(set(days))!=len(days) or any(pd.isna(day) or day.hour!=0 or day.minute!=0 or day.tz is not None for day in days):
        raise ValueError("Target dates must be distinct naive midnight timestamps.")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    raw,training,histories,imputation=prepare_all(days,tag)
    results,errors=[],{}
    for i,day in enumerate(days,1):
        print(f"[BM1 fixed beta] case {i}/{len(days)}: {day:%Y-%m-%d}",flush=True)
        try:
            case=one_date(day,histories[day],raw,training,
                type("Options",(),dict(degree=degree,quadrature=quadrature,
                    steps_per_hour=steps_per_hour,maxiter=maxiter,maxfun=maxfun,
                    oracle_maxiter=oracle_maxiter,quiet=quiet,strict=strict))())
            results.append(case)
        except Exception as exc:
            errors[str(day.date())]={"type":type(exc).__name__,"message":str(exc)}
            print(f"[BM1 fixed beta] FAILED {day:%Y-%m-%d}: {exc}",flush=True)
            if not continue_on_error: raise
    if not results: raise RuntimeError("No completed evaluations.")
    pd.DataFrame([case["row"] for case in results]).to_csv(output/"daily_scores.csv",index=False)
    pd.DataFrame([row for case in results for row in case["target_rows"]]).to_csv(output/"target_hourly.csv",index=False)
    pd.DataFrame([row for case in results for row in case["history_rows"]]).to_csv(output/"historical_hourly.csv",index=False)
    pooled_scores={period:pooled([case[f"{period}_score"] for case in results])
                   for period in ("history","target","complete")}
    aggregate={
        "study":"BM1 fixed beta Fokker-Planck + active-set QP",
        "dates_requested":[str(day.date()) for day in days],
        "dates_completed":[case["row"]["day"] for case in results],
        "dates_failed":errors,"tag":tag,"baseline":BASELINE,"N":N,"k":K,
        "model":"bm_1","degree":degree,"quadrature":quadrature,
        "steps_per_hour":steps_per_hour,"maxiter":maxiter,"maxfun":maxfun,
        "evaluation_contract":{
            "historical":"one-step-ahead filtered conditional observation densities",
            "target":"open-loop oracle-conditioned marginal observation densities",
            "combined":"pooled hours across historical and target despite different conditioning",
            "runtime":"excludes one-time data loading and plot/CSV generation",
        },
        "pooled":pooled_scores,
        "optimizer_converged_days":int(sum(case["fitted"]["converged"] for case in results)),
        "oracle_converged_days":int(sum(case["oracle"]["oracle_converged"] for case in results)),
        "imputation":imputation,
    }
    aggregate["mean_runtime_seconds"]=plot_timings(results,output,show=show)
    if len(results)>=6:
        best,worst=plot_best_worst(results,output,show=show)
        aggregate["best_days"]=best
        aggregate["worst_days"]=worst
    else:
        aggregate["best_days"]=[]
        aggregate["worst_days"]=[]
        print("[BM1 fixed beta] Fewer than six completed cases; skipping 3-best/3-worst plot.",flush=True)
    pd.DataFrame([
        {"period":period,"nll_total":score["nll_total"],"nll_mean_per_hour":score["nll_mean"],
         "expected_demand_rmse":score["rmse"],"empirical_coverage_95_pct":score["coverage_95_pct"],
         "covered_hours":score["covered"],"observations":score["observations"]}
        for period,score in pooled_scores.items()
    ]).to_csv(output/"pooled_scores.csv",index=False)
    (output/"summary.json").write_text(json.dumps(aggregate,indent=2,default=json_value,allow_nan=False),
                                      encoding="utf-8")
    print(f"\n[BM1 fixed beta] {len(results)}/{len(days)} completed; pooled scores:",flush=True)
    for name,score in pooled_scores.items():
        print(f"  {name:8s} NLL/h={score['nll_mean']:.6f} NLL(total)={score['nll_total']:.6f} "
              f"RMSE={score['rmse']:.6f} C95={score['coverage_95_pct']:.2f}% "
              f"({score['covered']}/{score['observations']})",flush=True)
    print("[BM1 fixed beta] Mean runtime (s): "+", ".join(
        f"{key}={value:.2f}" for key,value in aggregate["mean_runtime_seconds"].items()),flush=True)
    print(f"[BM1 fixed beta] Output: {output}",flush=True)
    return aggregate


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--days",nargs="+",default=list(DATES),help="Selected target dates (default: all 12).")
    parser.add_argument("--tag",default="ALL")
    parser.add_argument("--degree",type=int,default=6)
    parser.add_argument("--quadrature",type=int,default=48)
    parser.add_argument("--steps-per-hour",type=int,default=2)
    parser.add_argument("--maxiter",type=int,default=12)
    parser.add_argument("--maxfun",type=int,default=400)
    parser.add_argument("--oracle-maxiter",type=int,default=35)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--quiet",action="store_true")
    parser.add_argument("--strict",action="store_true")
    parser.add_argument("--continue-on-error",action="store_true")
    parser.add_argument("--show",action="store_true")
    args=parser.parse_args()
    run(args.days,tag=args.tag,degree=args.degree,quadrature=args.quadrature,
        steps_per_hour=args.steps_per_hour,maxiter=args.maxiter,maxfun=args.maxfun,
        oracle_maxiter=args.oracle_maxiter,quiet=args.quiet,strict=args.strict,
        continue_on_error=args.continue_on_error,show=args.show,output=args.output)


if __name__=="__main__": main()

"""Report best k=24 backlog for baseline-only and ARM oracle predictions.

Uses the saved pooled prediction and diffusion summaries. No drift, baseline,
oracle-state or diffusion refitting is performed. Selection independently
minimizes pooled k=24 prediction RMSE for each model/baseline combination.

The target ARM MAE-discrepancy check optionally replays saved oracle states;
this is a validation of the original prediction, not parameter estimation.

Run from repository root:
    python -m src.exe.run_model.wm_4.report_best_k24
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import (
    OUTPUT as DIFFUSION_OUTPUT, oracle_errors, variance_kernel,
)
from src.exe.run_model.wm_4.plot_moving_horizon_quality import OUTPUT as ARM_OUTPUT
from src.exe.run_model.wm_4.plot_baseline_moving_horizon_quality import OUTPUT as BASELINE_OUTPUT
from src.exe.run_model.wm_4.sweep_weekday_horizon import OUTPUT as SWEEP_OUTPUT


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"best_k24"
K = 24
BASELINES = ("weekly_average", "weekly_temperature")
LABELS = {"weekly_average":"Weekly average", "weekly_temperature":"Weekly temperature"}
KEY = ("day", "baseline", "training_weekdays", "horizon_hours")
MAE_FACTOR = float(np.sqrt(2./np.pi))


def summary(path, required):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Generate the corresponding quality results first.")
    data = pd.read_csv(path)
    missing = set(required)-set(data.columns)
    if missing: raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    data = data.loc[(data["horizon_hours"] == K) & data["baseline"].isin(BASELINES)].copy()
    if data.empty: raise ValueError(f"No k={K} data for either baseline in {path}")
    if data.duplicated(["baseline","training_weekdays","horizon_hours"]).any():
        raise ValueError(f"Duplicate N,k summary rows in {path}")
    for name in required:
        if name == "baseline": continue
        if not np.isfinite(pd.to_numeric(data[name],errors="coerce")).all():
            raise ValueError(f"Nonfinite {name} in {path}")
    return data


def rank_demand(data, model, n_min, n_max):
    data = data.loc[data["training_weekdays"].between(n_min,n_max)].copy()
    expected = set(range(n_min,n_max+1))
    for baseline in BASELINES:
        available = set(data.loc[data["baseline"]==baseline,"training_weekdays"].astype(int))
        if available != expected:
            raise ValueError(f"Incomplete {model}, {baseline} N grid: {sorted(expected-available)}")
    data["model"] = model
    data = data.sort_values(["baseline","rmse","training_weekdays"],kind="stable").reset_index(drop=True)
    data["rank"] = data.groupby("baseline").cumcount()+1
    return data


def winners(ranking):
    return ranking.loc[ranking["rank"]==1].set_index("baseline").loc[list(BASELINES)].reset_index()


def selected_saved(path, wanted, columns):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    selected = []
    usecols = list(dict.fromkeys([*KEY,*columns]))
    for batch in pd.read_csv(path,usecols=usecols,chunksize=100000):
        mask = (batch["horizon_hours"]==K)&np.logical_or.reduce([
            ((batch["baseline"]==baseline)&(batch["training_weekdays"]==n)).to_numpy()
            for baseline,n in wanted.items()
        ])
        if mask.any(): selected.append(batch.loc[mask].copy())
    if not selected: raise ValueError(f"No selected k=24 records in {path}")
    data = pd.concat(selected,ignore_index=True)
    data["day"] = pd.to_datetime(data["day"],format="mixed")
    if data.duplicated(list(KEY)).any():
        raise ValueError(f"Duplicate selected origin/N/k records in {path}")
    return data


def selected_parameters(diffusion_dir, sweep_dir, wanted):
    noise = selected_saved(diffusion_dir/"diffusion_interval_parameters.csv",wanted,
                           ("observation_variance","process_variance_rate"))
    drift = selected_saved(sweep_dir/"parameter_estimates.csv",wanted,
                           ("theta_F","theta_R","theta_U","baseline_coefficients",
                            "baseline_beta_C","baseline_chill_center"))
    result = drift.merge(noise,on=list(KEY),how="inner",validate="one_to_one")
    if len(result)!=len(drift) or len(result)!=len(noise):
        raise ValueError("Diffusion and ARM drift saved origins do not agree")
    return result


def predicted_mae(rows, steps):
    """Exact pooled model-implied Gaussian E|e| from saved tau^2, sigma^2."""
    results = {}
    for baseline,group in rows.groupby("baseline",sort=True):
        hourly = np.zeros(K)
        for row in group.itertuples(index=False):
            kernel = variance_kernel(row.theta_F,row.theta_R,steps,K)
            variance = row.observation_variance+row.process_variance_rate*kernel
            if not np.isfinite(variance).all() or np.min(variance)<-1e-10:
                raise ValueError(f"Nonfinite/negative variance at {row.day} ({baseline})")
            hourly += MAE_FACTOR*np.sqrt(np.maximum(variance,0.))
        results[baseline] = {"predicted_mae":float(np.mean(hourly/len(group))),
                             "hourly_predicted_mae":hourly/len(group),
                             "origins":len(group)}
    return results


def replay_target_mae(rows,sweep_dir,steps,output):
    """Reconstruct oracle errors at saved initial states, evaluate hourly MAE."""
    from src.exe.run_model.wm_4.moving_horizon_common import group_frame
    wanted = {baseline:int(group["training_weekdays"].iloc[0])
              for baseline,group in rows.groupby("baseline")}
    states = selected_saved(sweep_dir/"oracle_horizon_scores.csv",wanted,
                            ("oracle_F0","oracle_R0","mse"))
    combined = rows.merge(states,on=list(KEY),how="inner",validate="one_to_one")
    if len(combined)!=len(rows) or len(combined)!=len(states):
        raise ValueError("Selected oracle and drift origin cohorts do not agree")
    frame = group_frame("ALL")
    results, hourly_rows = {}, []
    for baseline,part in combined.groupby("baseline",sort=True):
        actual_abs, predicted_abs, square, variances, covered = (np.zeros(K) for _ in range(5))
        for row in part.to_dict("records"):
            error = oracle_errors(frame,row,row,steps)
            kernel = variance_kernel(row["theta_F"],row["theta_R"],steps,K)
            variance = np.maximum(row["observation_variance"]+row["process_variance_rate"]*kernel,0.)
            actual_abs += np.abs(error)
            predicted_abs += MAE_FACTOR*np.sqrt(variance)
            square += error*error
            variances += variance
            covered += (np.abs(error)<=1.959963984540054*np.sqrt(variance))
        count = len(part)
        real_hourly, model_hourly = actual_abs/count, predicted_abs/count
        actual_rmse, expected_rmse = np.sqrt(square/count),np.sqrt(variances/count)
        results[baseline] = {
            "observed_mae":float(np.mean(real_hourly)),
            "predicted_mae":float(np.mean(model_hourly)),
            "pooled_mae_gap":float(abs(np.mean(real_hourly)-np.mean(model_hourly))),
            "mean_hourly_mae_gap":float(np.mean(np.abs(real_hourly-model_hourly))),
            "mean_hourly_rmse_gap":float(np.mean(np.abs(actual_rmse-expected_rmse))),
            "coverage_95_percent":float(100.*np.mean(covered/count)),
        }
        for h in range(K):
            hourly_rows.append({
                "baseline":baseline,"training_weekdays":wanted[baseline],"lead_hour":h+1,
                "observed_mae":real_hourly[h],"gaussian_expected_mae":model_hourly[h],
                "absolute_mae_gap":abs(real_hourly[h]-model_hourly[h]),
                "observed_rmse":actual_rmse[h],"predicted_rmse":expected_rmse[h],
                "absolute_rmse_gap":abs(actual_rmse[h]-expected_rmse[h]),
                "coverage_95_percent":100.*covered[h]/count,
            })
    pd.DataFrame(hourly_rows).to_csv(output/"arm_diffusion_hourly_k24.csv",index=False)
    return results


def run(args):
    required = ("baseline","training_weekdays","horizon_hours",
                "rmse","mae","bias","origins","observations")
    baseline = summary(args.baseline_dir/"prediction_cumulative_quality.csv",required)
    arm = summary(args.arm_dir/"prediction_cumulative_quality.csv",required)
    base_rank = rank_demand(baseline,"Baseline",args.min_weekdays,args.max_weekdays)
    arm_rank = rank_demand(arm,"ARM",args.min_weekdays,args.max_weekdays)
    best_baselines, best_arm = winners(base_rank),winners(arm_rank)
    chosen = dict(zip(best_arm["baseline"],best_arm["training_weekdays"].astype(int)))

    diff_required = (
        "baseline","training_weekdays","horizon_hours","origins","observations",
        "observed_rmse","expected_rmse","rmse_absolute_error",
        "mean_hourly_rmse_absolute_error","bias_observed_minus_predicted",
        "ks_similarity","ks_distance","coverage_95",
        "observation_sigma_mean","process_sigma_mean","process_identifiable_pct",
    )
    diffusion = []
    for kind,filename in (("historical","diffusion_historical_summary.csv"),
                          ("oracle","diffusion_oracle_summary.csv")):
        table = summary(args.diffusion_dir/filename,diff_required)
        for name,n in chosen.items():
            subset = table.loc[(table["baseline"]==name)&(table["training_weekdays"]==n)]
            if len(subset)!=1: raise ValueError(f"Missing {kind} diffusion at N={n},k=24,{name}")
            diffusion.append({"evaluation":kind,**subset.iloc[0].to_dict()})

    params = selected_parameters(args.diffusion_dir,args.sweep_dir,chosen)
    config_path = args.sweep_dir/"configuration.json"
    if not config_path.is_file(): raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("group")!="ALL": raise ValueError("The saved sweep must use group=ALL")
    steps = int(config["steps_per_hour"])
    expected = predicted_mae(params,steps)
    historical = summary(args.arm_dir/"fit_cumulative_quality.csv",required)
    args.output.mkdir(parents=True,exist_ok=True)
    replay = (replay_target_mae(params,args.sweep_dir,steps,args.output)
              if not args.skip_hourly_replay else {})

    if replay:
        for name,metrics in replay.items():
            entry = arm.loc[(arm["baseline"]==name)&(arm["training_weekdays"]==chosen[name])]
            if len(entry)!=1 or not np.isclose(metrics["observed_mae"],float(entry.iloc[0]["mae"]),
                                                atol=5e-8,rtol=1e-5):
                raise ValueError(f"Oracle MAE replay disagrees with saved prediction at {name}")

    diff_rows = []
    for item in diffusion:
        name,n,kind = item["baseline"],int(item["training_weekdays"]),item["evaluation"]
        comparison = historical if kind=="historical" else arm
        reference = comparison.loc[(comparison["baseline"]==name)&
                                   (comparison["training_weekdays"]==n)]
        if len(reference)!=1: raise ValueError(f"Missing saved demand MAE at {name}, N={n}")
        if int(item["origins"])!=int(reference.iloc[0]["origins"]) or \
           int(item["origins"])!=expected[name]["origins"]:
            raise ValueError(f"{kind} diffusion origin count differs from saved ARM demand cohort at {name}, N={n}. "
                             "Regenerate quality and diffusion results for the same origin cohort.")
        observed_mae = float(reference.iloc[0]["mae"])
        predicted = expected[name]["predicted_mae"]
        diff_rows.append({
            **item,"observed_mae":observed_mae,
            "gaussian_expected_mae":predicted,
            "pooled_mae_absolute_discrepancy":abs(observed_mae-predicted),
            "mean_hourly_mae_absolute_discrepancy":(
                replay[name]["mean_hourly_mae_gap"] if kind=="oracle" and name in replay else np.nan),
            "coverage_95_percent":100.*float(item["coverage_95"]),
        })
    all_ranks = pd.concat([base_rank,arm_rank],ignore_index=True)
    winners_table = pd.concat([best_baselines,best_arm],ignore_index=True)
    diffusion_table = pd.DataFrame(diff_rows)
    all_ranks.to_csv(args.output/"all_k24_backlog_rankings.csv",index=False)
    winners_table.to_csv(args.output/"best_k24_demand.csv",index=False)
    diffusion_table.to_csv(args.output/"arm_diffusion_best_k24.csv",index=False)

    print("\nBEST k=24 POOLED TARGET DEMAND PERFORMANCE | ALL")
    print("Minimize pooled 24-hour RMSE independently for each baseline and ARM configuration.")
    print("ARM predictions use oracle initial states; baseline-only predictions do not.")
    for row in winners_table.itertuples(index=False):
        print(f"  {row.model:8s} | {LABELS[row.baseline]:20s} | N={int(row.training_weekdays):2d} | "
              f"RMSE={row.rmse:.8f} | MAE={row.mae:.8f} | bias(pred-obs)={row.bias:+.8f} | "
              f"origins={int(row.origins):,} | samples={int(row.observations):,}")
    print("\nARM gain versus the baseline-only model at its SAME selected N")
    for row in best_arm.itertuples(index=False):
        comparable = baseline.loc[(baseline["baseline"]==row.baseline)&
                                  (baseline["training_weekdays"]==row.training_weekdays)]
        if len(comparable)!=1: raise ValueError(f"Missing baseline N={row.training_weekdays} row")
        benchmark = float(comparable.iloc[0]["rmse"])
        print(f"  {LABELS[row.baseline]} | N={int(row.training_weekdays)} | "
              f"baseline RMSE={benchmark:.8f} | ARM RMSE={row.rmse:.8f} | "
              f"reduction={(1.-row.rmse/benchmark)*100:+.2f}%")

    print("\nARM DIFFUSION for ARM-selected N, k=24")
    print("Gaussian expected MAE = mean[sqrt(2/pi)*sqrt(tau^2+sigma^2*K_j)].")
    print("MAE discrepancy is absolute difference of pooled observed and model-implied MAE.")
    for row in diffusion_table.itertuples(index=False):
        print(f"  {LABELS[row.baseline]:20s} | {row.evaluation:10s} | N={int(row.training_weekdays):2d}")
        print(f"    observed RMSE={row.observed_rmse:.8f} | predicted RMSE={row.expected_rmse:.8f} | "
              f"pooled |RMSE gap|={row.rmse_absolute_error:.8f} | "
              f"mean hourly |RMSE gap|={row.mean_hourly_rmse_absolute_error:.8f}")
        print(f"    observed MAE={row.observed_mae:.8f} | expected Gaussian MAE={row.gaussian_expected_mae:.8f} | "
              f"pooled |MAE gap|={row.pooled_mae_absolute_discrepancy:.8f}")
        if row.evaluation=="oracle" and not args.skip_hourly_replay:
            print(f"    mean hourly |MAE gap|={row.mean_hourly_mae_absolute_discrepancy:.8f}")
        print(f"    Gaussian similarity(1-KS)={row.ks_similarity:.6f} | "
              f"KS distance={row.ks_distance:.6f} | mean empirical 95% coverage={row.coverage_95_percent:.2f}%")
        print(f"    bias(obs-pred)={row.bias_observed_minus_predicted:+.8f} | "
              f"process sigma={row.process_sigma_mean:.6f} | observation tau={row.observation_sigma_mean:.6f} | "
              f"identifiable={row.process_identifiable_pct:.1f}% | origins={int(row.origins):,}")
    print(f"\nSaved demand selection and full rankings: {args.output}")
    if not args.skip_hourly_replay:
        print(f"Also saved: {args.output/'arm_diffusion_hourly_k24.csv'}")
    return winners_table,diffusion_table


def main():
    parser = ArgumentParser(description="Report best N at k=24 for ALL baseline/ARM predictions and diffusion.")
    parser.add_argument("--baseline-dir",type=Path,default=BASELINE_OUTPUT/"all")
    parser.add_argument("--arm-dir",type=Path,default=ARM_OUTPUT/"all")
    parser.add_argument("--diffusion-dir",type=Path,default=DIFFUSION_OUTPUT/"all")
    parser.add_argument("--sweep-dir",type=Path,default=SWEEP_OUTPUT/"all")
    parser.add_argument("--output",type=Path,default=OUTPUT/"all")
    parser.add_argument("--min-weekdays",type=int,default=2)
    parser.add_argument("--max-weekdays",type=int,default=14)
    parser.add_argument("--skip-hourly-replay",action="store_true",
                        help="Skip exact oracle replay and report pooled MAE discrepancy only.")
    args = parser.parse_args()
    if not 2<=args.min_weekdays<=args.max_weekdays<=14:
        parser.error("Require 2 <= min-weekdays <= max-weekdays <= 14")
    return run(args)


if __name__=="__main__": main()

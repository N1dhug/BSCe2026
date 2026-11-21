"""Complete missing ARM oracle predictions for eligible historical-fitting days.

Only previously fitted constrained theta are reused; historical baselines are
reconstructed identically to sweep_weekday_horizon. Missing target-day oracle
initial states are optimized by the SAME box-constrained QP as the original
sweep. No theta refitting occurs. Supplemental scores and hourly forecast
components are cached separately from the original sweep files.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.method.moving_horizon import group_frame, metrics, window_data
from src.exe.run_model.wm_4.sweep_weekday_horizon import baseline_from_history
from src.tools.impute_hourly import impute_hourly_days


KEY = ["day","baseline","training_weekdays"]
SCORE_KEY = KEY+["horizon_hours"]
COMPONENT_KEY = KEY+["hour"]


def _read_cache(path, columns):
    if not path.is_file() or path.stat().st_size == 0: return pd.DataFrame(columns=columns)
    table = pd.read_csv(path)
    if not set(columns).issubset(table.columns):
        raise ValueError(f"Invalid saved oracle completion cache: {path}")
    table["day"] = pd.to_datetime(table["day"]).dt.normalize()
    for col in ("training_weekdays", "horizon_hours", "hour"):
        if col in table: table[col] = table[col].astype(int)
    return table


def _append(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,mode="a",header=not path.is_file(),index=False)


def complete(saved_scores, saved_components, eligible, *, input_dir, horizons=range(1,25),
             steps_per_hour=4, fail_fast=False, progress_every=25):
    """Return source + cached + newly computed oracle scores on fitted days.

    The fitting manifest is the exact set of (day, baseline, N) keys approved
    by plot_pooled_fit_rmse.py. The cache is stored under the prediction plot
    directory and does not alter the completed-sweep CSVs.
    """
    input_dir = Path(input_dir)
    group = input_dir.name.upper()
    if group not in ("ALL","MED"):
        raise ValueError("Oracle completion requires an ALL or MED sweep directory.")
    sph = int(steps_per_hour)
    if sph<2: raise ValueError("steps_per_hour must be >= 2.")
    config_path = input_dir/"configuration.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config.get("steps_per_hour",sph))!=sph:
            raise ValueError("Completion integration resolution differs from saved theta.")
        if str(config.get("group",group)).upper()!=group:
            raise ValueError("Completion data group differs from saved theta.")

    cache_dir = Path(__file__).resolve().parents[4]/"files"/"output"/"wm_4"/"pooled_prediction_rmse"/group.lower()/"oracle_cache"
    score_path,component_path = cache_dir/"scores.csv",cache_dir/"components.csv"
    cached_scores = _read_cache(score_path,SCORE_KEY+["oracle_F0","oracle_R0"])
    cached_components = _read_cache(component_path,COMPONENT_KEY+["observed","prediction_at_initial_zero","sensitivity_F0","sensitivity_R0"])
    scores = pd.concat((saved_scores,cached_scores),ignore_index=True).drop_duplicates(SCORE_KEY,keep="last")
    components = pd.concat((saved_components,cached_components),ignore_index=True).drop_duplicates(COMPONENT_KEY,keep="last")

    horizons = tuple(sorted(set(int(k) for k in horizons)))
    if not horizons or min(horizons)<1 or max(horizons)>24:
        raise ValueError("horizons must be within 1..24.")
    existing_scores = scores.loc[scores["horizon_hours"].isin(horizons)]
    score_count = existing_scores.groupby(KEY)["horizon_hours"].nunique().to_dict()
    component_count = components.groupby(KEY)["hour"].nunique().to_dict()
    pending = [(pd.Timestamp(day).normalize(),name,int(n)) for day,name,n in
               eligible[KEY].itertuples(index=False,name=None)
               if score_count.get((pd.Timestamp(day).normalize(),name,int(n)),0)<len(horizons)
               or component_count.get((pd.Timestamp(day).normalize(),name,int(n)),0)<24]
    if not pending:
        print(f"[ARM prediction] All {len(eligible)} fitted-day configurations already have oracle scores.")
        return scores,components

    parameters_path = input_dir/"parameter_estimates.csv"
    if not parameters_path.is_file(): raise FileNotFoundError(f"Missing saved fitted parameters: {parameters_path}")
    parameters = pd.read_csv(parameters_path)
    if not set(KEY+["theta_F","theta_U","theta_R"]).issubset(parameters):
        raise ValueError("Saved parameter estimates lack fit keys or theta coefficients.")
    parameters["day"] = pd.to_datetime(parameters["day"]).dt.normalize()
    parameters = parameters.drop_duplicates(KEY,keep="last")
    fit_lookup = {key:row for key,row in parameters.set_index(KEY).iterrows()}
    frame = group_frame(group)
    failures=[]
    completed=0
    print(f"[ARM prediction] Completing {len(pending)} missing fitted-day oracle configurations using saved theta.",flush=True)
    for pos,key in enumerate(pending,1):
        day,name,n = key
        try:
            if key not in fit_lookup: raise ValueError("Missing saved theta for historical fitting interval.")
            row = fit_lookup[key]
            theta = np.array([row.theta_F,row.theta_U,row.theta_R],dtype=float)
            if not np.isfinite(theta).all(): raise ValueError("Nonfinite saved theta.")
            history=[day-pd.Timedelta(weeks=i) for i in range(n,0,-1)]
            training,_=impute_hourly_days(frame,days=history,columns=("I","U","Temp"),require_days=True)
            baseline=baseline_from_history(training,history,name)
            if "baseline_coefficients" in row and isinstance(row.baseline_coefficients,str):
                coeff=np.asarray(json.loads(row.baseline_coefficients),dtype=float)
                if coeff.shape!=np.asarray(baseline.coef).shape or not np.allclose(coeff,baseline.coef,atol=1e-7,rtol=1e-6):
                    raise ValueError("Reconstructed baseline differs from saved fit coefficients.")
            model,forecast,_,observed=window_data(frame,day,baseline,sph)
            zero=model._solve_deterministic([0.0,0.0],forecast["U"],forecast["PE"],
                                            forecast["Time"],forecast["O"],theta)
            sample=np.arange(24,dtype=int)*sph
            base=np.asarray(zero["I"],dtype=float)[sample]
            sensitivity=np.asarray(model.sensitivity({"r":theta})["input"],dtype=float)[sample,:]
            baseline_hourly=np.asarray(forecast["O"],dtype=float)[sample]
            if (base.shape!=(24,) or sensitivity.shape!=(24,2) or
                not np.isfinite(base).all() or not np.isfinite(sensitivity).all() or
                not np.isfinite(observed).all()):
                raise ValueError("Nonfinite or incompatible target-day oracle inputs.")

            component_rows=[{
                "day":day,"baseline":name,"training_weekdays":n,"hour":int(h),
                "time":day+pd.Timedelta(hours=int(h)),"observed":float(observed[h]),
                "baseline_value":float(baseline_hourly[h]),
                "price":float(forecast["U"][sample[h]]),
                "prediction_at_F0_zero":float(base[h]),
                "prediction_at_initial_zero":float(base[h]),
                "sensitivity_F0":float(sensitivity[h,0]),
                "sensitivity_R0":float(sensitivity[h,1]),
            } for h in range(24)]
            score_rows=[]
            for k in horizons:
                desired=observed[:k]-base[:k]
                initial=np.asarray(model._least_squares_qp(
                    sensitivity[:k],desired,lb=np.array([-1.,-1.]),ub=np.array([1.,1.]),
                    ridge=model._FIT_RIDGE),dtype=float)
                fit=base[:k]+sensitivity[:k]@initial
                score=metrics(observed[:k],fit)
                bscore=metrics(observed[:k],baseline_hourly[:k])
                score_rows.append({
                    "day":day,"baseline":name,"training_weekdays":n,
                    "horizon_hours":k,
                    "theta_F":float(theta[0]),"theta_U":float(theta[1]),"theta_R":float(theta[2]),
                    "oracle_F0":float(initial[0]),"oracle_R0":float(initial[1]),
                    "rmse":score["rmse"],"mse":score["mse"],
                    "mae":score["mae"],"bias":score["bias"],
                    "baseline_rmse":bscore["rmse"],"baseline_mse":bscore["mse"],
                })
            _append(score_path,score_rows)
            _append(component_path,component_rows)
            completed+=1
        except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            failures.append({"day":day,"baseline":name,"training_weekdays":n,"error":str(exc)})
        if progress_every and pos%progress_every==0:
            print(f"[ARM prediction] {pos}/{len(pending)} missing fit windows processed; success={completed} failed={len(failures)}",flush=True)

    cache_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(failures,columns=KEY+["error"]).to_csv(cache_dir/"skipped.csv",index=False)
    scores=pd.concat((saved_scores,_read_cache(score_path,SCORE_KEY+["oracle_F0","oracle_R0"])),
                     ignore_index=True).drop_duplicates(SCORE_KEY,keep="last")
    components=pd.concat((saved_components,_read_cache(component_path,COMPONENT_KEY+["observed","prediction_at_initial_zero","sensitivity_F0","sensitivity_R0"])),
                         ignore_index=True).drop_duplicates(COMPONENT_KEY,keep="last")
    print(f"[ARM prediction] Reconstructed {completed}/{len(pending)} missing fitting windows; cache: {cache_dir}")
    return scores,components

"""Boundary study for BM1/BM2 with the existing logistic response family.

Run from the repository:
  python -B bm_boundary_study.py --repo . --study pilot --fit-missing
  python -B -m src.exe.run_model.bm_1.analyze_boundaries --study week --fit-missing

Reads existing prepared data and canonical fit.json files. Missing fits are
made only with --fit-missing, through the native five-window fitters.
Daily forecasts retain five equal deterministic paths; MPC uses the native
observation-updated prior and historical process-noise calibration.
Controlled scenarios never score counterfactual demand against observations.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import traceback

_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--repo", type=Path, default=Path.cwd())
_boot_args, _ = _bootstrap.parse_known_args()
ROOT = _boot_args.repo.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.exe.run_model.bm_1 import predict_daily as daily
from src.exe.run_model.bm_1 import predict_mpc as mpc
from src.exe.run_model.bm_1.compare_fg_week import write_json, sha, record, load_modules
from src.exe.run_model.bm_2 import predict as bm2_forecast
from src.model.bm_1 import bm_1
from src.model.bm_2 import bm_2
from src.params.bm_1 import lsq_params as BM1Fit
from src.params.bm_1 import historical_diffusion_params as BM1Diffusion
from src.params.bm_2 import lsq_params as BM2Fit
from src.params.bm_2 import historical_diffusion_params as BM2Diffusion
from src.params.bm_1_response import response_from_record
from src.solver.finite_volume import finite_volume_1d_cached
from src.tools.forecast_metrics import predictive_intervals

EPSILONS = (.025, .05, .10)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_value(value):
    """Preserve native diagnostic arrays when writing the analysis report."""
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {key:json_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_value(item) for item in value]
    return value


def fit_folder(root, day, kind, hour):
    return Path(root) / day / "logistic" / kind / (
        "forecast" if kind == "daily" else f"origin_{hour:02d}")


def restore(meta, model):
    if meta.get("family") != "logistic" or meta.get("status") != "complete":
        raise ValueError("Expected a complete canonical logistic fit.json (from compare_fg_week or compare_bm_week).")
    p = {**meta["theta"], **meta["shape_parameters"]}
    p["f"], p["g"] = response_from_record(meta["f"]), response_from_record(meta["g"])
    p["scalar_f"] = p["f"].scalar_evaluator()
    x = np.asarray(meta["state_X0"], dtype=float)
    if model == "bm2":
        x = np.column_stack((x, meta["state_R0"]))
    p.update(initial_states=x, initial_weights=np.full(5, .2),
             family="logistic", shape_parameters=meta["shape_parameters"],
             converged=bool(meta["converged"]),stop_reason=meta["stop_reason"],
             warning=None if meta["converged"] else meta["stop_reason"],
             mse=float(meta["training_rmse"])**2)
    if x.shape != ((5, 2) if model == "bm2" else (5,)):
        raise ValueError("The saved fit must contain five paired historical states.")
    return p


def find_fit(config, model, day, kind, hour):
    for root in config["fit_roots"][model]:
        folder = fit_folder(root, day, kind, hour)
        path = folder / "fit.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        # BM1 metadata may omit model; BM2 metadata is identified by R0/tau.
        is_bm2 = "state_R0" in meta and "tau" in meta.get("theta", {})
        if is_bm2 != (model == "bm2"):
            continue
        restore(meta, model)
        if meta.get("day") != day or meta.get("kind") != kind:
            raise ValueError(f"Inconsistent fit task identity: {path}")
        settings_path = Path(root) / "settings.json"
        if not settings_path.is_file():
            raise ValueError(f"Missing source settings: {settings_path}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if meta.get("settings_fingerprint") != settings.get("fingerprint"):
            raise ValueError(f"Fit/settings identity mismatch: {path}")
        source_prepared = Path(root) / "prepared" / f"{day}.pkl"
        if not source_prepared.is_file() and "prepared_root" in settings:
            source_prepared = Path(settings["prepared_root"]) / f"{day}.pkl"
        prepared = Path(config["prepared_root"]) / f"{day}.pkl"
        if not source_prepared.is_file() or sha(source_prepared) != sha(prepared):
            raise ValueError(f"Prepared data differ from the saved fit: {path}")
        return path, meta
    return None, None


def load_data(config, day):
    path = Path(config["prepared_root"]) / f"{day}.pkl"
    if sha(path) != config["prepared_sha256"][day]:
        raise ValueError("Prepared data changed since preflight: " + day)
    # Only the user's locally produced, trusted preparation files are loaded.
    with path.open("rb") as stream:
        return pickle.load(stream)


def windows(data, day, kind, hour):
    if kind == "daily":
        return data["training"], data["target"]
    origin = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    training = [mpc.checks.window(data["mpc_frame"], origin-pd.Timedelta(weeks=k),
                data["mpc_baseline"], hours=24) for k in range(5, 0, -1)]
    future = mpc.checks.window(data["mpc_frame"], origin, data["mpc_baseline"],
                               hours=9, observations=False)
    return training, {k: np.asarray(future[k])[:9] for k in ("Time", "B", "U")}


def get_fit(config, model, day, kind, hour, data):
    local = Path(config["output"]) / "fits" / model
    path = fit_folder(local, day, kind, hour) / "fit.json"
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("analysis_fingerprint") != config["fingerprint"]:
            raise ValueError("Cached fit belongs to different analysis settings.")
        return restore(meta, model), meta
    source, meta = find_fit(config, model, day, kind, hour)
    if source is not None:
        if sha(source) != config["fit_sha256"].get(str(source)):
            raise ValueError("Reference fit changed since preflight: " + str(source))
        meta = {**meta, "reused_from": str(source), "reused_sha256": sha(source)}
    else:
        if not config["fit_missing"]:
            raise ValueError(f"Missing fit: {model} {day} {kind} {hour}")
        training, _ = windows(data, day, kind, hour)
        native_model = bm_1 if model == "bm1" else bm_2
        obj = native_model(daily.held_signal(training[0]["B"]), lambda t: .5,
                           (0., 24.), 25)
        options = dict(max_iterations=config[kind+"_max_iterations"],
                       max_nfev=config[kind+"_max_nfev"],
                       state_maxiter=100 if kind == "daily" else 60, substeps=2)
        if kind == "mpc":
            options["fit_bounds"] = {"C": (.05, 100.)}
        if model == "bm1":
            fit = BM1Fit(obj).fit_many(training, **options)
            meta = record(fit, "logistic", load_modules(ROOT))
        else:
            # Preserve native paired-state fitting; warm-start from BM1 when available.
            seed_path, seed_meta = find_fit(config, "bm1", day, kind, hour)
            if seed_meta is not None:
                options["seed_fit"] = restore(seed_meta, "bm1")
            fit = BM2Fit(obj, "logistic").fit_many(training, **options)
            meta = dict(theta={k:float(fit[k]) for k in ("C", "Xi", "k", "tau")},
                shape_parameters=fit["shape_parameters"], f=fit["f"].to_record(),
                g=fit["g"].to_record(), state_X0=fit["initial_states"][:,0].tolist(),
                state_R0=fit["initial_states"][:,1].tolist(),
                training_rmse=float(np.sqrt(fit["mse"])),
                converged=bool(fit["converged"]), stop_reason=fit["stop_reason"])
            meta["bm1_seed_source"] = str(seed_path) if seed_path else None
        meta.update(status="complete", family="logistic", day=day, kind=kind,
                    new_fit=True, numerical_source_substeps=2)
    meta["analysis_fingerprint"] = config["fingerprint"]
    write_json(path, meta)
    return restore(meta, model), meta


def initial_law(config, model, fit, future, kind, data, origin, diffusion):
    if kind == "daily":
        states = fit["initial_states"]
        return (states[:,0] if model == "bm2" else states,
                states[:,1] if model == "bm2" else None, np.full(5, .2))
    observed = mpc.checks.observation(data["mpc_frame"], origin)
    x = (np.arange(config["x_steps"])+.5)/config["x_steps"]
    if model == "bm1":
        prediction = bm_2.demand(x, float(future["B"][0]), float(future["U"][0]), fit)
        weights = mpc.posterior_mass(fit["initial_states"], x, prediction, observed,
                                     diffusion["observation_sd"], .025)
        return x, None, weights
    x, r = bm_2.grid(config["x_steps"], config["r_steps"])
    weights = bm2_forecast.posterior_mass(fit["initial_states"], x, r, fit,
        float(future["B"][0]), float(future["U"][0]), observed,
        diffusion["observation_sd"], .025, .05)
    return x, r, weights


def evolve(config, model, future, fit, initial, sigma, refined=False):
    """Native deterministic paths or FV laws; no clipping or renormalizing."""
    x0, r0, weights = initial
    times = np.asarray(future["Time"], dtype=float)
    if sigma == 0:
        if weights.ndim == 2:
            X, R = np.meshgrid(x0, r0, indexing="ij")
            starts = np.column_stack((X.ravel(), R.ravel()))
            weights = weights.ravel()
        else:
            starts = np.column_stack((x0, r0)) if model == "bm2" else x0
        paths = []
        rpaths = []
        if model == "bm2" and initial[2].ndim == 2:
            current_x,current_r=starts[:,0].copy(),starts[:,1].copy()
            X=np.empty((len(times),len(current_x)));R=np.empty_like(X)
            for i in range(len(times)):
                X[i],R[i]=current_x,current_r
                if i+1<len(times):
                    current_x,current_r=bm_2.advance(current_x,current_r,
                        float(times[i+1]-times[i]),float(future["B"][i]),
                        float(future["U"][i]),fit,32 if refined else 2)
            paths=list(X.T);rpaths=list(R.T)
        for start in ([] if model=="bm2" and initial[2].ndim==2 else starts):
            if model == "bm1":
                states, demand = BM1Fit._predict(future, float(start), fit, 32 if refined else 2)
                paths.append(states)
            else:
                states, demand = bm_2.predict(future, start, fit, 32 if refined else 2)
                paths.append(states[:,0])
                rpaths.append(states[:,1])
        X = np.asarray(paths).T
        W = np.broadcast_to(weights, X.shape).copy()
        R = np.asarray(rpaths).T if rpaths else None
        rmean = (R*W).sum(axis=1) if R is not None else np.zeros(len(times))
        blocked = np.sum(W*(((X<=1e-12)&(R<0))|((X>=1-1e-12)&(R>0))),axis=1) if R is not None else np.zeros(len(times))
        diagnostics = dict(method="native deterministic paths", maximum_mass_error=0.)
        cell_uniform = False
    else:
        step = config["fp_max_step"]/2 if refined else config["fp_max_step"]
        if model == "bm1":
            obj = bm_1(daily.held_signal(future["B"]), lambda t:.5,
                       (float(times[0]),float(times[-1])),len(times))
            obj.fokker_planc_reference = finite_volume_1d_cached()
            result = obj.fokker_planc_reference_solver(weights,
                [daily.held_signal(future["U"]),fit["f"],fit["g"],fit["C"],fit["k"],fit["Xi"]],
                sigma, t_eval=times, max_step=step)
            _, grid, _, W, *rest = result
            diagnostics = result[-1]
            rmean = np.zeros(len(times))
        else:
            result = bm_2.evolve(future,fit,weights,sigma,max_step=step)
            grid = result["x_grid"]; joint=result["cell_mass"]
            W=joint.sum(axis=2);rmean=(joint.sum(axis=1)*result["r_grid"]).sum(axis=1)
            diagnostics=result["diagnostics"]
        X=np.broadcast_to(grid,W.shape).copy()
        blocked=np.full(len(times),np.nan)
        cell_uniform=True
    if np.any(X < -1e-10) or np.any(X > 1+1e-10):
        raise FloatingPointError("Native deterministic state left [0,1]; retained as a failure.")
    if not np.all(np.isfinite(W)) or W.min() < -1e-12 or np.max(abs(W.sum(axis=1)-1))>1e-8:
        raise FloatingPointError("Invalid native state law.")
    D=np.array([bm_2.demand(x,float(b),float(u),fit)
                for x,b,u in zip(X,future["B"],future["U"])])
    return dict(time=times,x=X,weights=W,demand=D,r_mean=rmean,
                boundary_blocked_probability=blocked,cell_uniform=cell_uniform,
                diagnostics=diagnostics)


def boundary(law, epsilon):
    x, w=law["x"],law["weights"]
    if law["cell_uniform"]:
        width=1./x.shape[1]
        low=np.clip((epsilon-(x-width/2))/width,0,1)
        high=np.clip(((x+width/2)-(1-epsilon))/width,0,1)
    else:
        low=x<=epsilon; high=x>=1-epsilon
    return (w*low).sum(axis=1),(w*high).sum(axis=1)


def describe(law):
    w,x,d=law["weights"],law["x"],law["demand"]
    mean=(x*w).sum(axis=1); variance=((x-mean[:,None])**2*w).sum(axis=1)
    sd=np.sqrt(np.maximum(variance,0))
    skew=np.divide(((x-mean[:,None])**3*w).sum(axis=1),sd**3,
                   out=np.zeros_like(sd),where=sd>1e-12)
    demand=(d*w).sum(axis=1)
    demand_sd=np.sqrt(np.maximum(((d-demand[:,None])**2*w).sum(axis=1),0))
    intervals=predictive_intervals(list(d),list(w))
    return dict(mean_X=mean,sd_X=sd,skew_X=skew,mean_R=law["r_mean"],
                predicted=demand,sd_demand=demand_sd,intervals=intervals)


def refine_initial(config, model, fit, future, kind, data, origin, diffusion, scenario):
    fine={**config,"x_steps":2*config["x_steps"]-1,"r_steps":2*config["r_steps"]-1}
    if scenario is None:
        return fine,initial_law(fine,model,fit,future,kind,data,origin,diffusion)
    return fine,point_initial(fine,model,scenario,diffusion["sigma_x"])


def point_initial(config, model, scenario, sigma):
    x0,r0=scenario["x0"],scenario["r0"]
    if sigma==0:
        return np.array([x0]),np.array([r0]) if model=="bm2" else None,np.array([1.])
    x=(np.arange(config["x_steps"])+.5)/config["x_steps"]
    if model=="bm1":
        mass=np.zeros(len(x));mass[np.argmin(abs(x-x0))]=1.
        return x,None,mass
    x,r=bm_2.grid(config["x_steps"],config["r_steps"])
    mass=np.zeros((len(x),len(r)))
    mass[np.argmin(abs(x-x0)),np.argmin(abs(r-r0))]=1.
    return x,r,mass


def job_id(task):
    if task["type"]=="forecast":
        return f'{task["model"]}/{task["day"]}/{task["kind"]}_{task["hour"]:02d}'
    return f'{task["model"]}/scenarios/{task["label"]}'


def complete(config, task):
    folder=Path(config["output"])/"jobs"/job_id(task)
    try:
        done=json.loads((folder/"done.json").read_text())
        return done["fingerprint"]==config["fingerprint"] and all(
            sha(folder/name)==digest for name,digest in done["sha256"].items())
    except (OSError,ValueError,KeyError):
        return False


def worker(config, task):
    folder=Path(config["output"])/"jobs"/job_id(task);folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    try:
        model,day=task["model"],task["day"];data=load_data(config,day)
        scenario=task if task["type"]=="scenario" else None
        kind="daily" if scenario else task["kind"];hour=0 if scenario else task["hour"]
        fit,meta=get_fit(config,model,day,kind,hour,data)
        training,future=windows(data,day,kind,hour)
        future={k:np.asarray(future[k]) for k in ("Time","B","U")}
        origin=pd.Timestamp(day)+pd.Timedelta(hours=hour)
        diffusion=dict(sigma_x=0.,observation_sd=0.)
        if scenario:
            diffusion=scenario["diffusion"]
        elif kind=="mpc":
            estimator=BM1Diffusion if model=="bm1" else BM2Diffusion
            diffusion=estimator().fit_many(training,fit,substeps=2)
        sigma=diffusion["sigma_x"] if kind=="mpc" else 0.
        if scenario:
            # Refine sampling of held hourly inputs, without new observations.
            old=future;times=np.arange(97,dtype=float)/4
            idx=np.minimum(np.searchsorted(old["Time"],times,side="right")-1,len(old["Time"])-1)
            future=dict(Time=times,B=old["B"][idx],U=old["U"][idx])
            if scenario["price"]!="actual":
                future["U"]=np.full(len(times),float(scenario["price"]))
            sigma=diffusion["sigma_x"] if scenario["noise"]=="estimated" else 0.
            initial=point_initial(config,model,scenario,sigma)
        else:
            initial=initial_law(config,model,fit,future,kind,data,origin,diffusion)
        law=evolve(config,model,future,fit,initial,sigma)
        stats=describe(law);rows=[]
        score_indices=range(len(law["time"])) if scenario else (range(24) if kind=="daily" else range(1,9))
        observed=None
        if not scenario:
            index=data["target_index"] if kind=="daily" else pd.DatetimeIndex(
                [origin+pd.Timedelta(hours=i) for i in range(1,9)])
            observed=data["target_observed"] if kind=="daily" else data["mpc_frame"]["I"].reindex(index).to_numpy()
        for epsilon in EPSILONS:
            low,high=boundary(law,epsilon)
            for obs_i,i in enumerate(score_indices):
                row=dict(model=model,day=day,kind="scenario" if scenario else kind,
                    origin=str(origin),lead_hours=float(law["time"][i]),epsilon=epsilon,
                    p_low=float(low[i]),p_high=float(high[i]),p_boundary=float(low[i]+high[i]),
                    sigma_x=float(sigma),fit_converged=bool(meta["converged"]),
                    fit_stop_reason=meta["stop_reason"],
                    source_numeric_passed=meta.get("numerically_passed"),
                    boundary_blocked_probability=float(law["boundary_blocked_probability"][i]))
                row.update({key:float(stats[key][i]) for key in
                    ("mean_X","sd_X","skew_X","mean_R","predicted","sd_demand")})
                for level,(lo,hi) in stats["intervals"].items():
                    row[f"lower_{level}"],row[f"upper_{level}"]=float(lo[i]),float(hi[i])
                if scenario:
                    row.update(label=scenario["label"],X0=scenario["x0"],R0=scenario["r0"],
                               price=scenario["price"],noise=scenario["noise"])
                else:
                    y=float(observed[obs_i]);row.update(observed=y,baseline=float(future["B"][i]),
                        error=row["predicted"]-y,baseline_error=float(future["B"][i])-y)
                    row["error_original"]=row["error"]*data["raw_scale"]
                    for level in (68,95):
                        row[f"covered_{level}"]=row[f"lower_{level}"]<=y<=row[f"upper_{level}"]
                rows.append(row)
        pd.DataFrame(rows).to_csv(folder/"boundaries.csv",index=False)
        np.savez_compressed(folder/"state_law.npz",time=law["time"],x=law["x"],
                            weights=law["weights"],demand=law["demand"],
                            r_mean=law["r_mean"],cell_uniform=law["cell_uniform"])
        report=dict(task=task,sigma_x=float(sigma),diffusion=diffusion,fit=meta,
                    native_solver=law["diagnostics"],refinement=None,
                    seconds=time.perf_counter()-started)
        if not scenario and meta.get("reused_from"):
            source_csv=Path(meta["reused_from"]).with_name("hourly.csv" if kind=="daily" else "forecasts.csv")
            if source_csv.is_file():
                saved=pd.read_csv(source_csv)
                column="I_predicted" if kind=="daily" else "I_"+model.upper()
                if len(saved)!=len(score_indices) or not np.allclose(saved.I_observed,observed,rtol=0,atol=1e-12):
                    raise ValueError("Saved forecast observations differ from prepared data.")
                gap=float(np.max(abs(saved[column].to_numpy()-stats["predicted"][list(score_indices)])))
                report["saved_forecast_max_difference"]=gap
                report["saved_forecast_changed"]=gap>1e-4
        if task.get("refine"):
            fine_config,fine_initial=refine_initial(config,model,fit,future,kind,data,origin,diffusion,scenario)
            fine=evolve(fine_config,model,future,fit,fine_initial,sigma,refined=True)
            fs=describe(fine);lo,hi=boundary(law,.05);fl,fh=boundary(fine,.05)
            gaps=dict(mean_X=float(np.max(abs(stats["mean_X"]-fs["mean_X"]))),
                predicted=float(np.max(abs(stats["predicted"]-fs["predicted"]))),
                sd_X=float(np.max(abs(stats["sd_X"]-fs["sd_X"]))),
                p_boundary=float(np.max(abs(lo+hi-fl-fh))))
            report["refinement"]=dict(gaps=gaps,passed=bool(
                gaps["predicted"]<=.01 and gaps["mean_X"]<=.01 and
                gaps["sd_X"]<=.01 and gaps["p_boundary"]<=.05),
                thresholds=dict(predicted=.01,mean_X=.01,sd_X=.01,p_boundary=.05),
                interpretation="Pilot screening tolerances; not a convergence proof.")
        report["seconds"]=time.perf_counter()-started
        write_json(folder/"report.json",json_value(report))
        files=("boundaries.csv","state_law.npz","report.json")
        write_json(folder/"done.json",dict(fingerprint=config["fingerprint"],
                                         sha256={name:sha(folder/name) for name in files}))
        return dict(job=job_id(task),seconds=report["seconds"],status="complete")
    except Exception:
        write_json(folder/"error.json",dict(task=task,traceback=traceback.format_exc()))
        return dict(job=job_id(task),status="failed",error=traceback.format_exc())


def run_jobs(config, tasks):
    pending=[t for t in tasks if not complete(config,t)]
    failures=[];active={};started=time.perf_counter();finished=len(tasks)-len(pending)
    successful=finished
    print(f"{len(tasks)} jobs: {finished} checkpoints, {len(pending)} remaining; workers={config['workers']}; BLAS=1.",flush=True)
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        while pending or active:
            stopped=(Path(config["output"])/"STOP").exists()
            while pending and len(active)<config["workers"] and not stopped:
                task=pending.pop(0);active[pool.submit(worker,config,task)]=task
            if not active:
                break
            ready,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in ready:
                task=active.pop(future)
                try:
                    result=future.result()
                except Exception:
                    result=dict(status="failed",job=job_id(task),error=traceback.format_exc())
                finished+=1
                if result["status"]=="complete" and not complete(config,task):
                    result=dict(status="failed",job=job_id(task),error="Incomplete checkpoint files.")
                if result["status"]!="complete":
                    failures.append(result);print(result["error"],flush=True)
                else:
                    successful+=1
                print(f"{finished}/{len(tasks)} {result['status']}: {result['job']}",flush=True)
                write_json(Path(config["output"])/"progress.json",dict(
                    completed=successful,expected=len(tasks),
                    failed=failures,seconds=time.perf_counter()-started))
    return failures


def summarize(config, tasks):
    out=Path(config["output"]);tables=[];reports=[]
    for task in tasks:
        if complete(config,task):
            folder=out/"jobs"/job_id(task)
            tables.append(pd.read_csv(folder/"boundaries.csv"))
            reports.append(json.loads((folder/"report.json").read_text()))
    if not tables:
        return
    allrows=pd.concat(tables,ignore_index=True);allrows.to_csv(out/"all_boundaries.csv",index=False)
    real=allrows[allrows.kind!="scenario"].copy();summary=[]
    if len(real):
        real["risk"]=pd.cut(real.p_boundary,[-1e-12,.1,.5,1.00000001],
                            labels=["below_10pct","10_to_50pct","above_50pct"])
        for keys,part in real.groupby(["model","kind","epsilon","risk"],observed=True):
            summary.append(dict(zip(("model","kind","epsilon","risk"),keys),
                n=len(part),origins=part.origin.nunique(),rmse=float(np.sqrt(np.mean(part.error**2))),
                baseline_rmse=float(np.sqrt(np.mean(part.baseline_error**2))),
                bias=float(part.error.mean()),coverage_68=float(part.covered_68.mean()),
                coverage_95=float(part.covered_95.mean()),
                width_95=float((part.upper_95-part.lower_95).mean()),
                nonconverged_rows=int((~part.fit_converged).sum())))
    pd.DataFrame(summary).to_csv(out/"forecast_by_boundary_risk.csv",index=False)
    selected=allrows[allrows.epsilon==.05];scenarios=selected[selected.kind=="scenario"];srows=[]
    for (model,label),part in (scenarios.groupby(["model","label"]) if len(scenarios) else []):
        part=part.sort_values("lead_hours");times=part.lead_hours.to_numpy()
        occupancy=float(np.trapz(part.p_boundary,times))
        exits=part[part.p_boundary<.5]
        initially_near=bool(part.p_boundary.iloc[0]>=.5)
        srows.append(dict(model=model,label=label,
            expected_boundary_hours=occupancy,
            first_sample_below_50pct_boundary_probability_hours=float(exits.lead_hours.iloc[0]) if initially_near and len(exits) else None,
            starts_near_boundary=initially_near,
            end_mean_X=float(part.mean_X.iloc[-1]),end_sd_X=float(part.sd_X.iloc[-1]),
            max_p_boundary=float(part.p_boundary.max())))
    pd.DataFrame(srows).to_csv(out/"scenario_summary.csv",index=False)
    refinements=[dict(model=r["task"]["model"],job=job_id(r["task"]),
        **r["refinement"]["gaps"],passed=r["refinement"]["passed"])
        for r in reports if r["refinement"]]
    pd.DataFrame(refinements).to_csv(out/"numerical_refinement.csv",index=False)
    write_json(out/"completion.json",dict(expected=config["planned_jobs"],completed=len(reports),
        complete=config["planned_jobs"]==len(reports),
        refinements_passed=bool(refinements) and all(r["passed"] for r in refinements),
        reused_forecast_fits=sum(bool(r["fit"].get("reused_from")) for r in reports if r["task"]["type"]=="forecast"),
        note="Daily intervals describe initial-state spread; MPC intervals are native conditional demand intervals."))
    import matplotlib.pyplot as plt
    for model in config["models"]:
        tab=selected[selected.model==model]
        if tab.empty:
            continue
        real=tab[tab.kind!="scenario"];scenario=tab[tab.kind=="scenario"]
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        if len(real):
            for kind,part in real.groupby("kind"):
                bylead=part.groupby("lead_hours")[["p_low","p_high"]].mean()
                axes[0,0].plot(bylead.index,bylead.p_low,label=kind+" lower")
                axes[0,0].plot(bylead.index,bylead.p_high,"--",label=kind+" upper")
            axes[0,1].scatter(real.p_boundary,abs(real.error),s=8,alpha=.3)
        if len(scenario):
            for label,part in scenario.groupby("label"):
                chosen={t["label"] for t in tasks if t["type"]=="scenario" and t["model"]==model and t.get("refine")}
                if label in chosen or label.replace("_zero","_estimated") in chosen or ("_uactual_" in label and "_r0_" in label):
                    axes[1,0].plot(part.lead_hours,part.mean_X,alpha=.7,label=label)
                    axes[1,1].plot(part.lead_hours,part.predicted,alpha=.7)
        for ax in axes.ravel():
            ax.grid(alpha=.2)
        axes[0,0].set(title="Predicted boundary probability",xlabel="Lead hours",ylabel="Probability")
        axes[0,0].legend(fontsize=7)
        axes[0,1].set(title="Forecast error vs predicted boundary risk",xlabel="Boundary probability",ylabel="Absolute normalized error")
        axes[1,0].set(title="Controlled state trajectories",xlabel="Hours",ylabel="Mean X")
        axes[1,1].set(title="Controlled demand trajectories",xlabel="Hours",ylabel="Normalized demand")
        fig.suptitle(model.upper()+" logistic boundary study");fig.tight_layout()
        fig.savefig(out/f"{model}_boundary_overview.png",dpi=160);plt.close(fig)
        scenario_tasks=[t for t in tasks if t["model"]==model and t["type"]=="scenario" and t.get("refine") and complete(config,t)]
        if scenario_tasks:
            fig,axes=plt.subplots(len(scenario_tasks),3,figsize=(12,3*len(scenario_tasks)),squeeze=False)
            for row,task in zip(axes,scenario_tasks):
                law=np.load(out/"jobs"/job_id(task)/"state_law.npz")
                for ax,hour in zip(row,(1,8,24)):
                    i=int(np.argmin(abs(law["time"]-hour)))
                    ax.hist(law["x"][i],bins=np.linspace(0,1,41),weights=law["weights"][i])
                    ax.set(title=f'{task["label"]} / {hour}h',xlabel="X",ylabel="Probability mass")
            fig.tight_layout();fig.savefig(out/f"{model}_boundary_distributions.png",dpi=160);plt.close(fig)


def scenario_tasks(config):
    tasks=[];xs=(.01,.99) if config["study"]=="pilot" else (.01,.05,.5,.95,.99)
    prices=("0.0","1.0") if config["study"]=="pilot" else ("0.0","0.5","1.0","actual")
    for model in config["models"]:
        data=load_data(config,config["anchor_day"])
        fit,meta=get_fit(config,model,config["anchor_day"],"daily",0,data)
        historic=float(np.mean(fit["initial_states"][:,1])) if model=="bm2" else 0.
        calibration=Path(config["output"])/"calibration"/f"{model}.json"
        if calibration.is_file():
            saved=json.loads(calibration.read_text())
            if saved["fingerprint"]!=config["fingerprint"]:
                raise ValueError("Anchor calibration identity mismatch.")
            diffusion=saved["diffusion"]
        else:
            training,_=windows(data,config["anchor_day"],"daily",0)
            estimator=BM1Diffusion if model=="bm1" else BM2Diffusion
            diffusion=estimator().fit_many(training,fit,substeps=2)
            write_json(calibration,json_value(dict(fingerprint=config["fingerprint"],diffusion=diffusion)))
        for x0 in xs:
            rs=(0.,) if model=="bm1" else ((-.5 if x0<.5 else .5),historic) if config["study"]=="pilot" else (-.5,historic,.5)
            for r_i,r0 in enumerate(rs):
                for price in prices:
                    for noise in ("zero","estimated"):
                        label=f"x{x0:.2f}_r{r_i}_u{price}_{noise}"
                        refine=(x0 in (.01,.99) and price==("1.0" if x0<.5 else "0.0")
                                and noise=="estimated" and (model=="bm1" or config["study"]=="week" or r_i==0))
                        if config["study"]=="week" and model=="bm2":
                            refine=refine and r0==(-.5 if x0<.5 else .5)
                        tasks.append(dict(type="scenario",model=model,day=config["anchor_day"],
                            label=label,x0=x0,r0=float(r0),price=price,noise=noise,refine=refine,diffusion=diffusion))
    return tasks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=ROOT)
    parser.add_argument("--study",choices=("pilot","week"),default="pilot")
    parser.add_argument("--reference",type=Path,default=Path("files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08"))
    parser.add_argument("--bm1-fit-root",type=Path,nargs="+")
    parser.add_argument("--bm2-fit-root",type=Path,nargs="+")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--models",choices=("bm1","bm2"),nargs="+",default=["bm1","bm2"])
    parser.add_argument("--fit-missing",action="store_true")
    parser.add_argument("--preflight",action="store_true",help="Inspect only; no fits or forecasts.")
    parser.add_argument("--x-steps",type=int,default=101)
    parser.add_argument("--r-steps",type=int,default=41)
    parser.add_argument("--fp-max-step",type=float,default=.25)
    args=parser.parse_args()
    if not 1<=args.workers<=8 or args.x_steps<3 or args.r_steps<3 or args.fp_max_step<=0:
        parser.error("workers must be 1..8; grids >=3; fp-max-step positive.")
    reference=args.reference.resolve()
    original=json.loads((reference/"settings.json").read_text(encoding="utf-8"))
    if original.get("schema") not in ("bm1-response-week-v1","bm1-joint-week-v1","bm1-bm2-week-v1"):
        parser.error("Use the existing prepared five-window week run as --reference.")
    days=["2022-05-05"] if args.study=="pilot" else [str(d.date()) for d in pd.date_range("2022-05-02","2022-05-08")]
    if not set(days)<=set(original["days"]):
        parser.error("Requested days are absent from reference.")
    for name,digest in original["inputs_sha256"].items():
        if sha(ROOT/"files"/"input"/"data"/name)!=digest:
            raise ValueError("Input workbook changed since reference: "+name)
    out=(args.output or Path(f"files/output/bm_1/runs/boundary_logistic_{args.study}")).resolve()
    candidates=dict(
        bm1=args.bm1_fit_root or [reference,ROOT/"files/output/bm_1/runs/fg_week_joint_2022-05-02_2022-05-08"],
        bm2=args.bm2_fit_root or [ROOT/"files/output/bm_2/runs/week_2022-05-02_2022-05-08"])
    roots={model:[str(p.resolve()) for p in paths] for model,paths in candidates.items()}
    sources=[reference]+[Path(p) for paths in roots.values() for p in paths]
    if any(out==p or out in p.parents or p in out.parents for p in sources):
        parser.error("Choose an independent output directory beside the reference runs.")
    prepared=reference/"prepared"
    config=dict(schema="bm-boundary-logistic-v1",study=args.study,days=days,anchor_day="2022-05-05",
        models=list(dict.fromkeys(args.models)),workers=args.workers,output=str(out),
        prepared_root=str(prepared),prepared_sha256={day:sha(prepared/f"{day}.pkl") for day in days},
        reference_settings_sha256=sha(reference/"settings.json"),fit_roots=roots,
        runner_sha256=sha(Path(__file__)),
        fit_missing=args.fit_missing,x_steps=args.x_steps,r_steps=args.r_steps,
        fp_max_step=args.fp_max_step,BLAS_threads=1,epsilons=list(EPSILONS),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"src").rglob("*.py"))})
    for key in ("daily_max_iterations","daily_max_nfev","mpc_max_iterations","mpc_max_nfev"):
        config[key]=original[key]
    hours=(0,8,16) if args.study=="pilot" else range(24)
    tasks=[dict(type="forecast",model=model,day=day,kind=kind,hour=hour,refine=(day=="2022-05-05" and kind=="mpc" and hour==0))
        for day in days for model in config["models"]
        for kind,hour in [("daily",0)]+[("mpc",h) for h in hours]]
    scenario_counts={"bm1":8,"bm2":16} if args.study=="pilot" else {"bm1":40,"bm2":120}
    config["planned_jobs"]=len(tasks)+sum(scenario_counts[m] for m in config["models"])
    found=0;missing=[];config["fit_sha256"]={}
    for task in tasks:
        path,meta=find_fit(config,task["model"],task["day"],task["kind"],task["hour"])
        if path:
            config["fit_sha256"][str(path)]=sha(path);found+=1
        else:
            missing.append(job_id(task))
    print(f"Prepared days: {len(days)}; ordinary jobs: {len(tasks)}; reusable fits: {found}; missing: {len(missing)}.",flush=True)
    if missing:
        print("Missing fits:\n"+"\n".join(missing),flush=True)
    if args.preflight:
        return 0
    if missing and not args.fit_missing:
        parser.error("Supply existing fit roots or explicitly use --fit-missing.")
    # Worker count is operational; permit resume with a different number of workers.
    config["fingerprint"]=fingerprint({k:v for k,v in config.items() if k!="workers"})
    out.mkdir(parents=True,exist_ok=True)
    path=out/"settings.json"
    if path.exists() and json.loads(path.read_text())["fingerprint"]!=config["fingerprint"]:
        raise ValueError("Source/data/settings changed. Use a new output directory.")
    write_json(path,config)
    failures=run_jobs(config,tasks)
    if not (out/"STOP").exists():
        scenarios=scenario_tasks(config)
        tasks+=scenarios
        failures+=run_jobs(config,scenarios)
    summarize(config,tasks)
    print("Saved:",out,flush=True)
    return 1 if failures else 0


if __name__=="__main__":
    sys.exit(main())

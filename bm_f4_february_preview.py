#!/usr/bin/env python3
"""BM-only real-data F4 H/I/J output preview for BSCe2026.

Run from the repository root:
  python -B bm_f4_february_preview.py --audit-only
  python -B bm_f4_february_preview.py --observation-contract hourly-point-approximation

Default: 7-13 February 2022, fixed origin, 168 hours, MED meters. Reads the
real files/input/data/input.xlsx and prices.xlsx; no synthetic data, WM
execution, weather downloads or network calls. Future historical DK1 prices
are known oracle inputs, not price forecasts.

Provisional FV application, NOT completed/accepted F4:
* small DISCRETE rho/Xi prior, not the validated continuous hat bank;
* shared monotone f/g jointly fit across TRAIN days; linear/2-piece/4-piece
  comparison on earlier MODEL_SELECTION; selected pair frozen for forecast/control;
* fixed k/sigma_x/sigma_y; conditional curve fit at declared rho/Xi before the
  final rho/Xi bank; no joint recovery or curve-uncertainty claim;
* Gaussian point likelihood; hourly-point-approximation is explicitly provisional
  when hourly consumption/average-power readings represent intervals;
* no mixture/noise-bank/population calibration or accepted numerical convergence;
* optional restricted day/night tariff control with plug-in parameters,
  conservative marginal-state handoff and conditional-bank forward forecasts;
* control target is a history-only baseline, never held-out measured I.

WM4-compatible output types: observed and predicted demand, reference baseline,
price, model-specific state, MSE/RMSE/MAE/bias. BM X is not WM4 F/R. Figures are
hourly. Normalization is fitted before MODEL_SELECTION; separate WM4 comparisons
must use identical independently declared scales or raw-unit metrics.

Fresh directory files/output/bm_1/runs/f4_bm_preview/<week>_<timestamp>/:
bm_weekly_prediction.png, hourly_results.csv, daily_results.csv, summary.json,
parameter_posterior.csv, origin_bank.npz, response_curves.json/.csv/.png,
curve_validation.csv, curve_trials.csv and optional control_hourly.csv/control.png.
Default TRAIN: 1-30 January; MODEL_SELECTION: 31 January-6 February; forecast:
7-13 February 2022. Illustrative chronological split, not a registered blind
final holdout. Curve baseline is TRAIN-only weekday/hour mean; forecast baseline
is the mandatory five-previous-weekday reference. Every curve candidate replays
the complete continuous record without state recycling or midnight resets.
Retained curves get finer FV checks; reported checks do not complete H-f/g or I/J.
Compatible main source inspected: a74fe448f736ecc430852ecfedec2c35a5daf51f.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
from time import perf_counter

# Set before importing numerical libraries, including in spawned Windows workers.
for _thread_name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_name] = "1"

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize

INSPECTED_COMMIT = "a74fe448f736ecc430852ecfedec2c35a5daf51f"
TZ = "Europe/Copenhagen"


def repository(path):
    root = Path(path).resolve()
    if not (root / "src/model/bm_1.py").is_file():
        raise ValueError("Run from the BSCe2026 root or provide --repo PATH.")
    sys.path.insert(0, str(root))
    return root


def phase_progress(started, completed, total, message):
    """Cheap phase-scoped ETA; no extra model evaluations or checkpoints."""
    from src.exe.run_model.progress import runtime_text
    print(f"{message}; phase {runtime_text(started, completed, total)}", flush=True)


class PiecewiseResponse:
    """One immutable, shared curve: argument selects the segment, never date."""
    def __init__(self, knots, values):
        self.knots = np.asarray(knots, dtype=float).copy()
        self.values = np.asarray(values, dtype=float).copy()
        if (self.knots.ndim != 1 or self.knots.shape != self.values.shape or
            len(self.knots) < 2 or not np.all(np.isfinite(self.knots)) or
            not np.all(np.isfinite(self.values)) or self.knots[0] != 0 or
            self.knots[-1] != 1 or np.any(np.diff(self.knots) <= 0) or
            self.values[0] != 1 or self.values[-1] != -1 or
            np.any(np.diff(self.values) > 0)):
            raise ValueError("Curves require increasing [0,1] knots, endpoints 1/-1 and nonincreasing heights.")
        self.knots.setflags(write=False)
        self.values.setflags(write=False)

    def __call__(self, argument):
        z = np.asarray(argument, dtype=float)
        if not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(z > 1):
            raise ValueError("f/g arguments must be finite in [0,1]; no extrapolation.")
        return np.interp(z.ravel(), self.knots, self.values).reshape(z.shape)


def linear_curve_spec():
    return dict(family="linear", knots=[0.0, 1.0], f=[1.0, -1.0], g=[1.0, -1.0])


def curve_pair(spec):
    spec = spec or linear_curve_spec()
    return (PiecewiseResponse(spec["knots"], spec["f"]),
            PiecewiseResponse(spec["knots"], spec["g"]))


def shape_spec(logits, segments):
    """Positive falls with total 2 enforce all shape constraints exactly."""
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (2*(segments-1),) or not np.all(np.isfinite(logits)):
        raise ValueError("One bounded logit per independent segment fall is required.")
    knots = np.linspace(0, 1, segments+1)
    curves = []
    for values in np.split(logits, 2):
        values = np.r_[values, 0.0]
        falls = 2*np.exp(values-logsumexp(values))
        heights = np.r_[1.0, 1-np.cumsum(falls)]
        heights[-1] = -1.0
        curves.append(heights.tolist())
    spec = dict(family=f"{segments}-piece", knots=knots.tolist(), f=curves[0], g=curves[1])
    curve_pair(spec)
    return spec


def curve_dates(start, args):
    first = pd.Timestamp(getattr(args, "curve_train_start", "2022-01-01")).normalize()
    selection = start-pd.Timedelta(days=7)
    if first.tzinfo is not None or first >= selection-pd.Timedelta(days=7):
        raise ValueError("Curve TRAIN needs at least seven complete days before the selection week.")
    return first, selection


def hold(values, hours_per_bin=1.0):
    values = np.asarray(values, dtype=float)
    def evaluate(t):
        index = np.minimum(np.floor(np.asarray(t) * 24.0 / hours_per_bin + 1e-10)
                           .astype(int), len(values) - 1)
        if np.any(index < 0):
            raise ValueError("Signal requested before its origin.")
        return values[index]
    return evaluate


def metric(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    if not valid.any():
        raise ValueError("No observed values are available for scoring.")
    e = p[valid] - a[valid]
    return dict(n=int(valid.sum()), mse=float(np.mean(e*e)),
                rmse=float(np.sqrt(np.mean(e*e))), mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_info(root):
    def run(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(commit=run("rev-parse", "HEAD"),
                working_tree_status=run("status", "--porcelain"),
                inspected_commit=INSPECTED_COMMIT)


def load_raw(root, tag, end):
    from src.tools.import_data.import_input import input_data
    data_dir = root / "files/input/data"
    reader = input_data(data_dir / "input.xlsx")
    columns = [c for c in reader.measurement_columns
               if not tag or reader.measurement_tag(c).upper() == tag.upper()]
    if not columns:
        raise ValueError(f"No measurement columns for tag {tag!r}.")
    readings = reader.observed_data[columns].apply(pd.to_numeric, errors="coerce")
    if not readings.index.is_unique:
        raise ValueError("Duplicate input timestamps require an explicit data decision.")
    # Keep the selected population fixed: partial meter rows are missing, not
    # means of a different subset. Never use reader.data's interpolation.
    consumption = readings.mean(axis=1).where(readings.notna().all(axis=1))
    consumption = consumption.sort_index().rename("I_raw")
    # WM's spot-price importer interpolates its public data view. Read the
    # same workbook columns directly to retain the genuine observation mask.
    raw = pd.read_excel(data_dir/"prices.xlsx", engine="openpyxl")
    raw.columns = raw.columns.astype(str).str.strip()
    if not {"SpotPriceDKK", "HourUTC", "PriceArea"}.issubset(raw.columns):
        raise ValueError("Expected HourUTC, PriceArea and SpotPriceDKK in prices.xlsx.")
    raw = raw.loc[raw.PriceArea.astype(str).str.strip().str.upper().eq("DK1")].copy()
    utc = pd.to_datetime(raw.HourUTC, utc=True, errors="raise")
    u = pd.Series(pd.to_numeric(raw.SpotPriceDKK, errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(utc).tz_convert(TZ).tz_localize(None))
    # Only the used historical/forecast window needs a local-time contract.
    # A full-year workbook legitimately repeats 02:00 at October's DST end;
    # that unrelated hour must not prevent this January/February preview.
    u = u.loc[u.index < end]
    if not u.index.is_unique:
        raise ValueError("Duplicate DK1 timestamps require an explicit data decision.")
    frame = pd.concat([consumption, u.rename("price_raw")], axis=1).sort_index()
    frame = frame.loc[frame.index < end]
    metadata = dict(tag=tag, measurement_columns=columns,
                    population_policy="fixed selected meters; any missing meter => missing aggregate",
                    source_start=str(consumption.index.min()),
                    source_end=str(consumption.index.max()),
                    source_hashes={p.name: sha256(p) for p in
                                   (data_dir/"input.xlsx", data_dir/"prices.xlsx")})
    return frame, metadata


def prepare(frame, start, args):
    from src.tools.bm_1_data import previous_weekday_mean
    end = start + pd.Timedelta(days=7)
    train_start = start - pd.Timedelta(days=7)
    # This preview avoids DST; the underlying H adapter needs a richer interval
    # and timestamp contract before extending to arbitrary transition weeks.
    times = pd.date_range(train_start, end, freq="h", inclusive="left")
    aware = times.tz_localize(TZ, ambiguous="raise", nonexistent="raise")
    if len(aware) != 336 or not np.all(np.diff(aware.asi8) == 3600*10**9):
        raise ValueError("Preview requires two complete 168-hour weeks without DST.")
    scale_cutoff = curve_dates(start, args)[1] if getattr(args, "curve_mode", "linear") == "auto" else start
    history = frame.loc[frame.index < scale_cutoff, "I_raw"].dropna()
    if history.empty or (history < 0).any():
        raise ValueError("A nonnegative historical consumption population is required.")
    scale = float(args.power_scale if args.power_scale is not None else 2*history.max())
    if not np.isfinite(scale) or scale <= 0 or history.max() > scale:
        raise ValueError("Power scale must be positive and contain historical consumption.")
    price_history = frame.loc[frame.index < scale_cutoff, "price_raw"].dropna()
    pmin, pmax = float(price_history.min()), float(price_history.max())
    span = pmax - pmin
    lo = float(args.price_low if args.price_low is not None else pmin - 0.5*span)
    hi = float(args.price_high if args.price_high is not None else pmax + 0.5*span)
    if not np.isfinite(lo+hi) or hi <= lo:
        raise ValueError("Price bounds must be finite with low < high.")
    selected = frame.reindex(times).copy()
    if selected["price_raw"].isna().any():
        raise ValueError("Missing observed price inside training/forecast window; no imputation.")
    selected["I"] = selected["I_raw"] / scale
    selected["U_bm"] = (selected["price_raw"] - lo) / (hi-lo)
    if selected["U_bm"].min() < 0 or selected["U_bm"].max() > 1:
        raise ValueError("A price is outside the frozen bounds. Supply independently chosen physical --price-low/--price-high; prices are never clipped.")
    target = pd.date_range(start, end, freq="h", inclusive="left", tz=TZ)
    hs = frame["I_raw"].copy()
    hs.index = hs.index.tz_localize(TZ, ambiguous="raise", nonexistent="raise")
    baseline, source_dates = previous_weekday_mean(hs, target, origin=start.tz_localize(TZ))
    base = baseline.to_numpy()/scale
    if np.any((base < 0) | (base > 1)):
        raise ValueError("Historical baseline is outside BM's physical power scale.")
    # Offline baseline fit uses all information available at forecast origin.
    # Repeat the frozen weekly profile over the previous fitting week. It is
    # not advertised as prequential baseline estimation during that fit.
    selected["B"] = np.tile(base, 2)
    scaling = dict(power_scale=scale, consumption_transform="raw / power_scale",
                   power_scale_source="explicit" if args.power_scale else "2 * maximum before normalization cutoff",
                   price_low=lo, price_high=hi, price_transform_bm="(raw-low)/(high-low)",
                   fitted_before=str(scale_cutoff))
    return selected, scaling, [[t.isoformat() for t in row] for row in source_dates]


def curve_dataset(frame, start, scaling, args):
    first, selection = curve_dates(start, args)
    index = pd.date_range(first, start, freq="h", inclusive="left")
    utc = index.tz_localize(TZ, ambiguous="raise", nonexistent="raise")
    if not np.all(np.diff(utc.asi8) == 3600*10**9):
        raise ValueError("Curve preview's declared record must not cross DST.")
    data = frame.reindex(index).copy()
    data["I"] = data.I_raw/scaling["power_scale"]
    data["U_bm"] = (data.price_raw-scaling["price_low"])/(scaling["price_high"]-scaling["price_low"])
    if data.U_bm.isna().any() or np.any((data.U_bm < 0)|(data.U_bm > 1)):
        raise ValueError("Curve record needs complete prices in the frozen physical domain.")
    training = data.loc[data.index < selection]
    if training.I.dropna().empty or np.any(training.I.dropna() < 0):
        raise ValueError("No usable nonnegative curve TRAIN observations.")
    # This baseline is estimated ONLY on TRAIN. The separate five-weekday
    # forecast reference remains in prepare(), and is never used to choose curves.
    profile = training.I.groupby([training.index.dayofweek, training.index.hour]).mean()
    counts = training.I.groupby([training.index.dayofweek, training.index.hour]).count()
    keys = pd.MultiIndex.from_arrays([index.dayofweek,index.hour])
    data["B"] = profile.reindex(keys).to_numpy()
    if data.B.isna().any() or np.any((data.B < 0)|(data.B > 1)):
        raise ValueError("Curve TRAIN must cover every weekday/hour with a valid baseline.")
    meta = dict(train_start=first.isoformat(), train_end_exclusive=selection.isoformat(),
                model_selection_start=selection.isoformat(), model_selection_end_exclusive=start.isoformat(),
                forecast_holdout_start=start.isoformat(), forecast_holdout_end_exclusive=(start+pd.Timedelta(days=7)).isoformat(),
                split_status="illustrative chronological split; not a registered blind FINAL_HOLDOUT",
                baseline="TRAIN-only weekday/hour arithmetic mean; separate five-weekday forecast reference",
                baseline_minimum_training_values_per_bin=int(counts.min()),
                fixed_rho=args.curve_rho, fixed_Xi=args.curve_xi, fixed_k=args.k,
                shared_population="same selected meter set and normalization across every training day",
                initial_law="Beta(20,20) at TRAIN origin; no midnight reset",
                initial_hour_excluded=True)
    return data, len(training), meta


def curve_filter(data, spec, args, *, cells, max_step, initial_mass=None,
                 initial_hour=0, observation_hours=None, keep_history=True):
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_cached
    from src.method.bm_1_estimation import run_fv_filter
    from src.params.bm_1 import BM1ObservationSeries
    from src.control.bm_1 import beta_initial_cell_mass
    hours = np.arange(1,len(data)) if observation_hours is None else np.asarray(observation_hours)
    if hours.size == 0:
        raise ValueError("A curve record needs observations after its initial prior.")
    f,g = curve_pair(spec)
    model = bm_1(hold(data.B.to_numpy()), hold(data.B.to_numpy()), (0.0,(len(data)-1)/24),2)
    model.fokker_planc_reference = finite_volume_1d_cached()
    theta = [hold(data.U_bm.to_numpy()),f,g,1/args.curve_rho,args.k,args.curve_xi]
    y = data.I.to_numpy()[hours]
    obs = BM1ObservationSeries(hours/24,y,"day",np.isfinite(y),"point","right")
    if initial_mass is None:
        initial_mass = beta_initial_cell_mass(np.linspace(0,1,cells+1))
    return run_fv_filter(model,theta,args.sigma_x,args.sigma_y,initial_mass,initial_hour/24,obs,
            breakpoints=np.arange(1,len(data)-1)/24,max_step=max_step,
            likelihood_quadrature_order=3, keep_history=keep_history)


def shape_penalty(spec):
    k = np.asarray(spec["knots"])
    linear = 1-2*k
    deviations = np.r_[np.asarray(spec["f"])[1:-1]-linear[1:-1],
                       np.asarray(spec["g"])[1:-1]-linear[1:-1]]
    return 0.0 if len(deviations)==0 else float(np.mean(deviations**2))


def fit_curve_family(job):
    repository(job["repo"])
    args, data, segments = job["args"], job["training"], job["segments"]
    trials, cache = [], {}
    best = None
    clock = perf_counter()
    def objective(logits):
        nonlocal best
        key = tuple(np.asarray(logits,float))
        if key not in cache:
            spec = shape_spec(logits,segments)
            fit = curve_filter(data,spec,args,cells=args.curve_cells,max_step=args.curve_max_step,
                               keep_history=False)
            observed = int(fit.observed.sum())
            if observed == 0:
                raise ValueError("No observed curve training samples.")
            value = -fit.total_log_likelihood/observed+args.curve_regularization*shape_penalty(spec)
            record = dict(family=spec["family"],evaluation=len(trials)+1,objective=float(value),
                    mean_training_log_score=float(fit.total_log_likelihood/observed),
                    penalty=shape_penalty(spec),maximum_mass_error=fit.maximum_abs_mass_error,
                    elapsed_seconds=perf_counter()-clock,logits=list(key))
            trials.append(record)
            cache[key] = value
            if best is None or value < best["objective"]:
                best = dict(objective=float(value),spec=spec,logits=list(key))
        return cache[key]
    dim = 2*(segments-1)
    starts = [np.zeros(dim), np.r_[np.linspace(-.5,.5,segments-1),np.linspace(.5,-.5,segments-1)]]
    solvers = []
    for start_number, initial in enumerate(starts, 1):
        fit = minimize(objective,initial,method="Powell",bounds=[(-4.,4.)]*dim,
                options=dict(maxiter=8,maxfev=args.curve_evaluations,xtol=2e-3,ftol=1e-4))
        solvers.append(dict(success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev),
                            logits=fit.x.tolist(),objective=float(fit.fun)))
        phase_progress(clock, start_number, len(starts),
                       f"H-f/g {segments}-piece: optimizer start {start_number}/{len(starts)}")
    return dict(family=f"{segments}-piece",curve=best["spec"],objective=best["objective"],
                trials=trials,solvers=solvers,distinct_evaluations=len(trials),
                fit_converged=any(s["success"] and np.array_equal(s["logits"],best["logits"])
                                  for s in solvers),seconds=perf_counter()-clock,
                status="best tested shared curve within declared budget; convergence not presumed")


def validate_curve(spec, data, n_train, args):
    training = data.iloc[:n_train]
    coarse = curve_filter(training,spec,args,cells=args.curve_cells,max_step=args.curve_max_step)
    fine = curve_filter(training,spec,args,cells=args.cells,max_step=args.max_step)
    validation_hours = np.arange(n_train,len(data))
    coarse_val = curve_filter(data,spec,args,cells=args.curve_cells,max_step=args.curve_max_step,
                             initial_mass=coarse.posterior_mass[-1],initial_hour=n_train-1,
                             observation_hours=validation_hours)
    fine_val = curve_filter(data,spec,args,cells=args.cells,max_step=args.max_step,
                           initial_mass=fine.posterior_mass[-1],initial_hour=n_train-1,
                           observation_hours=validation_hours)
    obs = fine_val.observed
    if not obs.any():
        raise ValueError("MODEL_SELECTION contains no observed readings.")
    # Open seven-day selection forecast: propagate TRAIN's last posterior to
    # midnight and then run without any selection-week observation updates.
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_cached
    f,g = curve_pair(spec)
    last = training.iloc[-1]
    tail = bm_1(hold([last.B]),hold([last.B]),(0.0,1/24),2)
    tail.fokker_planc_reference = finite_volume_1d_cached()
    theta = [hold([last.U_bm]),f,g,1/args.curve_rho,args.k,args.curve_xi]
    origin_mass = tail.fokker_planc_reference_solver(fine.posterior_mass[-1],theta,args.sigma_x,
                                t_eval=[0.,1/24],max_step=args.max_step)[3][-1]
    fixed = pd.DataFrame(dict(rho=[args.curve_rho],Xi=[args.curve_xi],weight=[1.0]))
    validation = data.iloc[n_train:]
    forecast, distribution = bm_forecast(fixed,origin_mass[None,:],validation.B.to_numpy(),
                                       validation.U_bm.to_numpy(),args,curve=spec)
    stats = metric(validation.I,forecast["I_bm"])
    d,p = distribution
    residual = (validation.I.to_numpy()[None,:,None]-d)/args.sigma_y
    with np.errstate(divide="ignore"):
        log_scores = logsumexp(np.log(p)-.5*residual**2-np.log(args.sigma_y)-.5*np.log(2*np.pi),axis=(0,2))
    report = dict(family=spec["family"],free_shape_parameters=2*(len(spec["knots"])-2),
        mean_training_log_score=float(fine.total_log_likelihood/fine.observed.sum()),
        mean_selection_predictive_log_score=float(fine_val.total_log_likelihood/obs.sum()),
        mean_selection_open_log_score=float(np.mean(log_scores[np.isfinite(validation.I)])),
        selection_open_rmse=stats["rmse"],selection_open_mae=stats["mae"],
        train_total_score_refinement=float(abs(fine.total_log_likelihood-coarse.total_log_likelihood)),
        selection_total_score_refinement=float(abs(fine_val.total_log_likelihood-coarse_val.total_log_likelihood)),
        maximum_prefix_score_refinement=float(np.max(np.abs(np.cumsum(fine_val.log_evidence[obs])-np.cumsum(coarse_val.log_evidence[obs])))),
        maximum_mass_error=max(fine.maximum_abs_mass_error,fine_val.maximum_abs_mass_error),
        minimum_cell_mass=min(fine.minimum_cell_mass,fine_val.minimum_cell_mass),
        numerical_acceptance=False)
    # Interval coverage is about covariate/state visitation, not identification.
    edges = np.linspace(0,1,args.cells+1)
    centers = .5*(edges[:-1]+edges[1:])
    coverage = dict(price_fraction=[],posterior_state_fraction=[])
    for left,right in zip([0,.25,.5,.75],[.25,.5,.75,1]):
        price = training.U_bm.to_numpy()
        pmask = (price>=left)&((price<=right) if right==1 else (price<right))
        xmask = (centers>=left)&((centers<=right) if right==1 else (centers<right))
        coverage["price_fraction"].append(float(np.mean(pmask)))
        coverage["posterior_state_fraction"].append(float(np.mean(fine.posterior_mass[:,xmask].sum(axis=1))))
    return report,coverage


def validate_curve_job(job):
    repository(job["repo"])
    return validate_curve(job["curve"],job["data"],job["n_train"],job["args"])


def learn_curves(frame, start, scaling, root, args):
    if args.curve_mode == "linear":
        return linear_curve_spec(),dict(mode="explicit linear-only baseline; H-f/g comparison skipped"),[],[],[]
    data,n_train,meta = curve_dataset(frame,start,scaling,args)
    jobs = [dict(repo=str(root),args=args,training=data.iloc[:n_train],segments=n) for n in (2,4)]
    fitted = []
    fit_clock = perf_counter()
    # Whole curve fits are independent; no nested pool inside a candidate replay.
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(2,args.workers),mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(fit_curve_family,j) for j in jobs]
            for future in futures:
                result = future.result()
                fitted.append(result)
                phase_progress(fit_clock, len(fitted), len(jobs),
                    f"H-f/g: {result['family']} shared fit; {result['distinct_evaluations']} complete replays; converged={result['fit_converged']}")
    else:
        for job in jobs:
            fitted.append(fit_curve_family(job))
            phase_progress(fit_clock, len(fitted), len(jobs), "H-f/g shared curve fits")
    specs = [linear_curve_spec()]+[f["curve"] for f in fitted]
    validation_clock = perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(3,args.workers),mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(validate_curve_job,dict(repo=str(root),curve=s,data=data,
                          n_train=n_train,args=args)) for s in specs]
            validation_results = []
            for future in futures:
                validation_results.append(future.result())
                phase_progress(validation_clock, len(validation_results), len(specs),
                               "H-f/g retained curve checks")
    else:
        validation_results = []
        for spec in specs:
            validation_results.append(validate_curve(spec,data,n_train,args))
            phase_progress(validation_clock, len(validation_results), len(specs),
                           "H-f/g retained curve checks")
    rows,coverages = [],{}
    for spec,(row,coverage) in zip(specs,validation_results):
        rows.append(row)
        coverages[spec["family"]] = coverage
        print(f"H-f/g: {spec['family']} selection mean predictive log score={row['mean_selection_predictive_log_score']:.6f}; open RMSE={row['selection_open_rmse']:.6f}",flush=True)
    # Fixed pre-holdout rule: highest fine sequential selection score; within
    # 1e-8 per observation choose fewer shape parameters, never test-week RMSE.
    best_score = max(r["mean_selection_predictive_log_score"] for r in rows)
    eligible = [r for r in rows if r["mean_selection_predictive_log_score"]>=best_score-1e-8]
    chosen_row = min(eligible,key=lambda r:r["free_shape_parameters"])
    chosen = next(s for s in specs if s["family"]==chosen_row["family"])
    meta.update(mode="shared TRAIN curve fits; chronological MODEL_SELECTION; curves frozen before target week",
        selected_family=chosen["family"],selection_rule="largest fine sequential predictive Y log score; within 1e-8 prefer fewer parameters",
        candidates=specs,regularization=args.curve_regularization,logit_bounds=[-4,4],
        budget_per_start=args.curve_evaluations,starts_per_flexible_family=2,
        fitted=[{k:v for k,v in f.items() if k!="trials"} for f in fitted],
        coverage=coverages,weak_interval_threshold=0.05,
        coarse_numerics=dict(cells=args.curve_cells,max_step=args.curve_max_step,q=3),
        checked_numerics=dict(cells=args.cells,max_step=args.max_step,q=3),
        curve_uncertainty_in_forecast=False,joint_curve_parameter_recovery_claim=False,
        training_refit_on_model_selection=False,completed_H_fg_claim=False,
        identification="conditional fit; interval visitation is not proof of shape identification")
    meta["weak_intervals"] = {name:{key:[i for i,v in enumerate(values) if v<.05]
                                   for key,values in coverage.items()} for name,coverage in coverages.items()}
    trials = [row for f in fitted for row in f["trials"]]
    return chosen,meta,rows,trials,specs


def conditional_fit(job):
    repository(job["repo"])
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_cached
    from src.method.bm_1_estimation import run_fv_filter
    from src.params.bm_1 import BM1ObservationSeries
    from src.control.bm_1 import beta_initial_cell_mass
    t = np.arange(1, 168, dtype=float)/24
    b, u, y = job["B"], job["U"], job["I"]
    model = bm_1(hold(b), hold(b), (0.0, t[-1]), 2)
    model.fokker_planc_reference = finite_volume_1d_cached()
    f,g = curve_pair(job.get("curve"))
    theta = [hold(u), f, g, 1/job["rho"], job["k"], job["Xi"]]
    edges = np.linspace(0, 1, job["cells"]+1)
    observations = BM1ObservationSeries(t, y[1:], "day", np.isfinite(y[1:]), "point", "right")
    fit = run_fv_filter(model, theta, job["sigma_x"], job["sigma_y"],
                        beta_initial_cell_mass(edges), 0.0, observations,
                        breakpoints=t[1:-1], max_step=job["max_step"],
                        likelihood_quadrature_order=3)
    # Propagate the last 23:00 posterior to midnight without touching the
    # first held-out observation. Price/baseline stay on the last historic bin.
    tail = bm_1(hold(b[-1:]), hold(b[-1:]), (0.0, 1/24), 2)
    tail.fokker_planc_reference = finite_volume_1d_cached()
    theta[0] = hold(u[-1:])
    state = tail.fokker_planc_reference_solver(fit.posterior_mass[-1], theta,
                job["sigma_x"], t_eval=[0.0, 1/24], max_step=job["max_step"])
    mass = state[3][-1]
    return dict(id=job["id"], rho=job["rho"], Xi=job["Xi"], mass=mass,
                scores=fit.log_evidence,
                mass_error=max(float(fit.maximum_abs_mass_error), float(np.max(np.abs(state[3].sum(axis=1)-1)))),
                minimum_mass=min(float(fit.minimum_cell_mass), float(state[3].min())))


def estimate(selected, root, args, *, curve=None):
    clock = perf_counter()
    rho = np.linspace(args.rho_low, args.rho_high, args.nodes)
    xi = np.linspace(args.xi_low, args.xi_high, args.nodes)
    jobs = []
    for r in rho:
        for x in xi:
            jobs.append(dict(repo=str(root), id=len(jobs), rho=float(r), Xi=float(x),
                 B=selected["B"].to_numpy()[:168], U=selected["U_bm"].to_numpy()[:168],
                 I=selected["I"].to_numpy()[:168], k=args.k, sigma_x=args.sigma_x,
                 sigma_y=args.sigma_y, cells=args.cells, max_step=args.max_step,curve=curve))
    results = [None]*len(jobs)
    workers = min(args.workers, len(jobs))
    if workers == 1:
        for j in jobs:
            results[j["id"]] = conditional_fit(j)
            phase_progress(clock, j['id']+1, len(jobs),
                           f"F4 training: {j['id']+1}/{len(jobs)} hypotheses")
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(conditional_fit, j) for j in jobs]
            for n, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results[result["id"]] = result
                phase_progress(clock, n, len(jobs),
                               f"F4 training: {n}/{len(jobs)} hypotheses")
    scores = np.stack([r["scores"] for r in results])
    # Missing observations supply no likelihood factor, not an artificial zero Y.
    observed = np.isfinite(jobs[0]["I"][1:])
    logs = np.nansum(scores, axis=1) - np.log(len(jobs))
    logs -= logsumexp(logs)
    weights = np.exp(logs)
    mass = np.stack([r["mass"] for r in results])
    if np.max(np.abs(mass.sum(axis=1)-1)) > 1e-10 or mass.min() < -1e-14:
        raise FloatingPointError("Origin probability law failed conservation/positivity.")
    posterior = pd.DataFrame({"rho": [j["rho"] for j in jobs],
                              "Xi": [j["Xi"] for j in jobs], "weight": weights})
    return posterior, mass, scores, dict(observed_hours=int(observed.sum()),
         missing_hours=int((~observed).sum()), excluded_initial_hour=True,
         initial_law="Beta(20,20)", prior="uniform discrete joint node masses",
         maximum_mass_error=max(r["mass_error"] for r in results),
         minimum_cell_mass=min(r["minimum_mass"] for r in results))


def bm_forecast(posterior, masses, baseline, prices, args, *, curve=None):
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_cached
    times = np.arange(169, dtype=float)/24
    model = bm_1(hold(baseline), hold(baseline), (0.0, 7.0), 2)
    model.fokker_planc_reference = finite_volume_1d_cached()
    f,g = curve_pair(curve)
    demands, states, laws = [], [], []
    for j, row in posterior.iterrows():
        theta = [hold(prices), f, g, 1/row.rho, args.k, row.Xi]
        result = model.fokker_planc_reference_solver(masses[j], theta, args.sigma_x,
                     t_eval=times, breakpoints=times[1:-1], max_step=args.max_step)
        if np.max(np.abs(result[3].sum(axis=1)-1)) > 1e-10 or result[3].min() < -1e-14:
            raise FloatingPointError("Forecast probability law failed; no repair is applied.")
        laws.append(result[3][:-1])
        demands.append(np.stack([model.eval(t, result[1], theta[0], f, g,
                                             args.k, row.Xi)[-1] for t in times[:-1]]))
        states.append(result[1])
    # Every parameter hypothesis retains its own conditional state law.
    d = np.stack(demands)
    p = np.stack(laws) * posterior.weight.to_numpy()[:, None, None]
    mean = np.sum(p*d, axis=(0, 2))
    xmean = np.sum(p*np.stack(states)[:, None, :], axis=(0, 2))
    lower, upper = [], []
    for h in range(168):
        values, weights = d[:, h, :].ravel(), p[:, h, :].ravel()
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(weights[order])
        lower.append(float(values[order[min(np.searchsorted(cumulative, 0.05), len(order)-1)]]))
        upper.append(float(values[order[min(np.searchsorted(cumulative, 0.95), len(order)-1)]]))
    return dict(I_bm=mean, X_bm=xmean, D_p05=lower, D_p95=upper), (d, p)


def optimize_tariff(posterior, masses, baseline, observed, args, *, curve=None):
    clock = perf_counter()
    from src.control.bm_1 import BM1Parameters, BM1ControlProblem, BM1ControlNumerics
    from src.params.bm_1 import conservative_mass_transfer
    # Explicit plug-in approximation allowed by I; forecast above keeps conditionals.
    rho = float(posterior.rho @ posterior.weight)
    xi = float(posterior.Xi @ posterior.weight)
    marginal = posterior.weight.to_numpy() @ masses
    edges = np.linspace(0, 1, args.cells+1)
    f,g = curve_pair(curve)
    def initial_mass(new_edges):
        return conservative_mass_transfer(edges, marginal, new_edges)
    problem = BM1ControlProblem(baseline=hold(baseline), target=hold(baseline),
       horizon=(0.0, 7.0), price_edges=np.arange(169)/24,
       parameters=BM1Parameters(f, g, 1/rho, args.k, xi, args.sigma_x),
       initial_mass_builder=initial_mass, breakpoints=np.arange(1,168)/24,
       observed_prices=observed)
    numerics = BM1ControlNumerics(x_steps=args.cells, objective_max_step=1/192,
                                 solver_max_step=1/192)
    evaluator = problem.build_evaluator(numerics)
    night = (np.arange(168)%24 < 7) | (np.arange(168)%24 >= 22)
    def policy(values):
        return np.where(night, values[0], values[1])
    cache = {}
    def cost(values):
        key = tuple(np.asarray(values, float))
        if key not in cache:
            cache[key] = float(evaluator(policy(values)))
        return cache[key]
    candidates = [dict(values=np.array([0.5,0.5]), loss=cost([0.5,0.5]), success=True, status="neutral feasible candidate")]
    for start_number, initial in enumerate(([0.3,0.7], [0.7,0.3]), 1):
        fit = minimize(cost, initial, method="Powell", bounds=[(0,1),(0,1)],
                       options=dict(maxiter=8, maxfev=args.control_evaluations, xtol=1e-3, ftol=1e-5))
        candidates.append(dict(values=fit.x, loss=float(fit.fun), success=bool(fit.success), status=str(fit.message)))
        phase_progress(clock, start_number, 2,
                       f"Control: {len(cache)} distinct tariff evaluations; {fit.message}")
    best = min(candidates, key=lambda c:c["loss"])
    prices = policy(best["values"])
    coarse = evaluator.evaluate(prices)
    fine_numerics = BM1ControlNumerics(x_steps=args.cells, objective_max_step=1/384, solver_max_step=1/384)
    fine = problem.build_evaluator(fine_numerics).evaluate(prices)
    return prices, dict(method="plug-in; two recurring day/night tariff variables; best feasible tested candidate",
          curve_family=(curve or linear_curve_spec())["family"],curves_refitted=False,
          rho=rho, Xi=xi, target="frozen five-previous-weekday baseline",
          prices_night_day=best["values"].tolist(), solver_success=best["success"],
          solver_status=best["status"], distinct_evaluations=len(cache),
          total_cost=float(coarse.total), refined_total_cost=float(fine.total),
          absolute_time_refinement=float(abs(coarse.total-fine.total)),
          observed_price_cost=float(evaluator.evaluate(observed).total),
          neutral_price_cost=float(candidates[0]["loss"]),
          convergence_accepted=False, counterfactual_not_scored_against_observed_I=True)


def save_plot(hourly, directory, start):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(hourly.index, hourly.I, label="Observed hourly input I", color="black", lw=1)
    axes[0].scatter(hourly.index, hourly.I, s=10, color="black")
    axes[0].plot(hourly.index, hourly.I_bm, label="BM preview predicted mean demand D")
    axes[0].fill_between(hourly.index, hourly.D_p05, hourly.D_p95, alpha=.17,
                         label="BM marginal 90% latent-D interval")
    axes[0].plot(hourly.index, hourly.B, label="Five previous weekday/hour mean B", ls="--")
    axes[0].set_ylabel("Normalized input / demand")
    axes[1].step(hourly.index, hourly.U_bm, where="post", label="Known historical DK1 price U [0,1]")
    axes[1].set_ylabel("Normalized price")
    axes[2].plot(hourly.index, hourly.X_bm, label="Predicted BM mean battery state X [0,1]")
    axes[2].set_ylabel("Battery state X")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Physical time (Europe/Copenhagen)")
    for axis in axes:
        axis.legend(fontsize=9, loc="best")
        axis.grid(alpha=.25)
    fig.suptitle(f"BM F4 preview | fixed origin {start:%Y-%m-%d} | known historical prices")
    fig.tight_layout()
    fig.savefig(directory/"bm_weekly_prediction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_control_plot(hourly, controlled, directory):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(hourly.index, hourly.U_bm, label="Observed historical DK1 price")
    axes[0].step(controlled.index, controlled.U_optimized, where="post", label="Selected day/night tariff")
    axes[0].set_ylabel("Normalized price U [0,1]")
    axes[1].plot(hourly.index, hourly.B, label="History-only target B", ls="--")
    axes[1].plot(hourly.index, hourly.I_bm, label="Bank mean D at observed price")
    axes[1].plot(controlled.index, controlled.I_bm, label="Bank mean D at selected tariff")
    axes[1].set_ylabel("Model demand / target")
    axes[2].plot(hourly.index, hourly.X_bm, label="Bank mean X at observed price")
    axes[2].plot(controlled.index, controlled.X_bm, label="Bank mean X at selected tariff")
    axes[2].set_ylabel("Battery state X")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Physical time (Europe/Copenhagen)")
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(alpha=.25)
    fig.suptitle("F4 preview control | restricted plug-in optimization | conditional-bank forward predictions")
    fig.tight_layout()
    fig.savefig(directory/"control.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_curves(selected, metadata, rows, trials, specs, directory):
    (directory/"response_curves.json").write_text(json.dumps(dict(selected=selected,
                         metadata=metadata),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    pd.DataFrame(rows).to_csv(directory/"curve_validation.csv",index=False)
    pd.DataFrame([{**r,"logits":json.dumps(r["logits"])} for r in trials]).to_csv(directory/"curve_trials.csv",index=False)
    records = []
    for spec in specs or [selected]:
        for z,f,g in zip(spec["knots"],spec["f"],spec["g"]):
            records.append(dict(family=spec["family"],argument=z,f=f,g=g,
                                selected=spec["family"]==selected["family"]))
    pd.DataFrame(records).to_csv(directory/"response_curves.csv",index=False)
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    z = np.linspace(0,1,401)
    for spec in specs or [selected]:
        f,g = curve_pair(spec)
        label = spec["family"]+(" (selected)" if spec["family"]==selected["family"] else "")
        for axis,curve in zip(axes,[f,g]):
            axis.plot(z,curve(z),label=label,ls="--" if spec["family"]=="linear" else "-")
            axis.scatter(spec["knots"],curve(spec["knots"]),s=20)
    for axis,name in zip(axes,["f(x): state response","g(u): price response"]):
        axis.set_title(name)
        axis.set_xlabel("x" if name.startswith("f") else "normalized u")
        axis.set_xlim(0,1)
        axis.set_ylim(-1.05,1.05)
        axis.grid(alpha=.25)
        axis.legend(fontsize=9)
    axes[0].set_ylabel("Shared response")
    fig.suptitle("H-f/g preview | pooled TRAIN fit | frozen before forecast week")
    fig.tight_layout()
    fig.savefig(directory/"response_curves.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--week", default="2022-02-07", help="Exact forecast start, not silently snapped to Monday.")
    parser.add_argument("--tag", default="MED", help="Same default population as WM4; empty string selects all meters.")
    parser.add_argument("--observation-contract", choices=["point", "hourly-point-approximation"])
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--power-scale", type=float)
    parser.add_argument("--price-low", type=float)
    parser.add_argument("--price-high", type=float)
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--rho-low", type=float, default=0.5)
    parser.add_argument("--rho-high", type=float, default=1.5)
    parser.add_argument("--xi-low", type=float, default=0.0)
    parser.add_argument("--xi-high", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--sigma-x", type=float, default=0.5)
    parser.add_argument("--sigma-y", type=float, default=0.03)
    parser.add_argument("--curve-mode", choices=["auto","linear"], default="auto",help="Default compares pooled linear/2-piece/4-piece f/g before target week.")
    parser.add_argument("--curve-train-start", default="2022-01-01")
    parser.add_argument("--curve-rho",type=float,default=1.0,help="Declared conditional rho during shape fitting, in day^-1.")
    parser.add_argument("--curve-xi",type=float,default=.75,help="Declared conditional Xi during shape fitting; final bank is refitted afterwards.")
    parser.add_argument("--curve-evaluations",type=int,default=24,help="Maximum optimizer function calls per start; two starts per flexible family.")
    parser.add_argument("--curve-cells",type=int,default=101,help="Coarse, explicitly provisional curve-search FV grid.")
    parser.add_argument("--curve-max-step",type=float,default=1/192,help="Coarse curve-search step in days; retained curves are checked on --cells/--max-step.")
    parser.add_argument("--curve-regularization",type=float,default=.01,help="Fixed mean knot-deviation penalty toward 1-2z.")
    parser.add_argument("--cells", type=int, default=401)
    parser.add_argument("--max-step", type=float, default=1/1536, help="Days; provisional numerical setting.")
    parser.add_argument("--workers", type=int, default=min(7, max(1, (os.cpu_count() or 2)-1)))
    parser.add_argument("--skip-optimize", action="store_true")
    parser.add_argument("--control-evaluations", type=int, default=40, help="Budget per Powell start, no success invented when exhausted.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if (args.nodes < 2 or args.cells < 3 or args.workers < 1 or
        args.control_evaluations < 1 or args.max_step <= 0 or not np.isfinite(args.max_step) or
        not 0 < args.rho_low < args.rho_high or not 0 <= args.xi_low < args.xi_high <= 1 or
        args.sigma_x < 0 or args.sigma_y <= 0 or args.k <= 0 or
        not np.all(np.isfinite([args.rho_low,args.rho_high,args.xi_low,args.xi_high,args.k,args.sigma_x,args.sigma_y]))):
        parser.error("Invalid bank, noise, grid or worker settings.")
    if (args.curve_evaluations < 1 or args.curve_cells < 3 or args.curve_max_step <= 0 or
        args.curve_rho <= 0 or not 0 <= args.curve_xi <= 1 or args.curve_regularization < 0 or
        not np.all(np.isfinite([args.curve_rho,args.curve_xi,args.curve_max_step,args.curve_regularization]))):
        parser.error("Invalid curve-search budget, numerical or conditional parameter settings.")
    if args.curve_mode == "auto" and (args.curve_cells>args.cells or args.curve_max_step<args.max_step):
        parser.error("Curve search must be no finer than its reference: curve-cells <= cells and curve-max-step >= max-step.")
    root = repository(args.repo)
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("Output directory already exists; choose a fresh directory. Existing results are retained.")
    start = pd.Timestamp(args.week).normalize()
    if start.tzinfo is not None:
        parser.error("Provide a local calendar date without timezone.")
    end = start + pd.Timedelta(days=7)
    frame, sources = load_raw(root, args.tag, end)
    selected, scaling, baseline_dates = prepare(frame, start, args)
    print(f"Target {start} -> {end}; fixed forecast origin; {len(sources['measurement_columns'])} {args.tag} meters", flush=True)
    print(f"Frozen power scale={scaling['power_scale']:.8g}; price bounds=[{scaling['price_low']:.8g}, {scaling['price_high']:.8g}]", flush=True)
    print(f"Training missing hours={selected.I.iloc[:168].isna().sum()}; target missing hours={selected.I.iloc[168:].isna().sum()}", flush=True)
    if args.audit_only:
        print(json.dumps(dict(sources=sources, scaling=scaling, baseline_first_hour_sources=baseline_dates[0]), indent=2))
        return 0
    if not args.observation_contract:
        parser.error("Declare --observation-contract point only for established point-power readings, or hourly-point-approximation for an explicitly provisional output preview. Interval likelihoods are not implemented.")
    clock = perf_counter()
    curve_clock = perf_counter()
    curve,curve_metadata,curve_rows,curve_trials,curve_candidates = learn_curves(frame,start,scaling,root,args)
    curve_metadata.update(schema_version="f4-preview-response-curves-v1",normalization=scaling,
                          population=sources,source=git_info(root),frozen_before=start.isoformat(),
                          observation_contract=args.observation_contract,
                          fixed_noise=dict(k=args.k,sigma_x=args.sigma_x,sigma_y=args.sigma_y))
    curve_seconds = perf_counter()-curve_clock
    bm_clock = perf_counter()
    posterior, masses, scores, fit_metadata = estimate(selected, root, args,curve=curve)
    estimation_seconds = perf_counter()-bm_clock
    forecast_clock = perf_counter()
    actual = selected.iloc[168:].copy()
    bm, (d,p) = bm_forecast(posterior, masses, actual.B.to_numpy(), actual.U_bm.to_numpy(), args,curve=curve)
    forecast_seconds = perf_counter()-forecast_clock
    hourly = actual.assign(**bm)
    hourly.index.name = "timestamp"
    valid = hourly.I.notna().to_numpy()
    if not valid.any():
        raise ValueError("No held-out I observations; a comparison cannot be scored.")
    # Predictive Gaussian Y density at real held-out observations, with frozen
    # weights/laws; these data never enter an update or candidate policy.
    residual = (hourly.I.to_numpy()[None,:,None]-d)/args.sigma_y
    log_density = -.5*residual*residual-np.log(args.sigma_y)-.5*np.log(2*np.pi)
    with np.errstate(divide="ignore"):
        hourly["bm_Y_log_score"] = logsumexp(np.log(p)+log_density, axis=(0,2))
    names = dict(BM="I_bm", B="B")
    metrics = {n: metric(hourly.I, hourly[c]) for n,c in names.items()}
    raw_metrics = {n: metric(hourly.I_raw, hourly[c]*scaling["power_scale"]) for n,c in names.items()}
    daily = pd.DataFrame([dict(day=str(day.date()), model=n, **metric(group.I,group[c]))
                          for day,group in hourly.groupby(hourly.index.normalize())
                          for n,c in names.items()])
    control_clock = perf_counter()
    control_metadata = None
    control_hourly = None
    if not args.skip_optimize:
        optimal, control_metadata = optimize_tariff(posterior, masses, actual.B.to_numpy(), actual.U_bm.to_numpy(), args,curve=curve)
        optimized_forecast, _ = bm_forecast(posterior, masses, actual.B.to_numpy(), optimal, args,curve=curve)
        control_hourly = pd.DataFrame(optimized_forecast, index=hourly.index)
        control_hourly["U_optimized"] = optimal
        control_hourly["target_B"] = hourly.B
    control_seconds = perf_counter()-control_clock
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.output_dir or root/f"files/output/bm_1/runs/f4_bm_preview/{start:%Y-%m-%d}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    posterior.to_csv(directory/"parameter_posterior.csv", index=False)
    np.savez_compressed(directory/"origin_bank.npz", conditional_mass=masses,
                        conditional_log_evidence=scores, posterior_weights=posterior.weight.to_numpy(),
                        parameter_nodes=posterior[["rho","Xi"]].to_numpy(), x_edges=np.linspace(0,1,args.cells+1))
    hourly.to_csv(directory/"hourly_results.csv")
    daily.to_csv(directory/"daily_results.csv", index=False)
    if control_hourly is not None:
        control_hourly.to_csv(directory/"control_hourly.csv")
    summary = dict(status="provisional real-data H/I/J preview; completed F4 NOT claimed",
        origin=start.isoformat(), target_end_exclusive=end.isoformat(), timezone=TZ,
        information="fixed origin; no target I assimilation; known historical oracle DK1 prices",
        models_executed=["BM1"],
        observation_contract=args.observation_contract,
        normalization=scaling, data=sources, source=git_info(root), script_sha256=sha256(__file__),
        core_source_hashes={p:sha256(root/p) for p in (
            "src/model/bm_1.py", "src/method/bm_1_estimation.py", "src/solver/finite_volume.py",
            "src/control/bm_1.py", "src/exe/run_model/progress.py")},
        environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
        settings={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        baseline_source_times=baseline_dates, fit=fit_metadata,
        fixed_parameters=dict(k=args.k, sigma_x=args.sigma_x, sigma_y=args.sigma_y),
        response_curves=curve,curve_estimation=curve_metadata,
        estimated_means=dict(rho=float(posterior.rho@posterior.weight), Xi=float(posterior.Xi@posterior.weight)),
        posterior_boundary_weight=float(posterior.loc[(posterior.rho.eq(args.rho_low)|posterior.rho.eq(args.rho_high)|posterior.Xi.eq(args.xi_low)|posterior.Xi.eq(args.xi_high)),"weight"].sum()),
        metrics=metrics, raw_unit_metrics=raw_metrics,
        mean_predictive_Y_log_score=float(hourly.bm_Y_log_score.loc[valid].mean()),
        uncertainty="marginal latent D intervals conditional on frozen selected f/g; no curve uncertainty, measured-Y intervals or simultaneous week coverage",
        control=control_metadata,
        seconds=dict(curve_fit_and_selection=curve_seconds, estimation=estimation_seconds, open_forecast=forecast_seconds,
                     control=control_seconds, elapsed_before_save=perf_counter()-clock),
        limitations=["coarse discrete parameter prior; no accepted parameter-grid/FV convergence",
                     "shared f/g conditional fit with finite search budget; shape uncertainty omitted",
                     "fixed noise; no mixture or noise-bank comparison",
                     "metadata/point versus interval observation contract not established by workbook importer",
                     "historical DK1 spot prices are a proxy for the unknown price paid by these meters",
                     "no causal demand-response claim; no population-calibration claim"])
    (directory/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save_plot(hourly, directory, start)
    save_curves(curve,curve_metadata,curve_rows,curve_trials,curve_candidates,directory)
    if control_hourly is not None:
        save_control_plot(hourly, control_hourly, directory)
    print(f"Fitting week={start-pd.Timedelta(days=7)} -> {start}")
    print(f"Prediction week={start} -> {end}; no new I measurements")
    print(f"BM posterior means: rho={summary['estimated_means']['rho']:.8f}, Xi={summary['estimated_means']['Xi']:.8f}")
    print(f"BM fixed assumptions: k={args.k:.8f}, sigma_x={args.sigma_x:.8f}, sigma_y={args.sigma_y:.8f}")
    print(f"Shared f/g: selected {curve['family']}; frozen across all forecast hours and control candidates")
    print(f"Curve knots={curve['knots']}; f={curve['f']}; g={curve['g']}")
    print(f"\n{'':<8}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    for name, item in metrics.items():
        print(f"{name:<8}{item['mse']:>14.10f}{item['rmse']:>14.10f}{item['mae']:>14.10f}{item['bias']:>14.10f}")
    print(f"Output={directory.resolve()}")
    print("PROVISIONAL: real data; point/interval contract and full F4 validation remain separate.")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

"""Exploratory price-signal identifiability diagnostics for the WM4 N8/H7 study.

Uses SAVED historical-only WM4 parameter fits; no parameter refitting.
The relevant price for identifying theta_U is the eight-day TRAINING price
history, not the seven target-day prices. Target prices are shown separately
to describe realized forcing and forecast-error conditions.

Association is descriptive, NOT causal evidence of a consumer price effect.
Overlapping eight-week training sets create dependence; moving-block bootstrap
SEs and Benjamini-Hochberg adjustments are exploratory approximations.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from src.exe.run_model.wm_4.analyze_oracle_8weekday import (
    HORIZON, N_WEEKDAYS, OUTPUT as ORACLE_OUTPUT, check_tables, save_figure,
)
from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.tools.impute_hourly import impute_hourly_days


PRICE_FEATURES = (
    "historical_price_mean", "historical_price_mean_abs",
    "historical_price_std", "historical_within_day_std",
    "historical_within_day_abs_change",
)
RESPONSE_OUTCOMES = ("price_active", "price_coefficient_magnitude")


def historical_price_features(frame, days):
    """Exactly eight corresponding historical days, with within-day U imputation.

    Price changes at midnight between separate training days are excluded
    from volatility calculations to avoid artificial adjacent-week jumps.
    """
    dates = pd.DatetimeIndex(days).normalize()
    if len(dates) != N_WEEKDAYS or dates.has_duplicates:
        raise ValueError("Expected eight distinct corresponding historical weekdays.")
    filled, imputation = impute_hourly_days(frame, days=dates, columns=("U",), require_days=True)
    profiles = []
    for day in dates:
        values = filled.reindex(pd.date_range(day, periods=24, freq="h"))["U"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0+1e-8):
            raise ValueError(f"Historical normalized price is missing or outside [-1,1] on {day:%Y-%m-%d}.")
        profiles.append(values)
    matrix = np.vstack(profiles)
    prices = matrix.ravel()
    deltas = np.diff(matrix, axis=1)
    return {
        "historical_price_mean": float(np.mean(prices)),
        "historical_price_mean_abs": float(np.mean(np.abs(prices))),
        "historical_price_std": float(np.std(prices)),
        "historical_price_range": float(np.ptp(prices)),
        "historical_price_rms": float(np.sqrt(np.mean(prices**2))),
        "historical_within_day_std": float(np.mean(np.std(matrix, axis=1))),
        "historical_within_day_range": float(np.mean(np.ptp(matrix, axis=1))),
        "historical_within_day_abs_change": float(np.mean(np.abs(deltas))),
        "historical_within_day_nonzero_change_fraction": float(np.mean(np.abs(deltas)>1e-10)),
        "historical_price_imputed_hours": int(imputation["imputed_by_column"]["U"]),
    }


def price_summary(daily, hourly, frame, threshold=1e-6):
    """One row per fitting day. Never treat target price as a training input."""
    daily, hourly = check_tables(daily, hourly)
    if "training_days" not in daily:
        raise ValueError("Saved daily_metrics.csv must include the historical 'training_days' column.")
    hgroup = hourly.groupby("day")
    records = []
    for row in daily.to_dict("records"):
        day = pd.Timestamp(row["day"]).normalize()
        history = pd.to_datetime([part for part in str(row["training_days"]).split(";") if part])
        if len(history) != N_WEEKDAYS or any(past >= day for past in history):
            raise ValueError(f"{day:%Y-%m-%d} has invalid historical training dates.")
        if any(past.weekday() != day.weekday() for past in history):
            raise ValueError(f"{day:%Y-%m-%d} does not use corresponding historical weekdays.")
        historical = historical_price_features(frame, history)
        current = hgroup.get_group(row["day"]).sort_values("hour")
        forecast = current["U"].to_numpy(dtype=float)
        if len(forecast) != HORIZON or not np.isfinite(forecast).all():
            raise ValueError(f"{day:%Y-%m-%d} lacks seven valid target-hour prices.")
        theta = float(row["theta_U"])
        records.append({
            "day": day, "weekday": day.day_name(), "rmse": float(row["rmse"]),
            "baseline_rmse": float(np.sqrt(float(row["baseline_mse"]))),
            "theta_U": theta,
            "price_active": int(theta < -threshold),
            "price_boundary": int(abs(theta) <= threshold),
            "price_positive_above_tolerance": int(theta > threshold),
            "price_coefficient_magnitude": float(max(-theta, 0.0)),
            "price_forcing_rms": float(row["price_rms"]),
            "price_forcing_mean": float(row["price_mean"]),
            "forecast_price_mean": float(np.mean(forecast)),
            "forecast_price_mean_abs": float(np.mean(np.abs(forecast))),
            "forecast_price_std": float(np.std(forecast)),
            "forecast_price_range": float(np.ptp(forecast)),
            "forecast_price_mean_abs_change": float(np.mean(np.abs(np.diff(forecast)))),
            **historical,
        })
    table = pd.DataFrame(records).sort_values("day").reset_index(drop=True)
    if table.empty: raise ValueError("No eligible daily records for price analysis.")
    return table


def block_bootstrap_spearman(x, y, *, block_size=7, draws=1000, seed=271828):
    """Spearman rank correlation with calendar-ordered moving-block bootstrap.

    Bootstrap samples aligned (x,y) row pairs. Standard error and normal
    p value are rough serial-dependence corrections, not exact inference.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x)&np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    outcome = {"days": n, "spearman_rho": np.nan,
               "block_size": int(block_size), "bootstrap_draws_used": 0,
               "bootstrap_se": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan,
               "p_value_block_approx": np.nan}
    if n < max(10, block_size+1) or np.unique(x).size < 2 or np.unique(y).size < 2:
        return outcome
    rho = float(spearmanr(x, y).statistic)
    outcome["spearman_rho"] = rho
    rng = np.random.default_rng(int(seed))
    boot = []
    nb = int(np.ceil(n/block_size))
    for _ in range(int(draws)):
        starts = rng.integers(0, n, size=nb)
        indices = np.concatenate([(start+np.arange(block_size)) % n for start in starts])[:n]
        a, b = x[indices], y[indices]
        if np.unique(a).size > 1 and np.unique(b).size > 1:
            boot.append(float(spearmanr(a,b).statistic))
    if len(boot) < max(30, draws//2): return outcome
    boot = np.asarray(boot)
    se = float(np.std(boot, ddof=1))
    outcome.update({"bootstrap_draws_used": len(boot), "bootstrap_se": se,
                    "ci95_lower": float(np.quantile(boot, .025)),
                    "ci95_upper": float(np.quantile(boot, .975)),
                    "p_value_block_approx": float(2*norm.sf(abs(rho/se))) if se > 1e-12 else np.nan})
    return outcome


def benjamini_hochberg(pvalues):
    values = np.asarray(pvalues, dtype=float)
    q = np.full_like(values, np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    if indices.size == 0: return q
    ordered = indices[np.argsort(values[indices])]
    raw = values[ordered]*len(ordered)/np.arange(1,len(ordered)+1)
    q[ordered] = np.minimum.accumulate(raw[::-1])[::-1].clip(0,1)
    return q


def association_tests(table, *, block_size=7, draws=1000, seed=271828):
    results = []
    # Predeclared: only historical fitting-window prices, NOT forecast prices.
    for outcome in RESPONSE_OUTCOMES:
        for name in PRICE_FEATURES:
            record = {"outcome": outcome, "predictor": name, "period": "eight_historical_weekdays",
                      **block_bootstrap_spearman(table[name], table[outcome], block_size=block_size,
                                                 draws=draws, seed=seed)}
            results.append(record)
    result = pd.DataFrame(results)
    result["q_bh_across_all_tests"] = benjamini_hochberg(result["p_value_block_approx"])
    result["ci_excludes_zero"] = ((result["ci95_lower"]>0)|(result["ci95_upper"]<0))
    return result


def quartile_summary(table, feature="historical_price_std"):
    # Quantile 1 has the lowest historical price level/volatility.
    ranks = table[feature].rank(method="first")
    bins = pd.qcut(ranks, q=min(4,len(table)), labels=False, duplicates="drop")
    grouped = table.assign(price_quantile=bins+1).groupby("price_quantile")
    records = []
    for quantile, group in grouped:
        records.append({
            "price_quantile": int(quantile), "feature": feature, "days": len(group),
            "feature_mean": float(group[feature].mean()),
            "price_active_fraction": float(group["price_active"].mean()),
            "theta_U_median": float(group["theta_U"].median()),
            "coefficient_strength_mean": float(group["price_coefficient_magnitude"].mean()),
            "forecast_rmse_median": float(group["rmse"].median()),
        })
    return pd.DataFrame(records)


def error_quartiles(table):
    bins = pd.qcut(table["rmse"].rank(method="first"), q=min(4,len(table)),
                   labels=False, duplicates="drop")
    return table.assign(error_quantile=bins+1).groupby("error_quantile").agg(
        days=("day","size"), median_rmse=("rmse","median"),
        price_active_fraction=("price_active","mean"),
        coefficient_strength_mean=("price_coefficient_magnitude","mean"),
        historical_price_std_mean=("historical_price_std","mean"),
    ).reset_index()


def plot_price_analysis(table, stats, volatility, level, errors, output):
    output = Path(output)
    # 1. Coefficient magnitude versus historical price level and variation.
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    for ax, x, xlabel in zip(axes.flat,
                             ("historical_price_mean", "historical_price_std",
                              "historical_price_mean_abs", "historical_within_day_abs_change"),
                             ("Historical mean U (signed price level)",
                              "Historical U standard deviation", "Historical mean |U|",
                              "Historical within-day mean |ΔU|")):
        for active, group in table.groupby("price_active"):
            ax.scatter(group[x], group["price_coefficient_magnitude"], alpha=.7, s=24,
                       label="θU negative" if active else "θU at boundary")
        ax.set_xlabel(xlabel); ax.set_ylabel("Fitted response magnitude −θU")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Historical price levels/variation versus fitted price-response magnitude")
    save_figure(fig, output, "historical_price_vs_coefficient.png")

    # 2. Training-price variation versus activity rate.
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    for ax, group, xlabel in ((axes[0],volatility,"Historical std(U)"),
                              (axes[1],level,"Historical mean U (signed)")):
        ax.plot(group["feature_mean"],group["price_active_fraction"],marker="o")
        for row in group.itertuples(index=False):
            ax.annotate(f"Q{row.price_quantile} (n={row.days})",
                        (row.feature_mean,row.price_active_fraction),
                        xytext=(4,5),textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel("Fraction with θU < −1e−6")
        ax.set_ylim(-.05,1.05); ax.grid(alpha=.2)
    axes[2].bar(errors["error_quantile"],errors["price_active_fraction"])
    axes[2].set_xlabel("Seven-hour oracle RMSE quartile (1 = lowest error)")
    axes[2].set_ylabel("Fraction with θU < −1e−6")
    axes[2].set_ylim(0,1);axes[2].grid(alpha=.2,axis="y")
    fig.suptitle("Active price-response estimates by historical variation, level, and oracle error")
    save_figure(fig,output,"price_activation_by_quantiles.png")

    # 3. Distinguish price variation in FITTING window from target horizon.
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
    for ax,feature,label in ((axes[0],"historical_price_std","Historical 8-weekday std(U)"),
                             (axes[1],"forecast_price_std","Target 7-hour std(U)")):
        ax.plot(table["day"],table[feature],linewidth=1,label=label)
        marker=table.loc[table["price_active"]==1]
        ax.scatter(marker["day"],marker[feature],s=13,label="θU < −1e−6")
        ax.set_ylabel(label); ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel("Prediction date")
    fig.suptitle("Price volatility during fitting versus the seven-hour prediction")
    save_figure(fig,output,"historical_vs_forecast_price_volatility.png")

    # 4. Conditional coefficient/error scatter; forecast price enters as context only.
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for active,group in table.groupby("price_active"):
        label="Active price coefficient" if active else "Near-zero price coefficient"
        axes[0].scatter(group["rmse"],group["historical_price_std"],
                        label=label,s=30,alpha=.7)
        axes[1].scatter(group["forecast_price_std"],group["price_forcing_rms"],
                        label=label,s=30,alpha=.7)
    axes[0].set_xlabel("Seven-hour oracle RMSE")
    axes[0].set_ylabel("Historical 8-weekday std(U)")
    axes[1].set_xlabel("Target seven-hour std(U)")
    axes[1].set_ylabel("Price contribution RMS |θU U|")
    axes[0].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("Price identification, error, and realized forcing (right panel partly mechanical)")
    save_figure(fig,output,"price_identification_and_error.png")

    # 5. Direction and uncertainty. Exploratory association, multiple-testing adjusted.
    plot = stats.loc[stats["outcome"]=="price_active"].copy().reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(plot))
    for i,row in plot.iterrows():
        ax.plot([row["ci95_lower"],row["ci95_upper"]],[i,i],lw=2)
        ax.plot(row["spearman_rho"],i,"o")
    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(x,plot["predictor"])
    ax.set_xlabel("Spearman ρ with active price coefficient (moving-block 95% CI)")
    ax.set_title("Historical price features and price-response detection")
    ax.grid(alpha=.2,axis="x")
    save_figure(fig,output,"price_activity_association_intervals.png")


def analyze(input_dir, output_dir, group="MED", threshold=1e-6,
            block_size=7, draws=1000, seed=271828):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(input_dir/"daily_metrics.csv",parse_dates=["day"])
    hourly = pd.read_csv(input_dir/"hourly_predictions.csv",parse_dates=["day","time"])
    frame = group_frame(group)
    result = price_summary(daily,hourly,frame,threshold)
    associations = association_tests(result,block_size=block_size,draws=draws,seed=seed)
    volatility = quartile_summary(result, "historical_price_std")
    level = quartile_summary(result, "historical_price_mean")
    errors = error_quartiles(result)
    result.to_csv(output_dir/"daily_price_identification.csv",index=False)
    associations.to_csv(output_dir/"historical_price_associations.csv",index=False)
    volatility.to_csv(output_dir/"price_volatility_quartiles.csv",index=False)
    level.to_csv(output_dir/"price_level_quartiles.csv",index=False)
    errors.to_csv(output_dir/"error_quartiles_price_activity.csv",index=False)
    plot_price_analysis(result,associations,volatility,level,errors,output_dir)
    active=int(result["price_active"].sum())
    boundary=int(result["price_boundary"].sum())
    violation=int(result["price_positive_above_tolerance"].sum())
    print(f"\nWM4 price identifiability: {len(result)} fitted days, N={N_WEEKDAYS}, H={HORIZON}, group={group}")
    print(f"Negative fitted θU (<−{threshold:.1g}): {active}/{len(result)}; "
          f"boundary (|θU|≤{threshold:.1g}): {boundary}/{len(result)}; "
          f"positive beyond tolerance: {violation}/{len(result)}")
    print("\nHistorical price correlations (exploratory moving-block bootstrap):")
    print(associations[["outcome","predictor","days","spearman_rho",
                        "ci95_lower","ci95_upper","p_value_block_approx","q_bh_across_all_tests"]]
          .to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nPrice effect detection by historical price-volatility quartile:")
    print(volatility.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nPrice effect detection by historical mean normalized price quartile:")
    print(level.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nPrice effect detection by seven-hour oracle-error quartile:")
    print(errors.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nInterpretation: θU is trained ONLY on historical days; target price and oracle RMSE are context.")
    print("Neither nonzero θU nor its price correlation alone proves an observed causal demand response.")
    print("Eight-week training windows overlap; block-bootstrap CI/p-values and BH q-values are exploratory.")
    print(f"\nSaved tables and plots: {output_dir}")
    return result, associations


def main():
    parser=ArgumentParser(description="Relate historical price information to fitted WM4 price response, N8/H7.")
    parser.add_argument("--group",choices=("MED","ALL"),default="MED")
    parser.add_argument("--input",default=None,help="Existing N8/H7 oracle output folder.")
    parser.add_argument("--output",default=None,help="Price-identification results folder.")
    parser.add_argument("--threshold",type=float,default=1e-6)
    parser.add_argument("--block-size",type=int,default=7)
    parser.add_argument("--bootstrap-draws",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=271828)
    args=parser.parse_args()
    if not 0<args.threshold<0.1 or args.block_size<2 or args.bootstrap_draws<100:
        parser.error("Require 0<threshold<0.1, block-size>=2 and bootstrap-draws>=100.")
    source=Path(args.input) if args.input else ORACLE_OUTPUT/args.group.lower()/"weekly_temperature"
    dest=Path(args.output) if args.output else source/"price_identification"
    analyze(source,dest,args.group,args.threshold,args.block_size,args.bootstrap_draws,args.seed)


if __name__=="__main__": main()

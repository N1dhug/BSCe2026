"""Plot six fitted historical weekdays and their oracle-conditioned target.

The default figure uses seven rows (six historical Mondays 4 July--8 August
2022 followed by Monday 15 August 2022) and two columns (IM left, ARM right).

Each historical panel shows observed demand, a common weekly-temperature
baseline, its model-specific historical fitted mean and marginal 95% band.
IM historical means are MLE one-step-ahead predictions updated using only
past observations within that historical day; ARM historical means are
free-running joint-QP trajectories. The seventh row uses each model's
saved open-loop oracle-conditioned prediction and predictive 95% interval.

Historical results are persisted by both seven-day fitters in
historical_fitted_predictions.csv after the timed fitting procedure. This
plotter loads saved results and never refits model parameters.

Run from repository root:
    python -m src.exe.run_model.wm_4.plot_selected_bm_fit_history --show
    python -m src.exe.run_model.wm_4.plot_selected_bm_fit_history --date 2022-08-16
    python -m src.exe.run_model.wm_4.plot_selected_bm_fit_history --week 2 --date 2022-08-22
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.selected_bm_dates_arm import K, reference_week


ROOT=Path(__file__).resolve().parents[4]
IM_OUTPUT=ROOT/"files"/"output"/"bm_1"/"selected_bm_dates_beta"
ARM_OUTPUT=ROOT/"files"/"output"/"wm_4"/"selected_bm_dates_arm"
OUTPUT=ROOT/"files"/"output"/"model_comparison"/"selected_bm_dates"


def _read_csv(path,model,fields):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{model}: missing {path}. Run the updated {model} seven-day sample fitter "
            "to save the six fitted historical intervals. No fitting is performed by this plotter.")
    df=pd.read_csv(path)
    absent=set(fields)-set(df.columns)
    if absent: raise ValueError(f"{model}: {path.name} missing {sorted(absent)}")
    return df


def _numeric(df,model,source):
    columns=["observed","baseline_predicted","fitted_demand","lower_95","upper_95"]
    for col in columns:
        df[col]=pd.to_numeric(df[col],errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{model} {source}: nonfinite {col}")
    if (df["lower_95"]>df["upper_95"]).any():
        raise ValueError(f"{model} {source}: lower_95 exceeds upper_95")
    return df


def historical(path,model,day,history):
    fields=("target_day","historical_day","historical_number","hour",
            "observed","baseline_predicted","fitted_demand","lower_95","upper_95")
    df=_read_csv(path,model,fields)
    df["target_day"]=pd.to_datetime(df["target_day"],errors="raise").dt.normalize()
    df["historical_day"]=pd.to_datetime(df["historical_day"],errors="raise").dt.normalize()
    df=df.loc[df["target_day"]==day].copy()
    if len(df)!=len(history)*K:
        raise ValueError(f"{model}: expected {len(history)*K} historical hours for "
                         f"{day:%Y-%m-%d}, found {len(df)} in {path}.")
    df["hour"]=pd.to_numeric(df["hour"],errors="raise")
    df["historical_number"]=pd.to_numeric(df["historical_number"],errors="raise")
    if df.duplicated(["historical_day","hour"]).any():
        raise ValueError(f"{model}: duplicated historical date/hour entries.")
    result=[]
    for slot,past in enumerate(history,1):
        part=df.loc[df["historical_day"]==past].sort_values("hour").reset_index(drop=True)
        if len(part)!=K or not np.array_equal(part["hour"].to_numpy(),np.arange(K)):
            raise ValueError(f"{model}: missing or extra hours for {past:%Y-%m-%d}.")
        if not (part["historical_number"]==slot).all():
            raise ValueError(f"{model}: historical number order mismatched at {past}.")
        result.append(_numeric(part,model,f"training {past:%Y-%m-%d}"))
    return result


def target(path,model,day):
    fields=("day","hour","observed","lower_95","upper_95")
    df=_read_csv(path,model,fields)
    df["day"]=pd.to_datetime(df["day"],errors="raise").dt.normalize()
    if "baseline" in df:
        df=df.loc[df["baseline"]=="weekly_temperature"].copy()
    df=df.loc[df["day"]==day].copy()
    if len(df)!=K: raise ValueError(f"{model}: expected {K} forecast hours for {day}, found {len(df)}.")
    df["hour"]=pd.to_numeric(df["hour"],errors="raise")
    if not np.array_equal(df.sort_values("hour")["hour"].to_numpy(),np.arange(K)):
        raise ValueError(f"{model}: hourly target times are not 0..{K-1}.")
    mean="beta_expected_demand" if model=="IM" else "predicted"
    baseline="weekly_temperature_baseline" if model=="IM" else "baseline_predicted"
    absent={mean,baseline}-set(df.columns)
    if absent: raise ValueError(f"{model}: missing target columns {sorted(absent)}")
    df=df.sort_values("hour").reset_index(drop=True)
    df=df.rename(columns={mean:"fitted_demand",baseline:"baseline_predicted"})
    return _numeric(df,model,f"target {day:%Y-%m-%d}")


def plot(day,week=1,im_dir=IM_OUTPUT,arm_dir=ARM_OUTPUT,output=OUTPUT,show=False):
    dates,histories=reference_week(week)
    day=pd.Timestamp(day).normalize()
    if day not in dates:
        raise ValueError(f"Selected day {day:%Y-%m-%d} must belong to the seven "
                         f"sample days: {', '.join(d.strftime('%Y-%m-%d') for d in dates)}.")
    history=histories[dates.index(day)]
    if len(history)!=6: raise ValueError("Expected exactly six corresponding historical weekdays.")
    paths=[]
    for model,directory in (("IM",Path(im_dir)),("ARM",Path(arm_dir))):
        past=historical(directory/"historical_fitted_predictions.csv",model,day,history)
        forecast=target(directory/"hourly_predictions.csv",model,day)
        paths.append(past+[forecast])

    # Verify that the models are compared against the same observed demand.
    for slot in range(7):
        if not np.allclose(paths[0][slot]["observed"],paths[1][slot]["observed"],
                           rtol=1e-8,atol=1e-8):
            raise ValueError(f"IM and ARM observed demand differ in panel {slot+1}.")
        if not np.allclose(paths[0][slot]["baseline_predicted"],
                           paths[1][slot]["baseline_predicted"],rtol=1e-7,atol=1e-8):
            raise ValueError(f"IM and ARM weekly-temperature baselines differ in panel {slot+1}.")

    with plt.rc_context({"font.family":"serif",
                         "font.serif":["Times New Roman","DejaVu Serif"],
                         "font.size":14,"axes.titlesize":14,
                         "axes.labelsize":15,"xtick.labelsize":13,
                         "ytick.labelsize":13}):
        fig,axes=plt.subplots(7,2,figsize=(21,23),sharex=True,sharey=True)
        h=np.arange(K)
        # Each row is the same calendar date; IM is on the left, ARM on the right.
        # The seventh (final) row is the oracle-conditioned target prediction.
        for row in range(7):
            is_target=row==6
            stamp=history[row] if row<6 else day
            title=f"Historical {row+1}" if not is_target else "PREDICTION"
            for col,(model,color) in enumerate((("IM","tab:orange"),("ARM","tab:blue"))):
                data=paths[col][row]
                ax=axes[row,col]
                if is_target: ax.set_facecolor("#F4F5F8")
                ax.fill_between(h,data["lower_95"].to_numpy(),data["upper_95"].to_numpy(),
                                color=color,alpha=0.22 if is_target else 0.12)
                ax.plot(h,data["observed"].to_numpy(),color="black",
                        lw=1.55,marker="o",markersize=2.6)
                ax.plot(h,data["baseline_predicted"].to_numpy(),color="tab:green",
                        ls="--",lw=1.5)
                ax.plot(h,data["fitted_demand"].to_numpy(),color=color,lw=1.9)
                ax.set_title(f"{title} | {stamp:%d %b %Y} | {model}",pad=7,
                             fontweight="bold" if is_target else "normal",loc="left")
                ax.set_xlim(0,K-1)
                ax.set_xticks([0,4,8,12,16,20,23])
                ax.grid(alpha=0.2)
                if col==0: ax.set_ylabel("Normalized demand")
                if row==6: ax.set_xlabel("Hour of day")
        # Set the shared scale once, after every panel has been populated;
        # preserve 95% tails even when they extend beyond [0,1].
        all_panels=[data for group in paths for data in group]
        ymin=min(float(data["lower_95"].min()) for data in all_panels)
        ymax=max(max(float(data[col].max()) for col in
                     ("observed","baseline_predicted","fitted_demand","upper_95"))
                 for data in all_panels)
        margin=max(0.02,0.04*(ymax-ymin))
        axes[0,0].set_ylim(min(0.,ymin)-margin,ymax+margin)
        fig.legend(handles=[
            Line2D([],[],color="black",lw=1.6,marker="o",markersize=3,label="Observed demand"),
            Line2D([],[],color="tab:green",lw=1.6,ls="--",label="Weekly-temperature baseline"),
            Line2D([],[],color="tab:orange",lw=2.,label="IM fitted/predicted demand"),
            Line2D([],[],color="tab:blue",lw=2.,label="ARM fitted/predicted demand"),
            Patch(facecolor="0.5",alpha=0.2,label="Model-specific 95% predictive interval"),
        ],loc="upper center",bbox_to_anchor=(0.5,0.973),ncol=3,frameon=False,fontsize=15)
        fig.suptitle(
            f"IM vs ARM | six corresponding historical weekdays and {day:%A %d %B %Y} prediction",
            fontsize=20,y=0.994)
        fig.subplots_adjust(left=0.075,right=0.987,top=0.91,bottom=0.06,
                            wspace=0.15,hspace=0.47)
        output=Path(output)
        output.mkdir(parents=True,exist_ok=True)
        filename=f"im_arm_{day:%Y_%m_%d}_historical_6_plus_prediction_7x2"
        png=output/(filename+".png")
        pdf=output/(filename+".pdf")
        fig.savefig(png,dpi=200,bbox_inches="tight")
        fig.savefig(pdf,bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
    print(f"IM/ARM historical fits and oracle prediction:\n  {png}\n  {pdf}")
    return png,pdf


def main():
    parser=ArgumentParser(description="Compare six fitted training weekdays and one oracle prediction for IM and ARM (7x2).")
    parser.add_argument("--date",default="2022-08-15",
                        help="Target date in YYYY-MM-DD, default is Monday 15 August 2022.")
    parser.add_argument("--week",type=int,choices=(1,2),default=1)
    parser.add_argument("--im-dir",type=Path,default=None,
                        help="Folder with IM hourly_predictions.csv and historical_fitted_predictions.csv.")
    parser.add_argument("--arm-dir",type=Path,default=None,
                        help="Folder with ARM hourly_predictions.csv and historical_fitted_predictions.csv.")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--show",action="store_true")
    args=parser.parse_args()
    im_dir=args.im_dir or (IM_OUTPUT/("week_2" if args.week==2 else ""))
    arm_dir=args.arm_dir or (ARM_OUTPUT/("week_2" if args.week==2 else ""))
    plot(args.date,args.week,im_dir,arm_dir,args.output,args.show)


if __name__=="__main__": main()

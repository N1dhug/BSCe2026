"""WM4 single-day fit: historical joint QP and retrospective oracle prediction.

Fit a shared drift on N corresponding weekly-offset k-hour intervals with
independent historical F0 states. Fit the Gaussian uncertainty on historical
residuals only; then optimize the target F0/R0 using target observations.
"""
from argparse import ArgumentParser
from pathlib import Path

from src.method.moving_horizon import BASELINES
from src.method.single_day import selection
from src.exe.run_model.wm_4.plot_corresponding_interval_fit import run as fit_and_plot


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"single_day"


def run(day, origin_point=0, k=24, N=6, baseline="weekly_temperature", *,
        group="ALL", steps_per_hour=4, output=None):
    """Fit the report's ARM methodology to one k-hour interval.

    day: YYYY-MM-DD; origin_point: 0..23; k: 2..24 hourly observations;
    N: number of weekly-offset histories; baseline: registered model name.

    The historical QP and noise calibration do not see target observations.
    Target optimization is explicitly retrospective, not a causal forecast.
    """
    origin,_=selection(day,origin_point,k,N,baseline)
    output=Path(output) if output is not None else OUTPUT/group.lower()/(
        f"{origin:%Y-%m-%d_%H}h_k{k}_N{N}_{baseline}")
    output.mkdir(parents=True,exist_ok=True)
    result=fit_and_plot(day=day,hour=origin_point,k=k,N=N,B=baseline,
                        group=group,steps_per_hour=steps_per_hour,
                        output=output/"historical.png")
    print(f"[WM4] Single-day fitting output: {output}")
    return {"output":output,**result}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--day",required=True,help="Target calendar day YYYY-MM-DD.")
    parser.add_argument("--origin-point",type=int,default=0,help="Hour of day, 0..23.")
    parser.add_argument("--k",type=int,default=24,help="Interval length in hours (2..24).")
    parser.add_argument("--N",type=int,default=6,help="Number of historical weekly intervals.")
    parser.add_argument("--baseline",choices=tuple(BASELINES),default="weekly_temperature")
    parser.add_argument("--group",choices=("ALL","MED"),default="ALL")
    parser.add_argument("--steps-per-hour",type=int,default=4)
    parser.add_argument("--output",type=Path,help="Output directory (default: model/date-specific).")
    args=parser.parse_args()
    return run(args.day,args.origin_point,args.k,args.N,args.baseline,
               group=args.group,steps_per_hour=args.steps_per_hour,output=args.output)


if __name__=="__main__": main()

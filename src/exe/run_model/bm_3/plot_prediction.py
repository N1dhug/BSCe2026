"""Plot a saved BM3 distribution using BM1's demand-heatmap renderer, without refitting."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.exe.run_model.bm_1.plot_prediction import demand_heatmap
from src.exe.run_model.bm_3.predict_daily import OUTPUT


def plot_saved(directory):
    directory = Path(directory)
    hourly = pd.read_csv(directory/'hourly.csv',parse_dates=['time']).set_index('time')
    with np.load(directory/'distribution.npz') as distribution:
        edges,pdf = demand_heatmap(distribution['demand_supports'],distribution['probabilities'])
    times = np.arange(len(hourly)+1)
    fig,ax = plt.subplots(figsize=(13,6))
    heatmap = ax.pcolormesh(times,edges,pdf.T,shading='flat',cmap='Blues')
    centers = times[:-1]+.5
    ax.plot(centers,hourly.I_predicted,label='BM3 forecast',color='orange')
    ax.plot(centers,hourly.B,'--',label='Baseline',color='green')
    if 'I_observed' in hourly: ax.plot(centers,hourly.I_observed,label='Observed demand',color='black')
    ax.set(xlabel='Hours from forecast start',ylabel='Normalized demand',title='BM3 saved demand distribution')
    ax.legend(); fig.colorbar(heatmap,ax=ax,label='Display-smoothed demand density')
    fig.tight_layout(); path = directory/'demand_heatmap.png'; fig.savefig(path,dpi=160); plt.close(fig)
    return path


def main():
    p = ArgumentParser(description='Plot saved BM3 demand distribution; no new fit.')
    p.add_argument('--day',default='2022-08-22'); p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--run',type=Path,help='Exact daily result folder, including a week-run forecast folder.')
    args = p.parse_args(); print(plot_saved(args.run or args.output/args.day))


if __name__=='__main__': main()

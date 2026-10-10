"""Plot saved BM fits without importing data or rerunning parameter estimation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import numpy as np
import pandas as pd


def save_plots(run, *, day=None):
    run = Path(run)
    with (run/'fit.json').open(encoding='utf-8') as handle:
        summary = json.load(handle)
    frame = pd.read_csv(run/'trajectory.csv')
    # Saved hourly observations plus the final state boundary. Older automatic
    # baseline runs wrote fine-grid offsets; reconstruct the actual hourly axis.
    hours = summary['input_time_start']+np.arange(len(frame), dtype=float)
    origin = summary.get('data_import', {}).get('start')
    if origin is not None:
        axis = pd.DatetimeIndex(pd.Timestamp(origin)+pd.to_timedelta(hours-hours[0], unit='h'))
        groups = axis[:-1].normalize()
        xlabel = 'Date / local time'
    else:
        axis = hours
        groups = np.floor((hours[:-1]-hours[0])/24).astype(int)
        xlabel = 'Physical elapsed hours'
    hourly = frame.iloc[:-1].copy()
    observed = hourly.observed_demand.to_numpy(dtype=float)
    measured = np.isfinite(observed)
    fit_error = hourly.fitted_demand.to_numpy(dtype=float)-observed
    baseline_error = hourly.baseline.to_numpy(dtype=float)-observed
    fitted_rmse = float(np.sqrt(np.mean(fit_error[measured]**2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_error[measured]**2)))
    paths = []
    if day is None:
        keep = np.ones(len(axis), dtype=bool)
        name = 'fit_overview.png'
    else:
        if origin is None:
            raise ValueError('--day requires a historical run with calendar timestamps.')
        chosen = pd.Timestamp(day).date()
        keep = np.array([stamp.date() == chosen for stamp in axis])
        if not keep[:-1].any():
            raise ValueError('The selected day has no hourly results.')
        name = f'fit_{chosen.isoformat()}.png'
    sample_keep = keep[:-1]
    x = axis[:-1][sample_keep]
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(x, observed[sample_keep], label='Observed demand', color='black', linewidth=1, marker='.', markersize=2)
    axes[0].plot(x, hourly.fitted_demand.to_numpy()[sample_keep], label='BM fitted demand', linewidth=1)
    axes[0].plot(x, hourly.baseline.to_numpy()[sample_keep], label='Baseline B', alpha=.8, linewidth=1)
    contract = 'hourly mean' if summary['observation_contract'] == 'hourly-mean' else 'hour-start sample'
    axes[0].set_ylabel(f'Demand ({contract})\noriginal input units')
    axes[1].step(x, hourly.price.to_numpy()[sample_keep], where='post', label='Input price (DKK)' if origin else 'Input price')
    axes[1].set_ylabel('Price (input units)')
    axes[2].plot(axis[keep], frame.X.to_numpy()[keep], label='State X', color='tab:green')
    axes[2].set_ylim(-.03, 1.03)
    axes[2].set_ylabel('X [0,1]')
    axes[3].plot(x, fit_error[sample_keep], label='BM minus observed', linewidth=1)
    axes[3].plot(x, baseline_error[sample_keep], label='Baseline minus observed', alpha=.7, linewidth=1)
    axes[3].axhline(0, color='black', linewidth=.7)
    axes[3].set_ylabel('Residual (input units)')
    axes[3].set_xlabel(xlabel)
    if origin is not None:
        locator = AutoDateLocator(tz=axis.tz)
        axes[3].xaxis.set_major_locator(locator)
        axes[3].xaxis.set_major_formatter(ConciseDateFormatter(locator, tz=axis.tz))
    for ax in axes:
        ax.legend(loc='upper right')
        ax.grid(True, alpha=.25)
    p = summary['parameters']
    fig.suptitle(f"BM deterministic fit | {summary['status']} | C={p['C_hours']:.4g} h, k={p['k']:.4g}, "
                 f"Xi={p['Xi']:.4g}, X0={summary['x0']:.4g}\n"
                 f"Full-run RMSE: BM={fitted_rmse:.5g}; baseline={baseline_rmse:.5g}")
    fig.tight_layout()
    paths.append(run/name)
    fig.savefig(paths[-1], dpi=200, bbox_inches='tight')
    plt.close(fig)
    if day is None:
        if summary.get('response_fitted', False):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, name, xlabel_curve in zip(axes, ('f', 'g'), ('State X', 'Normalized price u')):
                curve = summary['response'][name]
                ax.plot(curve['knots'], curve['values'], marker='o', label=f'Fitted {name}')
                ax.plot([0, 1], [1, -1], '--', alpha=.5, label='Linear reference')
                ax.set(xlabel=xlabel_curve, ylabel=name, ylim=(-1.05, 1.05))
                ax.grid(True, alpha=.25)
                ax.legend()
            fig.tight_layout()
            paths.append(run/'fit_response.png')
            fig.savefig(paths[-1], dpi=200, bbox_inches='tight')
            plt.close(fig)
        errors = pd.DataFrame({'group': groups, 'BM': fit_error, 'Baseline': baseline_error})
        daily = errors.loc[measured].groupby('group')[['BM', 'Baseline']].agg(lambda v: np.sqrt(np.mean(v**2)))
        fig, ax = plt.subplots(figsize=(15, 4))
        for column in daily:
            ax.plot(daily.index, daily[column], marker='o', markersize=3, label=column)
        ax.set_ylabel('Daily RMSE (input units)')
        ax.set_xlabel('Date' if origin else 'Day index (0 = first day)')
        if origin is not None:
            locator = AutoDateLocator(tz=axis.tz)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(locator, tz=axis.tz))
        ax.set_title('Historical fit error: BM and baseline versus measured demand')
        ax.legend()
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        paths.append(run/'fit_daily_errors.png')
        fig.savefig(paths[-1], dpi=200, bbox_inches='tight')
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Folder containing fit.json and trajectory.csv.')
    parser.add_argument('--day', help='Optional historical date for a one-day detail plot, e.g. 2022-01-03.')
    args = parser.parse_args()
    for path in save_plots(args.run, day=args.day):
        print(path)


if __name__ == '__main__':
    main()

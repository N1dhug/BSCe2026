"""Reusable BM1 figure/raw export; no simulation, normalization or fitting.

Joint masses have shape (hypothesis, time, cell) and already include the
parameter weights. Demand must retain each hypothesis's own response values.
The exporter never replaces this joint law by demand at a mean parameter/state.
"""

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np


MODES = ('reference_simulation', 'fixed_origin_forecast', 'predictive_filter',
         'posterior_filter', 'retrospective_fit', 'counterfactual_control')


def _calendar_days(timestamps, hours):
    """Group exact slices by their declared local date, retaining elapsed time.

    ISO timestamps must agree with the numeric time grid. An offset-aware
    repeated clock hour remains two separate samples; missing hours stay absent.
    """
    try:
        stamps = [datetime.fromisoformat(stamp) for stamp in timestamps]
        elapsed = np.array([(stamp - stamps[0]).total_seconds() / 3600 for stamp in stamps])
    except (ValueError, TypeError) as exc:
        raise ValueError('timestamps must be consistent ISO datetimes.') from exc
    if not np.allclose(elapsed, hours - hours[0], rtol=0, atol=1e-8):
        raise ValueError('timestamps must align with the stored numeric time grid.')
    dates = np.array([stamp.date().isoformat() for stamp in stamps])
    days = []
    for date in dict.fromkeys(dates):
        indices = np.flatnonzero(dates == date)
        midnight = stamps[indices[0]].replace(hour=0, minute=0, second=0, microsecond=0)
        day_hours = np.array([(stamps[i] - midnight).total_seconds() / 3600 for i in indices])
        days.append((date, indices, day_hours))
    return days


def _plot_demand(axis, hours, result, moments, indices=slice(None)):
    """Use the same stored predictions, measured values and bands in all views."""
    axis.fill_between(hours, moments['p05'][indices], moments['p95'][indices],
                      color='tab:blue', alpha=.2, label='Latent D: pointwise 5–95%')
    axis.plot(hours, moments['mean'][indices], color='tab:blue', linewidth=1.8,
              label='Predicted demand E[D]')
    axis.plot(hours, result['baseline'][indices], '--', color='tab:orange', label='Baseline B')
    if np.isfinite(result['observations']).any():
        axis.plot(hours, result['observations'][indices], 'o-', color='black',
                  markersize=3, linewidth=1.3, label='Observed demand I')


def _moments(values, masses):
    mean = np.sum(values * masses, axis=1)
    variance = np.sum(masses * (values - mean[:, None]) ** 2, axis=1)
    order = np.argsort(values, axis=1, kind='stable')
    sorted_values = np.take_along_axis(values, order, axis=1)
    cdf = np.cumsum(np.take_along_axis(masses, order, axis=1), axis=1)
    quantiles = []
    for q in (.05, .95):
        indices = np.sum(cdf < q, axis=1).clip(max=values.shape[1] - 1)
        quantiles.append(sorted_values[np.arange(len(mean)), indices])
    return dict(mean=mean, std=np.sqrt(variance), p05=quantiles[0], p95=quantiles[1])


def summarize_distribution(times, x_edges, joint_mass, demand_values, baseline, price,
                           *, observations=None):
    """Check exact stored slices and summarize latent X/D, without repairs.

Two-dimensional masses/demand are the one-hypothesis convenience form.
NaN observations are missing; metrics use only observed matching time slices.
No interpolation, observation update, clipping or renormalization takes place.
"""
    times = np.asarray(times, dtype=float)
    edges = np.asarray(x_edges, dtype=float)
    mass = np.asarray(joint_mass, dtype=float)
    demand = np.asarray(demand_values, dtype=float)
    B, U = np.asarray(baseline, dtype=float), np.asarray(price, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('times must contain at least two finite increasing stored slices.')
    if (edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0) or edges[0] != 0 or edges[-1] != 1):
        raise ValueError('x_edges must be a strictly increasing [0, 1] cell grid.')
    if mass.ndim == 2:
        mass = mass[None, :, :]
    if demand.ndim == 2:
        demand = demand[None, :, :]
    expected = (len(times), len(edges) - 1)
    if mass.ndim != 3 or mass.shape[1:] != expected or demand.shape != mass.shape:
        raise ValueError('Joint masses and conditional demand must share (hypothesis, time, cell).')
    if not np.isfinite(mass).all() or not np.isfinite(demand).all() or np.any(mass < 0):
        raise ValueError('Joint masses/demand must be finite; masses must be nonnegative.')
    totals = mass.sum(axis=(0, 2))
    error = float(np.max(np.abs(totals - 1)))
    if error > 1e-10:
        raise ValueError('Joint mass must sum to one at every time; no repair applied.')
    if B.shape != times.shape or U.shape != times.shape or not np.isfinite(B).all() or not np.isfinite(U).all():
        raise ValueError('baseline and price must be finite vectors matching times.')
    if np.any((U < 0) | (U > 1)):
        raise ValueError('Normalized price must lie in [0, 1].')
    x = (edges[:-1] + edges[1:]) / 2
    state_mass = mass.sum(axis=0)
    X = _moments(np.broadcast_to(x, state_mass.shape), state_mass)
    # Move time to the first dimension before flattening hypothesis/cell.
    D = _moments(demand.transpose(1, 0, 2).reshape(len(times), -1),
                 mass.transpose(1, 0, 2).reshape(len(times), -1))
    metrics = {}
    observed = np.full(times.shape, np.nan) if observations is None else np.asarray(observations, dtype=float)
    if observed.shape != times.shape or np.isinf(observed).any():
        raise ValueError('Observations must match stored times; NaN denotes missing values.')
    valid = np.isfinite(observed)
    if valid.any():
        for name, prediction in (('BM', D['mean']), ('baseline', B)):
            residual = prediction[valid] - observed[valid]
            mse = float(np.mean(residual ** 2))
            metrics[name] = dict(n=int(valid.sum()), MSE=mse, RMSE=float(np.sqrt(mse)),
                                 MAE=float(np.mean(np.abs(residual))), bias=float(np.mean(residual)))
    return dict(times=times.copy(), x_edges=edges.copy(), x=x, joint_mass=mass.copy(),
                demand_values=demand.copy(), state_mass=state_mass, X=X, D=D,
                baseline=B.copy(), price=U.copy(), observations=observed.copy(),
                metrics=metrics, maximum_abs_mass_error=error)


def export_distribution_outputs(directory, *, times, x_edges, joint_mass, demand_values,
                                baseline, price, mode, metadata, time_unit='days',
                                observations=None, timestamps=None, deterministic_state=None,
                                deterministic_demand=None, time_origin=None,
                                price_edges=None, price_levels=None):
    """Save overview figures, dated daily comparisons, CSV, NPZ and metadata.

Bands are pointwise 5--95% quantiles of latent X/D, on the stored discrete
supports. They exclude observation noise and simultaneous trajectory coverage.
The caller must declare units, information set and omitted uncertainty in metadata.
An existing destination is never overwritten. Plotting uses stored time slices;
it does not refine sparse forecasts or change the model's output/time grid.
ISO timestamps additionally create one daily figure per represented local date.
All daily demand panels share a vertical scale; missing observations stay NaN.
"""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}.')
    time_unit = {'day': 'days', 'hour': 'hours'}.get(time_unit, time_unit)
    if time_unit not in ('days', 'hours'):
        raise ValueError('time_unit must be days or hours.')
    if not isinstance(metadata, dict) or not metadata.get('interpretation') or not metadata.get('demand_unit'):
        raise ValueError('metadata must declare interpretation and demand_unit.')
    # Validate JSON and all optional vectors before creating any outputs.
    json.dumps(metadata, allow_nan=False)
    result = summarize_distribution(times, x_edges, joint_mass, demand_values, baseline, price,
                                    observations=observations)
    if mode == 'counterfactual_control' and np.isfinite(result['observations']).any():
        raise ValueError('Do not score a counterfactual tariff against consumption under the observed tariff.')
    times = result['times']
    time_origin = times[0] if time_origin is None else float(time_origin)
    if not np.isfinite(time_origin) or time_origin > times[0]:
        raise ValueError('time_origin must be finite and no later than the first stored slice.')
    factor = 24 if time_unit == 'days' else 1
    hours = (times - time_origin) * factor
    if (price_edges is None) != (price_levels is None):
        raise ValueError('price_edges and price_levels must be supplied together.')
    price_arrays = {}
    price_hours, price_plot = hours, result['price']
    if price_edges is not None:
        pe, pv = np.asarray(price_edges, dtype=float), np.asarray(price_levels, dtype=float)
        if (pe.ndim != 1 or pv.ndim != 1 or len(pe) != len(pv) + 1
                or len(pv) < 1 or not np.isfinite(pe).all() or not np.isfinite(pv).all()
                or np.any(np.diff(pe) <= 0) or pe[0] > times[0] or pe[-1] < times[-1]
                or np.any((pv < 0) | (pv > 1))):
            raise ValueError('Explicit piecewise price plan must cover the stored times with U in [0, 1].')
        indices = np.minimum(np.searchsorted(pe, times, side='right') - 1, len(pv) - 1)
        if not np.allclose(pv[indices], result['price'], rtol=0, atol=1e-14):
            raise ValueError('Stored price samples disagree with the explicit right-continuous plan.')
        price_hours, price_plot = (pe - time_origin) * factor, np.r_[pv, pv[-1]]
        price_arrays = dict(price_edges=pe, price_levels=pv)
    optional = {}
    for name, value in (('deterministic_state', deterministic_state),
                        ('deterministic_demand', deterministic_demand)):
        if value is not None:
            value = np.asarray(value, dtype=float)
            if value.shape != times.shape or not np.isfinite(value).all():
                raise ValueError(f'{name} must be finite and match times.')
            optional[name] = value
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=str)
        if timestamps.shape != times.shape:
            raise ValueError('timestamps must match times.')
    calendar_days = [] if timestamps is None else _calendar_days(timestamps, hours)
    X, D = result['X'], result['D']
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    title = str(metadata['interpretation'])
    unit = str(metadata['demand_unit'])
    import matplotlib.pyplot as plt

    def save(fig, name):
        try:
            fig.tight_layout()
            fig.savefig(directory / name, dpi=180, bbox_inches='tight')
        finally:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    # Voronoi time bins display each stored slice; do not invent intermediate laws.
    time_edges = np.r_[hours[0], (hours[:-1] + hours[1:]) / 2, hours[-1]]
    mesh = ax.pcolormesh(time_edges, result['x_edges'],
                        (result['state_mass'] / np.diff(result['x_edges'])).T,
                        shading='flat', cmap='viridis')
    ax.plot(hours, X['mean'], color='white', linewidth=1.5, label='E[X]')
    ax.set(title='BM1 state density — ' + title, xlabel='Hours from declared time origin', ylabel='State X')
    ax.legend(loc='best')
    fig.colorbar(mesh, ax=ax, label='Probability density p(X,t)')
    save(fig, 'bm_state_density.png')

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].fill_between(hours, X['p05'], X['p95'], alpha=.2, label='Latent X: pointwise 5–95%')
    axes[0].plot(hours, X['mean'], label='E[X]')
    axes[0].step(price_hours, price_plot, where='post', label='Normalized price U', linestyle=':')
    if 'deterministic_state' in optional:
        axes[0].plot(hours, optional['deterministic_state'], label='Deterministic X (initial mean)', linestyle='--')
    axes[0].set(ylabel='State / price', title='BM1 response — ' + title)
    axes[1].plot(hours, D['mean'] - result['baseline'], label='E[D] − B: net charging demand')
    axes[1].axhline(0, color='grey', linewidth=.7)
    axes[1].set_ylabel(unit)
    _plot_demand(axes[2], hours, result, D)
    if 'deterministic_demand' in optional:
        axes[2].plot(hours, optional['deterministic_demand'], label='Deterministic D', linestyle=':')
    axes[2].set(ylabel=unit, xlabel='Hours from declared time origin')
    for ax in axes:
        ax.grid(True, alpha=.25)
        ax.legend(loc='best')
    save(fig, 'bm_response.png')

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    _plot_demand(axes[0], hours, result, D)
    axes[0].set(title='BM1 demand — ' + title, ylabel=unit)
    axes[1].step(price_hours, price_plot, where='post', label='Normalized price U')
    axes[1].set_ylabel('Price U')
    axes[2].plot(hours, X['mean'], label='BM1 E[X]')
    axes[2].fill_between(hours, X['p05'], X['p95'], alpha=.2, label='Latent X: pointwise 5–95%')
    axes[2].set(ylabel='State X', xlabel='Hours from declared time origin')
    for ax in axes:
        ax.grid(True, alpha=.25)
        ax.legend(loc='best')
    save(fig, 'bm_demand.png')

    daily_files = []
    if calendar_days:
        (directory / 'daily').mkdir()
        demand_range = np.r_[D['mean'], D['p05'], D['p95'], result['baseline'], result['observations']]
        low, high = np.nanmin(demand_range), np.nanmax(demand_range)
        padding = .05 * max(high - low, .01)
        for date, indices, day_hours in calendar_days:
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            _plot_demand(axes[0], day_hours, result, D, indices)
            axes[0].set(title=f'BM1 demand — {date}\n{title}', ylabel=unit,
                        ylim=(low - padding, high + padding))
            # Retain full declared price changes, even between sparse slices.
            midnight_hours = hours[indices[0]] - day_hours[0]
            axes[1].step(price_hours - midnight_hours, price_plot, where='post',
                         label='Normalized price U', color='tab:orange')
            axes[1].set(ylabel='Price U', ylim=(-.05, 1.05))
            axes[2].fill_between(day_hours, X['p05'][indices], X['p95'][indices],
                                 color='tab:blue', alpha=.2, label='Latent X: pointwise 5–95%')
            axes[2].plot(day_hours, X['mean'][indices], label='BM1 E[X]', color='tab:blue')
            axes[2].set(ylabel='State X', ylim=(0, 1),
                        xlabel='Elapsed hours from this day’s midnight',
                        xlim=(0, max(24, float(np.ceil(day_hours[-1] + .5)))))
            axes[2].set_xticks(np.arange(0, 25, 3))
            for ax in axes:
                ax.grid(True, alpha=.25)
                ax.legend(loc='best')
            name = f'daily/bm_demand_{date}.png'
            save(fig, name)
            daily_files.append(name)

    columns = dict(time=times, elapsed_hours=hours, baseline=result['baseline'], price=result['price'],
                   observed=result['observations'], net_charging_demand=D['mean'] - result['baseline'])
    if timestamps is not None:
        columns['timestamp'] = timestamps
    for name, moments in (('X', X), ('D', D)):
        columns.update({name + '_' + key: value for key, value in moments.items()})
    with (directory / 'moments.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    arrays = {key: result[key] for key in ('times', 'x_edges', 'joint_mass', 'demand_values',
                                         'baseline', 'price', 'observations')}
    arrays.update(optional)
    arrays.update(price_arrays)
    arrays['time_origin'] = np.asarray(time_origin)
    if timestamps is not None:
        arrays['timestamps'] = timestamps
    np.savez_compressed(directory / 'distribution.npz', **arrays)
    summary = dict(schema='bm1-distribution-output-v1', mode=mode, time_unit=time_unit, time_origin=time_origin,
                   metadata=metadata, maximum_abs_mass_error=result['maximum_abs_mass_error'],
                   metrics=result['metrics'], bias_convention='prediction minus observation',
                   uncertainty='Pointwise latent X/D 5–95% discrete-support quantiles; observation noise and simultaneous path coverage excluded. See metadata for omitted parameter/curve uncertainty.',
                   density_display='Each stored time slice occupies its nearest-time bin; no intermediate law is computed.',
                   daily_figures=daily_files,
                   daily_display='Exact stored slices grouped by ISO local date; common demand scale; no daily refit, reset or observation update.',
                   files=['bm_state_density.png', 'bm_response.png', 'bm_demand.png',
                          'moments.csv', 'distribution.npz', 'summary.json'] + daily_files)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary

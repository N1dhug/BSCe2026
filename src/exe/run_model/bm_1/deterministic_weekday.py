"""Small rolling-origin G4 development study; no target-demand state fitting."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def metrics(observed, prediction):
    mask = np.isfinite(observed)
    if not mask.any():
        return {'samples': 0, 'mse': None, 'rmse': None, 'mae': None, 'bias': None}
    e = prediction[mask] - observed[mask]
    return {'samples': int(mask.sum()), 'mse': float(np.mean(e*e)),
            'rmse': float(np.sqrt(np.mean(e*e))), 'mae': float(np.mean(abs(e))),
            'bias': float(np.mean(e))}


def history_inputs(frame, baseline, day, steps):
    from src.method.bm_1_deterministic_response import BM1ResponseInputs
    from src.method.bm_1_deterministic_weekday import BM1WeekdayInputs
    from src.tools.bm_1_deterministic_data import common_inputs

    trajectories = []
    days = [day - pd.Timedelta(weeks=k) for k in range(1, 6)]
    for historical_day in days:
        start = historical_day - pd.Timedelta(hours=24)
        signals = common_inputs(frame[['U', 'Temp']], baseline, start,
            historical_day + pd.Timedelta(hours=24), steps_per_hour=steps)
        observed = frame.reindex(signals['index']).I.to_numpy(dtype=float)
        trajectories.append(BM1ResponseInputs(signals['time'], signals['baseline'],
            (signals['price_hour']+1)/2, observed, observation_mask=np.isfinite(observed),
            warmup_hours=24, steps_per_hour=steps))
    return BM1WeekdayInputs(trajectories), days


def run(raw, reference, start, days, output, *, steps=60, seconds_per_day=180,
        calibration_end=None):
    from src.exe.run_model.progress import runtime_text
    from src.exe.run_model.wm_4.output_model import statistical_output
    from src.method.bm_1_deterministic_response import BM1ResponseInputs, fit_response
    from src.params.bm_1_response import BM1PiecewiseResponse
    from src.tools.bm_1_deterministic_data import common_inputs, prepare_frame

    start = pd.Timestamp(start).normalize()
    if not 1 <= days <= 28 or not 2 <= steps <= 120:
        raise ValueError('Development study requires 1..28 days and 2..120 substeps/h.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Calibration is fixed BEFORE the first prediction; subsequent baseline fits
    # expand, but affine units do not change between days or lambda candidates.
    calibration_end = start if calibration_end is None else pd.Timestamp(calibration_end)
    if calibration_end > start:
        raise ValueError('Calibration must finish no later than the first origin.')
    # A declared zero-price floor also admits known zero-price scenarios; the
    # upper bound comes ONLY from the pre-calibration history. No future maximum.
    price_max = float(raw.loc[raw.index < calibration_end, 'Price'].max())
    frame, scales = prepare_frame(raw, raw.index.min(), calibration_end,
                                price_bounds=(0., price_max))
    if start-pd.Timedelta(days=36) < frame.index.min():
        raise ValueError('Need five previous matching weekdays plus warm-up.')
    save_json(output/'protocol.json', {'start': start, 'days': days, 'scales': scales,
        'reference_parameters': reference, 'regularizations': [0., .001],
        'starts': [0., .25], 'steps_per_hour': steps, 'warmup_hours': 24,
        'calibration_end_exclusive': calibration_end,
        'price_domain': 'declared zero-price floor; upper bound from calibration TRAIN',
        'baseline': 'WM4 expanding pre-origin history',
        'initial_state': 'x=0.5 at each historical warm-up; median x at hour 24 for forecast',
        'history': 'five separate previous matching weekdays; pooled measured target hours',
        'horizon': 'daily rolling origin, NOT one frozen 168-hour forecast',
        'observation_contract': 'legacy hourly-point approximation; development',
        'seconds_per_day': seconds_per_day})
    begun = perf_counter()
    results = []
    for number, day in enumerate(pd.date_range(start, periods=days, freq='D')):
        dest = output/f'{day:%Y-%m-%d}'
        dest.mkdir()
        train = frame.loc[frame.index < day, ['I', 'Temp']]
        baseline = statistical_output().fit(train, day)
        pooled, historical_days = history_inputs(frame.loc[frame.index < day], baseline, day, steps)
        fitted = fit_response(pooled, reference, fixed_x0=.5, regularizations=(0., .001),
            starts=(0., .25), max_calls=3000, calls_per_start=750, seconds=seconds_per_day,
            checkpoint=lambda value: save_json(dest/'fit.json', value))
        if fitted['status'] != 'complete':
            raise RuntimeError(f'Incomplete fit on {day}; checkpoint preserved, no winner declared.')
        signals = common_inputs(frame[['U', 'Temp']], baseline, day,
                               day+pd.Timedelta(hours=24), steps_per_hour=steps)
        forecast = BM1ResponseInputs(signals['time'], signals['baseline'],
            (signals['price_hour']+1)/2, steps_per_hour=steps)
        predictions = pd.DataFrame({'O': signals['baseline'][np.arange(24)*steps]},
                                   index=signals['index'])
        records = []
        for j, candidate in enumerate(fitted['candidates']):
            best = candidate['best_converged']
            g = BM1PiecewiseResponse.from_inner_values([best['g_mid']], pieces=2)
            historical = pooled.solve(best['parameters'], price_response=g)
            fine_history = pooled.refined().solve(best['parameters'], price_response=g)
            x0 = float(np.median(historical['historical_start_states']))
            refined_x0 = float(np.median(fine_history['historical_start_states']))
            parameters = list(best['parameters']); parameters[3] = x0
            simulated = forecast.solve(parameters, price_response=g)
            refined_parameters = parameters.copy(); refined_parameters[3] = refined_x0
            fine_forecast = forecast.refined().solve(refined_parameters, price_response=g)
            error = max(float(np.max(abs(historical['demand']-fine_history['demand']))),
                        float(np.max(abs(simulated['demand']-fine_forecast['demand']))))
            if error > 1e-5:
                raise RuntimeError(f'Time refinement failed: {error}; no scores accepted.')
            name = 'without_penalty' if j == 0 else 'with_penalty'
            predictions[name] = simulated['demand']
            records.append({'name': name, 'lambda': candidate['regularization'],
                'parameters': best['parameters'], 'g_mid': best['g_mid'], 'x_forecast': x0,
                'historical_start_states': historical['historical_start_states'].tolist(),
                'refinement_max_error': error, 'training_mse': best['mse'],
                'active_bounds': best['active_bounds']})
        # Save frozen forecasts before opening target observations for scoring.
        prediction_file = dest/'forecast_normalized.csv'
        predictions.to_csv(prediction_file, index_label='Datetime')
        observed = raw.reindex(predictions.index).Demand.to_numpy(dtype=float)
        scored = predictions*scales['demand_scale']+scales['demand_min']
        scored['observed'] = observed
        scored.to_csv(dest/'scored.csv', index_label='Datetime')
        record = {'day': str(day), 'training_days': list(map(str, historical_days)),
            'observed_training_hours': int(pooled.observation_mask.sum()),
            'baseline': baseline.metadata, 'candidates': records,
            'scores': {name: metrics(observed, scored[name].to_numpy()) for name in predictions},
            'forecast_sha256': hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
            'counts': fitted['counts'], 'fit_seconds': fitted['seconds']}
        save_json(dest/'result.json', record)
        results.append(record)
        save_json(output/'results.json', {'status': 'running', 'daily': results})
        print(f'{number+1}/{days} {day:%Y-%m-%d}: {runtime_text(begun, number+1, days)}', flush=True)
    save_json(output/'results.json', {'status': 'complete', 'seconds': perf_counter()-begun, 'daily': results})
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--raw-csv', required=True, help='Datetime,Demand,Price,Temp in original units.')
    parser.add_argument('--reference-fit', required=True, help='Earlier G fit.json; fixes k and seeds C/Xi.')
    parser.add_argument('--start', default='2022-02-07')
    parser.add_argument('--days', type=int, default=21)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps-per-hour', type=int, default=60)
    parser.add_argument('--seconds-per-day', type=float, default=180)
    parser.add_argument('--calibration-end', help='Optional common pre-origin scale cutoff for independent week shards.')
    args = parser.parse_args()
    raw = pd.read_csv(args.raw_csv, index_col=0, parse_dates=True)
    reference_path = Path(args.reference_fit)
    manifest = json.loads((reference_path.parent/'manifest.json').read_text(encoding='utf-8'))
    if pd.Timestamp(manifest['training_end_exclusive']) > pd.Timestamp(args.start):
        raise ValueError('Reference fit must finish before the first forecast origin.')
    reference = json.loads(reference_path.read_text(encoding='utf-8'))['best_converged']['parameters']
    run(raw, reference, args.start, args.days, args.output,
        steps=args.steps_per_hour, seconds_per_day=args.seconds_per_day,
        calibration_end=args.calibration_end)


if __name__ == '__main__':
    main()

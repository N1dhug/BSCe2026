"""One bounded, registered frozen-fit price audit, without optimization/refits."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.method.bm_1_g7_matched import MatchedDeterministicControl


def dump(path, value):
    Path(path).write_text(json.dumps(value, separators=(',', ':'), allow_nan=False), encoding='utf-8')


def compact(forward):
    """Hourly raw state/demand receipts plus full-grid cost; no dense trace claim."""
    steps = forward['steps_per_hour']
    return dict(model=forward['model'], steps_per_hour=steps,
                block_prices=forward['block_prices'].tolist(),
                objective_normalized=forward['objective_normalized'],
                objective_raw=forward['objective_raw'],
                demand_energy_normalized=forward['demand_energy_normalized'],
                target_energy_normalized=forward['target_energy_normalized'],
                state_names=forward['state_names'],
                state_hourly=forward['state'][::steps].tolist(),
                demand_first_midpoint_each_hour=forward['demand'][::steps].tolist(),
                state_min=forward['state'].min(axis=0).tolist(),
                state_max=forward['state'].max(axis=0).tolist(),
                demand_min=float(forward['demand'].min()),
                demand_max=float(forward['demand'].max()))


def run(packet_path, output):
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    packet_bytes = Path(packet_path).read_bytes()
    packet = json.loads(packet_bytes)
    protocol = packet['response_protocol']
    if (protocol['grids'] != [60, 120] or protocol['price_levels'] != [.25, .5, .75]
            or protocol['max_forwards'] != 18 or protocol['max_seconds'] != 30.):
        raise ValueError('This runner only executes the registered small response audit.')
    control = MatchedDeterministicControl(packet)
    if set(control.models) != {'bm_linear_january', 'bm_g4_january', 'wm4_january'}:
        raise ValueError('Matched January pair and original linear BM reference required.')
    source_hashes = packet['execution_source_sha256']
    root = Path(__file__).resolve().parents[4]
    for path, digest in source_hashes.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen audit source changed.')
    # Export immutable protocol before the first actual forward.
    dump(output/'packet.json', packet)
    report = dict(schema='g7-matched-response-raw-v1',
                  packet_sha256=hashlib.sha256(packet_bytes).hexdigest(),
                  source_commit=packet['source_commit'], forwards=0, results=[],
                  precision=[], response=[], status='running',
                  stochastic_run=False, refit=False, optimized_policy=False,
                  historical_model_ranking=False, full_control_accept=False)
    saved, fine = {}, {}
    try:
        for name in control.models:
            for level in protocol['price_levels']:
                prices = [level]*(len(control.edges)-1)
                for steps in protocol['grids']:
                    if report['forwards'] >= 18 or perf_counter()-started >= 30:
                        raise RuntimeError('Registered response budget exhausted; no retry.')
                    report['forwards'] += 1
                    forward = control.evaluate(name, prices, steps)
                    saved[name, level, steps] = forward
                    report['results'].append(compact(forward))
                    dump(output/'report.json', report)
                    if perf_counter()-started >= 30:
                        raise RuntimeError('Response budget exceeded after completed forward.')
                coarse, refined = (saved[name, level, steps] for steps in (60, 120))
                # Same physical midpoints: interpolate only within a constant-price policy.
                fine_demand_at_coarse = np.interp(coarse['midpoints'], refined['midpoints'], refined['demand'])
                state_error = float(np.max(np.abs(coarse['state']-refined['state'][::2])))
                demand_error = float(np.max(np.abs(coarse['demand']-fine_demand_at_coarse)))
                cost_error = abs(coarse['objective_normalized']-refined['objective_normalized'])
                cost_gate = cost_error <= 1e-4+1e-3*abs(refined['objective_normalized'])
                passed = state_error <= 1e-5 and demand_error <= 1e-5 and cost_gate
                report['precision'].append(dict(model=name, price=level,
                                               state_error=state_error,
                                               demand_error=demand_error,
                                               cost_error=cost_error, passed=bool(passed)))
                fine[name, level] = refined
        for name in control.models:
            low, middle, high = (fine[name, level] for level in (.25, .5, .75))
            difference = high['demand']-low['demand']
            report['response'].append(dict(model=name,
                low_to_high_demand_change_min=float(difference.min()),
                low_to_high_demand_change_max=float(difference.max()),
                low_to_high_demand_change_rms=float(np.sqrt(np.mean(difference**2))),
                low_to_high_energy_change_normalized=high['demand_energy_normalized']-low['demand_energy_normalized'],
                affine_midpoint_error=float(np.max(np.abs(middle['demand']-.5*(low['demand']+high['demand']))))))
        report['precision_accept'] = all(r['passed'] for r in report['precision'])
        report['status'] = 'complete' if report['precision_accept'] else 'precision-rejected'
    except Exception as exc:
        report['status'] = 'stopped'
        report['reason'] = str(exc)
    finally:
        report['invocation_seconds_before_export'] = perf_counter()-started
        dump(output/'report.json', report)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.packet, args.output)
    print(json.dumps({k: report[k] for k in ('status', 'forwards', 'invocation_seconds_before_export')}))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())

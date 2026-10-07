"""One registered G7 deterministic fixed-supply day; no historical fit or FP.

Run with BLAS=1. All residual/Jacobian forwards count, checkpoint per completed
forward, no automatic retry, no process resume or increased study budget.
"""
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import signal
import sys

import numpy as np
import scipy

from src.control.bm_1 import BM1ControlNumerics
from src.method.bm_1_distribution_panel import build_problem
from src.method.bm_1_g7_deterministic import ControlBudgetStop, DeterministicResidualControl


def save(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def source_inventory(root):
    paths = {Path(__file__).resolve(),
             root/'src/method/bm_1_g7_deterministic.py'}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(root):
                paths.add(path)
    return {str(p.relative_to(root)):dict(
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        git_blob=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest())
        for p in sorted(paths)}


def raw_evaluation(evaluator, result):
    return dict(prices=result.prices.tolist(), objective=result.total,
        times=evaluator.solver_t_eval.tolist(), state=result.state.tolist(),
        demand=result.demand.tolist(), midpoints=evaluator.midpoints.tolist(),
        target=evaluator.target.tolist(), widths=evaluator.widths.tolist(),
        terminal_state=float(result.state[-1]),
        state_min=float(np.min(result.state)), state_max=float(np.max(result.state)))


def run(output, source_commit):
    started = perf_counter()
    deadline = started+240.  # final 60 s of the 300 s ceiling is export reserve
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    inventory = source_inventory(root)
    protocol = dict(study_id='g7_deterministic_fixed_supply_20261007_v1',
        source_commit=source_commit, source_hashes=inventory,
        scipy_version=scipy.__version__, time_unit='hours', horizon=[0.,24.],
        model=dict(C_hours=4., k=2., Xi=.4, f='1-2*x', g='1-2*u',
                   initial_state=.5, sigma_x=0.),
        baseline='.5+.1*sin(2*pi*t/24)', target='.5 fixed ex ante',
        price_edges=[0.,6.,12.,18.,24.], price_bounds=[0.,1.],
        starts=[[.5,.5,.5,.5],[.7,.7,.3,.3]],
        objective='sum widths*(D-A)**2; original BM fixed-supply objective',
        optimizer=dict(method='least_squares/trf', jac='2-point', loss='linear',
            diff_step=1e-4, ftol=1e-7, xtol=1e-7, gtol=1e-7,
            max_new_forwards_per_start=44, max_total_forwards=100),
        numerics=dict(coarse_quad=1/24, fine_quad=1/48,
            coarse_solver_step=1/12, fine_solver_step=1/24,
            coarse_rtol=1e-9, coarse_atol=1e-11, fine_rtol=1e-10, fine_atol=1e-12),
        gates=dict(objective_atol=1e-4, objective_rtol=1e-3,
            state_at_coarse_times_atol=1e-5, demand_at_coarse_midpoints_atol=1e-5,
            residual_identity_atol=1e-13, residual_identity_rtol=1e-12,
            fine_selected_not_worse_than_neutral=True),
        budget=dict(pilot_seconds=60., compute_seconds=240., total_seconds=300.,
            previous_conservative_panel_debit_seconds=1045.580078,
            previous_remaining_seconds=754.419922,
            note='This run extends the G ledger; no F/WM budget or old study is reset.'),
        scope='synthetic deterministic numerical development; not WM comparison, historical control or stochastic G7 acceptance',
        terminal_weight=0., smoothing_weight=0., historical_fit=False)
    save(dest/'protocol.json', protocol)
    report = dict(protocol=protocol, status='running', searches=[], comparisons={},
        deterministic_pilot_accept=False, wm_comparison_accept=False,
        historical_control_accept=False, stochastic_G7_accept=False)
    control = None
    fine_calls = 0
    problem = replace(build_problem(), parameters=replace(build_problem().parameters, sigma_x=0.))
    numerics = BM1ControlNumerics(objective_max_step=1/24, solver_max_step=1/12)
    coarse = problem.build_deterministic_evaluator(numerics)
    fine = problem.build_deterministic_evaluator(replace(numerics,
        objective_max_step=1/48, solver_max_step=1/24,
        deterministic_rtol=1e-10, deterministic_atol=1e-12))
    coarse_log = dest/'completed_forwards.jsonl'

    def record_forward(record):
        with coarse_log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')

    def hard_stop(*_):
        raise ControlBudgetStop('Process compute timer expired; export partials.')

    old_handler = None
    if hasattr(signal, 'setitimer'):
        old_handler = signal.signal(signal.SIGALRM, hard_stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-perf_counter()))
    try:
        control = DeterministicResidualControl(coarse, max_forwards=100,
            deadline=deadline, on_forward=record_forward)
        _, neutral = control.evaluate(np.full(4,.5))
        save(dest/'coarse_neutral.json', raw_evaluation(coarse, neutral))
        pilot_seconds = perf_counter()-started
        report['pilot_seconds'] = pilot_seconds
        save(dest/'report.json', report)
        if pilot_seconds >= 60:
            raise ControlBudgetStop('Pilot exceeds its 60 s admission cap.')
        # Forecast from one point origin. Each candidate restarts the same x0.
        for start in protocol['starts']:
            search = control.search(start, max_new_forwards=44)
            report['searches'].append(search)
            save(dest/'report.json', report)
        converged = [s for s in report['searches'] if s['success']]
        candidates = [s for s in report['searches'] if s['prices'] is not None]
        if not candidates:
            raise RuntimeError('No completed price candidate.')
        chosen = min(converged or candidates, key=lambda s:s['objective'])
        report['selection_rule'] = 'lowest coarse objective among converged starts; incumbent only if none converged'
        report['chosen'] = chosen
        policies = {'neutral':neutral.prices, 'selected':np.array(chosen['prices'])}
        # Validate both reported local iterates, not an additional fit or family search.
        for index, item in enumerate(report['searches']):
            if item['prices'] is not None:
                policies[f'start_{index+1}'] = np.array(item['prices'])
        fine_cache = {}
        for name, prices in policies.items():
            _, base = control.evaluate(prices)
            save(dest/f'coarse_{name}.json', raw_evaluation(coarse, base))
            key = tuple(float(v) for v in prices)
            if key not in fine_cache:
                if perf_counter() >= deadline or control.forward_calls+fine_calls >= 100:
                    raise ControlBudgetStop('No reserve for fine validation.')
                fine_calls += 1
                value = fine.evaluate(prices)
                fine_cache[key] = value
            value = fine_cache[key]
            save(dest/f'fine_{name}.json', raw_evaluation(fine, value))
            x_error = float(np.max(abs(base.state-np.interp(coarse.solver_t_eval,
                fine.solver_t_eval, value.state))))
            # No demand interpolation across a price jump: compare midpoint laws
            # using fine-interpolated state and the unchanged BM demand map.
            fine_x = np.interp(coarse.midpoints, fine.solver_t_eval, value.state)
            parameters = fine.parameters_for_prices(prices)
            fine_d = fine.model.eval(coarse.midpoints, fine_x,
                *[parameters[i] for i in (0,1,2,4,5)])[-1]
            d_error = float(np.max(abs(base.objective.expected_demand-fine_d)))
            j_error = abs(base.total-value.total)
            passed = (j_error <= 1e-4+1e-3*abs(value.total)
                      and x_error <= 1e-5 and d_error <= 1e-5
                      and np.min(value.state) >= 0 and np.max(value.state) <= 1)
            report['comparisons'][name] = dict(coarse_J=base.total, fine_J=value.total,
                objective_error=j_error, state_error=x_error, demand_error=d_error,
                prices=np.asarray(prices).tolist(), passed=bool(passed))
            save(dest/'report.json', report)
        neutral_j = report['comparisons']['neutral']['fine_J']
        selected_j = report['comparisons']['selected']['fine_J']
        report['relative_improvement'] = (neutral_j-selected_j)/neutral_j if neutral_j else None
        report['deterministic_pilot_accept'] = bool(converged
            and all(item['passed'] for item in report['comparisons'].values())
            and selected_j <= neutral_j)
        report['status'] = 'complete' if report['deterministic_pilot_accept'] else 'inconclusive'
    except (ControlBudgetStop, RuntimeError, ValueError) as exc:
        report['status'] = 'budget_stop' if isinstance(exc, ControlBudgetStop) else 'failed'
        report['error'] = str(exc)
    finally:
        if hasattr(signal, 'setitimer'):
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        report['source_unchanged'] = source_inventory(root) == inventory
        report['coarse_forward_calls'] = 0 if control is None else control.forward_calls
        report['fine_forward_calls'] = fine_calls
        report['total_forward_calls'] = report['coarse_forward_calls']+fine_calls
        report['cache_hits'] = 0 if control is None else control.cache_hits
        elapsed = perf_counter()-started
        report['invocation_seconds_before_final_export'] = elapsed
        report['conservative_panel_debit_seconds'] = 1045.580078+elapsed
        report['remaining_panel_seconds'] = 754.419922-elapsed
        if not report['source_unchanged']:
            report['deterministic_pilot_accept'] = False
            report['status'] = 'failed'
            report['error'] = 'Source inventory changed during invocation.'
        save(dest/'report.json', report)
    return report


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    result = run(args.output, args.source_commit)
    print(json.dumps({key:result.get(key) for key in ('status','deterministic_pilot_accept',
        'pilot_seconds','total_forward_calls','invocation_seconds_before_final_export',
        'relative_improvement','error')}, indent=2))

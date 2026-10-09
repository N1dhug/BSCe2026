"""Bounded C/k/Xi/a least squares for deterministic G; no strategy or penalty studies."""
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from src.method.bm_1_deterministic_fit import BudgetStop, _parameters
from src.params.bm_1_response import BM1PiecewiseResponse


def fit_strategy(inputs, reference, *, max_calls=1500,
                 calls_per_start=750, seconds=600, checkpoint=None, initial_records=None):
    p = _parameters(reference)
    mask = inputs.observation_mask.copy()
    n = int(mask.sum())
    if not n:
        raise ValueError('Measured training demand is required.')
    bounds=([0.,np.log(.05),0.,-1.],[np.log(744.),np.log(10.),1.,1.])
    seeds=[np.r_[np.log(p[:2]),p[2],0.],np.array([np.log(12.),np.log(2.),.5,.25])]
    if initial_records is not None:
        if not 1 <= len(initial_records) <= 2:
            raise ValueError('A checkpoint continuation needs one or two saved seeds.')
        seeds=[]
        for record in initial_records:
            saved=_parameters(record['parameters'])
            seeds.append(np.r_[np.log(saved[:2]),saved[2],record['g_mid']])
    begun = perf_counter(); deadline = begun+seconds
    counts = {'residual_calls': 0, 'pooled_solves': 0, 'rhs_evaluations': 0}
    records = []; best = None; provisional = None; last_save = begun
    def decode(v):
        return np.r_[np.exp(v[:2]),v[2],.5],float(v[3])
    def snapshot(status):
        return {'status': status, 'mode': 'joint', 'lambda': 0.,
                'seconds': perf_counter()-begun, 'counts': counts.copy(),
                'starts': records.copy(), 'best_converged': best,
                'best_provisional': provisional, 'observed_samples': n,
                'budget': {'max_calls': max_calls, 'calls_per_start': calls_per_start, 'seconds': seconds}}
    for number, seed in enumerate(seeds):
        initial_count = counts['residual_calls']
        def residual(v):
            nonlocal best, provisional, last_save
            if counts['residual_calls'] >= max_calls:
                raise BudgetStop('total residual-call limit')
            if counts['residual_calls']-initial_count >= calls_per_start:
                raise BudgetStop('per-start residual-call limit')
            if perf_counter() >= deadline:
                raise BudgetStop('wall-time limit')
            parameters, a = decode(v)
            g = BM1PiecewiseResponse.from_inner_values([a], pieces=2)
            counts['residual_calls'] += 1; counts['pooled_solves'] += 1
            try:
                solved = inputs.solve(parameters, deadline=deadline, price_response=g)
            finally:
                counts['rhs_evaluations'] += inputs.last_rhs_evaluations
            values = (solved['demand']-inputs.observations)[mask]/np.sqrt(n)
            objective = float(values@values)
            if provisional is None or objective < provisional['objective']:
                provisional = {'parameters': parameters.tolist(), 'g_mid': a,
                    'mse': float(values@values), 'objective': objective}
            if checkpoint is not None and perf_counter()-last_save >= 2:
                checkpoint(snapshot('running')); last_save = perf_counter()
            return values
        try:
            fit = least_squares(residual, seed, bounds=bounds, method='trf',
                jac='2-point', diff_step=1e-4, loss='linear', x_scale='jac',
                ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=calls_per_start)
            parameters, a = decode(fit.x)
            data_mse = float(fit.fun@fit.fun)
            singular = np.linalg.svd(fit.jac, compute_uv=False)
            record = {'start': number, 'parameters': parameters.tolist(), 'g_mid': a,
                'mse': data_mse, 'objective': data_mse,
                'converged': bool(fit.success), 'stop': fit.message,
                'residual_calls': counts['residual_calls']-initial_count,
                'active_bounds': fit.active_mask.tolist(),
                'data_singular_values': singular.tolist(),
                'data_local_rank': int(np.count_nonzero(singular > singular[0]*1e-8)) if singular[0] else 0}
            if fit.success and (best is None or record['objective'] < best['objective']):
                best = record.copy()
            records.append(record)
        except BudgetStop as exc:
            records.append({'start': number, 'converged': False, 'stop': str(exc),
                            'residual_calls': counts['residual_calls']-initial_count})
            if 'per-start' not in str(exc):
                break
        if checkpoint is not None:
            checkpoint(snapshot('between-starts'))
    result = snapshot('converged' if best is not None else 'unfinished')
    if checkpoint is not None:
        checkpoint(result)
    return result

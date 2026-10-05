"""G4: shared monotone responses on G's existing input grid and bounded LS.

The original G3 kernel remains independently restorable. Linear geometry
delegates to that kernel; the curved path uses the same RK4/grid convention.
"""
from bisect import bisect_right
from math import tanh
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares

from src.method.bm_1_deterministic_fit import BM1DeterministicInputs, BudgetStop, _parameters
from src.params.bm_1_response import BM1PiecewiseResponse


def is_linear(curve):
    return np.array_equal(curve.values, 1-2*curve.knots)


def _scalar_response(curve):
    knots, values = tuple(curve.knots), tuple(curve.values)
    slopes = tuple(np.diff(curve.values)/np.diff(curve.knots))
    def evaluate(x):
        if not np.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('RK4 stage left the response domain; no extrapolation.')
        if is_flat_linear:
            return 1-2*x
        j = min(bisect_right(knots, x)-1, len(slopes)-1)
        return values[j]+slopes[j]*(x-knots[j])
    is_flat_linear = is_linear(curve)
    return evaluate


class BM1ResponseInputs(BM1DeterministicInputs):
    def __init__(self, *args, state_response=None, price_response=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_response = state_response or BM1PiecewiseResponse.linear(pieces=2)
        self.price_response = price_response or BM1PiecewiseResponse.linear(pieces=2)
        if not all(isinstance(c, BM1PiecewiseResponse) for c in (self.state_response, self.price_response)):
            raise ValueError('Use validated shared response geometry.')

    def refined(self):
        return BM1ResponseInputs(self.time, self.baseline, self.prices, self.observations,
            state_response=self.state_response, price_response=self.price_response,
            observation_contract=self.contract, warmup_hours=self.warmup,
            steps_per_hour=2*self.steps, observation_mask=self.observation_mask)

    def solve(self, parameters, *, deadline=None, keep_fine=False, price_response=None):
        g = self.price_response if price_response is None else price_response
        if not isinstance(g, BM1PiecewiseResponse):
            raise ValueError('Use validated shared price response geometry.')
        if is_linear(self.state_response) and is_linear(g):
            return super().solve(parameters, deadline=deadline, keep_fine=keep_fine)
        self.last_rhs_evaluations = 0
        C, k, xi, x = _parameters(parameters)
        slope = float(np.max(np.abs(np.diff(self.state_response.values)/np.diff(self.state_response.knots))))
        if .5*k*xi*self.h*slope/C > 1:
            raise ValueError('Refine the declared RK4 grid for this state-response slope.')
        f = _scalar_response(self.state_response)
        hourly_g = g(self.prices)
        fine_g = np.repeat(hourly_g, self.steps)
        def demand(x, gv, baseline):
            delta = tanh(.5*k*(f(x)+gv))
            return baseline+xi*delta*((1-baseline) if delta >= 0 else baseline)
        states = np.empty(self.n_hours+1); states[0] = x
        fine_states = np.empty(len(fine_g)+1) if keep_fine else None
        if keep_fine: fine_states[0] = x
        h = self.h
        for j, gv in enumerate(fine_g):
            if j % (24*self.steps) == 0 and deadline is not None and perf_counter() >= deadline:
                raise BudgetStop('wall-time limit')
            b0, bm, b1 = self.b_edge[j], self.b_mid[j], self.b_edge[j+1]
            self.last_rhs_evaluations += 4
            v1 = (demand(x, gv, b0)-b0)/C
            v2 = (demand(x+h*v1/2, gv, bm)-bm)/C
            v3 = (demand(x+h*v2/2, gv, bm)-bm)/C
            v4 = (demand(x+h*v3, gv, b1)-b1)/C
            x += h*(v1+2*v2+2*v3+v4)/6
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError('RK4 state left [0,1]; no clipping.')
            if keep_fine: fine_states[j+1] = x
            if (j+1) % self.steps == 0: states[(j+1)//self.steps] = x
        point_b = np.interp(np.arange(self.n_hours), self.time, self.baseline)
        point = np.array([demand(x, gv, b) for x, gv, b in zip(states[:-1], hourly_g, point_b)])
        mean = self.b_mean+C*np.diff(states)
        result = {'state': states, 'demand': point if self.contract == 'hourly-point' else mean,
                  'point_demand': point, 'mean_demand': mean, 'rhs_evaluations': self.last_rhs_evaluations}
        if keep_fine:
            result.update(fine_time=np.arange(len(fine_states))*h, fine_state=fine_states,
                fine_demand=np.array([demand(x, gv, b) for x, gv, b in
                    zip(fine_states, np.r_[fine_g, hourly_g[-1]], self.b_edge)]))
        return result


def fit_response(inputs, reference_parameters, *, fixed_x0=.5, C_bounds=(1.,744.),
                 Xi_bounds=(0.,1.), x0_bounds=(0.,1.), regularizations=(0.,1e-3),
                 starts=(0.,.25), max_calls=3000, calls_per_start=750, seconds=600.,
                 checkpoint=None, progress=None):
    """One shared two-piece g; k and linear f fixed from the reference.

    All strengths/starts share the residual/deadline budget, including Jacobian
    probes. Return a candidate per strength; TRAIN never chooses the strength.
    """
    p = _parameters(reference_parameters)
    if not isinstance(inputs, BM1ResponseInputs) or not is_linear(inputs.state_response):
        raise ValueError('The first G4 pilot requires response inputs with linear f.')
    lambdas = tuple(float(v) for v in regularizations)
    seeds = tuple(float(v) for v in starts)
    if not 1 <= len(lambdas) <= 2 or len(set(lambdas)) != len(lambdas) or not all(np.isfinite(v) and v >= 0 for v in lambdas):
        raise ValueError('Declare at most two distinct nonnegative strengths, including zero.')
    if 0. not in lambdas or not 1 <= len(seeds) <= 2 or not all(np.isfinite(v) and -1 <= v <= 1 for v in seeds):
        raise ValueError('G4 requires the zero-penalty reference and at most two valid starts.')
    for value, upper in ((max_calls,3000), (calls_per_start,3000)):
        if isinstance(value,bool) or int(value) != value or not 1 <= value <= upper:
            raise ValueError('Residual budgets must be positive integers <= 3000.')
    if not np.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError('The complete G4 fit budget is at most 600 seconds.')
    bounds = np.array([C_bounds, Xi_bounds, (-1.,1.), x0_bounds], dtype=float)
    if bounds.shape != (4,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0] >= bounds[:,1]) or bounds[0,0] <= 0 or np.any(bounds[[1,3],0] < 0) or np.any(bounds[[1,3],1] > 1):
        raise ValueError('Invalid fixed G4 physical bounds.')
    active = 4 if fixed_x0 is None else 3
    seed = np.array([p[0],p[2],0.,p[3] if fixed_x0 is None else fixed_x0])
    if not np.isfinite(seed).all() or np.any(seed < bounds[:,0]) or np.any(seed > bounds[:,1]):
        raise ValueError('Reference/start state must be inside declared bounds.')
    mask = inputs.observation_mask.copy(); mask[:inputs.warmup] = False
    if inputs.observations is None or not mask.any():
        raise ValueError('Measured training observations are required.')
    n = int(mask.sum())
    started = perf_counter(); deadline = started+seconds
    counts = {'residual_calls':0, 'forward_solves':0, 'rhs_evaluations':0}
    candidates = []; last_save = started; stopped = None
    def pack(v):
        out = np.asarray(v[:active], dtype=float).copy(); out[0] = np.log(out[0]); return out
    def unpack(v):
        out = np.r_[np.exp(v[0]), v[1:]]
        return out if active == 4 else np.r_[out,fixed_x0]
    def report(status):
        return {'status':status, 'seconds':perf_counter()-started, 'counts':dict(counts),
            'candidates':candidates, 'fixed_k':float(p[1]), 'fixed_x0':fixed_x0,
            'optimized_coordinates':['log_C_hours','Xi','g_mid','x0'][:active],
            'observed_samples':n, 'selection':'not-selected; compare on DEVELOPMENT',
            'policy':{'regularizations':list(lambdas), 'starts':list(seeds), 'bounds':bounds.tolist(),
                      'max_calls':max_calls,'calls_per_start':calls_per_start,'seconds':seconds,
                      'penalty':'lambda * g(0.5)^2', 'family':'linear f; shared two-piece g'}}
    for strength in lambdas:
        item = {'regularization':strength,'starts':[], 'best_converged':None,'best_provisional':None}
        candidates.append(item)
        for number, middle in enumerate(seeds):
            begin_calls = counts['residual_calls']
            def residual(v):
                nonlocal last_save
                if counts['residual_calls'] >= max_calls: raise BudgetStop('total residual-call limit')
                if counts['residual_calls']-begin_calls >= calls_per_start: raise BudgetStop('per-start residual-call limit')
                if perf_counter() >= deadline: raise BudgetStop('wall-time limit')
                counts['residual_calls'] += 1; counts['forward_solves'] += 1
                C, xi, inner, x0 = unpack(v)
                g = BM1PiecewiseResponse.from_inner_values([inner],pieces=2)
                try:
                    out = inputs.solve([C,p[1],xi,x0],price_response=g,deadline=deadline)
                finally:
                    counts['rhs_evaluations'] += inputs.last_rhs_evaluations
                values = (out['demand']-inputs.observations)[mask]/np.sqrt(n)
                mse = float(values@values); penalty = float(strength*inner**2)
                record = {'parameters':[float(C),float(p[1]),float(xi),float(x0)],
                          'g_mid':float(inner),'mse':mse,'penalty':penalty,
                          'regularized_objective':mse+penalty,'start':number}
                previous = item['best_provisional']
                if previous is None or record['regularized_objective'] < previous['regularized_objective']:
                    item['best_provisional'] = record
                now = perf_counter()
                if checkpoint is not None and now-last_save >= 2:
                    checkpoint(report('running')); last_save = now
                if progress is not None and counts['residual_calls'] % 25 == 0: progress(report('running'))
                return np.r_[values, np.sqrt(strength)*inner]
            try:
                initial = seed.copy(); initial[2] = middle
                fit = least_squares(residual,pack(initial),bounds=(pack(bounds[:,0]),pack(bounds[:,1])),
                    method='trf',jac='2-point',diff_step=1e-4,loss='linear',x_scale='jac',
                    ftol=1e-7,xtol=1e-7,gtol=1e-7,max_nfev=calls_per_start)
                C, xi, inner, x0 = unpack(fit.x)
                mse = float(fit.fun[:-1]@fit.fun[:-1]); penalty = float(strength*inner**2)
                record = {'parameters':[float(C),float(p[1]),float(xi),float(x0)],
                    'g_mid':float(inner),'mse':mse,'penalty':penalty,'regularized_objective':mse+penalty,
                    'converged':bool(fit.success),'stop':fit.message,'start':number,
                    'residual_calls':counts['residual_calls']-begin_calls,'active_bounds':fit.active_mask.tolist()}
                if fit.success:
                    singular = np.linalg.svd(fit.jac[:-1],compute_uv=False)
                    record['data_jacobian_singular_values'] = singular.tolist()
                    record['data_local_rank'] = int(np.count_nonzero(singular > singular[0]*1e-8)) if singular[0] else 0
                    old = item['best_converged']
                    if old is None or record['regularized_objective'] < old['regularized_objective']:
                        item['best_converged'] = dict(record)
                item['starts'].append(record)
            except BudgetStop as exc:
                item['starts'].append({'converged':False,'stop':str(exc),'residual_calls':counts['residual_calls']-begin_calls})
                if 'per-start' not in str(exc): stopped = str(exc); break
            if checkpoint is not None: checkpoint(report('between-starts'))
        if stopped: break
    result = report('budget-stopped' if stopped else 'complete' if all(c['best_converged'] is not None for c in candidates) else 'unfinished')
    result['stop'] = stopped
    if checkpoint is not None: checkpoint(result)
    return result

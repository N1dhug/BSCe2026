"""G1/G2: linear-response BM1, cheap deterministic propagation and bounded LS.

The scalar RK4 kernel is the standard BM1 equation, with parity checked against
bm_1.deterministic_solver. Fixed input interpolation avoids adaptive-step noise
in finite differences. No clipping, filtering, stochastic or price optimization.
"""
from math import tanh, sinh, asinh, exp
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from src.params.bm_1_deterministic import BM1DeterministicConfig


class BudgetStop(RuntimeError):
    """A stopped fit is not a converged fit."""


def _parameters(parameters):
    p = np.asarray(parameters, dtype=float)
    if p.shape != (4,) or not np.all(np.isfinite(p)):
        raise ValueError('Expected finite [C_hours,k,Xi,x0].')
    if p[0] <= 0 or p[1] <= 0 or not np.all((p[2:] >= 0) & (p[2:] <= 1)):
        raise ValueError('Invalid BM1 parameters.')
    return p


def linear_demand(x, u, baseline, k, xi):
    """Scalar specialization of bm.l/bm.D with f=g=1-2*argument."""
    delta = tanh(k * (1.0 - x - u))
    return baseline + xi * delta * ((1.0 - baseline) if delta >= 0 else baseline)


def exact_linear_interval(x, baseline, u, hours, C_hours, k, xi):
    """F1 reference for genuinely constant inputs; not used for WM baseline fit."""
    _parameters((C_hours, k, xi, x))
    if not (0 <= baseline <= 1 and 0 <= u <= 1 and np.isfinite(hours) and hours > 0):
        raise ValueError('Invalid constant interval.')
    q = 1.0 - u
    if xi == 0 or x == q:
        return x, baseline
    gamma = 1.0 - baseline if x < q else baseline
    # Declared G bounds k<=10 keep sinh away from overflow.
    if abs(k * (x - q)) > 700:
        raise ValueError('Exact reference argument exceeds supported range.')
    next_x = q + asinh(sinh(k * (x - q)) * exp(-k * xi * gamma * hours / C_hours)) / k
    return next_x, baseline + C_hours * (next_x - x) / hours


class BM1DeterministicInputs:
    """One continuous hourly trajectory and linearly interpolated fine baseline.

    prices[j] is held on [j,j+1); baseline covers both endpoints. observations
    are optional: preparing a forecast does not need any future demand.
    """
    def __init__(self, baseline_time, baseline, prices, observations=None, *,
                 observation_contract='hourly-point', warmup_hours=0, steps_per_hour=12, observation_mask=None):
        self.time = np.asarray(baseline_time, dtype=float).copy()
        self.baseline = np.asarray(baseline, dtype=float).copy()
        self.prices = np.asarray(prices, dtype=float).copy()
        self.n_hours = self.prices.size
        if self.time.ndim != 1 or self.baseline.shape != self.time.shape or self.time.size < 2:
            raise ValueError('Matching baseline time/value vectors are required.')
        if self.prices.ndim != 1 or self.n_hours < 1:
            raise ValueError('At least one hourly price is required.')
        if not np.all(np.isfinite(self.time)) or np.any(np.diff(self.time) <= 0):
            raise ValueError('Baseline time must be finite and strictly increasing.')
        if self.time[0] != 0 or self.time[-1] != self.n_hours:
            raise ValueError('Baseline must cover [0, number of price hours].')
        if not np.all(np.isfinite(self.baseline)) or np.any((self.baseline < 0) | (self.baseline > 1)):
            raise ValueError('Baseline must lie in [0,1].')
        if not np.all(np.isfinite(self.prices)) or np.any((self.prices < 0) | (self.prices > 1)):
            raise ValueError('BM prices must lie in [0,1]; no clipping.')
        if observation_contract not in ('hourly-point', 'hourly-mean'):
            raise ValueError('Declare point or interval-mean observations.')
        if int(warmup_hours) != warmup_hours or not 0 <= warmup_hours < self.n_hours:
            raise ValueError('Invalid warm-up length.')
        if isinstance(steps_per_hour, bool) or int(steps_per_hour) != steps_per_hour or steps_per_hour < 1:
            raise ValueError('steps_per_hour must be a positive integer.')
        self.contract, self.warmup = observation_contract, int(warmup_hours)
        self.steps = int(steps_per_hour)
        self.observations = None if observations is None else np.asarray(observations, dtype=float).copy()
        self.observation_mask = np.ones(self.n_hours,dtype=bool) if observation_mask is None else np.asarray(observation_mask,dtype=bool).copy()
        if self.observation_mask.shape != self.prices.shape:
            raise ValueError('One observation mask value per hour is required.')
        if self.observations is not None and (self.observations.shape != self.prices.shape or not np.all(np.isfinite(self.observations[self.observation_mask]))):
            raise ValueError('Scored observations must be finite; mark unmeasured hours explicitly.')
        self.h = 1. / self.steps
        times = np.arange(self.n_hours * self.steps + 1) * self.h
        self.b_edge = np.interp(times, self.time, self.baseline)
        self.b_mid = np.interp(times[:-1] + self.h / 2, self.time, self.baseline)
        self.u_step = np.repeat(self.prices, self.steps)
        # Exact integral of the declared piecewise-linear baseline per hour.
        area = .5 * (self.baseline[1:] + self.baseline[:-1]) * np.diff(self.time)
        cumulative = np.r_[0., np.cumsum(area)]
        # All hour edges must exist for this exact integration contract.
        indices = np.searchsorted(self.time, np.arange(self.n_hours + 1))
        if not np.allclose(self.time[indices], np.arange(self.n_hours + 1), atol=1e-12, rtol=0):
            raise ValueError('Baseline grid must include each hour edge.')
        self.b_mean = np.diff(cumulative[indices])

    def refined(self):
        return BM1DeterministicInputs(self.time, self.baseline, self.prices, self.observations,
                                     observation_contract=self.contract, warmup_hours=self.warmup,
                                     steps_per_hour=2 * self.steps, observation_mask=self.observation_mask)

    def solve(self, parameters, *, deadline=None, keep_fine=False):
        self.last_rhs_evaluations = 0
        C, k, xi, x = _parameters(parameters)
        # Conservative RK4 stability guard at maximum local restoring slope.
        if k * xi * self.h / C > 1.0:
            raise ValueError('Refine the declared RK4 grid for these bounds.')
        states = np.empty(self.n_hours + 1)
        states[0] = x
        fine_states = np.empty(len(self.u_step)+1) if keep_fine else None
        if keep_fine: fine_states[0] = x
        h = self.h
        for j, u in enumerate(self.u_step):
            if j % (24 * self.steps) == 0 and deadline is not None and perf_counter() >= deadline:
                raise BudgetStop('wall-time limit')
            b0, bm, b1 = self.b_edge[j], self.b_mid[j], self.b_edge[j+1]
            self.last_rhs_evaluations += 4
            v1 = (linear_demand(x, u, b0, k, xi) - b0) / C
            v2 = (linear_demand(x + h*v1/2, u, bm, k, xi) - bm) / C
            v3 = (linear_demand(x + h*v2/2, u, bm, k, xi) - bm) / C
            # Right endpoint uses the left interval's held price.
            v4 = (linear_demand(x + h*v3, u, b1, k, xi) - b1) / C
            x += h * (v1 + 2*v2 + 2*v3 + v4) / 6
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError('RK4 state left [0,1]; no clipping is applied.')
            if keep_fine: fine_states[j+1] = x
            if (j+1) % self.steps == 0:
                states[(j+1)//self.steps] = x
        point_b = np.interp(np.arange(self.n_hours), self.time, self.baseline)
        point = np.array([linear_demand(x, u, b, k, xi) for x, u, b in zip(states[:-1], self.prices, point_b)])
        mean = self.b_mean + C * np.diff(states)
        demand = point if self.contract == 'hourly-point' else mean
        result = {'state': states, 'demand': demand, 'point_demand': point,
                  'mean_demand': mean, 'rhs_evaluations': 4*len(self.u_step)}
        if keep_fine:
            fine_u = np.r_[self.u_step, self.prices[-1]]
            result.update(fine_time=np.arange(len(fine_states))*self.h, fine_state=fine_states,
                          fine_demand=np.array([linear_demand(x,u,b,k,xi)
                                                for x,u,b in zip(fine_states,fine_u,self.b_edge)]))
        return result


def fit_deterministic(inputs, config=None, *, checkpoint=None, progress=None):
    """Bounded TRF with explicit counts including every finite-difference solve."""
    config = config or BM1DeterministicConfig()
    if inputs.observations is None:
        raise ValueError('Fitting requires observed demand.')
    started = perf_counter()
    deadline = started + config.seconds
    counts = {'residual_calls': 0, 'forward_solves': 0, 'rhs_evaluations': 0}
    completed, provisional, best = [], None, None
    last_save = started
    mask = inputs.observation_mask.copy()
    mask[:inputs.warmup] = False
    n_observed = int(mask.sum())
    if not n_observed or not np.isfinite(inputs.observations[mask]).all():
        raise ValueError('Fit needs finite measured demand after warm-up.')
    lower, upper = np.array(config.lower), np.array(config.upper)
    # log C/log k; amplitudes/state remain in their bounded physical domains.
    active = 4 if config.fixed_x0 is None else 3
    coordinate_names = ['log_C_hours','log_k','Xi','x0'][:active]
    def pack(p): return np.r_[np.log(p[:2]), p[2:active]]
    def unpack(p):
        values = np.r_[np.exp(p[:2]), p[2:]]
        return values if active == 4 else np.r_[values, config.fixed_x0]

    def snapshot(status):
        return {'status': status, 'counts': dict(counts), 'seconds': perf_counter()-started,
                'starts': list(completed), 'best_converged': best, 'best_provisional': provisional,
                'optimized_coordinates': coordinate_names, 'fixed_x0': config.fixed_x0,
                'observed_samples':n_observed, 'unscored_hours':inputs.n_hours-n_observed}

    for start_index, start in enumerate(config.starts):
        begin_calls = counts['residual_calls']
        def residual(v):
            nonlocal provisional, last_save
            if counts['residual_calls'] >= config.max_calls:
                raise BudgetStop('total residual-call limit')
            if counts['residual_calls'] - begin_calls >= config.calls_per_start:
                raise BudgetStop('per-start residual-call limit')
            if perf_counter() >= deadline:
                raise BudgetStop('wall-time limit')
            counts['residual_calls'] += 1
            counts['forward_solves'] += 1
            p = unpack(v)
            try:
                out = inputs.solve(p, deadline=deadline)
            finally:
                counts['rhs_evaluations'] += inputs.last_rhs_evaluations
            values = (out['demand'] - inputs.observations)[mask]
            mse = float(np.mean(values**2))
            if provisional is None or mse < provisional['mse']:
                provisional = {'parameters': p.tolist(), 'mse': mse, 'start': start_index}
            now = perf_counter()
            if checkpoint is not None and now-last_save >= 2.:
                checkpoint(snapshot('running'))
                last_save = now
            if progress is not None and counts['residual_calls'] % 25 == 0:
                progress(snapshot('running'))
            return values / np.sqrt(len(values))
        try:
            result = least_squares(residual, pack(np.asarray(start)), bounds=(pack(lower), pack(upper)),
                                   method='trf', jac='2-point', diff_step=1e-4, loss='linear',
                                   x_scale='jac', ftol=1e-7, xtol=1e-7, gtol=1e-7,
                                   max_nfev=config.calls_per_start)
            record = {'start': start_index, 'parameters': unpack(result.x).tolist(),
                      'mse': float(np.mean((result.fun*np.sqrt(n_observed))**2)),
                      'converged': bool(result.success), 'stop': result.message,
                      'residual_calls': counts['residual_calls']-begin_calls,
                      'active_bounds': result.active_mask.tolist()}
            if result.success:
                singular = np.linalg.svd(result.jac, compute_uv=False)
                record['scaled_jacobian_singular_values'] = singular.tolist()
                record['local_rank'] = int(np.count_nonzero(singular > singular[0]*1e-8)) if singular[0] else 0
                if best is None or record['mse'] < best['mse']:
                    best = dict(record)
            completed.append(record)
        except BudgetStop as exc:
            completed.append({'start': start_index, 'converged': False, 'stop': str(exc),
                              'residual_calls': counts['residual_calls']-begin_calls})
            if 'per-start' not in str(exc):
                break
        if checkpoint is not None:
            checkpoint(snapshot('between-starts'))
    status = 'converged' if best is not None else 'unfinished'
    answer = snapshot(status)
    if checkpoint is not None:
        checkpoint(answer)
    return answer

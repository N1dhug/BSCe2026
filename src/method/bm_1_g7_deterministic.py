"""G7 deterministic controls and independent energy/QP checks.

Residual least squares uses the original BM evaluator; matched control compares
BM hours with unchanged WM4 day dynamics. Their different objectives stay explicit.
"""
from copy import deepcopy
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.model.wm_4 import wm_4
from src.params.bm_1_response import BM1PiecewiseResponse


class ControlBudgetStop(RuntimeError):
    """An explicit call/time cap, not optimizer convergence."""


class DeterministicResidualControl:
    """Weighted tracking residuals with true forward counts and exact caching.

All finite-difference probes count. SciPy's nfev alone excludes numerical
Jacobian probes. Deadlines are cooperative: a running ODE returns before
the post-call deadline check; the research runner also has a process timer.
"""

    def __init__(self, evaluator, *, max_forwards, deadline, on_forward=None):
        if evaluator.terminal_weight != 0 or evaluator.price_smoothing_weight != 0:
            raise ValueError('This adapter represents pure tracking only.')
        if (isinstance(max_forwards, bool) or int(max_forwards) != max_forwards
                or max_forwards < 1 or not np.isfinite(deadline)):
            raise ValueError('Declare a positive integer cap and finite deadline.')
        self.evaluator = evaluator
        self.max_forwards = int(max_forwards)
        self.deadline = float(deadline)
        self.on_forward = on_forward
        self.forward_calls = 0
        self.cache_hits = 0
        self.records = []
        self._cache = {}

    def evaluate(self, prices):
        prices = self.evaluator.validate_prices(prices)
        if perf_counter() >= self.deadline:
            raise ControlBudgetStop('Deterministic control time cap.')
        key = tuple(float(v) for v in prices)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if self.forward_calls >= self.max_forwards:
            raise ControlBudgetStop('Deterministic control forward cap.')
        self.forward_calls += 1
        started = perf_counter()
        result = self.evaluator.evaluate(prices)
        residual = np.sqrt(self.evaluator.widths) * (
            result.objective.expected_demand - self.evaluator.target)
        cost = float(residual @ residual)
        if not np.isclose(cost, result.total, rtol=1e-12, atol=1e-13):
            raise RuntimeError('Residual cost disagrees with legacy objective.')
        if (not np.all(np.isfinite(residual)) or not np.all(np.isfinite(result.state))
                or np.min(result.state) < 0 or np.max(result.state) > 1):
            raise RuntimeError('Non-finite residual or invalid raw state; no clipping.')
        residual.setflags(write=False)
        item = (residual, result)
        self._cache[key] = item
        record = dict(prices=prices.tolist(), objective=cost,
                      seconds=perf_counter()-started, forward=self.forward_calls)
        self.records.append(record)
        if self.on_forward is not None:
            self.on_forward(record)
        if perf_counter() >= self.deadline:
            raise ControlBudgetStop('Time cap reached after completed forward.')
        return item

    def residual(self, prices):
        return self.evaluate(prices)[0].copy()

    def search(self, start, *, max_new_forwards, diff_step=1e-4):
        """One bounded TRF start; return an honest stop and its best incumbent."""
        start = self.evaluator.validate_prices(start).copy()
        if (isinstance(max_new_forwards, bool)
                or int(max_new_forwards) != max_new_forwards or max_new_forwards < 1):
            raise ValueError('Each start requires a positive integer forward cap.')
        if not np.isfinite(diff_step) or diff_step <= 0:
            raise ValueError('Finite positive difference step required.')
        before = self.forward_calls
        local_limit = before + int(max_new_forwards)
        best = None

        def fun(prices):
            nonlocal best
            key = tuple(float(v) for v in prices)
            if key not in self._cache and self.forward_calls >= local_limit:
                raise ControlBudgetStop('Per-start forward cap.')
            residual, evaluation = self.evaluate(prices)
            if best is None or evaluation.total < best.total:
                best = evaluation
            return residual.copy()

        try:
            fit = least_squares(fun, start, bounds=(0., 1.), method='trf',
                jac='2-point', diff_step=diff_step, loss='linear', x_scale=1.,
                ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=max_new_forwards)
            # Always report the optimizer iterate, not a Jacobian probe, on convergence.
            residual, final = self.evaluate(fit.x)
            return dict(status='converged' if fit.success else 'optimizer_stop',
                success=bool(fit.success), prices=fit.x.tolist(), objective=final.total,
                scipy_half_cost=float(fit.cost), optimality=float(fit.optimality),
                scipy_status=int(fit.status), message=str(fit.message),
                nfev=int(fit.nfev), njev=int(fit.njev),
                new_forwards=self.forward_calls-before,
                factor_two_error=abs(2*float(fit.cost)-float(residual@residual)))
        except ControlBudgetStop as exc:
            return dict(status='budget_stop', success=False, message=str(exc),
                prices=None if best is None else best.prices.tolist(),
                objective=None if best is None else best.total,
                new_forwards=self.forward_calls-before)


class MatchedDeterministicControl:
    def __init__(self, packet):
        if packet.get('schema') != 'g7-matched-fixed-target-v1':
            raise ValueError('Unsupported matched control packet.')
        self.time = np.asarray(packet['baseline_time_hours'], dtype=float).copy()
        self.baseline = np.asarray(packet['baseline'], dtype=float).copy()
        self.target = np.asarray(packet['target'], dtype=float).copy()
        self.edges = np.asarray(packet['price_edges_hours'], dtype=float).copy()
        if (self.time.ndim != 1 or self.time.size < 2 or self.time[0] != 0
                or not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0)):
            raise ValueError('Ordered finite baseline grid starting at zero required.')
        self.hours = int(self.time[-1])
        if self.hours != self.time[-1] or self.hours < 1:
            raise ValueError('Integer physical-hour horizon required.')
        for values in (self.baseline, self.target):
            if (values.shape != self.time.shape or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))):
                raise ValueError('Shared baseline/target must be finite in [0,1].')
        if (self.edges.ndim != 1 or self.edges.size < 2
                or not np.isfinite(self.edges).all() or self.edges[0] != 0
                or self.edges[-1] != self.hours or np.any(np.diff(self.edges) <= 0)
                or np.any(self.edges != np.floor(self.edges))):
            raise ValueError('Price edges must be ordered integer hours covering horizon.')
        self.scales = dict(packet['scales'])
        if not all(np.isfinite(self.scales[k]) for k in
                   ('demand_min', 'demand_scale', 'price_min', 'price_scale')):
            raise ValueError('Finite shared affine scales required.')
        if self.scales['demand_scale'] <= 0 or self.scales['price_scale'] <= 0:
            raise ValueError('Positive scale widths required.')
        self.models = deepcopy(packet['models'])
        for values in (self.time, self.baseline, self.target, self.edges):
            values.setflags(write=False)

    def hourly_prices(self, block_prices):
        u = np.asarray(block_prices, dtype=float)
        if (u.shape != (len(self.edges)-1,) or not np.isfinite(u).all()
                or np.any((u < 0) | (u > 1))):
            raise ValueError('One bounded price per shared block required; no clipping.')
        blocks = np.searchsorted(self.edges, np.arange(self.hours), side='right')-1
        return u[blocks]

    def evaluate(self, name, block_prices, steps_per_hour):
        if (isinstance(steps_per_hour, bool) or int(steps_per_hour) != steps_per_hour
                or steps_per_hour < 2):
            raise ValueError('At least two integer steps per physical hour required.')
        steps = int(steps_per_hour)
        hourly_u = self.hourly_prices(block_prices)
        times = np.arange(self.hours*steps+1, dtype=float)/steps
        midpoints = .5*(times[:-1]+times[1:])
        widths = np.diff(times)
        baseline = np.interp(times, self.time, self.baseline)
        record = self.models[name]
        if record['family'] == 'BM1':
            g = BM1PiecewiseResponse.from_record(record['g'])
            f = BM1PiecewiseResponse.linear(pieces=2)
            inputs = BM1ResponseInputs(self.time, self.baseline, hourly_u,
                                       state_response=f, price_response=g,
                                       steps_per_hour=steps)
            C, k, xi = record['parameters']
            forward = inputs.solve([C, k, xi, record['x_origin']], keep_fine=True)
            state = forward['fine_state'][:, None]
            x_mid = .5*(state[:-1, 0]+state[1:, 0])
            delta = np.tanh(.5*k*(f(x_mid)+g(np.repeat(hourly_u, steps))))
            b_mid = np.interp(midpoints, self.time, self.baseline)
            demand = b_mid+xi*delta*np.where(delta >= 0, 1-b_mid, b_mid)
            state_names = ['x']
            valid_state = bool(np.all((state >= 0) & (state <= 1)))
        elif record['family'] == 'WM4':
            # r is unchanged: d/d(hours)=(1/24)*d/d(days), through grid alone.
            model = wm_4((0., self.hours/24.), len(times), lambda _t: .5)
            signed = 2*np.r_[np.repeat(hourly_u, steps), hourly_u[-1]]-1
            forward = model._solve_deterministic(record['state_origin'], signed,
                                                np.zeros_like(times), times,
                                                baseline, record['r'])
            state = np.column_stack((forward['F'], forward['R']))
            demand = np.interp(midpoints, self.time, self.baseline)+.5*(state[:-1, 1]+state[1:, 1])
            state_names = ['F', 'R']
            valid_state = bool(np.all((state[:, 0] >= -1) & (state[:, 0] <= 1)))
        else:
            raise ValueError('Only frozen BM1 and WM4 deterministic families supported.')
        target = np.interp(midpoints, self.time, self.target)
        if not valid_state or not np.isfinite(state).all() or not np.isfinite(demand).all():
            raise ValueError('Invalid raw trajectory; no clipping or state reset.')
        # WM4 has no flow clipping; demand outside common scale is a domain failure.
        if np.any((demand < 0) | (demand > 1)):
            raise ValueError('Demand outside the declared shared normalization domain.')
        residual = np.sqrt(widths)*(demand-target)
        normalized_cost = float(residual@residual)
        return dict(model=name, steps_per_hour=steps, time_hours=times,
                    state=state, state_names=state_names, midpoints=midpoints,
                    demand=demand, residual=residual, target=target,
                    objective_normalized=normalized_cost,
                    objective_raw=normalized_cost*self.scales['demand_scale']**2,
                    demand_energy_normalized=float(widths@demand),
                    target_energy_normalized=float(widths@target),
                    block_prices=np.asarray(block_prices, dtype=float).copy())


def energy_gap(forward):
    """Same midpoint measure as tracking; A has the frozen integral of B."""
    return forward['demand_energy_normalized']-forward['target_energy_normalized']


def box_quadratic(H, c, equality=None):
    """Exhaust all 3**n box faces for min ||c+H u||², n<=4.

    Each face solves its equality-constrained stationary equations. A finite
    feasible minimum exists on some face; positive-definite H.T@H ensures a
    unique free-face solution. The returned KKT residual is independent of
    the active-face search. No generic QP dependency or iterative tolerance.
    """
    H, c = np.asarray(H, float), np.asarray(c, float)
    n = H.shape[1]
    if not 1 <= n <= 4 or H.shape[0] != len(c):
        raise ValueError('This small enumerator requires 1..4 controls.')
    Q, q = 2*H.T@H, 2*H.T@c
    eigenvalues = np.linalg.eigvalsh(Q)
    if not np.isfinite(Q).all() or eigenvalues[0] <= 1e-12*eigenvalues[-1]:
        raise ValueError('A well-conditioned strictly convex control QP is required.')
    a, b = (None, None) if equality is None else equality
    if a is not None:
        a = np.asarray(a, float)
    candidates = []
    for face in product((-1, 0, 1), repeat=n):  # -1 free; 0/1 fixed bound
        free = np.flatnonzero(np.array(face) == -1)
        fixed = np.flatnonzero(np.array(face) != -1)
        u = np.array([max(0, v) for v in face], dtype=float)
        if free.size:
            A = Q[np.ix_(free, free)]
            rhs = -q[free]-Q[np.ix_(free, fixed)]@u[fixed]
            if a is None:
                u[free] = np.linalg.solve(A, rhs)
            else:
                ae = a[free]
                K = np.block([[A, ae[:, None]], [ae[None, :], np.zeros((1, 1))]])
                rhs = np.r_[rhs, b-a[fixed]@u[fixed]]
                sol, _, _, _ = np.linalg.lstsq(K, rhs, rcond=None)
                if np.max(np.abs(K@sol-rhs)) > 1e-10:
                    continue
                u[free] = sol[:-1]
        if np.any(u < 0) or np.any(u > 1):
            continue
        if a is not None and abs(a@u-b) > 1e-10:
            continue
        candidates.append((float(np.linalg.norm(c+H@u)**2), u))
    if not candidates:
        raise ValueError('No feasible box/equality face.')
    cost, u = min(candidates, key=lambda x: x[0])
    grad = Q@u+q
    interior = (u > 1e-8) & (u < 1-1e-8)
    if a is None:
        multiplier = 0.
    elif np.any(interior):
        multiplier = -float(a[interior]@grad[interior]) / float(a[interior]@a[interior])
    else:
        lower, upper = -float('inf'), float('inf')
        for aj, gj, uj in zip(a, grad, u):
            if aj == 0:
                continue
            bound = -gj/aj
            if (uj <= 1e-8 and aj > 0) or (uj >= 1-1e-8 and aj < 0):
                lower = max(lower, bound)
            else:
                upper = min(upper, bound)
        if lower > upper+1e-8:
            raise ValueError('No equality multiplier satisfies bound KKT conditions.')
        multiplier = min(max(0., lower), upper)
    lag = grad if a is None else grad+multiplier*a
    violation = np.where(u <= 1e-8, np.minimum(lag, 0),
                         np.where(u >= 1-1e-8, np.maximum(lag, 0), lag))
    return dict(prices=u.tolist(), objective=cost, eigenvalues=eigenvalues.tolist(),
                faces=3**n, feasible_faces=len(candidates), multiplier=multiplier,
                kkt_residual=float(np.max(np.abs(violation))),
                equality_error=0. if a is None else float(abs(a@u-b)))

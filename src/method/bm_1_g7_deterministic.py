"""G7 residual adapter for the original BM fixed-supply deterministic OCP.

The shared evaluator supplies all ODE dynamics and midpoint quadrature.
Only prices are optimized. No WM purchasing map or fitted parameters enter.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares


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

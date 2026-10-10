"""Deterministic G response on the existing RK4 input grid.

State and price responses use the shared monotone piecewise-linear geometry.
"""
from bisect import bisect_right
from math import tanh
from time import perf_counter
import numpy as np

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

    def solve(self, parameters, *, deadline=None, keep_fine=False, price_response=None, state_response=None):
        state_curve = self.state_response if state_response is None else state_response
        g = self.price_response if price_response is None else price_response
        if not all(isinstance(curve, BM1PiecewiseResponse) for curve in (state_curve, g)):
            raise ValueError('Use validated shared response geometry.')
        if is_linear(state_curve) and is_linear(g):
            return super().solve(parameters, deadline=deadline, keep_fine=keep_fine)
        self.last_rhs_evaluations = 0
        C, k, xi, x = _parameters(parameters)
        slope = float(np.max(np.abs(np.diff(state_curve.values)/np.diff(state_curve.knots))))
        if .5*k*xi*self.h*slope/C > 1:
            raise ValueError('Refine the declared RK4 grid for this state-response slope.')
        f = _scalar_response(state_curve)
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

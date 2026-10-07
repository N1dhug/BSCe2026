"""G-only fixed-target contract; original BM RK4 and WM4 Euler are reused.

One common price coordinate u in [0,1] means WM's signed U=2*u-1.
Physical time is hours; WM's existing solver receives a days grid.
No fitting, clipping, acquisition map or terminal penalty is introduced.
"""
from copy import deepcopy
import numpy as np

from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.model.wm_4 import wm_4
from src.params.bm_1_response import BM1PiecewiseResponse


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

"""BM3: BM1 demand response with additive noise and soft state limits.

dX = [(D-B)/C + R(X)] dt + sigma_x dW.
R(X) = lambda * ((-X)_+**2 - (X-1)_+**2).
X is unrestricted mathematically; demand remains normalized in [0,1].
CR is an explicit auxiliary energy flow, separately reported from demand.
The FV domain is a numerical truncation, never the physical [0,1] interval.
"""
import numpy as np
from scipy.integrate import solve_ivp
from src.model.bm_1 import bm_1
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation.model.bm_3 import configuration, finite_state


class bm_3(bm_1):
    def __init__(self, B, A, T, t_steps, *, restoring_strength=10.0,
                 state_bounds=(-1.0, 2.0), tail_tolerance=1e-4):
        super().__init__(B, A, T, t_steps)
        self.restoring_strength, self.state_bounds, self.tail_tolerance = configuration(
            restoring_strength, state_bounds, tail_tolerance)
        self.fokker_planc_reference = finite_volume_1d_cached()

    @staticmethod
    def restoring(x, strength):
        x = finite_state(x)
        return strength * (np.maximum(-x, 0.0)**2 - np.maximum(x-1.0, 0.0)**2)

    def _drift(self, t, X, parameters):
        _, b, _, _, _, d = self.eval(t, X, parameters[0], parameters[1], parameters[2],
                                     parameters[4], parameters[5])
        return (d-b)/parameters[3] + self.restoring(X, self.restoring_strength)

    def dXdt(self, t, state, U, f, g, C, k, Xi):
        state = finite_state(state)
        return self._drift(t, state, (U, f, g, C, k, Xi))

    @staticmethod
    def _diffusion(X, sigma_X):
        x = finite_state(X)
        if not np.isfinite(sigma_X) or sigma_X < 0:
            raise ValueError('BM3 sigma_x must be finite and nonnegative.')
        return np.full_like(x, float(sigma_X))

    def deterministic_solver(self, init_cond, parameters, *, t_eval=None, breakpoints=None,
                             rtol=1e-7, atol=1e-9, max_step=0.25):
        """Same return layout as BM1; permit finite initial states outside [0,1]."""
        times = self.t.copy() if t_eval is None else np.asarray(t_eval, dtype=float)
        x0 = finite_state(init_cond).reshape(-1)
        if x0.size != 1 or times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('BM3 requires one initial state and increasing finite output times.')
        if times[0] < self.T[0] or times[-1] > self.T[1]:
            raise ValueError('Output times must lie in the model time interval.')
        knots = np.asarray([] if breakpoints is None else breakpoints, dtype=float)
        if not np.isfinite(knots).all() or np.any((knots <= self.T[0]) | (knots >= self.T[1])):
            raise ValueError('Breakpoints must be finite and strictly inside the model interval.')
        edges = np.unique(np.r_[self.T[0], knots, self.T[1]])
        X = np.empty(len(times))
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            def rhs(t, x):
                query = np.nextafter(right, left) if t >= right and right < self.T[1] else t
                return self._drift(query, x, parameters)
            solution = solve_ivp(rhs, (left, right), x0, dense_output=True,
                                 rtol=rtol, atol=atol, max_step=max_step)
            if not solution.success:
                raise RuntimeError(solution.message)
            keep = (times >= left) & ((times <= right) if i == len(edges)-2 else (times < right))
            X[keep] = solution.sol(times[keep])[0]
            x0 = solution.y[:, -1]
        u, b, f, g, delta, d = self.eval(times, X, parameters[0], parameters[1], parameters[2],
                                        parameters[4], parameters[5])
        return times.copy(), X, u, b, f, g, delta, d

    def initial_mass(self, states, x_steps):
        """Deposit empirical atoms between adjacent FV centers, preserving their mean."""
        _, x, _ = self.fokker_planc_reference.grid(x_steps, self.state_bounds)
        states = finite_state(states).reshape(-1)
        if np.any((states < x[0]) | (states > x[-1])):
            raise ValueError('Initial atoms lie outside the numerical grid; widen state_bounds.')
        mass = np.zeros(len(x))
        for state in states:
            j = int(np.searchsorted(x, state))
            if j == 0: mass[0] += 1.0/len(states)
            elif j == len(x): mass[-1] += 1.0/len(states)
            else:
                weight = (state-x[j-1])/(x[j]-x[j-1])
                mass[j-1] += (1-weight)/len(states)
                mass[j] += weight/len(states)
        return mass

    def fokker_planc_reference_solver(self, initial_mass, parameters, sigma_x, *,
                                     t_eval=None, breakpoints=None, max_step=0.25):
        times = self.t if t_eval is None else np.asarray(t_eval, dtype=float)
        self._diffusion(0.0, sigma_x)
        result = self.fokker_planc_reference.solve(
            times, initial_mass, lambda t, x: self._drift(t, x, parameters),
            lambda t, x: self._diffusion(x, sigma_x), bounds=self.state_bounds,
            breakpoints=breakpoints, max_step=max_step,
            coefficient_key=lambda t: (float(self.B(t)), float(parameters[0](t))))
        x, mass = result['x_grid'], result['cell_mass']
        mean = mass @ x
        variance = np.maximum(mass @ (x*x)-mean*mean, 0)
        n_edge = max(1, int(np.ceil(len(x)*0.02)))
        tails = mass[:, :n_edge].sum(axis=1)+mass[:, -n_edge:].sum(axis=1)
        diagnostics = {**result['diagnostics'], 'state_bounds': list(self.state_bounds),
            'maximum_numerical_tail_mass': float(tails.max()), 'tail_tolerance': self.tail_tolerance,
            'tail_check_passed': bool(tails.max() <= self.tail_tolerance),
            'outside_probability': mass[:, (x < 0) | (x > 1)].sum(axis=1).tolist(),
            'auxiliary_flow_mean': (parameters[3]*(mass @ self.restoring(x, self.restoring_strength))).tolist()}
        if not diagnostics['tail_check_passed']:
            raise RuntimeError(f"BM3 numerical boundary mass {tails.max():.3g} exceeds "
                               f"{self.tail_tolerance:g}; widen --state-min/--state-max and refine --x-steps.")
        expected = []
        for i, t in enumerate(times):
            expected.append(float(np.dot(self.eval(t, x, parameters[0], parameters[1], parameters[2],
                                                   parameters[4], parameters[5])[-1], mass[i])))
        return (times.copy(), x.copy(), result['density'], mass, mean, variance,
                np.asarray(parameters[0](times)), np.asarray(self.B(times)), np.asarray(expected), diagnostics)

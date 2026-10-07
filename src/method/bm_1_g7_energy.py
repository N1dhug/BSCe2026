"""G-only energy-neutral four-block control; legacy dynamics are unchanged."""
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import minimize


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


def affine_box_range(base, slopes):
    """Exact coordinatewise extrema over the entire price box, no sampling."""
    return base+np.minimum(slopes, 0).sum(axis=-1), base+np.maximum(slopes, 0).sum(axis=-1)


class CachedSearch:
    def __init__(self, control, name, steps=60, cap=400, deadline=float('inf')):
        self.control, self.name, self.steps = control, name, steps
        self.cap, self.deadline = cap, deadline
        self.cache, self.records = {}, []

    def evaluate(self, u):
        if perf_counter() >= self.deadline:
            raise TimeoutError('Search deadline reached.')
        key = tuple(float(v) for v in u)
        if key not in self.cache:
            if len(self.records) >= self.cap:
                raise RuntimeError('Actual forward cap includes Jacobian probes.')
            f = self.control.evaluate(self.name, u, self.steps)
            self.cache[key] = f
            self.records.append(dict(prices=list(key), J=f['objective_normalized'],
                                     energy_gap=energy_gap(f)))
        return self.cache[key]

    def search(self, start, neutral):
        fun = lambda u: self.evaluate(u)['objective_normalized']
        constraints = ({'type': 'eq', 'fun': lambda u: energy_gap(self.evaluate(u))},) if neutral else ()
        fit = minimize(fun, start, method='SLSQP', bounds=[(0., 1.)]*4,
                       constraints=constraints,
                       options=dict(ftol=1e-11, eps=1e-5, maxiter=60))
        forward = self.evaluate(fit.x)
        return dict(success=bool(fit.success), status=int(fit.status),
                    message=str(fit.message), iterations=int(fit.nit),
                    prices=fit.x.tolist(), objective=fun(fit.x),
                    energy_gap=energy_gap(forward), actual_forwards=len(self.records),
                    trace=self.records), forward


def precision(coarse, fine, neutral):
    """Compare identical prices; midpoint interpolation never crosses a price jump."""
    if not np.array_equal(coarse['block_prices'], fine['block_prices']):
        raise ValueError('Precision comparison must use identical policies.')
    state = float(np.max(np.abs(coarse['state']-fine['state'][::2])))
    # Each coarse midpoint is exactly halfway between its own two fine midpoints.
    demand = float(np.max(np.abs(coarse['demand']-fine['demand'].reshape(-1, 2).mean(axis=1))))
    cost = abs(coarse['objective_normalized']-fine['objective_normalized'])
    energy = abs(energy_gap(fine))
    passed = (state <= 1e-5 and demand <= 1e-5
              and cost <= 1e-4+1e-3*abs(fine['objective_normalized'])
              and (not neutral or energy <= 1e-5))
    return dict(state_error=state, demand_error=demand, cost_error=cost,
                fine_energy_error=energy, passed=bool(passed))

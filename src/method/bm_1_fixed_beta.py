"""Fixed Bernstein-beta Fokker--Planck projection for stochastic BM1.

Follows Nielsen (2026), Sections 3.3 and 4.2: fixed normalized beta
basis, explicit Euler prediction, L2 simplex-QP projection at each step.
Observation updates use a second L2 projection, not moment matching.
"""
import numpy as np
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import norm
from src.solver.active_set import active_set


class bm_1_fixed_beta:
    def __init__(self, degree=6, quadrature=48, steps_per_hour=2, ridge=1e-9, model=None):
        self.degree, self.steps_per_hour = int(degree), int(steps_per_hour)
        self.model = model
        if self.degree < 2 or quadrature < 2 * self.degree + 5 or self.steps_per_hour < 1:
            raise ValueError("Use degree>=2, quadrature>=2*degree+5, steps_per_hour>=1.")
        self.x, self.qw = np.polynomial.legendre.leggauss(int(quadrature))
        self.x, self.qw = (self.x + 1.0) / 2.0, self.qw / 2.0
        self.phi = np.column_stack([self._basis(self.degree, m) for m in range(self.degree + 1)])
        self.dphi = (self.degree + 1) * self.degree * np.column_stack([
            (self._bern(self.degree - 1, m - 1) - self._bern(self.degree - 1, m))
            for m in range(self.degree + 1)])
        self.ddphi = (self.degree + 1) * self.degree * (self.degree - 1) * np.column_stack([
            self._bern(self.degree - 2, m - 2) - 2 * self._bern(self.degree - 2, m - 1) + self._bern(self.degree - 2, m)
            for m in range(self.degree + 1)])
        self.H = self.phi.T @ (self.qw[:, None] * self.phi)
        self.H = (self.H + self.H.T) / 2.0
        self.H += float(ridge) * np.eye(self.degree + 1)
        self.eq = np.ones((1, self.degree + 1))
        self.one = np.array([1.0])
        self.ridge = ridge

    def _bern(self, n, m):
        if m < 0 or m > n: return np.zeros_like(self.x)
        return np.exp(gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)) * self.x**m * (1.0 - self.x)**(n - m)

    def _basis(self, n, m):
        return np.exp(m * np.log(self.x) + (n - m) * np.log1p(-self.x) - betaln(m + 1, n - m + 1))

    def _qp(self, linear, old):
        answer = active_set.solve(self.H, linear, A_eq=self.eq, b_eq=self.one,
                                  lb=np.zeros(self.degree + 1), x0=old,
                                  tol=1e-8, max_iter=300)
        if not np.isfinite(answer).all() or np.min(answer) < -1e-7 or abs(np.sum(answer)-1) > 1e-7:
            raise RuntimeError("Beta projection produced invalid simplex weights.")
        return answer

    @staticmethod
    def demand(x, b, u, p):
        alpha, beta, k, xi = (p[name] for name in ("alpha", "beta", "k", "Xi"))
        fx = np.tanh(.5 * alpha * (.5 - x)) / np.tanh(alpha / 4.)
        gu = np.tanh(.5 * beta * (.5 - u)) / np.tanh(beta / 4.)
        delta = np.tanh(.5 * k * (fx + gu))
        return b + xi * delta * np.where(delta >= 0., 1. - b, b)

    def _fields(self, b, u, p):
        x = self.x
        alpha, beta, k, xi, capacity, sigma = (p[name] for name in ("alpha", "beta", "k", "Xi", "C", "sigma_x"))
        fx = np.tanh(.5 * alpha * (.5 - x)) / np.tanh(alpha / 4.)
        gu = np.tanh(.5 * beta * (.5 - u)) / np.tanh(beta / 4.)
        delta = np.tanh(.5 * k * (fx + gu))
        dfx = -.5 * alpha * (1. - np.tanh(.5 * alpha * (.5 - x))**2) / np.tanh(alpha / 4.)
        ddelta = .5 * k * (1. - delta**2) * dfx
        scale = np.where(delta >= 0., 1. - b, b)
        drift = xi * delta * scale / capacity
        ddrift = xi * ddelta * scale / capacity
        b2 = sigma**2 * x**2 * (1. - x)**2
        db2 = 2. * sigma**2 * x * (1. - x) * (1. - 2. * x)
        ddb2 = 2. * sigma**2 * (1. - 6. * x + 6. * x**2)
        operator = -ddrift[:, None]*self.phi - drift[:, None]*self.dphi
        operator += .5*(ddb2[:, None]*self.phi + 2.*db2[:, None]*self.dphi + b2[:, None]*self.ddphi)
        return self.phi.T @ (self.qw[:, None] * operator)

    def propagate(self, c, b0, b1, u, p):
        dt = 1. / self.steps_per_hour
        for h in range(self.steps_per_hour):
            baseline = b0 + (b1-b0) * (h + .5) * dt
            operator = self._fields(baseline, u, p)
            target = self.H @ c + dt * (operator @ c)
            c = self._qp(-target, c)
        return c

    def observation(self, c, y, b, u, p, update=True):
        demand = self.demand(self.x, b, u, p)
        likelihood = norm.logpdf(y, demand, p["sigma_y"])
        mass = self.qw * (self.phi @ c)
        logprob = logsumexp(np.log(np.maximum(mass, 1e-300)) + likelihood)
        if not np.isfinite(logprob): raise FloatingPointError("Nonfinite observation likelihood.")
        if update:
            density = np.exp(likelihood - logprob) * (self.phi @ c)
            c = self._qp(-(self.phi.T @ (self.qw * density)), c)
        return c, float(logprob)

    def simulate(self, data, p, initial, observations=False):
        b, u, y = (np.asarray(data[name], dtype=float) for name in ("B", "U", "I"))
        if len(b) != len(u) or len(b) != len(y) or len(b) < 2: raise ValueError("Invalid hourly BM1 interval.")
        c = np.asarray(initial, dtype=float).copy()
        if c.shape != (self.degree+1,) or not np.isfinite(c).all() or np.min(c)<0. or abs(c.sum()-1)>1e-7:
            raise ValueError("Initial coefficients must lie on the probability simplex.")
        means, supports, masses, loglik, coefficients = [], [], [], [], []
        for j in range(len(b)):
            density_mass = self.qw * (self.phi @ c)
            density_mass /= density_mass.sum()
            d = self.demand(self.x, b[j], u[j], p)
            means.append(float(density_mass @ d))
            supports.append(d.copy()); masses.append(density_mass.copy()); coefficients.append(c.copy())
            if observations: c, lp = self.observation(c, y[j], b[j], u[j], p)
            else: _, lp = self.observation(c, y[j], b[j], u[j], p, update=False)
            loglik.append(lp)
            if j < len(b)-1: c = self.propagate(c, b[j], b[j+1], u[j], p)
        return {"mean":np.array(means), "supports":supports, "weights":masses, "loglik":np.array(loglik),
                "nll":float(-sum(loglik)), "coefficients":np.array(coefficients)}

    @staticmethod
    def intervals(result, sigma_y):
        lower, upper = [], []
        for support, weight in zip(result["supports"], result["weights"]):
            def cdf(z): return float(weight @ norm.cdf((z-support)/sigma_y))
            from scipy.optimize import brentq
            lo, hi = float(np.min(support)-9*sigma_y), float(np.max(support)+9*sigma_y)
            lower.append(brentq(lambda z:cdf(z)-.025,lo,hi))
            upper.append(brentq(lambda z:cdf(z)-.975,lo,hi))
        return np.asarray(lower), np.asarray(upper)

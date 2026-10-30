from dataclasses import dataclass

import numpy as np

from src.model.base.wm import wm
from src.solver.active_set import active_set
from src.validation.model.wm_5 import wm_5_validation


@dataclass(frozen=True)
class response_function:
    """A centered, nondecreasing basis with *certified* derivative bounds on [-1,1].

    Supplied bounds must hold across the continuous domain, not merely on a grid.
    The validator checks representative points but cannot prove user certificates.
    """
    name: str
    value: object
    derivative: object
    derivative_bounds: tuple


def response_library():
    """Pre-certified centered bases, all monotone increasing on [-1,1]."""
    linear = response_function("linear", lambda x: x, lambda x: 1.0, (1.0, 1.0))
    cubic = response_function("cubic", lambda x: x ** 3, lambda x: 3.0 * x * x, (0.0, 3.0))
    quintic = response_function("quintic", lambda x: x ** 5, lambda x: 5.0 * x ** 4, (0.0, 5.0))
    k_exp = 1.0
    exponential = response_function(
        "exponential", lambda x: np.expm1(k_exp * x) / k_exp,
        lambda x: np.exp(k_exp * x), (float(np.exp(-k_exp)), float(np.exp(k_exp))),
    )
    k_log = 1.0
    logarithmic = response_function(
        "logarithmic",
        lambda x: np.sign(x) * np.log1p(k_log * abs(x)) / k_log,
        lambda x: 1.0 / (1.0 + k_log * abs(x)),
        (1.0 / (1.0 + k_log), 1.0),
    )
    return {basis.name: basis for basis in (linear, cubic, quintic, exponential, logarithmic)}


class wm_5(wm):
    """Deterministic nonlinear WM5 with parameter-affine monotone response bases.

    dF/dt=R, dR/dt=r_F*F + sum(a_i*phi_i(R)) + sum(b_j*psi_j(U)), I=O+R.
    The a_i and b_j are <=0, basis derivatives >=0, and g(0)=h(0)=0.
    Increasing price bases h_j(U) with b_j<=0 give nonpositive price sensitivity.
    A known or reconstructed R0 lets the sequential fitter optimize F0 by QP.
    """

    _FIT_EPS = 1e-8
    _FIT_RIDGE = 1e-10

    def __init__(self, T, t_steps, O, r_functions, u_functions):
        T, t_steps, O = wm_5_validation.constructor(T, t_steps, O)
        self.r_functions = wm_5_validation.basis_functions(r_functions, "r_functions")
        self.u_functions = wm_5_validation.basis_functions(u_functions, "u_functions")
        self.T, self.t_steps, self.O = T, t_steps, O
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = float(self.t[1] - self.t[0])
        self.n_r, self.n_u = len(self.r_functions), len(self.u_functions)
        self.parameter_names = ("r_F",) + tuple("r_" + b.name for b in self.r_functions) + tuple("u_" + b.name for b in self.u_functions)

    def coefficients(self, coefficients, stabilize=True):
        return wm_5_validation.coefficients(coefficients, self.r_functions, self.u_functions, self.dt, stabilize=stabilize)

    def response(self, R, U, coefficients):
        p = np.asarray(coefficients, dtype=float)
        rv = np.array([basis.value(float(R)) for basis in self.r_functions], dtype=float)
        uv = np.array([basis.value(float(U)) for basis in self.u_functions], dtype=float)
        return float(p[1:1+self.n_r] @ rv + p[1+self.n_r:] @ uv)

    def rate_derivative(self, R, coefficients):
        p = np.asarray(coefficients, dtype=float)
        dr = np.array([basis.derivative(float(R)) for basis in self.r_functions], dtype=float)
        return float(p[1:1+self.n_r] @ dr)

    def drift(self, F, R, U, coefficients):
        return np.array([R, float(coefficients[0]) * F + self.response(R, U, coefficients)], dtype=float)

    def expected_equilibrium(self, U, coefficients):
        self.coefficients(coefficients)
        U = wm_5_validation.signal_value(U, "U")
        return -self.response(0.0, U, coefficients) / float(coefficients[0])

    def _solve_deterministic(self, init_cond, U, Time, O, coefficients, with_sensitivity=False):
        init = wm_5_validation.initial_condition(init_cond)
        p = self.coefficients(coefficients)
        U, Time, O = (np.asarray(v, dtype=float) for v in (U, Time, O))
        n = self.t_steps
        if any(len(v) != n for v in (U, Time, O)):
            raise ValueError("WM5 simulation signals must match model.t_steps.")
        if np.any(U < -1.0) or np.any(U > 1.0) or not np.isfinite(np.r_[U, Time, O]).all():
            raise ValueError("WM5 requires finite signals and U in [-1,1].")
        F, R, R_dot = np.empty(n), np.empty(n), np.empty(n)
        F[0], R[0] = init
        S = np.empty((n, 2, 2), dtype=float) if with_sensitivity else None
        if S is not None: S[0] = np.eye(2)
        for k in range(n):
            R_dot[k] = p[0] * F[k] + self.response(R[k], U[k], p)
            if not np.isfinite(R_dot[k]): raise ValueError("WM5 simulation diverged: nonfinite rate drift.")
            if k < n - 1:
                F[k+1] = F[k] + self.dt * R[k]
                R[k+1] = R[k] + self.dt * R_dot[k]
                if S is not None:
                    jacobian = np.array([[1.0, self.dt], [self.dt * p[0], 1.0 + self.dt * self.rate_derivative(R[k], p)]])
                    S[k+1] = jacobian @ S[k]
        state = {"t": self.t.copy(), "Time": Time.copy(), "U": U.copy(), "O": O.copy(),
                 "F": F, "R": R, "R_dot": R_dot, "I": O + R, "coefficients": p.copy()}
        if S is not None: state["initial_sensitivity"] = S
        return state

    def deterministic_solver(self, init_cond, parameters):
        if not isinstance(parameters, dict) or not callable(parameters.get("U")) or not callable(parameters.get("Time")):
            raise TypeError("WM5 parameters require callable 'U' and 'Time' and a 'coefficients' vector.")
        U = np.array([wm_5_validation.signal_value(parameters["U"](tk), "U(t)") for tk in self.t], dtype=float)
        Time = np.array([wm_5_validation.finite_feature_input(parameters["Time"](tk), "Time(t)") for tk in self.t], dtype=float)
        O = np.array([wm_5_validation.feature_value(self.O(tk), "O(Time(t))") for tk in Time], dtype=float)
        return self._solve_deterministic(init_cond, U, Time, O, parameters["coefficients"])

    @staticmethod
    def _least_squares_qp(X, y, A_ub=None, b_ub=None, lb=None, ub=None, ridge=1e-10):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1)
        n, scale = X.shape[1], max(1, len(y))
        H = 2.0 * X.T @ X / scale + ridge * np.eye(n)
        f = -2.0 * X.T @ y / scale
        return active_set.solve(H, f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, tol=1e-9, max_iter=10000)

    def fit_params_joint_qp(self, input_data, initial_conditions=None, **kwargs):
        from src.params.wm_5 import joint_qp_params
        return joint_qp_params(self).fit(input_data, initial_conditions, **kwargs)
    def fit_params_qp(self, input_data, initial_conditions=None, **kwargs):
        from src.params.wm_5 import qp_params
        return qp_params(self).fit(input_data, initial_conditions=initial_conditions, **kwargs)

    def fit_params(self, input_data, initial_conditions=None, **kwargs):
        return self.fit_params_qp(input_data, initial_conditions, **kwargs)

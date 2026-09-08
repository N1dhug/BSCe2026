import numpy as np

from src.models.WM import WM
from src.models.Simulate import Simulate
from src.solvers.FokkerPlanc import FokkerPlanc


class WM1(WM):
    """Updated Water Model 1 from the WM1 appendix note.

    Causal ordering at each time step:
        (B_O, V, U) -> P_O -> O -> P_I -> I -> P_A -> A

    with
        P_O = [B_O, V, 1-U]
        P_I = [B_I, 1-V, O, 1-U]
        P_A = [1-U, I, V, 1-PE].

    The stochastic solver implements the no-inertia centered-error SDE
        dV = D(t, V, U) dt + sigma dW,
    where the additive diffusion is independent of both state and price.
    The centered error density is propagated by the generic Beta-mixture
    Fokker-Planck solver on a transformed finite computational interval.
    """

    def __init__(self, T, t_steps):
        self.simulate = Simulate()
        self.fp = FokkerPlanc()
        self.T = T
        self.t_steps = int(t_steps)
        self.t = np.linspace(float(T[0]), float(T[1]), self.t_steps)
        self.dt = float(self.t[1] - self.t[0])

    @staticmethod
    def _as_vector(values, length, name):
        values = np.asarray(values, dtype=float)
        if values.shape != (length,):
            raise ValueError(f"{name} must contain exactly {length} values.")
        return values

    @classmethod
    def validate_parameters(cls, alpha, beta, gamma, tol=1e-10):
        alpha = cls._as_vector(alpha, 4, "alpha")
        beta = cls._as_vector(beta, 3, "beta")
        gamma = cls._as_vector(gamma, 4, "gamma")

        for name, vector in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if np.any(vector < -tol):
                raise ValueError(f"{name} must be non-negative.")
            if not np.isclose(np.sum(vector), 1.0, atol=tol, rtol=0.0):
                raise ValueError(f"{name} must satisfy sum({name}) = 1.")

        return alpha, beta, gamma

    def I(self, P_I, alpha):
        """Aggregate input/purchased power I = alpha^T P_I.

        P_I = [B_I, 1-V, O, 1-U].
        """
        P_I = self._as_vector(P_I, 4, "P_I")
        alpha = self._as_vector(alpha, 4, "alpha")
        return float(alpha @ P_I)

    def O(self, P_O, beta):
        """Aggregate output/consumption O = beta^T P_O.

        P_O = [B_O, V, 1-U].
        """
        P_O = self._as_vector(P_O, 3, "P_O")
        beta = self._as_vector(beta, 3, "beta")
        return float(beta @ P_O)

    def A(self, P_A, gamma):
        """Aggregator available/bid power A = gamma^T P_A.

        P_A = [1-U, I, V, 1-PE].
        """
        P_A = self._as_vector(P_A, 4, "P_A")
        gamma = self._as_vector(gamma, 4, "gamma")
        return float(gamma @ P_A)

    @staticmethod
    def reduced_drift_coefficients(BtO, BtI, alpha, beta):
        """Return d0, dV, dU such that I-O = d0 + dV*V + dU*U."""
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        o_bar = beta[0] * BtO + beta[2]
        i_bar = alpha[0] * BtI + alpha[1] + alpha[2] * o_bar + alpha[3]

        d0 = i_bar - o_bar
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
        return d0, dV, dU

    def _evaluate_step(self, t, V, parameters):
        U = float(parameters["U"](t))
        BO = float(parameters["B_O"](t))
        BI = float(parameters["B_I"](t))
        PE = float(parameters["PE"](t))

        alpha, beta, gamma = self.validate_parameters(
            parameters["alpha"],
            parameters["beta"],
            parameters["gamma"],
        )

        P_O = np.array([BO, V, 1.0 - U], dtype=float)
        O = self.O(P_O, beta)

        P_I = np.array([BI, 1.0 - V, O, 1.0 - U], dtype=float)
        I = self.I(P_I, alpha)

        P_A = np.array([1.0 - U, I, V, 1.0 - PE], dtype=float)
        A = self.A(P_A, gamma)

        return U, BO, BI, PE, P_I, P_O, P_A, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        """Forward-Euler simulation of updated WM1.

        The dynamic price signal is supplied as parameters['U'].

        Set parameters['theta'] > 0 to retain the original discrete inertial
        rate recurrence

            R_k = I_k - O_k + theta R_{k-1}.

        With theta=0 the no-inertia model is

            V_{k+1} = V_k + dt (I_k - O_k).
        """
        init = np.asarray(init_cond, dtype=float).reshape(-1)
        V0 = float(init[0])
        R_prev = float(init[1]) if init.size > 1 else 0.0
        theta = float(parameters.get("theta", 0.0))

        V = np.empty(self.t_steps, dtype=float)
        R = np.empty(self.t_steps, dtype=float)
        U = np.empty(self.t_steps, dtype=float)
        BO = np.empty(self.t_steps, dtype=float)
        BI = np.empty(self.t_steps, dtype=float)
        PE = np.empty(self.t_steps, dtype=float)
        O = np.empty(self.t_steps, dtype=float)
        I = np.empty(self.t_steps, dtype=float)
        A = np.empty(self.t_steps, dtype=float)

        P_I = np.empty((self.t_steps, 4), dtype=float)
        P_O = np.empty((self.t_steps, 3), dtype=float)
        P_A = np.empty((self.t_steps, 4), dtype=float)

        V[0] = V0

        for k, tk in enumerate(self.t):
            (
                U[k],
                BO[k],
                BI[k],
                PE[k],
                P_I[k],
                P_O[k],
                P_A[k],
                O[k],
                I[k],
                A[k],
            ) = self._evaluate_step(tk, V[k], parameters)

            R[k] = I[k] - O[k] + theta * R_prev

            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]

            R_prev = R[k]

        return {
            "t": self.t.copy(),
            "V": V,
            "R": R,
            "U": U,
            "B_O": BO,
            "B_I": BI,
            "PE": PE,
            "P_I": P_I,
            "P_O": P_O,
            "P_A": P_A,
            "O": O,
            "I": I,
            "A": A,
        }

    def fokker_planc_solver(
        self,
        init_cond,
        parameters,
        sigma,
        degree=20,
        e_bounds=(-1.5, 1.5),
        x_steps=301,
        cfl=0.45,
    ):
        """Propagate the no-inertia stochastic WM1 centered-error density.

        Implements the no-inertia stochastic WM1 formulation

            V_t = m(t; U) + E_t,
            dE_t = dV E_t dt + sigma dW_t.

        The generic FokkerPlanc solver is defined on [0,1], so E is mapped to
        u=(E-e_min)/(e_max-e_min).
        """
        if float(parameters.get("theta", 0.0)) != 0.0:
            raise NotImplementedError(
                "The current WM1 Fokker-Planck implementation is the no-inertia model."
            )

        alpha, beta, _ = self.validate_parameters(
            parameters["alpha"],
            parameters["beta"],
            parameters["gamma"],
        )

        e_min, e_max = map(float, e_bounds)
        if not e_min < 0.0 < e_max:
            raise ValueError("e_bounds must satisfy e_min < 0 < e_max.")

        span = e_max - e_min
        sigma = float(sigma)
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")

        V0 = float(np.asarray(init_cond, dtype=float).reshape(-1)[0])
        mean_result = self.deterministic_solver([V0], parameters)
        mean = mean_result["V"]

        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]

        def drift_u(t, u):
            e = e_min + span * np.asarray(u, dtype=float)
            return dV * e / span

        def diffusion_u(t, u):
            return np.zeros_like(np.asarray(u, dtype=float)) + sigma / span

        u0 = (0.0 - e_min) / span
        u_grid, density_u, coefficients = self.fp.solve(
            self.t,
            [u0],
            drift_u,
            diffusion_u,
            degree=degree,
            x_steps=x_steps,
            cfl=cfl,
        )

        e_grid = e_min + span * u_grid
        density_e = density_u / span
        de = span / len(e_grid)

        error_mean = np.sum(density_e * e_grid[None, :], axis=1) * de
        error_second = np.sum(density_e * (e_grid[None, :] ** 2), axis=1) * de
        error_variance = np.maximum(error_second - error_mean**2, 0.0)

        lower_violation = np.empty(self.t_steps, dtype=float)
        upper_violation = np.empty(self.t_steps, dtype=float)

        for k in range(self.t_steps):
            lower_violation[k] = np.sum(density_e[k, e_grid < -mean[k]]) * de
            upper_violation[k] = np.sum(density_e[k, e_grid > 1.0 - mean[k]]) * de

        result = dict(mean_result)
        result.update(
            {
                "e_grid": e_grid,
                "density_e": density_e,
                "coefficients": coefficients,
                "error_mean": error_mean,
                "error_variance": error_variance,
                "lower_violation": lower_violation,
                "upper_violation": upper_violation,
                "sigma": sigma,
                "e_bounds": (e_min, e_max),
            }
        )
        return result

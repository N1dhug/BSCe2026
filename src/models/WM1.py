import numpy as np

from src.models.WM import WM
from src.models.Simulate import Simulate
from src.solvers.FokkerPlanc import FokkerPlanc


class WM1(WM):
    """Updated Water Model 1 from the WM1 appendix note.

    Causal ordering at each time step:
        (B_O, V, P) -> O -> I -> A

    The stochastic solver implements the no-inertia centered-error SDE
        dV = D(t, V, P) dt + sigma dW,
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
            raise ValueError(f"{name} must contain exactly {length} coefficients.")
        return values

    @classmethod
    def validate_parameters(cls, alpha, beta, gamma, tol=1e-10):
        alpha = cls._as_vector(alpha, 4, "alpha")
        beta = cls._as_vector(beta, 3, "beta")
        gamma = cls._as_vector(gamma, 4, "gamma")
        for name, vec in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if np.any(vec < -tol):
                raise ValueError(f"{name} must be non-negative.")
            if not np.isclose(np.sum(vec), 1.0, atol=tol, rtol=0.0):
                raise ValueError(f"{name} must satisfy sum({name}) = 1.")
        return alpha, beta, gamma

    def Ot(self, Vt, Pt, BtO, beta):
        """Aggregate output/consumption: O = beta_B B_O + beta_V V + beta_P (1-P)."""
        beta = self._as_vector(beta, 3, "beta")
        return beta[0] * BtO + beta[1] * Vt + beta[2] * (1.0 - Pt)

    def It(self, Vt, Pt, BtI, Ot, alpha):
        """Aggregate input/purchased power: I = alpha^T [B_I, 1-V, O, 1-P]."""
        alpha = self._as_vector(alpha, 4, "alpha")
        return (
            alpha[0] * BtI
            + alpha[1] * (1.0 - Vt)
            + alpha[2] * Ot
            + alpha[3] * (1.0 - Pt)
        )

    def At(self, Pt, It, PEt, Vt, gamma):
        """Aggregator available/bid power: A = gamma^T [1-P, I, V, 1-PE]."""
        gamma = self._as_vector(gamma, 4, "gamma")
        return (
            gamma[0] * (1.0 - Pt)
            + gamma[1] * It
            + gamma[2] * Vt
            + gamma[3] * (1.0 - PEt)
        )

    @staticmethod
    def reduced_drift_coefficients(BtO, BtI, alpha, beta):
        """Return d0, dV, dP such that I-O = d0 + dV*V + dP*P."""
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        o_bar = beta[0] * BtO + beta[2]
        i_bar = alpha[0] * BtI + alpha[1] + alpha[2] * o_bar + alpha[3]

        d0 = i_bar - o_bar
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        dP = (1.0 - alpha[2]) * beta[2] - alpha[3]
        return d0, dV, dP

    def _evaluate_step(self, t, V, parameters):
        P = float(parameters["P"](t))
        BO = float(parameters["B_O"](t))
        BI = float(parameters["B_I"](t))
        PE = float(parameters["PE"](t))
        alpha, beta, gamma = self.validate_parameters(
            parameters["alpha"], parameters["beta"], parameters["gamma"]
        )

        O = float(self.Ot(V, P, BO, beta))
        I = float(self.It(V, P, BI, O, alpha))
        A = float(self.At(P, I, PE, V, gamma))
        return P, BO, BI, PE, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        """Forward-Euler simulation of updated WM1.

        Set parameters['theta'] > 0 to retain the original discrete inertial
        rate recurrence R_k = I_k - O_k + theta R_{k-1}. With theta=0 this is
        the no-inertia model V_{k+1}=V_k+dt(I_k-O_k).
        """
        V0 = float(np.asarray(init_cond).reshape(-1)[0])
        R_prev = float(np.asarray(init_cond).reshape(-1)[1]) if np.size(init_cond) > 1 else 0.0
        theta = float(parameters.get("theta", 0.0))

        V = np.empty(self.t_steps, dtype=float)
        R = np.empty(self.t_steps, dtype=float)
        P = np.empty(self.t_steps, dtype=float)
        BO = np.empty(self.t_steps, dtype=float)
        BI = np.empty(self.t_steps, dtype=float)
        PE = np.empty(self.t_steps, dtype=float)
        O = np.empty(self.t_steps, dtype=float)
        I = np.empty(self.t_steps, dtype=float)
        A = np.empty(self.t_steps, dtype=float)

        V[0] = V0
        for k, tk in enumerate(self.t):
            P[k], BO[k], BI[k], PE[k], O[k], I[k], A[k] = self._evaluate_step(tk, V[k], parameters)
            R[k] = I[k] - O[k] + theta * R_prev
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
            R_prev = R[k]

        return {
            "t": self.t.copy(), "V": V, "R": R, "P": P,
            "B_O": BO, "B_I": BI, "PE": PE, "O": O, "I": I, "A": A,
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

        Implements Appendix Section 4:
            V_t = m(t;P) + E_t,
            dE_t = dV E_t dt + sigma dW_t.

        The generic FokkerPlanc solver is defined on [0,1], so E is mapped to
        u=(E-e_min)/(e_max-e_min). This is exactly the transformed finite
        computational interval used by the beta-mixture approximation.
        """
        if float(parameters.get("theta", 0.0)) != 0.0:
            raise NotImplementedError(
                "The current WM1 Fokker-Planck implementation is the no-inertia Section 4 model."
            )

        alpha, beta, _ = self.validate_parameters(
            parameters["alpha"], parameters["beta"], parameters["gamma"]
        )
        e_min, e_max = map(float, e_bounds)
        if not e_min < 0.0 < e_max:
            raise ValueError("e_bounds must satisfy e_min < 0 < e_max.")
        span = e_max - e_min
        sigma = float(sigma)
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")

        mean_result = self.deterministic_solver(
            [float(np.asarray(init_cond).reshape(-1)[0])], parameters
        )
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

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
        self.t = np.linspace(T[0], T[1], self.t_steps)
        self.dt = self.t[1] - self.t[0]

    @staticmethod
    def _as_vector(values, length, name):
        """Cast inputs to a 1D NumPy array while preserving the model vector shape.

        For a row-vector coefficient c and state vector x, the aggregate term is
        c^T x = sum_i c_i x_i.
        """
        values = np.asarray(values)
        return values

    def I(self, P_I, alpha):
        """Aggregate input/purchased power.

        The input term is the linear combination
            I = alpha^T P_I,
        where P_I = [B_I, 1-V, O, 1-U].
        """
        P_I = self._as_vector(P_I, 4, "P_I")
        alpha = self._as_vector(alpha, 4, "alpha")
        return alpha @ P_I

    def O(self, P_O, beta):
        """Aggregate output/consumption.

        The output term is the linear combination
            O = beta^T P_O,
        where P_O = [B_O, V, 1-U].
        """
        P_O = self._as_vector(P_O, 3, "P_O")
        beta = self._as_vector(beta, 3, "beta")
        return beta @ P_O

    def A(self, P_A, gamma):
        """Aggregator available/bid power.

        The available power is
            A = gamma^T P_A,
        where P_A = [1-U, I, V, 1-PE].
        """
        P_A = self._as_vector(P_A, 4, "P_A")
        gamma = self._as_vector(gamma, 4, "gamma")
        return gamma @ P_A

    @staticmethod
    def reduced_drift_coefficients(BtO, BtI, alpha, beta):
        """Return d0, dV, dU such that
            I - O = d0 + dV * V + dU * U.

        This is the reduced drift form obtained by substituting the affine state
        dependence of the bid and consumption vectors into the aggregate balance.
        """
        alpha = np.asarray(alpha)
        beta = np.asarray(beta)

        o_bar = beta[0] * BtO + beta[2]
        i_bar = alpha[0] * BtI + alpha[1] + alpha[2] * o_bar + alpha[3]

        d0 = i_bar - o_bar
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
        return d0, dV, dU

    def _evaluate_step(self, t, V, parameters):
        """Evaluate one WM1 time step at state V and time t.

        The model follows the causal chain
            (B_O, V, U) -> P_O -> O -> P_I -> I -> P_A -> A,
        with the algebraic balances defined in the appendix note.
        """
        U = parameters["U"](t)
        BO = parameters["B_O"](t)
        BI = parameters["B_I"](t)
        PE = parameters["PE"](t)

        alpha, beta, gamma = self.validate_parameters(
            parameters["alpha"],
            parameters["beta"],
            parameters["gamma"],
        )

        P_O = np.array([BO, V, 1.0 - U])
        O = self.O(P_O, beta)

        P_I = np.array([BI, 1.0 - V, O, 1.0 - U])
        I = self.I(P_I, alpha)

        P_A = np.array([1.0 - U, I, V, 1.0 - PE])
        A = self.A(P_A, gamma)

        return U, BO, BI, PE, P_I, P_O, P_A, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        """Forward-Euler simulation of updated WM1.

        The dynamic price signal is supplied as parameters['U'].

        The discrete-time balance is
            R_k = I_k - O_k + theta * R_{k-1},
            V_{k+1} = V_k + dt * R_k,
        where theta = 0 recovers the no-inertia case.
        """
        init = np.asarray(init_cond).reshape(-1)
        V0 = init[0]
        R_prev = init[1] if init.size > 1 else 0.0
        theta = parameters.get("theta", 0.0)

        V = np.empty(self.t_steps)
        R = np.empty(self.t_steps)
        U = np.empty(self.t_steps)
        BO = np.empty(self.t_steps)
        BI = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)
        O = np.empty(self.t_steps)
        I = np.empty(self.t_steps)
        A = np.empty(self.t_steps)

        P_I = np.empty((self.t_steps, 4))
        P_O = np.empty((self.t_steps, 3))
        P_A = np.empty((self.t_steps, 4))

        V[0] = V0

        for k, tk in enumerate(self.t):
            (U[k], BO[k], BI[k], PE[k], P_I[k], P_O[k], P_A[k], O[k], I[k], A[k]) = self._evaluate_step(tk, V[k], parameters)

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

    def fokker_planc_solver(self, init_cond, parameters, sigma, degree=20, e_bounds=(-1.5, 1.5), x_steps=301, cfl=0.45):
        """Propagate the no-inertia stochastic WM1 centered-error density.

        The centered error is defined by
            V(t) = m(t; U) + E(t),
            dE(t) = dV * E(t) dt + sigma dW_t.

        The generic Fokker-Planck solver works on the reference interval [0,1], so
        the error variable E is mapped by the affine transform
            u = (E - e_min) / (e_max - e_min).
        """

        alpha, beta = parameters["alpha"], parameters["beta"]
        e_min, e_max = e_bounds
        span = e_max - e_min

        V0 = np.asarray(init_cond).reshape(-1)[0]
        mean_result = self.deterministic_solver([V0], parameters)
        mean = mean_result["V"]

        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]

        def drift_u(t, u):
            e = e_min + span * np.asarray(u)
            return dV * e / span

        def diffusion_u(t, u):
            return np.zeros_like(np.asarray(u)) + sigma / span

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

        lower_violation = np.empty(self.t_steps)
        upper_violation = np.empty(self.t_steps)

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

from src.models.Simulate import Simulate
import numpy as np


class BM:
    # Constructor
    def __init__(self, B, A, T, t_steps):
        self.simulate = Simulate()
        self.B = B
        self.A = A
        self.T = T
        self.t_steps = int(t_steps)

        if self.t_steps < 2:
            raise ValueError("t_steps must be at least 2.")

        self.t = np.linspace(T[0], T[1], self.t_steps)
        self.dt = float(self.t[1] - self.t[0])

    # Internal functions (private)
    # Eval helper function
    def eval(self, t, Xt, U, f, g, k, Xi):
        Ut = U(t)
        Bt = self.B(t)
        fXt = f(Xt)
        gUt = g(Ut)
        dt = self.l(fXt, gUt, k)
        Dt = self.D(Bt, Xi, dt)
        return Ut, Bt, fXt, gUt, dt, Dt

    # Model functions:
    # Logistic function that outputs specific value of L(f(Xt; A) + g(u;B); k) (Eq. 2)
    def l(self, fXt, gUt, k):
        return 2 / (1 + np.exp(-k * (fXt + gUt))) - 1

    # Demand based on time function (Eq. 3)
    def D(self, Bt, Xi, dt):
        return Bt + dt * Xi * ((dt >= 0) * (1 - Bt) + (dt < 0) * Bt)

    # dXdt for the deterministic BM drift (Eq. 1).
    # The stochastic flag is retained for BM2 compatibility, but the base model
    # no longer depends on torch.
    def dXdt(self, t, state, U, f, g, C, k, Xi, stochastic=False):
        Xt = state[0]

        if stochastic and hasattr(Xt, "item"):
            try:
                Xt = Xt.item()
            except (ValueError, RuntimeError):
                pass

        _, Bt, _, _, _, Dt = self.eval(t, Xt, U, f, g, k, Xi)
        dXdt = (Dt - Bt) / C
        return [dXdt]

    # Diffusion coefficient for the stochastic BM:
    # b(X) = sigma_x * X * (1 - X)
    def diffusion_coefficient(self, X, sigma_x):
        return sigma_x * X * (1 - X)

    # Legacy stochastic helper retained for BM2/torchsde compatibility.
    def XtOx_dWt(self, t, X, sigma_x):
        diff = self.diffusion_coefficient(X, sigma_x)

        if hasattr(diff, "unsqueeze"):
            return diff.unsqueeze(-1)

        return np.expand_dims(np.asarray(diff), axis=-1)

    # Abstract functions
    def ode(self):
        raise NotImplementedError("Implement the ode in specific subclass")

    def phi(self):
        raise NotImplementedError("Implement the minimization function in specific subclass")

    def deterministic_solver(self):
        raise NotImplementedError("Implement the deterministic solver for the specific subclass")

    def stochastic_solver(self):
        raise NotImplementedError("Implement the stochastic solver for the specific subclass")

    def optimize(self):
        raise NotImplementedError("Implement the optimizer for the specific subclass")

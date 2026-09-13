from src.validation.model.BM import BMValidation
import numpy as np

class BM:
    def __init__(self, B, A, T, t_steps):
        B, A, T, t_steps = BMValidation.constructor(B, A, T, t_steps)
        self.B = B
        self.A = A
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], self.t_steps)
        self.dt = float(self.t[1] - self.t[0])

    def eval(self, t, Xt, U, f, g, k, Xi):
        U, f, g, _, k, Xi = BMValidation.dynamics(U, f, g, 1.0, k, Xi)
        Ut = U(t)
        Bt = self.B(t)
        fXt = f(Xt)
        gUt = g(Ut)
        Ut, Bt, fXt, gUt = BMValidation.response_values(Ut, Bt, fXt, gUt)
        dt = self.l(fXt, gUt, k)
        Dt = self.D(Bt, Xi, dt)
        return Ut, Bt, fXt, gUt, dt, Dt

    def l(self, fXt, gUt, k):
        fXt = BMValidation.bounded_array(fXt, -1.0, 1.0, "f(X)")
        gUt = BMValidation.bounded_array(gUt, -1.0, 1.0, "g(U)")
        k = BMValidation.positive_scalar(k, "k")
        return 2 / (1 + np.exp(-k * (fXt + gUt))) - 1

    def D(self, Bt, Xi, dt):
        Bt = BMValidation.bounded_array(Bt, 0.0, 1.0, "B(t)")
        dt = BMValidation.bounded_array(dt, -1.0, 1.0, "delta(t)")
        Xi = BMValidation.unit_scalar(Xi, "Xi")
        return Bt + dt * Xi * ((dt >= 0) * (1 - Bt) + (dt < 0) * Bt)

    def dXdt(self, t, state, U, f, g, C, k, Xi, stochastic=False):
        U, f, g, C, k, Xi = BMValidation.dynamics(U, f, g, C, k, Xi)
        state = BMValidation.state(state)
        Xt = state.reshape(-1)[0]
        if stochastic and hasattr(Xt, "item"):
            try:
                Xt = Xt.item()
            except (ValueError, RuntimeError):
                pass
        _, Bt, _, _, _, Dt = self.eval(t, Xt, U, f, g, k, Xi)
        return [(Dt - Bt) / C]

    def diffusion_coefficient(self, X, sigma_x):
        X, sigma_x = BMValidation.diffusion(X, sigma_x)
        return sigma_x * X * (1 - X)

    def XtOx_dWt(self, t, X, sigma_x):
        diff = self.diffusion_coefficient(X, sigma_x)
        if hasattr(diff, "unsqueeze"):
            return diff.unsqueeze(-1)
        return np.expand_dims(np.asarray(diff), axis=-1)

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

from src.validation.model.bm import bm_validation
import numpy as np

class bm:
    """Shared deterministic BM primitives and the F2 diffusion amplitude.

    F2 uses the Ito equation dX = (D - B) / C dt + b(X) dW, with
    b(X) = sigma_x * X * (1 - X) and Var(dW) = dt.  D and B are
    normalized powers; C = C_phys / P_scale has the same time unit as T,
    and sigma_x has units time**(-1/2).  Xi does not scale process noise.

    The canonical response functions decrease from 1 to -1 on [0, 1].
    Together with suitable Lipschitz regularity, inward boundary drift
    and vanishing boundary diffusion keep the continuous state in [0, 1].
    This is not a guarantee for a finite Euler-Maruyama step.
    See root/F2.md for the model contract and numerical limitations.
    """
    def __init__(self, B, A, T, t_steps):
        B, A, T, t_steps = bm_validation.constructor(B, A, T, t_steps)
        self.B = B
        self.A = A
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], self.t_steps)
        self.dt = float(self.t[1] - self.t[0])

    def eval(self, t, Xt, U, f, g, k, Xi):
        U, f, g, _, k, Xi = bm_validation.dynamics(U, f, g, 1.0, k, Xi)
        Ut = U(t)
        Bt = self.B(t)
        fXt = f(Xt)
        gUt = g(Ut)
        Ut, Bt, fXt, gUt = bm_validation.response_values(Ut, Bt, fXt, gUt)
        delta = self.l(fXt, gUt, k)
        Dt = self.D(Bt, Xi, delta)
        return Ut, Bt, fXt, gUt, delta, Dt

    def l(self, fXt, gUt, k):
        """Map state and price responses to signed flexibility in [-1, 1]."""
        fXt = bm_validation.bounded_array(fXt, -1.0, 1.0, "f(X)")
        gUt = bm_validation.bounded_array(gUt, -1.0, 1.0, "g(U)")
        k = bm_validation.positive_scalar(k, "k")
        return 2 / (1 + np.exp(-k * (fXt + gUt))) - 1

    def D(self, Bt, Xi, delta):
        """Apply the signed flexibility response delta, not a time increment."""
        Bt = bm_validation.bounded_array(Bt, 0.0, 1.0, "B(t)")
        delta = bm_validation.bounded_array(delta, -1.0, 1.0, "delta(t)")
        Xi = bm_validation.unit_scalar(Xi, "Xi")
        return Bt + delta * Xi * ((delta >= 0) * (1 - Bt) + (delta < 0) * Bt)

    def dXdt(self, t, state, U, f, g, C, k, Xi, stochastic=False):
        U, f, g, C, k, Xi = bm_validation.dynamics(U, f, g, C, k, Xi)
        state = bm_validation.state(state)
        Xt = state.reshape(-1)[0]
        if stochastic and hasattr(Xt, "item"):
            try:
                Xt = Xt.item()
            except (ValueError, RuntimeError):
                pass
        _, Bt, _, _, _, Dt = self.eval(t, Xt, U, f, g, k, Xi)
        return [(Dt - Bt) / C]

    def diffusion_coefficient(self, X, sigma_x):
        """Return b(X), not b(X)**2 or a time-scaled Brownian increment.

        The numerical method supplies sqrt(dt).  No clipping, reflection,
        capacity factor or Xi factor is introduced here.
        """
        X, sigma_x = bm_validation.diffusion(X, sigma_x)
        return sigma_x * X * (1 - X)

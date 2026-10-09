from src.validation.model.bm import bm_validation
import numpy as np

class bm:
    """Deterministic BM response and normalized energy balance."""
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
        # Default l/D already check f, g and B. Avoid checking those arrays
        # twice, but preserve the original entry checks for arbitrary overrides.
        Ut = bm_validation.bounded_array(Ut, 0.0, 1.0, "U(t)")
        Bt = np.asarray(Bt, dtype=float)
        fXt = np.asarray(fXt, dtype=float)
        gUt = np.asarray(gUt, dtype=float)
        response_map, demand_map = self.l, self.D
        if getattr(response_map, '__func__', None) is not bm.l:
            fXt = bm_validation.bounded_array(fXt, -1.0, 1.0, "f(X)")
            gUt = bm_validation.bounded_array(gUt, -1.0, 1.0, "g(U)")
        if getattr(demand_map, '__func__', None) is not bm.D:
            Bt = bm_validation.bounded_array(Bt, 0.0, 1.0, "B(t)")
        delta = response_map(fXt, gUt, k)
        Dt = demand_map(Bt, Xi, delta)
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

    def dXdt(self, t, state, U, f, g, C, k, Xi):
        U, f, g, C, k, Xi = bm_validation.dynamics(U, f, g, C, k, Xi)
        state = bm_validation.state(state)
        Xt = state.reshape(-1)[0]
        _, Bt, _, _, _, Dt = self.eval(t, Xt, U, f, g, k, Xi)
        return [(Dt - Bt) / C]

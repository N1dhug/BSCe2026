import numpy as np


class wm_5_params_validation:
    """Linear sufficient constraints for global monotonicity and signed equilibrium bounds.

    All basis functions have phi(0)=0, phi'(x)>=0 on [-1,1], and certified
    lower/upper derivative bounds. All nonlinear coefficients are nonpositive.
    The conservative derivative envelopes give Euler Schur stability on [-1,1].
    """

    @staticmethod
    def constraints(r_functions, u_functions, dt, stabilize=True, eps=1e-8):
        nr, nu = len(r_functions), len(u_functions)
        n = 1 + nr + nu
        ir = slice(1, 1 + nr)
        iu = slice(1 + nr, n)
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0: raise ValueError("dt must be finite and positive.")
        strict = 10.0 * float(eps)
        rlo = np.array([basis.derivative_bounds[0] for basis in r_functions], dtype=float)
        rhi = np.array([basis.derivative_bounds[1] for basis in r_functions], dtype=float)
        gm = np.array([basis.value(-1.0) for basis in r_functions], dtype=float)
        gp = np.array([basis.value(1.0) for basis in r_functions], dtype=float)
        hm = np.array([basis.value(-1.0) for basis in u_functions], dtype=float)
        hp = np.array([basis.value(1.0) for basis in u_functions], dtype=float)
        rows, rhs = [], []

        def inequality(f=0.0, r=None, u=None, bound=0.0):
            line = np.zeros(n, dtype=float); line[0] = f
            if r is not None: line[ir] = r
            if u is not None: line[iu] = u
            rows.append(line); rhs.append(float(bound))

        # Expected equilibrium F*(U) lies within [-1,1], for every U in [-1,1].
        inequality(f=1.0, u=-hp)       # r_F - h(1) <= 0
        inequality(f=1.0, u=hm)        # r_F + h(-1) <= 0

        # For each F,U in [-1,1], drift points inward at R=-1 and R=1.
        inequality(f=-1.0, r=-gm, u=-hp)  # -r_F - g(-1) - h(1) <= 0
        inequality(f=-1.0, r=gp, u=hm)    # -r_F + g(1) + h(-1) <= 0

        # Strict restoring and damping. With a_i<=0, max(g') <= a @ rlo.
        inequality(r=rlo, bound=-strict)
        if stabilize:
            # Euler Schur stability for all R in [-1,1]. Because a_i<=0,
            # g' >= a @ rhi and g' <= a @ rlo pointwise.
            inequality(f=dt * dt, r=-2.0 * dt * rhi, bound=4.0 - eps)
            inequality(f=-dt, r=rlo, bound=-strict)

        lb = np.full(n, -np.inf, dtype=float)
        ub = np.zeros(n, dtype=float)
        ub[0] = -strict
        return np.vstack(rows), np.asarray(rhs), lb, ub

    @staticmethod
    def initial_rate(observed_rate):
        value = float(observed_rate)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("The observed initial R0 must lie in [-1,1].")
        return value

    @staticmethod
    def fit_settings(max_state_iter, state_tol, f_bounds):
        if int(max_state_iter) < 1 or not np.isfinite(state_tol) or state_tol <= 0.0:
            raise ValueError("Invalid sequential fitter iteration limit or tolerance.")
        lo, hi = map(float, f_bounds)
        if not -1.0 <= lo < hi <= 1.0: raise ValueError("F0 bounds must be within [-1,1] and nonempty.")
        return int(max_state_iter), float(state_tol), (lo, hi)

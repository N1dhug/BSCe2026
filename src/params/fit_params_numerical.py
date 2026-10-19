from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.params.wm_4 import qp_params
from src.validation.model.wm_4 import wm_4_validation


class numerical_params:
    """Direct trajectory-error WM4 parameter fit with a fixed initial [F0, R0].

    Jointly optimize all structural coefficients against observed I=O+R.
    The optional constant r_0 is an additive term in dR/dt, not the initial R0.
    Initial conditions are fixed throughout; no alternating state optimization.
    """

    def __init__(self, model): self.model = model

    def _loss_and_gradient(self, coefficients, data, init_cond, fit_intercept=False):
        p = np.asarray(coefficients, dtype=float)
        r, r_0 = p[:3], (float(p[3]) if fit_intercept else 0.0)
        dt, U, O, observed = self.model.dt, data["U"], data["O"], data["I"]
        mask = np.asarray(data["observation_mask"], dtype=bool)
        n_obs = int(np.count_nonzero(mask))
        F, R = map(float, init_cond)
        S = np.zeros((2, p.size), dtype=float)
        loss, gradient = 0.0, np.zeros(p.size, dtype=float)

        for k in range(self.model.t_steps):
            if mask[k]:
                error = O[k] + R - observed[k]
                loss += error * error
                gradient += 2.0 * error * S[1]
            if k == self.model.t_steps - 1: break
            feature = np.array([F, U[k], R, 1.0], dtype=float)[:p.size]
            next_s0 = S[0] + dt * S[1]
            next_s1 = S[1] + dt * (r[0] * S[0] + r[2] * S[1] + feature)
            F, R = F + dt * R, R + dt * (r_0 + r[0] * F + r[1] * U[k] + r[2] * R)
            S[0], S[1] = next_s0, next_s1

        return float(loss / n_obs), gradient / n_obs

    def fit(self, input_data, initial_conditions=None, *, fit_intercept=False, stabilize=True,
            x0=None, use_qp_start=True, max_iter=300, ftol=1e-10, verbose=False):
        started = perf_counter()
        helper = qp_params(self.model)
        data = helper._prepare_data(input_data)
        initial_conditions = [0.0, 0.0] if initial_conditions is None else initial_conditions
        init = wm_4_validation.initial_condition(initial_conditions)
        A, b, lb, ub = helper._qp_constraints(stabilize=stabilize, fit_intercept=fit_intercept)
        n_params = 4 if fit_intercept else 3

        qp_seed_time, qp_seed_loss = 0.0, np.nan
        if x0 is not None:
            seed = np.asarray(x0, dtype=float).reshape(-1)
            if seed.size != n_params: raise ValueError(f"x0 must have exactly {n_params} coefficients.")
        elif use_qp_start:
            qp_start = perf_counter()
            seed_fit = helper.fit(input_data, init, fit_intercept=fit_intercept, stabilize=stabilize)
            qp_seed_time = perf_counter() - qp_start
            qp_seed_loss = float(seed_fit["deterministic_loss"])
            seed = np.r_[seed_fit["r"], seed_fit["r_0"]] if fit_intercept else seed_fit["r"]
        else:
            seed = np.array([-0.5, -0.1, -1.0] + ([0.0] if fit_intercept else []), dtype=float)

        if not np.all(np.isfinite(seed)): raise ValueError("Numerical WM4 initial coefficients must be finite.")
        bounds = Bounds(lb, ub)
        constraints = [LinearConstraint(A, -np.inf, b)]
        optimization_start = perf_counter()

        def objective(x): return self._loss_and_gradient(x, data, init, fit_intercept)

        result = minimize(
            objective, seed, method="SLSQP", jac=True, bounds=bounds, constraints=constraints,
            options={"maxiter": int(max_iter), "ftol": float(ftol), "disp": bool(verbose)},
        )
        numerical_time = perf_counter() - optimization_start
        solution = np.asarray(result.x, dtype=float)
        feasible = bool(np.all(A @ solution <= b + 2e-6) and np.all(solution >= lb - 2e-6) and np.all(solution <= ub + 2e-6))
        if not feasible: raise RuntimeError(f"Numerical WM4 optimizer returned an infeasible solution: {result.message}")

        r = wm_4_validation.recovered_structural_coefficients(solution[:3])
        r_0 = float(solution[3]) if fit_intercept else 0.0
        state = helper._state_from_parameters(data, init, r, full=True, r_0=r_0)
        mask = np.asarray(data["observation_mask"], dtype=bool)
        residual = np.asarray(data["I"][mask] - state["I"][mask], dtype=float)
        loss = float(np.mean(residual ** 2))
        q, kappa_R = helper._equivalent_q(r)
        branch, zeta = helper._damping_branch(r)
        output = {
            "r": r.copy(), "r_0": r_0, "fit_intercept": bool(fit_intercept),
            "fit_method": "numerical", "init_cond": init.copy(),
            "initial_condition_seed": init.copy(), "fit_initial_state": False,
            "deterministic_state": state, "observation_mask": mask.copy(),
            "transition_mask": np.asarray(data["transition_mask"], dtype=bool).copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(), "O": data["O"].copy(),
            "deterministic_loss": loss, "recovered_model_loss": loss, "numerical_loss": loss,
            "q": q, "kappa_R": float(kappa_R), "damping_ratio": float(zeta), "damping_branch": branch,
            "structural_viable": True, "stabilize_euler": bool(stabilize),
            "success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "evaluations": int(result.nfev),
            "gradient_evaluations": int(result.njev), "numerical_solve_time": float(numerical_time),
            "qp_seed_time": float(qp_seed_time), "qp_seed_loss": float(qp_seed_loss),
            "fit_time": float(perf_counter() - started),
        }
        if verbose: print(f"[numerical] success={output['success']} iterations={output['iterations']} loss={loss:.8g} time={output['fit_time']:.3f}s")
        return output

from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from src.validation.model.wm_4 import wm_4_validation


class qp_params:
    """Single constrained least-squares QP fitter for wm_4.

    With O(t) fixed and initial [F0, R0] supplied, measured input gives
    R = I - O and F is reconstructed from dF = R dt. The deterministic drift
    fit is therefore linear in r for

        P_R = [1, F, U, R],   dR/dt = r @ P_R.
    """

    def __init__(self, model): self.model = model

    def _prepare_data(self, input_data):
        data = wm_4_validation.fit_data(input_data, self.model.t_steps)
        if "O" in input_data:
            O = np.asarray(input_data["O"], dtype=float).reshape(-1)
            if O.size != self.model.t_steps or not np.all(np.isfinite(O)): raise ValueError(f"input_data['O'] must contain exactly {self.model.t_steps} finite samples.")
        else:
            O = np.array([wm_4_validation.feature_value(self.model.O(x), "O(Time)") for x in data["Time"]], dtype=float)
        return {**data, "O": O}

    @staticmethod
    def _equivalent_q(r):
        r = wm_4_validation.r_coefficients(r)
        kappa_R = -r[3]
        return r[:3] / kappa_R, kappa_R

    @staticmethod
    def _damping_branch(r, tol=1e-6):
        r = np.asarray(r, dtype=float)
        zeta = -float(r[3]) / (2.0 * np.sqrt(-float(r[1])))
        if abs(zeta - 1.0) <= tol: return "critical", zeta
        return ("under" if zeta < 1.0 else "over"), zeta

    def _observed_state(self, data, init_cond):
        init_cond = wm_4_validation.initial_condition(init_cond)
        n, dt = self.model.t_steps, self.model.dt
        R_observed = np.asarray(data["I"] - data["O"], dtype=float)
        R = R_observed.copy()
        R[0] = init_cond[1]
        F = np.empty(n, dtype=float)
        F[0] = init_cond[0]
        if n > 1: F[1:] = F[0] + np.cumsum(dt * R[:-1])
        P_R = np.column_stack((np.ones(n), F, data["U"], R))
        X = dt * P_R[:-1]
        y = np.diff(R)
        mask = np.asarray(data["transition_mask"], dtype=bool)
        return {
            "F": F, "R": R, "R_observed": R_observed, "P_R": P_R,
            "X": X, "y": y, "transition_mask": mask,
            "initial_rate_mismatch": float(R_observed[0] - init_cond[1]),
        }

    def _qp_constraints(self, stabilize=True):
        eps, dt = self.model._FIT_EPS, self.model.dt
        rows = [
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, -1.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 0.0]),
        ]
        rhs = [0.0, 0.0, 0.0]
        if stabilize:
            rows.extend([
                np.array([0.0, dt * dt, 0.0, -2.0 * dt]),
                np.array([0.0, -dt, 0.0, 1.0]),
            ])
            rhs.extend([4.0 - eps, -eps])
        lb = np.array([0.0, -np.inf, -np.inf, -np.inf], dtype=float)
        ub = np.array([np.inf, -eps, 0.0, -eps], dtype=float)
        return np.vstack(rows), np.asarray(rhs, dtype=float), lb, ub

    def _solve_qp(self, datasets, init_conds, stabilize=True):
        observed = [self._observed_state(data, init) for data, init in zip(datasets, init_conds)]
        X = np.vstack([state["X"][state["transition_mask"]] for state in observed])
        y = np.concatenate([state["y"][state["transition_mask"]] for state in observed])
        if X.shape[0] < 1: raise ValueError("At least one wm_4 state transition is required for fitting.")
        A_ub, b_ub, lb, ub = self._qp_constraints(stabilize=stabilize)
        start = perf_counter()
        r = self.model._least_squares_qp(X, y, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE)
        solve_time = perf_counter() - start
        r = wm_4_validation.recovered_structural_coefficients(r)
        residual = X @ r - y
        return r, observed, float(np.mean(residual ** 2)), solve_time

    def _state_from_parameters(self, data, init_cond, coefficients, kappa_r=None, full=False):
        if kappa_r is None:
            r = wm_4_validation.r_coefficients(coefficients)
        else:
            r = wm_4_validation.legacy_r(coefficients, kappa_r)
        state = self.model._solve_deterministic(init_cond, data["U"], data["PE"], data["Time"], data["O"], r)
        if full: return state
        return {key: value for key, value in state.items() if key not in ("P_R", "P_A")}

    def _fit_uncertainty_params(self, result, data):
        observed = self._observed_state(data, result["init_cond"])
        mask = observed["transition_mask"]
        R, F, r = observed["R"], observed["F"], np.asarray(result["r"], dtype=float)
        dt = np.full(np.count_nonzero(mask), self.model.dt, dtype=float)
        drift = observed["P_R"][:-1] @ r
        innovation = (np.diff(R) - self.model.dt * drift)[mask]
        diffusion_I = np.asarray(data["I"][:-1], dtype=float)[mask]
        if np.any(diffusion_I < 0.0): raise ValueError("wm_4 diffusion fitting requires observed I >= 0.")

        def nll(theta):
            sigma_0, sigma_I = map(float, theta)
            g = sigma_0 + sigma_I * diffusion_I
            if sigma_0 <= 0.0 or sigma_I < 0.0 or np.any(g <= 0.0): return np.inf
            variance = g * g * dt
            return 0.5 * float(np.sum(np.log(variance) + innovation * innovation / variance))

        constant = max(self.model._FIT_EPS, float(np.sqrt(np.mean(innovation * innovation / dt))))
        target = np.abs(innovation) * np.sqrt(np.pi / (2.0 * dt))
        X = np.column_stack((np.ones(diffusion_I.size), diffusion_I))
        linear = np.linalg.lstsq(X, target, rcond=None)[0]
        linear = np.array([max(self.model._FIT_EPS, float(linear[0])), max(0.0, float(linear[1]))])
        scale = max(float(np.max(diffusion_I)), self.model._FIT_EPS)
        starts = (np.array([constant, 0.0]), linear, np.array([0.5 * constant, 0.5 * constant / scale]))
        fits = [minimize(nll, x0, method="L-BFGS-B", bounds=((self.model._FIT_EPS, None), (0.0, None))) for x0 in starts]
        valid = [fit for fit in fits if fit.success and np.all(np.isfinite(fit.x)) and np.isfinite(fit.fun)]
        if not valid: raise RuntimeError("wm_4 diffusion maximum-likelihood fit failed.")
        fit = min(valid, key=lambda item: item.fun)
        sigma_0, sigma_I = map(float, fit.x)
        obs = np.asarray(data["observation_mask"], dtype=bool)
        residual = data["I"][obs] - result["deterministic_state"]["I"][obs]
        q, _ = self._equivalent_q(r)
        Q = q[0] + q[1] * F[:-1] + q[2] * data["U"][:-1]
        return {
            **result, "sigma_0": sigma_0, "sigma_I": sigma_I, "residual": residual,
            "diffusion_innovation": innovation, "diffusion_dt": dt, "diffusion_I": diffusion_I,
            "diffusion_F": F[:-1][mask], "diffusion_Q": Q[mask], "diffusion_nll": float(fit.fun),
        }

    def _fit_uncertainty_params_many(self, result, datasets, init_conds):
        innovations, dts, diffusion_inputs, observed_fs, qs, residuals = [], [], [], [], [], []
        r = np.asarray(result["r"], dtype=float)
        q, _ = self._equivalent_q(r)
        for state, data, init in zip(result["deterministic_states"], datasets, init_conds):
            observed = self._observed_state(data, init)
            mask = observed["transition_mask"]
            R, F = observed["R"], observed["F"]
            drift = observed["P_R"][:-1] @ r
            innovation = (np.diff(R) - self.model.dt * drift)[mask]
            diffusion_I = np.asarray(data["I"][:-1], dtype=float)[mask]
            if np.any(diffusion_I < 0.0): raise ValueError("wm_4 diffusion fitting requires observed I >= 0.")
            innovations.append(innovation)
            dts.append(np.full(np.count_nonzero(mask), self.model.dt, dtype=float))
            diffusion_inputs.append(diffusion_I)
            observed_fs.append(F[:-1][mask])
            Q = q[0] + q[1] * F[:-1] + q[2] * data["U"][:-1]
            qs.append(Q[mask])
            obs = np.asarray(data["observation_mask"], dtype=bool)
            residuals.append(data["I"][obs] - state["I"][obs])

        innovation, dt, diffusion_I = np.concatenate(innovations), np.concatenate(dts), np.concatenate(diffusion_inputs)

        def nll(theta):
            sigma_0, sigma_I = map(float, theta)
            g = sigma_0 + sigma_I * diffusion_I
            if sigma_0 <= 0.0 or sigma_I < 0.0 or np.any(g <= 0.0): return np.inf
            variance = g * g * dt
            return 0.5 * float(np.sum(np.log(variance) + innovation * innovation / variance))

        constant = max(self.model._FIT_EPS, float(np.sqrt(np.mean(innovation * innovation / dt))))
        target = np.abs(innovation) * np.sqrt(np.pi / (2.0 * dt))
        X = np.column_stack((np.ones(diffusion_I.size), diffusion_I))
        linear = np.linalg.lstsq(X, target, rcond=None)[0]
        linear = np.array([max(self.model._FIT_EPS, float(linear[0])), max(0.0, float(linear[1]))])
        scale = max(float(np.max(diffusion_I)), self.model._FIT_EPS)
        starts = (np.array([constant, 0.0]), linear, np.array([0.5 * constant, 0.5 * constant / scale]))
        fits = [minimize(nll, x0, method="L-BFGS-B", bounds=((self.model._FIT_EPS, None), (0.0, None))) for x0 in starts]
        valid = [fit for fit in fits if fit.success and np.all(np.isfinite(fit.x)) and np.isfinite(fit.fun)]
        if not valid: raise RuntimeError("wm_4 diffusion maximum-likelihood fit failed.")
        fit = min(valid, key=lambda item: item.fun)
        sigma_0, sigma_I = map(float, fit.x)
        return {
            **result, "sigma_0": sigma_0, "sigma_I": sigma_I, "residual": np.concatenate(residuals),
            "diffusion_innovation": innovation, "diffusion_dt": dt, "diffusion_I": diffusion_I,
            "diffusion_F": observed_fs, "diffusion_Q": qs, "diffusion_nll": float(fit.fun),
        }

    def _finish_result(self, result, qp_loss, qp_solve_time, fit_start, stabilize):
        r = np.asarray(result["r"], dtype=float)
        q, kappa_R = self._equivalent_q(r)
        branch, zeta = self._damping_branch(r)
        elapsed = perf_counter() - fit_start
        result.update({
            "q": q, "kappa_R": float(kappa_R),
            "fit_method": "qp", "damping_branch": branch, "damping_ratio": float(zeta),
            "qp_loss": float(qp_loss), "qp_solve_time": float(qp_solve_time),
            "qp_fit_time": elapsed, "stabilize_euler": bool(stabilize),
            "fit_initial_state": False,
        })
        return result

    def fit(self, input_data, initial_conditions, *, fit_initial_state=False, fit_diffusion=True, stabilize=True, verbose=False, **_legacy_kwargs):
        if fit_initial_state: raise ValueError("wm_4 QP fitting requires fixed initial [F0, R0]; fit_initial_state is no longer supported.")
        fit_start = perf_counter()
        data = self._prepare_data(input_data)
        init_cond = wm_4_validation.initial_condition(initial_conditions)
        r, observed, qp_loss, qp_solve_time = self._solve_qp([data], [init_cond], stabilize=stabilize)
        state = self._state_from_parameters(data, init_cond, r, full=True)
        obs = np.asarray(data["observation_mask"], dtype=bool)
        deterministic_loss = float(np.mean((data["I"][obs] - state["I"][obs]) ** 2))
        result = {
            "init_cond": init_cond.copy(), "initial_condition_seed": init_cond.copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(),
            "O": data["O"].copy(), "observation_mask": obs.copy(), "transition_mask": data["transition_mask"].copy(),
            "r": r.copy(), "deterministic_state": state, "observed_state": observed[0],
            "recovered_model_loss": deterministic_loss, "deterministic_loss": deterministic_loss,
            "structural_viable": True, "initial_rate_mismatch": observed[0]["initial_rate_mismatch"],
        }
        result = self._fit_uncertainty_params(result, data) if fit_diffusion else {**result, "sigma_0": self.model._FIT_EPS, "sigma_I": 0.0}
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        if verbose: print(f"[qp] transitions={np.count_nonzero(data['transition_mask'])} loss={qp_loss:.8g} free_run_mse={deterministic_loss:.8g} solve={qp_solve_time:.4f}s")
        return result

    def fit_many(self, input_data, initial_conditions, *, fit_initial_state=False, fit_diffusion=True, stabilize=True, verbose=False, **_legacy_kwargs):
        if fit_initial_state: raise ValueError("wm_4 QP fitting requires fixed initial [F0, R0]; fit_initial_state is no longer supported.")
        if not isinstance(input_data, (list, tuple)) or len(input_data) < 1: raise TypeError("input_data must be a non-empty list or tuple of wm_4 fitting dictionaries.")
        fit_start = perf_counter()
        datasets = [self._prepare_data(data) for data in input_data]
        raw_init = np.asarray(initial_conditions, dtype=float)
        if raw_init.shape == (2,): init_conds = [wm_4_validation.initial_condition(raw_init) for _ in datasets]
        elif raw_init.shape == (len(datasets), 2): init_conds = [wm_4_validation.initial_condition(row) for row in raw_init]
        else: raise ValueError("initial_conditions must be [F0,R0] or one [F0,R0] row per fitting trajectory.")

        r, observed, qp_loss, qp_solve_time = self._solve_qp(datasets, init_conds, stabilize=stabilize)
        states = [self._state_from_parameters(data, init, r, full=True) for data, init in zip(datasets, init_conds)]
        losses = []
        for data, state in zip(datasets, states):
            obs = np.asarray(data["observation_mask"], dtype=bool)
            losses.append(float(np.mean((data["I"][obs] - state["I"][obs]) ** 2)))
        residual = np.concatenate([data["I"][np.asarray(data["observation_mask"], dtype=bool)] - state["I"][np.asarray(data["observation_mask"], dtype=bool)] for data, state in zip(datasets, states)])
        deterministic_loss = float(np.mean(residual ** 2))
        result = {
            "init_cond": np.vstack(init_conds), "initial_condition_seed": raw_init.copy(),
            "r": r.copy(), "deterministic_state": states[0], "deterministic_states": states,
            "observed_states": observed, "per_trajectory_loss": np.asarray(losses, dtype=float),
            "recovered_model_loss": deterministic_loss, "deterministic_loss": deterministic_loss,
            "structural_viable": True, "fit_trajectories": len(datasets),
            "initial_rate_mismatch": np.asarray([state["initial_rate_mismatch"] for state in observed], dtype=float),
        }
        result = self._fit_uncertainty_params_many(result, datasets, init_conds) if fit_diffusion else {**result, "sigma_0": self.model._FIT_EPS, "sigma_I": 0.0}
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        result["fit_method"] = "qp_many"
        if verbose: print(f"[qp] trajectories={len(datasets)} transitions={sum(np.count_nonzero(data['transition_mask']) for data in datasets)} loss={qp_loss:.8g} free_run_mse={deterministic_loss:.8g} solve={qp_solve_time:.4f}s")
        return result


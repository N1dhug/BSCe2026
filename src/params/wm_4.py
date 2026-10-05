from time import perf_counter

import numpy as np
from src.validation.model.wm_4 import wm_4_validation


class qp_params:
    """Single constrained least-squares QP fitter for signed-domain wm_4.

    With O(t) fixed and initial [F0, R0] supplied, measured input gives
    R = I - O and F is reconstructed from dF = R dt. The deterministic drift
    fit is linear in r for

        P_R = [F, U, R],   dR/dt = r @ P_R.
    """

    def __init__(self, model): self.model = model

    @staticmethod
    def _reject_removed_diffusion_kwargs(kwargs):
        removed = [name for name in ("fit_diffusion", "deterministic") if name in kwargs]
        if removed: raise TypeError("Legacy WM4 diffusion fitting was removed; delete argument(s): " + ", ".join(removed) + ".")

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
        kappa_R = -r[2]
        return r[:2] / kappa_R, kappa_R

    @staticmethod
    def _damping_branch(r, tol=1e-6):
        r = wm_4_validation.r_coefficients(r)
        zeta = -float(r[2]) / (2.0 * np.sqrt(-float(r[0])))
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
        P_R = np.column_stack((F, data["U"], R))
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
        rows = [np.array([1.0, -1.0, 0.0])]
        rhs = [0.0]
        if stabilize:
            rows.extend([
                np.array([dt * dt, 0.0, -2.0 * dt]),
                np.array([-dt, 0.0, 1.0]),
            ])
            rhs.extend([4.0 - eps, -eps])
        lb = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        ub = np.array([-eps, 0.0, -eps], dtype=float)
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
        r = wm_4_validation.r_coefficients(coefficients) if kappa_r is None else wm_4_validation.legacy_r(coefficients, kappa_r)
        state = self.model._solve_deterministic(init_cond, data["U"], data["PE"], data["Time"], data["O"], r)
        if full: return state
        return {key: value for key, value in state.items() if key not in ("P_R", "P_A")}

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

    def fit(self, input_data, initial_conditions, *, fit_initial_state=False, stabilize=True, verbose=False, **_legacy_kwargs):
        self._reject_removed_diffusion_kwargs(_legacy_kwargs)
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
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        if verbose: print(f"[qp] transitions={np.count_nonzero(data['transition_mask'])} loss={qp_loss:.8g} free_run_mse={deterministic_loss:.8g} solve={qp_solve_time:.4f}s")
        return result

    def fit_many(self, input_data, initial_conditions, *, fit_initial_state=False, stabilize=True, verbose=False, **_legacy_kwargs):
        self._reject_removed_diffusion_kwargs(_legacy_kwargs)
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
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        result["fit_method"] = "qp_many"
        if verbose: print(f"[qp] trajectories={len(datasets)} transitions={sum(np.count_nonzero(data['transition_mask']) for data in datasets)} loss={qp_loss:.8g} free_run_mse={deterministic_loss:.8g} solve={qp_solve_time:.4f}s")
        return result


class optimized_qp_params(qp_params):
    """Alternating constrained-QP wm_4 fitter with optimized trajectory states.

    The shared structural coefficients fit

        dR/dt = r_F F + r_U U + r_R R,

    with signed F,U in [-1,1] and R = I - O. Each fitting trajectory receives
    its own [F0,R0], optimized against the same dR transition residual.
    """

    def _state_loss(self, data, init_cond, r):
        observed = self._observed_state(data, init_cond)
        mask = observed["transition_mask"]
        residual = (observed["X"] @ np.asarray(r, dtype=float) - observed["y"])[mask]
        return float(np.mean(residual ** 2)) if residual.size else np.inf

    def _optimize_initial_state(self, data, r, seed, f_bounds=(-1.0, 1.0), r_bounds=(-np.inf, np.inf)):
        seed = wm_4_validation.initial_condition(seed)
        r = wm_4_validation.r_coefficients(r)

        def residual(init):
            observed = self._observed_state(data, init)
            mask = observed["transition_mask"]
            return (observed["X"] @ r - observed["y"])[mask]

        base = residual(np.array([0.0, 0.0]))
        f_unit = residual(np.array([1.0, 0.0])) - base
        r_unit = residual(np.array([0.0, 1.0])) - base
        X = np.column_stack((f_unit, r_unit))
        y = -base
        lb = np.array([float(f_bounds[0]), float(r_bounds[0])], dtype=float)
        ub = np.array([float(f_bounds[1]), float(r_bounds[1])], dtype=float)
        init = self.model._least_squares_qp(X, y, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE, x0=np.clip(seed, lb, ub))
        return wm_4_validation.initial_condition(init)

    def optimize_prediction_initial_state(self, input_data, r, f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0)):
        """Oracle/diagnostic prediction-state fit for a fixed parameter vector."""
        data = self._prepare_data(input_data)
        r = wm_4_validation.r_coefficients(r)
        base = self._state_from_parameters(data, np.array([0.0, 0.0]), r)
        f_unit = self._state_from_parameters(data, np.array([1.0, 0.0]), r)
        r_unit = self._state_from_parameters(data, np.array([0.0, 1.0]), r)
        obs = np.asarray(data["observation_mask"], dtype=bool)
        X = np.column_stack(((f_unit["I"] - base["I"])[obs], (r_unit["I"] - base["I"])[obs]))
        y = data["I"][obs] - base["I"][obs]
        lb = np.array([float(f_bounds[0]), float(r_bounds[0])], dtype=float)
        ub = np.array([float(f_bounds[1]), float(r_bounds[1])], dtype=float)
        init = self.model._least_squares_qp(X, y, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE)
        state = self._state_from_parameters(data, init, r, full=True)
        loss = float(np.mean((state["I"][obs] - data["I"][obs]) ** 2))
        return {"init_cond": np.asarray(init, dtype=float), "deterministic_state": state, "loss": loss, "observation_mask": obs.copy()}

    @staticmethod
    def estimate_prediction_initial_state(initial_states):
        """Estimate Gaussian prediction-state uncertainty from optimized historical [F0,R0] states."""
        states = np.asarray(initial_states, dtype=float)
        if states.ndim != 2 or states.shape[1] != 2: raise ValueError("initial_states must have shape (n, 2) with columns [F0,R0].")
        if states.shape[0] < 2: raise ValueError("At least two optimized historical initial states are required to estimate uncertainty.")
        if not np.all(np.isfinite(states)): raise ValueError("initial_states must contain only finite values.")
        x_hat = np.mean(states, axis=0)
        covariance = np.cov(states, rowvar=False, ddof=1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        return {
            "x_hat": wm_4_validation.initial_condition(x_hat),
            "covariance": covariance,
            "std": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            "samples": int(states.shape[0]),
        }

    def _alternating_fit(self, datasets, init_conds, stabilize=True, max_state_iter=50, state_tol=1e-8, f_bounds=(-1.0, 1.0), r_bounds=(-np.inf, np.inf), verbose=False):
        init_conds = [wm_4_validation.initial_condition(init).copy() for init in init_conds]
        history, total_qp_time = [], 0.0
        previous_loss = np.inf

        for iteration in range(1, int(max_state_iter) + 1):
            r, _, _, solve_time = self._solve_qp(datasets, init_conds, stabilize=stabilize)
            total_qp_time += solve_time
            optimized = [self._optimize_initial_state(data, r, init, f_bounds=f_bounds, r_bounds=r_bounds) for data, init in zip(datasets, init_conds)]
            r, observed, qp_loss, solve_time = self._solve_qp(datasets, optimized, stabilize=stabilize)
            total_qp_time += solve_time
            max_state_change = max(float(np.max(np.abs(new - old))) for new, old in zip(optimized, init_conds))
            loss_change = abs(previous_loss - qp_loss) if np.isfinite(previous_loss) else np.inf
            history.append({
                "iteration": iteration, "qp_loss": float(qp_loss),
                "max_state_change": max_state_change, "loss_change": float(loss_change),
            })
            init_conds = optimized
            if verbose: print(f"[optimized_qp] iteration={iteration} loss={qp_loss:.8g} state_change={max_state_change:.3g}")
            if max_state_change <= state_tol and loss_change <= state_tol: break
            previous_loss = qp_loss

        return r, observed, init_conds, float(qp_loss), float(total_qp_time), history

    def fit(self, input_data, initial_conditions, *, fit_initial_state=True, stabilize=True, verbose=False, max_state_iter=50, state_tol=1e-8, f_bounds=(-1.0, 1.0), r_bounds=(-np.inf, np.inf), **_legacy_kwargs):
        self._reject_removed_diffusion_kwargs(_legacy_kwargs)
        fit_start = perf_counter()
        data = self._prepare_data(input_data)
        seed = wm_4_validation.initial_condition(initial_conditions)

        if fit_initial_state:
            r, observed, init_conds, qp_loss, qp_solve_time, history = self._alternating_fit(
                [data], [seed], stabilize=stabilize, max_state_iter=max_state_iter, state_tol=state_tol,
                f_bounds=f_bounds, r_bounds=r_bounds, verbose=verbose,
            )
            init = init_conds[0]
        else:
            init = seed
            r, observed, qp_loss, qp_solve_time = self._solve_qp([data], [init], stabilize=stabilize)
            history = []

        state = self._state_from_parameters(data, init, r, full=True)
        obs = np.asarray(data["observation_mask"], dtype=bool)
        deterministic_loss = float(np.mean((data["I"][obs] - state["I"][obs]) ** 2))
        result = {
            "init_cond": init.copy(), "initial_condition_seed": seed.copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(),
            "O": data["O"].copy(), "observation_mask": obs.copy(), "transition_mask": data["transition_mask"].copy(),
            "r": r.copy(), "deterministic_state": state, "observed_state": observed[0],
            "recovered_model_loss": deterministic_loss, "deterministic_loss": deterministic_loss,
            "structural_viable": True, "initial_rate_mismatch": observed[0]["initial_rate_mismatch"],
            "state_optimization_history": history, "state_iterations": len(history),
        }
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        result["fit_method"] = "optimized_qp" if fit_initial_state else "qp"
        result["fit_initial_state"] = bool(fit_initial_state)
        result["deterministic_fit"] = True
        if verbose: print(f"[optimized_qp] transitions={np.count_nonzero(data['transition_mask'])} loss={qp_loss:.8g} iterations={len(history)}")
        return result

    def fit_many(self, input_data, initial_conditions, *, fit_initial_state=True, stabilize=True, verbose=False, max_state_iter=50, state_tol=1e-8, f_bounds=(-1.0, 1.0), r_bounds=(-np.inf, np.inf), **_legacy_kwargs):
        self._reject_removed_diffusion_kwargs(_legacy_kwargs)
        if not isinstance(input_data, (list, tuple)) or len(input_data) < 1: raise TypeError("input_data must be a non-empty list or tuple of wm_4 fitting dictionaries.")
        fit_start = perf_counter()
        datasets = [self._prepare_data(data) for data in input_data]
        raw_init = np.asarray(initial_conditions, dtype=float)
        if raw_init.shape == (2,): seeds = [wm_4_validation.initial_condition(raw_init) for _ in datasets]
        elif raw_init.shape == (len(datasets), 2): seeds = [wm_4_validation.initial_condition(row) for row in raw_init]
        else: raise ValueError("initial_conditions must be [F0,R0] or one [F0,R0] row per fitting trajectory.")

        if fit_initial_state:
            r, observed, init_conds, qp_loss, qp_solve_time, history = self._alternating_fit(
                datasets, seeds, stabilize=stabilize, max_state_iter=max_state_iter, state_tol=state_tol,
                f_bounds=f_bounds, r_bounds=r_bounds, verbose=verbose,
            )
        else:
            init_conds = seeds
            r, observed, qp_loss, qp_solve_time = self._solve_qp(datasets, init_conds, stabilize=stabilize)
            history = []

        states = [self._state_from_parameters(data, init, r, full=True) for data, init in zip(datasets, init_conds)]
        losses = []
        for data, state in zip(datasets, states):
            obs = np.asarray(data["observation_mask"], dtype=bool)
            losses.append(float(np.mean((data["I"][obs] - state["I"][obs]) ** 2)))
        residual = np.concatenate([
            data["I"][np.asarray(data["observation_mask"], dtype=bool)] - state["I"][np.asarray(data["observation_mask"], dtype=bool)]
            for data, state in zip(datasets, states)
        ])
        deterministic_loss = float(np.mean(residual ** 2))
        result = {
            "init_cond": np.vstack(init_conds), "initial_condition_seed": raw_init.copy(),
            "r": r.copy(), "deterministic_state": states[0], "deterministic_states": states,
            "observed_states": observed, "per_trajectory_loss": np.asarray(losses, dtype=float),
            "recovered_model_loss": deterministic_loss, "deterministic_loss": deterministic_loss,
            "structural_viable": True, "fit_trajectories": len(datasets),
            "initial_rate_mismatch": np.asarray([state["initial_rate_mismatch"] for state in observed], dtype=float),
            "state_optimization_history": history, "state_iterations": len(history),
        }
        if fit_initial_state:
            prediction_state = self.estimate_prediction_initial_state(result["init_cond"])
            result.update({
                "x_hat": prediction_state["x_hat"], "x_cov": prediction_state["covariance"],
                "x_std": prediction_state["std"], "x_samples": prediction_state["samples"],
            })
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        result["fit_method"] = "optimized_qp_many" if fit_initial_state else "qp_many"
        result["fit_initial_state"] = bool(fit_initial_state)
        result["deterministic_fit"] = True
        if verbose: print(f"[optimized_qp] trajectories={len(datasets)} transitions={sum(np.count_nonzero(data['transition_mask']) for data in datasets)} loss={qp_loss:.8g} iterations={len(history)}")
        return result






class time_diffusion_qp:
    """Convex local QP for an automatically selected harmonic diffusion variance.

    For harmonic order K,

        q_K(t) = g(t)^2
               = beta_0
               + sum_{k=1}^K [
                   beta_c,k*cos(2*pi*k*t/24)
                   + beta_s,k*sin(2*pi*k*t/24)
                 ].

    The empirical target is hourly, so q_K is evaluated on the 24 physical
    hours and held constant within each hour during fine-grid covariance
    propagation. Starting from K=0, the fitter increments K until the relative
    reduction in propagated hourly variance MSE falls below a supplied border.
    """

    def __init__(self, model): self.model = model

    @staticmethod
    def design(time_hours, order):
        time_hours = np.asarray(time_hours, dtype=float).reshape(-1)
        order = int(order)
        if time_hours.size < 1 or not np.all(np.isfinite(time_hours)): raise ValueError("time_hours must contain finite values.")
        if order < 0 or order > 11: raise ValueError("harmonic order must lie in [0, 11] for 24 hourly samples.")
        omega = 2.0 * np.pi / 24.0
        columns = [np.ones(time_hours.size, dtype=float)]
        for k in range(1, order + 1):
            columns.extend((np.cos(k * omega * time_hours), np.sin(k * omega * time_hours)))
        return np.column_stack(columns)

    @staticmethod
    def amplitude_phase(beta):
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.size < 1 or beta.size % 2 != 1 or not np.all(np.isfinite(beta)): raise ValueError("beta must contain 1+2K finite harmonic coefficients.")
        order = (beta.size - 1) // 2
        omega = 2.0 * np.pi / 24.0
        harmonics = []
        for k in range(1, order + 1):
            beta_c, beta_s = float(beta[2 * k - 1]), float(beta[2 * k])
            harmonics.append({
                "order": k,
                "amplitude": float(np.hypot(beta_c, beta_s)),
                "phase_h": float(np.mod(np.arctan2(beta_s, beta_c) / (k * omega), 24.0 / k)),
            })
        return {"variance_baseline": float(beta[0]), "harmonics": harmonics}

    @staticmethod
    def _validate_fit_inputs(process_kernel, diffusion_time_hours, observed_variance, initial_variance, fit_mask):
        process_kernel = np.asarray(process_kernel, dtype=float)
        diffusion_time_hours = np.asarray(diffusion_time_hours, dtype=float).reshape(-1)
        observed_variance = np.asarray(observed_variance, dtype=float).reshape(-1)
        if process_kernel.ndim != 2: raise ValueError("process_kernel must be a two-dimensional matrix.")
        if process_kernel.shape[0] != observed_variance.size: raise ValueError("process_kernel rows must match observed_variance.")
        if process_kernel.shape[1] != diffusion_time_hours.size: raise ValueError("process_kernel columns must match diffusion_time_hours.")
        if not np.all(np.isfinite(process_kernel)): raise ValueError("process_kernel must contain finite values.")
        if not np.all(np.isfinite(observed_variance)) or np.any(observed_variance < 0.0): raise ValueError("observed_variance must contain finite nonnegative values.")

        if initial_variance is None: initial_variance = np.zeros_like(observed_variance)
        initial_variance = np.asarray(initial_variance, dtype=float).reshape(-1)
        if initial_variance.size != observed_variance.size or not np.all(np.isfinite(initial_variance)) or np.any(initial_variance < 0.0): raise ValueError("initial_variance must match observed_variance and be finite and nonnegative.")

        if fit_mask is None: fit_mask = np.ones(observed_variance.size, dtype=bool)
        fit_mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
        if fit_mask.size != observed_variance.size or np.count_nonzero(fit_mask) < 3: raise ValueError("fit_mask must select at least three observed variance samples.")
        return process_kernel, diffusion_time_hours, observed_variance, initial_variance, fit_mask

    def _fit_order(self, order, process_kernel, diffusion_time_hours, observed_variance, initial_variance, fit_mask):
        order = int(order)
        hour_nodes = np.arange(24, dtype=float)
        Phi_hour = self.design(hour_nodes, order)
        hour_bin = np.floor(np.mod(diffusion_time_hours, 24.0) + 1e-12).astype(int)
        hour_bin = np.clip(hour_bin, 0, 23)
        Phi_fine = Phi_hour[hour_bin]
        propagated_design = process_kernel @ Phi_fine
        X = propagated_design[fit_mask]
        y = (observed_variance - initial_variance)[fit_mask]

        constant_response = propagated_design[:, 0]
        response_fit = constant_response[fit_mask]
        denominator = float(response_fit @ response_fit)
        constant_q = float((response_fit @ y) / denominator) if denominator > 0.0 else float(np.mean(y))
        positivity_eps = max(1e-12, 1e-10 * max(1.0, abs(constant_q)))
        constant_q = max(constant_q, 10.0 * positivity_eps)

        if order == 0:
            beta = np.array([constant_q], dtype=float)
        else:
            x0 = np.zeros(1 + 2 * order, dtype=float)
            x0[0] = constant_q
            beta = self.model._least_squares_qp(
                X, y,
                A_ub=-Phi_hour, b_ub=-positivity_eps * np.ones(Phi_hour.shape[0], dtype=float),
                ridge=self.model._FIT_RIDGE, x0=x0,
            )

        hourly_variance = np.maximum(Phi_hour @ beta, 0.0)
        instantaneous_variance = hourly_variance[hour_bin]
        process_variance = np.maximum(propagated_design @ beta, 0.0)
        propagated_variance = np.maximum(initial_variance + process_variance, 0.0)
        propagated_std = np.sqrt(propagated_variance)
        observed_std = np.sqrt(observed_variance)
        variance_residual = propagated_variance - observed_variance
        std_residual = propagated_std - observed_std
        fit_variance_residual = variance_residual[fit_mask]
        fit_std_residual = std_residual[fit_mask]

        result = {
            "order": order, "beta": np.asarray(beta, dtype=float),
            "hourly_process_variance": hourly_variance,
            "instantaneous_variance": instantaneous_variance,
            "diffusion": np.sqrt(instantaneous_variance),
            "initial_variance": initial_variance.copy(),
            "process_variance": process_variance,
            "propagated_variance": propagated_variance,
            "propagated_std": propagated_std,
            "fit_mask": fit_mask.copy(),
            "variance_residual": variance_residual, "std_residual": std_residual,
            "variance_mse": float(np.mean(fit_variance_residual ** 2)),
            "variance_rmse": float(np.sqrt(np.mean(fit_variance_residual ** 2))),
            "std_rmse": float(np.sqrt(np.mean(fit_std_residual ** 2))),
            "std_mae": float(np.mean(np.abs(fit_std_residual))),
        }
        result.update(self.amplitude_phase(beta))
        return result

    def fit(self, process_kernel, diffusion_time_hours, observed_variance, initial_variance=None, fit_mask=None, max_order=11, min_relative_improvement=0.05):
        process_kernel, diffusion_time_hours, observed_variance, initial_variance, fit_mask = self._validate_fit_inputs(
            process_kernel, diffusion_time_hours, observed_variance, initial_variance, fit_mask,
        )
        max_order = int(max_order)
        min_relative_improvement = float(min_relative_improvement)
        if max_order < 1 or max_order > 11: raise ValueError("max_order must lie in [1, 11].")
        if not 0.0 <= min_relative_improvement < 1.0: raise ValueError("min_relative_improvement must lie in [0, 1).")

        selected = self._fit_order(0, process_kernel, diffusion_time_hours, observed_variance, initial_variance, fit_mask)
        selected["relative_improvement"] = np.nan
        selected["accepted"] = True
        history = [{
            "order": 0, "parameters": 1, "variance_mse": selected["variance_mse"],
            "variance_rmse": selected["variance_rmse"], "std_rmse": selected["std_rmse"],
            "relative_improvement": np.nan, "accepted": True, "reason": "constant baseline",
        }]
        previous = selected
        stop_reason = f"reached max_order={max_order}"

        for order in range(1, max_order + 1):
            try:
                candidate = self._fit_order(order, process_kernel, diffusion_time_hours, observed_variance, initial_variance, fit_mask)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                history.append({
                    "order": order, "parameters": 1 + 2 * order,
                    "variance_mse": np.nan, "variance_rmse": np.nan, "std_rmse": np.nan,
                    "relative_improvement": np.nan, "accepted": False, "reason": f"solver failed: {exc}",
                })
                stop_reason = f"stopped at K={order}: solver failed"
                break

            denominator = max(previous["variance_mse"], np.finfo(float).eps)
            relative_improvement = float((previous["variance_mse"] - candidate["variance_mse"]) / denominator)
            accepted = relative_improvement >= min_relative_improvement
            history.append({
                "order": order, "parameters": 1 + 2 * order,
                "variance_mse": candidate["variance_mse"], "variance_rmse": candidate["variance_rmse"],
                "std_rmse": candidate["std_rmse"], "relative_improvement": relative_improvement,
                "accepted": bool(accepted),
                "reason": "accepted" if accepted else f"relative improvement < {min_relative_improvement:.6g}",
            })
            if not accepted:
                stop_reason = f"stopped before K={order}: relative improvement {relative_improvement:.6g} < {min_relative_improvement:.6g}"
                break

            candidate["relative_improvement"] = relative_improvement
            candidate["accepted"] = True
            selected, previous = candidate, candidate

        selected["selected_order"] = int(selected["order"])
        selected["selection_history"] = history
        selected["min_relative_improvement"] = min_relative_improvement
        selected["max_order"] = max_order
        selected["stop_reason"] = stop_reason
        return selected

    def evaluate(self, time_hours, beta):
        time_hours = np.asarray(time_hours, dtype=float).reshape(-1)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.size < 1 or beta.size % 2 != 1: raise ValueError("beta must contain 1+2K coefficients.")
        order = (beta.size - 1) // 2
        Phi_hour = self.design(np.arange(24, dtype=float), order)
        hour_bin = np.floor(np.mod(time_hours, 24.0) + 1e-12).astype(int)
        hour_bin = np.clip(hour_bin, 0, 23)
        hourly_variance = np.maximum(Phi_hour @ beta, 0.0)
        q = hourly_variance[hour_bin]
        return {"variance": q, "std": np.sqrt(q), "hourly_variance": hourly_variance, "order": order}


class brute_force_params(qp_params):
    """Profile a separate F0 for every wm_4 fitting trajectory.

    The structural coefficients [r_F,r_U,r_R] remain shared across all
    trajectories. For fixed shared coefficients, each trajectory independently
    profiles the supplied F0 grid and optimizes its own R0. The shared
    coefficients are then refit against the selected trajectory states and the
    procedure alternates to convergence.
    """

    @staticmethod
    def _f0_grid(f0_grid=None, f0_min=-1.0, f0_max=1.0, f0_steps=81):
        if f0_grid is None:
            f0_min, f0_max, f0_steps = float(f0_min), float(f0_max), int(f0_steps)
            if not np.isfinite(f0_min) or not np.isfinite(f0_max) or f0_max < f0_min: raise ValueError("f0_min and f0_max must be finite with f0_min <= f0_max.")
            if f0_steps < 2: raise ValueError("f0_steps must be at least 2.")
            values = np.linspace(f0_min, f0_max, f0_steps)
        else:
            values = np.asarray(f0_grid, dtype=float).reshape(-1)
            if values.size < 1 or not np.all(np.isfinite(values)): raise ValueError("f0_grid must contain at least one finite value.")
        values = np.unique(values)
        if np.any(values < -1.0) or np.any(values > 1.0): raise ValueError("Every brute-force F0 candidate must lie in [-1, 1].")
        return values

    def _state_loss(self, data, init_cond, r):
        observed = self._observed_state(data, init_cond)
        mask = observed["transition_mask"]
        residual = (observed["X"] @ np.asarray(r, dtype=float) - observed["y"])[mask]
        return float(np.mean(residual ** 2)) if residual.size else np.inf

    def _optimize_rate_state(self, data, r, f0, seed_r0, r_bounds=(-np.inf, np.inf)):
        r = wm_4_validation.r_coefficients(r)
        f0, seed_r0 = float(f0), float(seed_r0)

        def residual(rate):
            observed = self._observed_state(data, np.array([f0, float(rate)], dtype=float))
            mask = observed["transition_mask"]
            return (observed["X"] @ r - observed["y"])[mask]

        base = residual(0.0)
        unit = residual(1.0) - base
        lb, ub = float(r_bounds[0]), float(r_bounds[1])
        if lb > ub: raise ValueError("r_bounds must satisfy lower <= upper.")
        rate = self.model._least_squares_qp(
            unit[:, None], -base, lb=np.array([lb]), ub=np.array([ub]),
            ridge=self.model._FIT_RIDGE, x0=np.array([np.clip(seed_r0, lb, ub)]),
        )
        return wm_4_validation.initial_condition(np.array([f0, float(rate[0])], dtype=float))

    def _profile_state(self, data, r, seed, grid, r_bounds=(-np.inf, np.inf)):
        seed = wm_4_validation.initial_condition(seed)
        candidates, profile = [], []
        for f0 in grid:
            try:
                init = self._optimize_rate_state(data, r, float(f0), seed[1], r_bounds=r_bounds)
                loss = self._state_loss(data, init, r)
                if not np.isfinite(loss): raise RuntimeError("non-finite structural loss")
                candidates.append((loss, init))
                profile.append({"F0": float(f0), "success": True, "qp_loss": float(loss), "R0": float(init[1]), "reason": ""})
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                profile.append({"F0": float(f0), "success": False, "qp_loss": np.nan, "R0": np.nan, "reason": str(exc)})
        if not candidates: raise RuntimeError("wm_4 brute-force F0 search found no feasible candidate for a fitting trajectory.")
        _, best = min(candidates, key=lambda item: item[0])
        return best, profile

    def _alternating_fit(self, datasets, seeds, grid, stabilize=True, max_state_iter=25, state_tol=1e-8, r_bounds=(-np.inf, np.inf), verbose=False):
        init_conds = []
        for seed in seeds:
            seed = wm_4_validation.initial_condition(seed)
            f0 = float(grid[np.argmin(np.abs(grid - seed[0]))])
            init_conds.append(wm_4_validation.initial_condition(np.array([f0, seed[1]], dtype=float)))

        history, total_qp_time, previous_loss = [], 0.0, np.inf
        profiles = [[] for _ in datasets]

        for iteration in range(1, int(max_state_iter) + 1):
            r, _, _, solve_time = self._solve_qp(datasets, init_conds, stabilize=stabilize)
            total_qp_time += solve_time

            optimized, profiles = [], []
            for data, init in zip(datasets, init_conds):
                best, profile = self._profile_state(data, r, init, grid, r_bounds=r_bounds)
                optimized.append(best); profiles.append(profile)

            r, observed, qp_loss, solve_time = self._solve_qp(datasets, optimized, stabilize=stabilize)
            total_qp_time += solve_time
            max_f0_change = max(abs(float(new[0] - old[0])) for new, old in zip(optimized, init_conds))
            max_r0_change = max(abs(float(new[1] - old[1])) for new, old in zip(optimized, init_conds))
            max_state_change = max(max_f0_change, max_r0_change)
            loss_change = abs(previous_loss - qp_loss) if np.isfinite(previous_loss) else np.inf
            history.append({
                "iteration": iteration, "qp_loss": float(qp_loss),
                "max_F0_change": float(max_f0_change), "max_R0_change": float(max_r0_change),
                "max_state_change": float(max_state_change), "loss_change": float(loss_change),
                "F0": np.asarray([init[0] for init in optimized], dtype=float),
            })
            init_conds = optimized
            if verbose:
                f0_text = np.array2string(np.asarray([init[0] for init in init_conds]), precision=4, separator=",")
                print(f"[brute_F0] iteration={iteration} loss={qp_loss:.8g} state_change={max_state_change:.3g} F0={f0_text}")
            if max_state_change <= state_tol and loss_change <= state_tol: break
            previous_loss = qp_loss

        return r, observed, init_conds, float(qp_loss), float(total_qp_time), history, profiles

    def fit_many(self, input_data, initial_conditions, *, f0_grid=None, f0_min=-1.0, f0_max=1.0, f0_steps=81, stabilize=True, max_state_iter=25, state_tol=1e-8, r_bounds=(-np.inf, np.inf), verbose=False, **_legacy_kwargs):
        self._reject_removed_diffusion_kwargs(_legacy_kwargs)
        if not isinstance(input_data, (list, tuple)) or len(input_data) < 1: raise TypeError("input_data must be a non-empty list or tuple of wm_4 fitting dictionaries.")
        fit_start = perf_counter()
        datasets = [self._prepare_data(data) for data in input_data]
        raw_init = np.asarray(initial_conditions, dtype=float)
        if raw_init.shape == (2,): seeds = [wm_4_validation.initial_condition(raw_init) for _ in datasets]
        elif raw_init.shape == (len(datasets), 2): seeds = [wm_4_validation.initial_condition(row) for row in raw_init]
        else: raise ValueError("initial_conditions must be [F0,R0] or one [F0,R0] row per fitting trajectory.")

        grid = self._f0_grid(f0_grid=f0_grid, f0_min=f0_min, f0_max=f0_max, f0_steps=f0_steps)
        r, observed, init_conds, qp_loss, qp_solve_time, history, profiles = self._alternating_fit(
            datasets, seeds, grid, stabilize=stabilize,
            max_state_iter=max_state_iter, state_tol=state_tol, r_bounds=r_bounds, verbose=verbose,
        )

        states = [self._state_from_parameters(data, init, r, full=True) for data, init in zip(datasets, init_conds)]
        losses, residuals = [], []
        for data, state in zip(datasets, states):
            obs = np.asarray(data["observation_mask"], dtype=bool)
            residual = data["I"][obs] - state["I"][obs]
            residuals.append(residual); losses.append(float(np.mean(residual ** 2)))
        deterministic_loss = float(np.mean(np.concatenate(residuals) ** 2))
        f0_values = np.asarray([init[0] for init in init_conds], dtype=float)
        feasible = np.asarray([sum(bool(row["success"]) for row in profile) for profile in profiles], dtype=int)

        result = {
            "init_cond": np.vstack(init_conds), "initial_condition_seed": raw_init.copy(),
            "r": np.asarray(r, dtype=float).copy(), "deterministic_state": states[0], "deterministic_states": states,
            "observed_states": observed, "per_trajectory_loss": np.asarray(losses, dtype=float),
            "recovered_model_loss": deterministic_loss, "deterministic_loss": deterministic_loss,
            "structural_viable": True, "fit_trajectories": len(datasets),
            "initial_rate_mismatch": np.asarray([state["initial_rate_mismatch"] for state in observed], dtype=float),
            "F0": f0_values, "f0_grid": grid.copy(), "f0_profile": profiles,
            "state_optimization_history": history, "state_iterations": len(history),
        }
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        result["fit_method"] = "brute_force_many"
        result["fit_initial_state"] = True
        result["fit_initial_rate"] = True
        result["deterministic_fit"] = True
        result["brute_force_candidates"] = len(grid)
        result["brute_force_feasible_candidates"] = feasible
        result["brute_force_evaluations"] = int(len(grid) * len(datasets) * len(history))
        if verbose: print(f"[brute_F0] trajectories={len(datasets)} selected_F0={np.array2string(f0_values, precision=4, separator=',')} loss={qp_loss:.8g}")
        return result

    def fit(self, input_data, initial_conditions, **kwargs):
        result = self.fit_many([input_data], np.asarray(initial_conditions, dtype=float), **kwargs)
        result["init_cond"] = result["init_cond"][0].copy()
        result["deterministic_state"] = result["deterministic_states"][0]
        result["observed_state"] = result["observed_states"][0]
        result["F0"] = float(result["F0"][0])
        result["f0_profile"] = result["f0_profile"][0]
        result["brute_force_feasible_candidates"] = int(result["brute_force_feasible_candidates"][0])
        result["fit_method"] = "brute_force"
        result["fit_trajectories"] = 1
        return result

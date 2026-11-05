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
        strict = 10.0 * eps
        rows = [
            np.array([1.0, -1.0, 0.0]),
            np.array([-1.0, -1.0, 1.0]),
        ]
        rhs = [0.0, 0.0]
        if stabilize:
            rows.extend([
                np.array([dt * dt, 0.0, -2.0 * dt]),
                np.array([-dt, 0.0, 1.0]),
            ])
            rhs.extend([4.0 - eps, -strict])
        lb = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        ub = np.array([-strict, 0.0, -strict], dtype=float)
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







class constant_diffusion_qp:
    """Separate constant demand-observation variance from accumulated WM4 process variance.

    D_k = B_k + R_k + eta_k, eta_k iid N(0, observation_variance)
    dF = R dt
    dR = (r_F F + r_U U + r_R R) dt + sigma dW

    For zero-mean demand residuals e, E[e_k**2] = observation_variance
    + initial_variance[k] + q * process_kernel[k].sum(), q=sigma**2.
    Both variances are fitted as nonnegative parameters in a convex QP.
    """

    def __init__(self, model): self.model = model

    def fit(self, process_kernel, observed_variance, initial_variance=None, fit_mask=None, residuals=None):
        process_kernel = np.asarray(process_kernel, dtype=float)
        observed_variance = np.asarray(observed_variance, dtype=float).reshape(-1)
        if process_kernel.ndim != 2: raise ValueError("process_kernel must be a two-dimensional matrix.")
        if process_kernel.shape[0] != observed_variance.size: raise ValueError("process_kernel rows must match observed_variance.")
        if not np.all(np.isfinite(process_kernel)): raise ValueError("process_kernel must contain finite values.")
        if not np.all(np.isfinite(observed_variance)) or np.any(observed_variance < 0.0): raise ValueError("observed_variance must contain finite nonnegative values.")

        if residuals is not None:
            residuals = np.asarray(residuals, dtype=float)
            if residuals.ndim != 2 or residuals.shape[1] != observed_variance.size or residuals.shape[0] < 1:
                raise ValueError("residuals must be a nonempty (days, hours) matrix.")
            if not np.all(np.isfinite(residuals)): raise ValueError("demand residuals must be finite.")
            observed_variance = np.mean(residuals ** 2, axis=0)

        if initial_variance is None: initial_variance = np.zeros_like(observed_variance)
        initial_variance = np.asarray(initial_variance, dtype=float).reshape(-1)
        if initial_variance.size != observed_variance.size or not np.all(np.isfinite(initial_variance)) or np.any(initial_variance < 0.0): raise ValueError("initial_variance must match observed_variance and be finite and nonnegative.")

        if fit_mask is None: fit_mask = np.ones(observed_variance.size, dtype=bool)
        fit_mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
        if fit_mask.size != observed_variance.size or np.count_nonzero(fit_mask) < 2: raise ValueError("fit_mask must select at least two variance samples.")

        response = process_kernel @ np.ones(process_kernel.shape[1], dtype=float)
        X = np.column_stack((np.ones(np.count_nonzero(fit_mask)), response[fit_mask]))
        y = (observed_variance - initial_variance)[fit_mask]
        if np.ptp(response[fit_mask]) <= 1e-12:
            raise ValueError("Cannot distinguish constant observation error from diffusion without distinct process horizons.")
        seed = np.maximum(np.linalg.lstsq(X, y, rcond=None)[0], 1e-12)
        fitted = self.model._least_squares_qp(
            X, y, lb=np.zeros(2, dtype=float),
            ridge=self.model._FIT_RIDGE, x0=seed,
        )
        observation_q, q = np.maximum(np.asarray(fitted, dtype=float), 0.0)
        observation_variance = np.full(observed_variance.size, observation_q, dtype=float)
        process_variance = np.maximum(response * q, 0.0)
        propagated_variance = np.maximum(initial_variance + process_variance + observation_variance, 0.0)
        propagated_std = np.sqrt(propagated_variance)
        observed_std = np.sqrt(observed_variance)
        variance_residual = propagated_variance - observed_variance
        std_residual = propagated_std - observed_std

        return {
            "q": float(q), "sigma": float(np.sqrt(q)),
            "observation_q": float(observation_q), "observation_sigma": float(np.sqrt(observation_q)),
            "observation_variance": observation_variance,
            "initial_variance": initial_variance.copy(),
            "process_variance": process_variance, "propagated_variance": propagated_variance,
            "propagated_std": propagated_std, "observed_variance": observed_variance.copy(),
            "fit_mask": fit_mask.copy(), "variance_residual": variance_residual,
            "std_residual": std_residual,
            "variance_mse": float(np.mean(variance_residual[fit_mask] ** 2)),
            "variance_rmse": float(np.sqrt(np.mean(variance_residual[fit_mask] ** 2))),
            "std_rmse": float(np.sqrt(np.mean(std_residual[fit_mask] ** 2))),
            "std_mae": float(np.mean(np.abs(std_residual[fit_mask]))),
        }

    @staticmethod
    def one_step_diffusion(residuals, dt):
        """Gaussian MLE sigma**2 = mean(e**2)/dt when observations are noiseless.

        If demand contains independent observation error, one-step residuals
        also contain errors from the state inferred at the previous sample.
        Use fit() on demand-residual moments across horizons to distinguish the
        constant observation floor from accumulating Brownian variance.
        """
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        dt = float(dt)
        if residuals.size == 0 or not np.all(np.isfinite(residuals)): raise ValueError("residuals must be nonempty and finite.")
        if not np.isfinite(dt) or dt <= 0.0: raise ValueError("dt must be positive.")
        q = float(np.mean(residuals ** 2) / dt)
        return {"q": q, "sigma": float(np.sqrt(q))}

    @staticmethod
    def evaluate(size, q, observation_q=0.0):
        size, q, observation_q = int(size), float(q), float(observation_q)
        if size < 1: raise ValueError("size must be positive.")
        if not np.isfinite(q) or q < 0.0: raise ValueError("q must be finite and nonnegative.")
        if not np.isfinite(observation_q) or observation_q < 0.0: raise ValueError("observation_q must be finite and nonnegative.")
        return {
            "variance": np.full(size, q, dtype=float),
            "std": np.full(size, np.sqrt(q), dtype=float),
            "observation_variance": np.full(size, observation_q, dtype=float),
            "observation_std": np.full(size, np.sqrt(observation_q), dtype=float),
        }


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

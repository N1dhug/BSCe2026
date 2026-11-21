from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

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

    def _observed_state(self, data, init_cond, fit_intercept=False):
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
        if fit_intercept: X = np.column_stack((X, np.full(n - 1, dt)))
        y = np.diff(R)
        mask = np.asarray(data["transition_mask"], dtype=bool)
        return {
            "F": F, "R": R, "R_observed": R_observed, "P_R": P_R,
            "X": X, "y": y, "transition_mask": mask,
            "initial_rate_mismatch": float(R_observed[0] - init_cond[1]),
        }

    def _qp_constraints(self, stabilize=True, fit_intercept=False):
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
        if fit_intercept:
            rows = [np.r_[row, 0.0] for row in rows]
            rows.extend((np.array([1.0, -1.0, 0.0, 1.0]), np.array([1.0, -1.0, 0.0, -1.0]),
                         np.array([-1.0, -1.0, 1.0, 1.0]), np.array([-1.0, -1.0, 1.0, -1.0])))
            rhs.extend((0.0, 0.0, 0.0, 0.0))
            lb, ub = np.r_[lb, -np.inf], np.r_[ub, np.inf]
        return np.vstack(rows), np.asarray(rhs, dtype=float), lb, ub

    def _solve_qp(self, datasets, init_conds, stabilize=True, fit_intercept=False):
        observed = [self._observed_state(data, init, fit_intercept=fit_intercept) for data, init in zip(datasets, init_conds)]
        X = np.vstack([state["X"][state["transition_mask"]] for state in observed])
        y = np.concatenate([state["y"][state["transition_mask"]] for state in observed])
        if X.shape[0] < 1: raise ValueError("At least one wm_4 state transition is required for fitting.")
        A_ub, b_ub, lb, ub = self._qp_constraints(stabilize=stabilize, fit_intercept=fit_intercept)
        start = perf_counter()
        r = self.model._least_squares_qp(X, y, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE)
        solve_time = perf_counter() - start
        r = np.asarray(r, dtype=float)
        residual = X @ r - y
        if fit_intercept:
            wm_4_validation.recovered_structural_coefficients(r[:3])
            return r[:3], observed, float(np.mean(residual ** 2)), solve_time, float(r[3])
        r = wm_4_validation.recovered_structural_coefficients(r)
        return r, observed, float(np.mean(residual ** 2)), solve_time

    def _state_from_parameters(self, data, init_cond, coefficients, kappa_r=None, full=False, r_0=0.0):
        r = wm_4_validation.r_coefficients(coefficients) if kappa_r is None else wm_4_validation.legacy_r(coefficients, kappa_r)
        state = self.model._solve_deterministic(init_cond, data["U"], data["PE"], data["Time"], data["O"], r, r_0=r_0)
        if full: return state
        return {key: value for key, value in state.items() if key not in ("P_R", "P_A")}

    def _finish_result(self, result, qp_loss, qp_solve_time, fit_start, stabilize):
        r = np.asarray(result["r"], dtype=float)
        q, kappa_R = self._equivalent_q(r)
        branch, zeta = self._damping_branch(r)
        elapsed = perf_counter() - fit_start
        result.update({
            "q": q, "kappa_R": float(kappa_R), "r_0": float(result.get("r_0", 0.0)),
            "fit_method": "qp", "damping_branch": branch, "damping_ratio": float(zeta),
            "qp_loss": float(qp_loss), "qp_solve_time": float(qp_solve_time),
            "qp_fit_time": elapsed, "stabilize_euler": bool(stabilize),
            "fit_initial_state": False,
        })
        return result

    def fit(self, input_data, initial_conditions, *, fit_initial_state=False, stabilize=True, fit_intercept=False, verbose=False, **_legacy_kwargs):
        self._reject_removed_diffusion_kwargs(_legacy_kwargs)
        if fit_initial_state: raise ValueError("wm_4 QP fitting requires fixed initial [F0, R0]; fit_initial_state is no longer supported.")
        fit_start = perf_counter()
        data = self._prepare_data(input_data)
        init_cond = wm_4_validation.initial_condition(initial_conditions)
        solve = self._solve_qp([data], [init_cond], stabilize=stabilize, fit_intercept=fit_intercept)
        r, observed, qp_loss, qp_solve_time = solve[:4]
        r_0 = solve[4] if fit_intercept else 0.0
        state = self._state_from_parameters(data, init_cond, r, full=True, r_0=r_0)
        obs = np.asarray(data["observation_mask"], dtype=bool)
        deterministic_loss = float(np.mean((data["I"][obs] - state["I"][obs]) ** 2))
        result = {
            "init_cond": init_cond.copy(), "initial_condition_seed": init_cond.copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(),
            "O": data["O"].copy(), "observation_mask": obs.copy(), "transition_mask": data["transition_mask"].copy(),
            "r": r.copy(), "r_0": float(r_0), "fit_intercept": bool(fit_intercept), "deterministic_state": state, "observed_state": observed[0],
            "recovered_model_loss": deterministic_loss, "deterministic_loss": deterministic_loss,
            "structural_viable": True, "initial_rate_mismatch": observed[0]["initial_rate_mismatch"],
        }
        result = self._finish_result(result, qp_loss, qp_solve_time, fit_start, stabilize)
        if verbose: print(f"[qp] transitions={np.count_nonzero(data['transition_mask'])} loss={qp_loss:.8g} free_run_mse={deterministic_loss:.8g} solve={qp_solve_time:.4f}s")
        return result

    def fit_many(self, input_data, initial_conditions, *, fit_initial_state=False, stabilize=True, fit_intercept=False, verbose=False, **_legacy_kwargs):
        self._reject_removed_diffusion_kwargs(_legacy_kwargs)
        if fit_initial_state: raise ValueError("wm_4 QP fitting requires fixed initial [F0, R0]; fit_initial_state is no longer supported.")
        if not isinstance(input_data, (list, tuple)) or len(input_data) < 1: raise TypeError("input_data must be a non-empty list or tuple of wm_4 fitting dictionaries.")
        fit_start = perf_counter()
        datasets = [self._prepare_data(data) for data in input_data]
        raw_init = np.asarray(initial_conditions, dtype=float)
        if raw_init.shape == (2,): init_conds = [wm_4_validation.initial_condition(raw_init) for _ in datasets]
        elif raw_init.shape == (len(datasets), 2): init_conds = [wm_4_validation.initial_condition(row) for row in raw_init]
        else: raise ValueError("initial_conditions must be [F0,R0] or one [F0,R0] row per fitting trajectory.")

        solve = self._solve_qp(datasets, init_conds, stabilize=stabilize, fit_intercept=fit_intercept)
        r, observed, qp_loss, qp_solve_time = solve[:4]
        r_0 = solve[4] if fit_intercept else 0.0
        states = [self._state_from_parameters(data, init, r, full=True, r_0=r_0) for data, init in zip(datasets, init_conds)]
        losses = []
        for data, state in zip(datasets, states):
            obs = np.asarray(data["observation_mask"], dtype=bool)
            losses.append(float(np.mean((data["I"][obs] - state["I"][obs]) ** 2)))
        residual = np.concatenate([data["I"][np.asarray(data["observation_mask"], dtype=bool)] - state["I"][np.asarray(data["observation_mask"], dtype=bool)] for data, state in zip(datasets, states)])
        deterministic_loss = float(np.mean(residual ** 2))
        result = {
            "init_cond": np.vstack(init_conds), "initial_condition_seed": raw_init.copy(),
            "r": r.copy(), "r_0": float(r_0), "fit_intercept": bool(fit_intercept), "deterministic_state": states[0], "deterministic_states": states,
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

    def optimize_prediction_initial_state(self, input_data, r, f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0), r_0=0.0):
        """Oracle/diagnostic prediction-state fit for a fixed parameter vector."""
        data = self._prepare_data(input_data)
        r = wm_4_validation.r_coefficients(r)
        base = self._state_from_parameters(data, np.array([0.0, 0.0]), r, r_0=r_0)
        f_unit = self._state_from_parameters(data, np.array([1.0, 0.0]), r, r_0=r_0)
        r_unit = self._state_from_parameters(data, np.array([0.0, 1.0]), r, r_0=r_0)
        obs = np.asarray(data["observation_mask"], dtype=bool)
        X = np.column_stack(((f_unit["I"] - base["I"])[obs], (r_unit["I"] - base["I"])[obs]))
        y = data["I"][obs] - base["I"][obs]
        lb = np.array([float(f_bounds[0]), float(r_bounds[0])], dtype=float)
        ub = np.array([float(f_bounds[1]), float(r_bounds[1])], dtype=float)
        init = self.model._least_squares_qp(X, y, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE)
        state = self._state_from_parameters(data, init, r, full=True, r_0=r_0)
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
            "state_optimization_history": history, "state_iterations": len(history), "state_tol": float(state_tol),
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
            "state_optimization_history": history, "state_iterations": len(history), "state_tol": float(state_tol),
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


class joint_qp_params(qp_params):
    """Single convex joint fit of [r_F*F0, r_F, r_U, r_R]."""

    def fit(self, input_data, initial_conditions=None, *, stabilize=True, enforce_path=True,
            f_bounds=(-1.0, 1.0), verbose=False):
        from src.solver.joint_drift_qp import joint_drift_qp
        start = perf_counter()
        data = self._prepare_data(input_data)
        R = data["I"] - data["O"]
        if initial_conditions is not None:
            seed = wm_4_validation.initial_condition(initial_conditions)
            if not np.isclose(seed[1], R[0], atol=1e-8):
                raise ValueError("Joint fit requires R0=I[0]-O[0].")
        A, b, lb, ub = self._qp_constraints(stabilize=stabilize)
        joint = joint_drift_qp.solve(
            R, data["U"], self.model.dt, data["U"][:-1, None], R[:-1, None],
            (A, b, lb, ub), transition_mask=data["transition_mask"],
            f_bounds=f_bounds, ridge=self.model._FIT_RIDGE, enforce_path=enforce_path)
        r = wm_4_validation.r_coefficients(joint["coefficients"])
        state = self.model._solve_deterministic(joint["init_cond"], data["U"], data["PE"],
                                               data["Time"], data["O"], r)
        mask = data["observation_mask"]
        result = {**joint, "r": r.copy(), "deterministic_state": state,
                  "observation_mask": mask.copy(), "fit_method": "wm4_joint_qp",
                  "deterministic_loss": float(np.mean((state["I"][mask]-data["I"][mask])**2)),
                  "fit_time": perf_counter()-start, "success": True}
        if verbose: print(f"[wm4 joint] qp_loss={joint['qp_loss']:.8g}, F0={joint['init_cond'][0]:.6g}")
        return result


    def fit_many(self, input_data, initial_conditions=None, *, stabilize=True, enforce_path=True,
                 f_bounds=(-1.0, 1.0), verbose=False):
        """Joint convex fit with shared [theta_F,theta_U,theta_R] and individual F0."""
        from src.solver.joint_drift_qp import joint_drift_qp
        if not isinstance(input_data, (list, tuple)) or not input_data:
            raise TypeError("WM4 joint QP requires a nonempty list of fitting days.")
        started = perf_counter()
        data = [self._prepare_data(item) for item in input_data]
        rates = [item["I"]-item["O"] for item in data]
        if initial_conditions is not None:
            init = np.asarray(initial_conditions, dtype=float)
            if init.shape != (len(data), 2):
                raise ValueError("Optional initial conditions require one [F0,R0] per day.")
            if any(not np.isclose(seed[1], R[0], atol=1e-8) for seed, R in zip(init, rates)):
                raise ValueError("Observed initial rates must match I[0]-O[0].")
        constraints = self._qp_constraints(stabilize=stabilize)
        joint = joint_drift_qp.solve_many(
            rates, [item["U"] for item in data], self.model.dt, constraints,
            transition_masks=[item["transition_mask"] for item in data],
            f_bounds=f_bounds, ridge=self.model._FIT_RIDGE, enforce_path=enforce_path)
        theta = wm_4_validation.r_coefficients(joint["coefficients"])
        states = [self.model._solve_deterministic(initial, item["U"], item["PE"],
                  item["Time"], item["O"], theta)
                  for initial, item in zip(joint["init_conds"], data)]
        errors = np.concatenate([(state["I"]-item["I"])[item["observation_mask"]]
                                 for state, item in zip(states, data)])
        result = {**joint, "r": theta.copy(), "init_cond": joint["init_conds"].copy(),
                  "deterministic_states": states, "fit_trajectories": len(data),
                  "fit_method": "wm4_joint_qp_many",
                  "deterministic_loss": float(np.mean(errors**2)),
                  "fit_time": perf_counter()-started, "success": True}
        if verbose:
            print(f"[wm4 joint many] days={len(data)} qp_loss={joint['qp_loss']:.8g} "
                  f"rank={joint['design_rank']}/{len(joint['beta'])}")
        return result



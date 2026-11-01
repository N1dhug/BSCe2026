from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from src.validation.model.wm_5 import wm_5_validation
from src.validation.params.wm_5 import wm_5_params_validation


class qp_params:
    """Alternating *convex* QPs for parameter coefficients and F0 with fixed observed R0.

    Although each block problem is convex, their joint problem can be nonconvex
    because r_F*F0 is bilinear. All response features are observed during fitting.
    """

    def __init__(self, model): self.model = model

    def _prepare_data(self, input_data):
        data = wm_5_validation.fit_data(input_data, self.model.t_steps)
        if "O" in input_data:
            O = np.asarray(input_data["O"], dtype=float).reshape(-1)
            if O.size != self.model.t_steps or not np.isfinite(O).all():
                raise ValueError("WM5 input_data['O'] has invalid length or nonfinite values.")
        else:
            O = np.array([wm_5_validation.feature_value(self.model.O(t), "O(Time)") for t in data["Time"]])
        return {**data, "O": O}

    def _observed_state(self, data):
        R = np.asarray(data["I"] - data["O"], dtype=float)
        R0 = wm_5_params_validation.initial_rate(R[0])
        C = np.r_[0.0, np.cumsum(self.model.dt * R[:-1])]
        g = np.column_stack([[b.value(float(x)) for x in R[:-1]] for b in self.model.r_functions])
        h = np.column_stack([[b.value(float(x)) for x in data["U"][:-1]] for b in self.model.u_functions])
        if not np.isfinite(g).all() or not np.isfinite(h).all():
            raise ValueError("WM5 fitted response basis is nonfinite at a reconstructed state.")
        return {"R": R, "R0": R0, "C": C, "rate_features": g,
                "price_features": h, "y": np.diff(R),
                "mask": np.asarray(data["transition_mask"], dtype=bool)}

    def _design(self, obs, f0):
        F = float(f0) + obs["C"][:-1]
        return self.model.dt * np.column_stack((F, obs["rate_features"], obs["price_features"]))

    def _solve_parameters(self, observed, f0, stabilize):
        X = self._design(observed, f0)[observed["mask"]]
        y = observed["y"][observed["mask"]]
        constraints = wm_5_params_validation.constraints(self.model.r_functions, self.model.u_functions,
                                                          self.model.dt, stabilize=stabilize)
        started = perf_counter()
        p = self.model._least_squares_qp(X, y, A_ub=constraints[0], b_ub=constraints[1],
                                         lb=constraints[2], ub=constraints[3], ridge=self.model._FIT_RIDGE)
        elapsed = perf_counter() - started
        p = self.model.coefficients(p, stabilize=stabilize)
        return p, float(np.mean((X @ p - y) ** 2)), elapsed

    def _solve_initial_f(self, observed, p, f_bounds):
        # All dependence on F0 is the scalar dt*r_F*F0, so the exact QP
        # minimizer is the clipped unconstrained least-squares solution.
        mask = observed["mask"]
        y = observed["y"][mask]
        fixed = self._design(observed, 0.0)[mask] @ p
        rFdt = self.model.dt * p[0]
        if abs(rFdt) < 1e-14: raise RuntimeError("WM5 F0 is unidentifiable when r_F is zero.")
        return float(np.clip(np.mean(y - fixed) / rFdt, *f_bounds))

    def fit(self, input_data, initial_conditions=None, *, fit_initial_state=True, stabilize=True,
            max_state_iter=50, state_tol=1e-8, f_bounds=(-1.0, 1.0), verbose=False):
        started = perf_counter()
        max_state_iter, state_tol, f_bounds = wm_5_params_validation.fit_settings(max_state_iter, state_tol, f_bounds)
        data = self._prepare_data(input_data)
        observed = self._observed_state(data)
        R0 = observed["R0"]
        if initial_conditions is None: initial_conditions = [0.0, R0]
        seed = wm_5_validation.initial_condition(initial_conditions)
        if not np.isclose(seed[1], R0, atol=1e-10):
            raise ValueError("WM5 sequential fit requires fixed R0=Iobs[0]-O[0].")
        if not f_bounds[0] <= seed[0] <= f_bounds[1]: raise ValueError("F0 initial guess is outside f_bounds.")
        f0, history, qp_time, previous = float(seed[0]), [], 0.0, np.inf
        steps = max_state_iter if fit_initial_state else 1
        for iteration in range(1, steps + 1):
            p, qp_loss, solve_time = self._solve_parameters(observed, f0, stabilize)
            qp_time += solve_time
            if not fit_initial_state: break
            optimized = self._solve_initial_f(observed, p, f_bounds)
            p, qp_loss, solve_time = self._solve_parameters(observed, optimized, stabilize)
            qp_time += solve_time
            delta = abs(optimized - f0)
            loss_change = abs(previous - qp_loss) if np.isfinite(previous) else np.inf
            history.append({"iteration": iteration, "qp_loss": qp_loss, "max_state_change": delta,
                            "loss_change": loss_change})
            f0 = optimized
            if verbose: print(f"[wm5 QP] iteration={iteration}, qp_loss={qp_loss:.8g}, F0_change={delta:.3g}")
            if delta <= state_tol and loss_change <= state_tol: break
            previous = qp_loss

        init = np.array([f0, R0], dtype=float)
        state = self.model._solve_deterministic(init, data["U"], data["Time"], data["O"], p)
        mask = data["observation_mask"]
        error = state["I"][mask] - data["I"][mask]
        converged = (not fit_initial_state or
                     bool(history and history[-1]["max_state_change"] <= state_tol and
                          history[-1]["loss_change"] <= state_tol))
        return {"coefficients": p.copy(), "coefficient_names": self.model.parameter_names,
                "init_cond": init, "initial_condition_seed": seed.copy(),
                "deterministic_state": state, "observed_state": observed,
                "observation_mask": mask.copy(), "transition_mask": observed["mask"].copy(),
                "qp_loss": qp_loss, "deterministic_loss": float(np.mean(error ** 2)),
                "fit_time": perf_counter() - started, "qp_solve_time": qp_time,
                "state_iterations": len(history), "state_optimization_history": history,
                "state_converged": converged, "state_tol": state_tol, "success": True,
                "fit_method": "wm5_sequential_qp" if fit_initial_state else "wm5_qp"}

    def optimize_prediction_initial_state(self, input_data, coefficients, *, f_bounds=(-1.0, 1.0),
                                          r_bounds=(-1.0, 1.0), starts=2, max_nfev=50):
        """Retrospective two-state 24h oracle. Nonlinear WM5 oracle is not a convex QP."""
        data = self._prepare_data(input_data)
        p = self.model.coefficients(coefficients)
        obs = np.asarray(data["observation_mask"], dtype=bool)
        lb = np.array([f_bounds[0], r_bounds[0]], dtype=float)
        ub = np.array([f_bounds[1], r_bounds[1]], dtype=float)
        if np.any(ub <= lb): raise ValueError("Prediction oracle bounds must be nonempty.")
        R0 = float(data["I"][0] - data["O"][0])
        seeds = [[0.0, np.clip(R0, *r_bounds)], [0.0, 0.0], [-0.5, -0.5], [0.5, 0.5]]
        best = None
        for seed in seeds[:max(1, min(int(starts), len(seeds)))]:
            def residual(init):
                state = self.model._solve_deterministic(init, data["U"], data["Time"], data["O"], p)
                return state["I"][obs] - data["I"][obs]

            def jacobian(init):
                state = self.model._solve_deterministic(init, data["U"], data["Time"], data["O"], p, with_sensitivity=True)
                return state["initial_sensitivity"][obs, 1, :]

            try:
                result = least_squares(residual, np.clip(seed, lb, ub), jac=jacobian, bounds=(lb, ub),
                                       max_nfev=int(max_nfev), ftol=1e-9, xtol=1e-9, gtol=1e-9)
            except ValueError:
                continue
            loss = float(np.mean(result.fun ** 2))
            if best is None or loss < best[0]: best = (loss, result.x)
        if best is None: raise RuntimeError("WM5 nonlinear oracle initial-state optimization failed.")
        state = self.model._solve_deterministic(best[1], data["U"], data["Time"], data["O"], p)
        return {"init_cond": best[1].copy(), "deterministic_state": state, "loss": best[0],
                "observation_mask": obs.copy()}


class joint_qp_params(qp_params):
    """Single convex joint fit of [r_F*F0, r_F, a_1,...,b_1,...]."""

    def fit(self, input_data, initial_conditions=None, *, stabilize=True, enforce_path=True,
            f_bounds=(-1.0, 1.0), verbose=False):
        from src.solver.joint_drift_qp import joint_drift_qp
        started = perf_counter()
        data = self._prepare_data(input_data)
        obs = self._observed_state(data)
        if initial_conditions is not None:
            seed = wm_5_validation.initial_condition(initial_conditions)
            if not np.isclose(seed[1], obs["R0"], atol=1e-8):
                raise ValueError("Joint fit requires observed R0.")
        constraints = wm_5_params_validation.constraints(
            self.model.r_functions, self.model.u_functions, self.model.dt, stabilize=stabilize)
        joint = joint_drift_qp.solve(
            obs["R"], data["U"], self.model.dt, obs["rate_features"], obs["price_features"],
            constraints, transition_mask=obs["mask"], f_bounds=f_bounds,
            ridge=self.model._FIT_RIDGE, enforce_path=enforce_path)
        p = self.model.coefficients(joint["coefficients"], stabilize=stabilize)
        state = self.model._solve_deterministic(joint["init_cond"], data["U"], data["Time"], data["O"], p)
        mask = data["observation_mask"]
        result = {**joint, "coefficients": p.copy(), "coefficient_names": self.model.parameter_names,
                  "deterministic_state": state, "observation_mask": mask.copy(),
                  "fit_method": "wm5_joint_qp", "deterministic_loss": float(np.mean((state["I"][mask]-data["I"][mask])**2)),
                  "fit_time": perf_counter()-started, "success": True}
        if verbose: print(f"[wm5 joint] qp_loss={joint['qp_loss']:.8g}, F0={joint['init_cond'][0]:.6g}")
        return result


    def fit_many(self, input_data, initial_conditions=None, *, stabilize=True, enforce_path=True,
                 f_bounds=(-1.0, 1.0), verbose=False):
        """Shared nonlinear-response coefficients and one fitted F0 per historical day.

        Fixed response functions g(R), h(U) are evaluated on the observed
        historical rate and price trajectories, leaving a joint convex QP.
        Parameter constraints and simulation stay in the WM5 model/params
        layer; the generic solver only constructs and solves the QP.
        """
        from src.solver.joint_drift_qp import joint_drift_qp
        if not isinstance(input_data, (list, tuple)) or not input_data:
            raise TypeError("WM5 joint multi-day fitting requires a nonempty list of day inputs.")
        started = perf_counter()
        datasets = [self._prepare_data(item) for item in input_data]
        observed = [self._observed_state(data) for data in datasets]
        if initial_conditions is not None:
            raw = np.asarray(initial_conditions, dtype=float)
            if raw.shape != (len(observed), 2):
                raise ValueError("WM5 multi-day initial_conditions requires one [F0,R0] per day.")
            for seed, obs in zip(raw, observed):
                state = wm_5_validation.initial_condition(seed)
                if not np.isclose(state[1], obs["R0"], atol=1e-8):
                    raise ValueError("WM5 multi-day fit requires observed R0=I[0]-O[0].")

        constraints = wm_5_params_validation.constraints(
            self.model.r_functions, self.model.u_functions, self.model.dt, stabilize=stabilize)
        joint = joint_drift_qp.solve_many_features(
            [item["R"] for item in observed], [item["U"] for item in datasets],
            self.model.dt,
            [basis.value for basis in self.model.r_functions],
            [basis.value for basis in self.model.u_functions],
            constraints, masks=[item["mask"] for item in observed],
            f_bounds=f_bounds, ridge=self.model._FIT_RIDGE, enforce_path=enforce_path)
        p = self.model.coefficients(joint["coefficients"], stabilize=stabilize)
        states = [self.model._solve_deterministic(init, data["U"], data["Time"], data["O"], p)
                  for init, data in zip(joint["init_conds"], datasets)]
        errors = np.concatenate([
            (state["I"]-data["I"])[data["observation_mask"]]
            for state, data in zip(states, datasets)
        ])
        result = {**joint, "coefficients": p.copy(),
                  "coefficient_names": self.model.parameter_names,
                  "init_cond": joint["init_conds"].copy(),
                  "deterministic_states": states,
                  "fit_trajectories": len(datasets), "fit_method": "wm5_joint_qp_many",
                  "deterministic_loss": float(np.mean(errors**2)),
                  "fit_time": perf_counter()-started, "success": True}
        if verbose:
            print(f"[wm5 joint many] days={len(datasets)} qp_loss={joint['qp_loss']:.8g} "
                  f"rank={joint['design_rank']}/{joint['design_columns']}")
        return result

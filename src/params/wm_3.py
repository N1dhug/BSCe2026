import itertools
from time import perf_counter

import numpy as np
from scipy.signal import lfiltic, lfilter

from src.validation.model.wm_3 import wm_3_validation


class brute_params:
    """Reduced-coordinate brute-force estimator for the complete wm_3 model."""

    def __init__(self, model): self.model = model

    @staticmethod
    def _transition_stable(dt, d_v, kappa_r, tol=1e-10):
        A = np.array([[1.0, dt], [kappa_r * dt * d_v, 1.0 - kappa_r * dt]], dtype=float)
        return float(np.max(np.abs(np.linalg.eigvals(A)))) < 1.0 + tol

    @staticmethod
    def _local_std(residual):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        n, window = residual.size, max(5, residual.size // 20)
        if window % 2 == 0: window += 1
        window = min(window, n if n % 2 else max(1, n - 1))
        half, result = window // 2, np.empty(n)
        for k in range(n):
            segment = residual[max(0, k - half):min(n, k + half + 1)]
            result[k] = np.std(segment, ddof=1) if segment.size > 1 else 0.0
        return result

    def _prepare_data(self, input_data):
        data = wm_3_validation.fit_data(input_data, self.model.t_steps)
        return {
            **data,
            "L": np.array([wm_3_validation.feature_value(self.model.L(x), "L(Time)") for x in data["Time"]], dtype=float),
            "H": np.array([wm_3_validation.feature_value(self.model.H(x), "H(Temp)") for x in data["Temp"]], dtype=float),
        }

    def _recover_alpha_beta(self, input_coeff, drift_coeff):
        c, d, tol = np.asarray(input_coeff, dtype=float), np.asarray(drift_coeff, dtype=float), 1e-7
        beta = np.array([c[2] - d[2], c[0] - d[0], c[1] - d[1], c[4] - d[4]], dtype=float)

        if not np.isclose(d[3] - c[3], beta[1], atol=tol, rtol=0.0): raise ValueError("Reduced wm_3 fit violates beta_U consistency.")
        if np.any(beta < -tol) or np.any(beta > 1.0 + tol): raise ValueError("Reduced wm_3 fit implies beta outside [0, 1].")
        if not np.isclose(np.sum(beta), 1.0, atol=tol, rtol=0.0): raise ValueError("Reduced wm_3 fit implies beta that does not sum to one.")

        beta = np.clip(beta, 0.0, 1.0)
        beta /= np.sum(beta)
        M = np.array([[1.0, 1.0, beta[1]], [0.0, 0.0, beta[2]], [-1.0, 0.0, beta[0]], [0.0, -1.0, -beta[1]], [0.0, 0.0, beta[3]]], dtype=float)
        alpha = self.model._least_squares_qp(M, c, A_eq=np.vstack((np.ones(3), M[2])), b_eq=np.array([1.0, c[2]]), lb=np.zeros(3), ub=np.ones(3), ridge=self.model._FIT_RIDGE)
        alpha, beta = wm_3_validation.recovered_structural_coefficients(alpha, beta)
        recovered_input, recovered_drift = self.model.reduced_coefficients(alpha, beta)

        if not np.isclose(recovered_input[2], c[2], atol=tol, rtol=0.0): raise ValueError("Recovered wm_3 c_V does not match the brute-force candidate.")
        if not np.isclose(recovered_drift[2], d[2], atol=tol, rtol=0.0): raise ValueError("Recovered wm_3 d_V does not match the brute-force candidate.")

        projection_error = float(np.mean((recovered_input - c) ** 2) + np.mean((recovered_drift - d) ** 2))
        return alpha, beta, recovered_input, recovered_drift, projection_error

    @staticmethod
    def _homogeneous_state(n, dt, d_v, kappa_r, v0, r0):
        a1 = 2.0 - kappa_r * dt
        a2 = -(1.0 - kappa_r * dt) + kappa_r * dt * dt * d_v
        den = np.array([1.0, -a1, -a2], dtype=float)
        num = np.array([1.0], dtype=float)
        v = np.empty(n + 1, dtype=float)
        v[0], v[1] = float(v0), float(v0) + dt * float(r0)
        if n > 1:
            zi = lfiltic(num, den, y=[v[1], v[0]], x=[0.0, 0.0])
            v[2:], _ = lfilter(num, den, np.zeros(n - 1, dtype=float), zi=zi)
        return v[:n], np.diff(v) / dt

    @staticmethod
    def _forced_state_map(features, dt, d_v, kappa_r):
        n = features.shape[0]
        a1 = 2.0 - kappa_r * dt
        a2 = -(1.0 - kappa_r * dt) + kappa_r * dt * dt * d_v
        gain = kappa_r * dt * dt
        forcing = np.zeros((n + 1, features.shape[1]), dtype=float)
        if n > 1: forcing[2:] = features[:-1]
        v = lfilter(np.array([gain], dtype=float), np.array([1.0, -a1, -a2], dtype=float), forcing, axis=0)
        return v[:n], np.diff(v, axis=0) / dt

    def _state_from_reduced(self, data, init_cond, input_coeff, drift_coeff, kappa_r, full=False):
        n, dt = self.model.t_steps, self.model.dt
        input_coeff, drift_coeff = np.asarray(input_coeff, dtype=float), np.asarray(drift_coeff, dtype=float)
        features = np.column_stack((np.ones(n), data["L"], data["U"], data["H"]))
        c_bar, d_bar = input_coeff[[0, 1, 3, 4]], drift_coeff[[0, 1, 3, 4]]
        c_v, d_v = float(input_coeff[2]), float(drift_coeff[2])

        V_b, R_b = self._homogeneous_state(n, dt, d_v, kappa_r, init_cond[0], init_cond[1])
        forced_V, forced_R = self._forced_state_map(features, dt, d_v, kappa_r)
        V, R = V_b + forced_V @ d_bar, R_b + forced_R @ d_bar
        I = features @ c_bar + c_v * V
        D = features @ d_bar + d_v * V
        O = I - D
        R_dot = kappa_r * (D - R)

        state = {
            "t": self.model.t.copy(), "V": V, "R": R, "R_dot": R_dot,
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(), "Temp": data["Temp"].copy(),
            "L": data["L"].copy(), "H": data["H"].copy(), "O": O, "I": I, "D": D,
            "input_coeff": input_coeff.copy(), "drift_coeff": drift_coeff.copy(),
        }
        if not full: return state

        P_I = np.column_stack((1.0 - V, 1.0 - data["U"], O))
        P_O = np.column_stack((V, 1.0 - data["U"], data["L"], data["H"]))
        state.update({
            "P_I": P_I, "P_O": P_O, "P_A": np.full((n, 4), np.nan),
            "A": np.full(n, np.nan),
        })
        return state

    def _solve_constraint_generated_qp(self, X, y, I_X, I_b, structural_A, structural_b, A_eq, b_eq, lb, ub, obs, max_rounds=20, batch=32, tol=1e-9):
        n = I_X.shape[0]
        seed = np.unique(np.concatenate((
            np.flatnonzero(obs),
            np.linspace(0, n - 1, min(257, n), dtype=int),
            np.array([0, n - 1], dtype=int),
        )))
        active = set(map(int, seed))
        solution, qp_time, rounds = None, 0.0, 0

        for rounds in range(1, max_rounds + 1):
            idx = np.array(sorted(active), dtype=int)
            A_ub = np.vstack((structural_A, I_X[idx], -I_X[idx]))
            b_ub = np.concatenate((structural_b, np.ones(idx.size) - I_b[idx], I_b[idx]))
            t0 = perf_counter()
            solution = self.model._least_squares_qp(
                X, y, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                lb=lb, ub=ub, ridge=self.model._FIT_RIDGE
            )
            qp_time += perf_counter() - t0

            values = I_b + I_X @ solution
            upper = values - 1.0
            lower = -values
            max_violation = max(float(np.max(upper)), float(np.max(lower)), 0.0)
            if max_violation <= tol: return solution, qp_time, rounds, len(active), max_violation

            candidates = []
            upper_idx = np.flatnonzero(upper > tol)
            lower_idx = np.flatnonzero(lower > tol)
            if upper_idx.size:
                order = upper_idx[np.argsort(upper[upper_idx])[-batch:]]
                candidates.extend(map(int, order))
            if lower_idx.size:
                order = lower_idx[np.argsort(lower[lower_idx])[-batch:]]
                candidates.extend(map(int, order))
            before = len(active)
            active.update(candidates)
            if len(active) == before: break

        raise RuntimeError("wm_3 constraint-generation QP did not satisfy all fine-grid input bounds.")

    def _solve_inner(self, data, initial_conditions, c_v, d_v, kappa_r, fit_initial_state, r0_range):
        n, dt = self.model.t_steps, self.model.dt
        obs = np.asarray(data["observation_mask"], dtype=bool)
        n_var = 10 if fit_initial_state else 8
        i_c, i_d = slice(0, 4), slice(4, 8)
        i_v0, i_r0 = (8, 9) if fit_initial_state else (None, None)
        features = np.column_stack((np.ones(n), data["L"], data["U"], data["H"]))

        V_X, R_X = np.zeros((n, n_var), dtype=float), np.zeros((n, n_var), dtype=float)
        forced_V, forced_R = self._forced_state_map(features, dt, d_v, kappa_r)
        V_X[:, i_d], R_X[:, i_d] = forced_V, forced_R
        V_b, R_b = np.zeros(n, dtype=float), np.zeros(n, dtype=float)

        if fit_initial_state:
            V_X[:, i_v0], R_X[:, i_v0] = self._homogeneous_state(n, dt, d_v, kappa_r, 1.0, 0.0)
            V_X[:, i_r0], R_X[:, i_r0] = self._homogeneous_state(n, dt, d_v, kappa_r, 0.0, 1.0)
        else:
            V_b, R_b = self._homogeneous_state(n, dt, d_v, kappa_r, initial_conditions[0], initial_conditions[1])

        I_X = c_v * V_X
        I_X[:, i_c] += features
        I_b = c_v * V_b
        X, y = I_X[obs], data["I"][obs] - I_b[obs]

        eq_rows, eq_rhs = [], []
        row = np.zeros(n_var); row[0], row[2], row[4], row[6] = 1.0, 1.0, -1.0, -1.0
        eq_rows.append(row); eq_rhs.append(0.0)
        row = np.zeros(n_var); row[0], row[1], row[3], row[4], row[5], row[7] = 1.0, 1.0, 1.0, -1.0, -1.0, -1.0
        eq_rows.append(row); eq_rhs.append(1.0 - (c_v - d_v))

        if np.allclose(data["H"], 0.0, atol=1e-12, rtol=0.0):
            row = np.zeros(n_var); row[3] = 1.0; eq_rows.append(row); eq_rhs.append(0.0)
            row = np.zeros(n_var); row[7] = 1.0; eq_rows.append(row); eq_rhs.append(0.0)

        structural_rows, structural_rhs = [], []
        for U_value, L_value, H_value in itertools.product((0.0, 1.0), repeat=3):
            f = np.array([1.0, L_value, U_value, H_value], dtype=float)
            row = np.zeros(n_var); row[i_d] = -f
            structural_rows.append(row); structural_rhs.append(0.0)
            row = np.zeros(n_var); row[i_d] = f
            structural_rows.append(row); structural_rhs.append(-d_v)

        for c_idx, d_idx in ((0, 4), (1, 5), (3, 7)):
            row = np.zeros(n_var); row[d_idx], row[c_idx] = 1.0, -1.0
            structural_rows.append(row); structural_rhs.append(0.0)

        A_eq, b_eq = np.vstack(eq_rows), np.asarray(eq_rhs, dtype=float)
        structural_A, structural_b = np.vstack(structural_rows), np.asarray(structural_rhs, dtype=float)
        lb, ub = np.full(n_var, -np.inf), np.full(n_var, np.inf)
        if fit_initial_state:
            lb[i_v0], ub[i_v0] = 0.0, 1.0
            lb[i_r0], ub[i_r0] = map(float, r0_range)

        solution, qp_solve_time, constraint_rounds, active_constraints, max_input_violation = self._solve_constraint_generated_qp(
            X, y, I_X, I_b, structural_A, structural_b, A_eq, b_eq, lb, ub, obs
        )

        c_bar, d_bar = solution[i_c], solution[i_d]
        input_coeff = np.array([c_bar[0], c_bar[1], c_v, c_bar[2], c_bar[3]], dtype=float)
        drift_coeff = np.array([d_bar[0], d_bar[1], d_v, d_bar[2], d_bar[3]], dtype=float)
        estimated_init = np.array([solution[i_v0], solution[i_r0]], dtype=float) if fit_initial_state else np.asarray(initial_conditions, dtype=float).copy()

        latent_V = V_b + V_X @ solution
        latent_R = R_b + R_X @ solution
        reduced_I = I_b + I_X @ solution

        alpha, beta, recovered_input, recovered_drift, projection_error = self._recover_alpha_beta(input_coeff, drift_coeff)
        state = self._state_from_reduced(data, estimated_init, recovered_input, recovered_drift, kappa_r, full=False)
        reduced_loss = float(np.mean((data["I"][obs] - reduced_I[obs]) ** 2))
        recovered_loss = float(np.mean((data["I"][obs] - state["I"][obs]) ** 2))

        return {
            "init_cond": estimated_init, "initial_condition_seed": np.asarray(initial_conditions, dtype=float).copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(), "Temp": data["Temp"].copy(),
            "L": data["L"].copy(), "H": data["H"].copy(), "observation_mask": obs.copy(), "alpha": alpha, "beta": beta,
            "I_coeff": recovered_input, "d_coeff": recovered_drift, "reduced_fit_input_coeff": input_coeff, "reduced_fit_d_coeff": drift_coeff,
            "kappa_R": float(kappa_r), "latent_V": latent_V, "latent_R": latent_R,
            "deterministic_state": state, "reduced_fit_loss": reduced_loss, "recovered_model_loss": recovered_loss,
            "structural_projection_error": projection_error, "structural_viable": True,
            "brute_c_V": float(c_v), "brute_d_V": float(d_v), "brute_kappa_R": float(kappa_r), "brute_alpha_O": float(alpha[2]),
            "qp_solve_time": qp_solve_time, "constraint_rounds": constraint_rounds,
            "active_input_constraints": active_constraints, "max_input_violation": max_input_violation,
        }

    def _fit_uncertainty_params(self, result, data):
        state, obs = result["deterministic_state"], np.asarray(data["observation_mask"], dtype=bool)
        residual = data["I"][obs] - state["I"][obs]
        local_std = self._local_std(residual)
        X = np.column_stack((np.ones(np.count_nonzero(obs)), state["O"][obs]))
        sigma = self.model._least_squares_qp(X, local_std, A_ub=-X, b_ub=-np.full(X.shape[0], self.model._FIT_EPS), lb=np.array([self.model._FIT_EPS, 0.0]), ub=np.array([np.inf, np.inf]), ridge=self.model._FIT_RIDGE)
        return {**result, "sigma_0": float(sigma[0]), "sigma_O": float(sigma[1]), "residual": residual, "local_std": local_std}

    @staticmethod
    def _refined_interval(values, best, global_lo, global_hi):
        values = np.unique(np.asarray(values, dtype=float))
        width = 0.1 * max(1.0, global_hi - global_lo) if values.size <= 1 else float(np.max(np.diff(values)))
        lo, hi = max(global_lo, best - width), min(global_hi, best + width)
        if hi <= lo: hi = min(global_hi, lo + max(width, 1e-8))
        return lo, hi

    def _search_branch(self, branch, data, initial_conditions, grid_points, refinements, c_v_range, d_v_range, ratio_eps, fit_initial_state, r0_range, verbose, progress_every):
        c_bounds, d_bounds = tuple(map(float, c_v_range)), tuple(map(float, d_v_range))
        rho_global = {"under": (ratio_eps, 1.0 - ratio_eps), "critical": (1.0, 1.0)}[branch]
        c_local, d_local, rho_local = c_bounds, d_bounds, rho_global
        best, history, branch_start = None, [], perf_counter()

        for level in range(refinements):
            c_grid, d_grid = np.linspace(*c_local, grid_points), np.linspace(*d_local, grid_points)
            rho_grid = np.array([1.0]) if branch == "critical" else np.linspace(*rho_local, grid_points)
            level_best, tested, pruned, qp_times, level_start = None, 0, 0, [], perf_counter()
            if verbose: print(f"[brute] branch={branch} refinement={level + 1}/{refinements} candidates={len(c_grid) * len(d_grid) * len(rho_grid)}")

            for d_v in d_grid:
                if d_v >= -self.model._FIT_EPS: continue
                for c_v in c_grid:
                    beta_v = c_v - d_v
                    if beta_v < -1e-10 or beta_v > 1.0 + 1e-10:
                        pruned += len(rho_grid)
                        continue

                    for rho in rho_grid:
                        kappa_r = 4.0 * float(rho) * abs(float(d_v))
                        if kappa_r <= 0.0 or not self._transition_stable(self.model.dt, d_v, kappa_r):
                            pruned += 1
                            continue

                        tested, candidate_start = tested + 1, perf_counter()
                        try: candidate = self._solve_inner(data, initial_conditions, c_v, d_v, kappa_r, fit_initial_state, r0_range)
                        except (ValueError, RuntimeError, np.linalg.LinAlgError):
                            if verbose and tested % progress_every == 0: print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={perf_counter() - level_start:.2f}s last_candidate={perf_counter() - candidate_start:.4f}s status=failed")
                            continue

                        qp_times.append(float(candidate["qp_solve_time"]))
                        if verbose and (tested == 1 or tested % progress_every == 0): print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={perf_counter() - level_start:.2f}s last_qp={qp_times[-1]:.4f}s mean_qp={np.mean(qp_times):.4f}s")

                        score = float(candidate["recovered_model_loss"])
                        if level_best is None or score < level_best["score"]: level_best = {"score": score, "c_v": float(c_v), "d_v": float(d_v), "rho": float(rho), "kappa_R": float(kappa_r), "alpha": candidate["alpha"].copy(), "beta": candidate["beta"].copy(), "result": candidate}

            level_elapsed = perf_counter() - level_start
            mean_qp = float(np.mean(qp_times)) if qp_times else np.nan
            history.append({"level": level, "tested": tested, "pruned": pruned, "best_loss": np.inf if level_best is None else level_best["score"], "elapsed": level_elapsed, "mean_qp_solve_time": mean_qp, "qp_solves": len(qp_times)})
            if verbose: print(f"[brute] branch={branch} refinement={level + 1} complete tested={tested} pruned={pruned} qp_solves={len(qp_times)} elapsed={level_elapsed:.2f}s mean_qp={mean_qp:.4f}s")
            if level_best is None: break

            best = level_best
            c_local = self._refined_interval(c_grid, best["c_v"], *c_bounds)
            d_local = self._refined_interval(d_grid, best["d_v"], *d_bounds)
            if branch != "critical": rho_local = self._refined_interval(rho_grid, best["rho"], *rho_global)

        if best is not None: best.update({"history": history, "branch": branch, "elapsed": perf_counter() - branch_start})
        return best

    def _search_all(self, data, initial_conditions, grid_points, refinements, c_v_range, d_v_range, ratio_eps, fit_initial_state, r0_range, verbose, progress_every):
        branch_results = {branch: self._search_branch(branch, data, initial_conditions, grid_points, refinements, c_v_range, d_v_range, ratio_eps, fit_initial_state, r0_range, verbose, progress_every) for branch in ("under", "critical")}
        valid = [item for item in branch_results.values() if item is not None]
        if not valid: raise RuntimeError("wm_3 brute-force parameter search found no structurally admissible underdamped or critically damped grid candidate.")
        return min(valid, key=lambda item: item["score"]), branch_results

    def fit(self, input_data, initial_conditions, *, grid_points=4, refinements=2, c_v_range=(-1.0, 1.0), d_v_range=(-2.0, -1e-4), ratio_eps=1e-3, fit_initial_state=True, r0_range=(-1.0, 1.0), fit_diffusion=True, verbose=True, progress_every=25, alpha_points=None, alpha_o_range=None):
        data = self._prepare_data(input_data)
        initial_conditions = wm_3_validation.initial_condition(initial_conditions)
        grid_points, refinements, progress_every = int(grid_points), int(refinements), int(progress_every)
        r0_range = tuple(map(float, r0_range))

        if grid_points < 2 or refinements < 1: raise ValueError("grid_points >= 2 and refinements >= 1 are required.")
        if progress_every < 1: raise ValueError("progress_every must be at least 1.")
        if not c_v_range[0] < c_v_range[1]: raise ValueError("c_v_range must be increasing.")
        if not d_v_range[0] < d_v_range[1] < 0.0: raise ValueError("d_v_range must be increasing and strictly negative.")
        if not 0.0 < ratio_eps < 0.5: raise ValueError("ratio_eps must lie in (0, 0.5).")
        if len(r0_range) != 2 or not r0_range[0] < r0_range[1]: raise ValueError("r0_range must contain increasing lower/upper bounds.")

        fit_start = perf_counter()
        best, branches = self._search_all(data, initial_conditions, grid_points, refinements, c_v_range, d_v_range, ratio_eps, bool(fit_initial_state), r0_range, verbose, progress_every)
        result = {**best["result"], "alpha": best["alpha"].copy(), "beta": best["beta"].copy()}
        result["deterministic_state"] = self._state_from_reduced(
            data, result["init_cond"], result["I_coeff"], result["d_coeff"], result["kappa_R"], full=True
        )
        result = self._fit_uncertainty_params(result, data) if fit_diffusion else {**result, "sigma_0": self.model._FIT_EPS, "sigma_O": 0.0}
        result.update({"fit_method": "brute", "brute_branch": best["branch"], "brute_score": float(best["score"]), "brute_grid_points": grid_points, "brute_refinements": refinements, "brute_search_history": branches, "brute_fit_time": perf_counter() - fit_start, "fit_initial_state": bool(fit_initial_state), "deterministic_loss": float(result["recovered_model_loss"])})
        return result

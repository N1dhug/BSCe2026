from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfiltic, lfilter

from src.validation.model.wm_4 import wm_4_validation


class brute_params:
    """Outer brute search with a convex inner QP for wm_4."""

    def __init__(self, model): self.model = model

    @staticmethod
    def _transition_stable(dt, q_f, kappa_r, tol=1e-10):
        A = np.array([[1.0, dt], [kappa_r * dt * q_f, 1.0 - kappa_r * dt]], dtype=float)
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
        data = wm_4_validation.fit_data(input_data, self.model.t_steps)
        return {**data, "O": np.array([wm_4_validation.feature_value(self.model.O(x), "O(Time)") for x in data["Time"]], dtype=float)}

    @staticmethod
    def _homogeneous_state(n, dt, q_f, kappa_r, f0, r0):
        a1 = 2.0 - kappa_r * dt
        a2 = -(1.0 - kappa_r * dt) + kappa_r * dt * dt * q_f
        den, num = np.array([1.0, -a1, -a2], dtype=float), np.array([1.0], dtype=float)
        F = np.empty(n + 1, dtype=float)
        F[0], F[1] = float(f0), float(f0) + dt * float(r0)
        if n > 1:
            zi = lfiltic(num, den, y=[F[1], F[0]], x=[0.0, 0.0])
            F[2:], _ = lfilter(num, den, np.zeros(n - 1, dtype=float), zi=zi)
        return F[:n], np.diff(F) / dt

    @staticmethod
    def _forced_state_map(features, dt, q_f, kappa_r):
        n = features.shape[0]
        a1 = 2.0 - kappa_r * dt
        a2 = -(1.0 - kappa_r * dt) + kappa_r * dt * dt * q_f
        gain = kappa_r * dt * dt
        forcing = np.zeros((n + 1, features.shape[1]), dtype=float)
        if n > 1: forcing[2:] = features[:-1]
        F = lfilter(np.array([gain], dtype=float), np.array([1.0, -a1, -a2], dtype=float), forcing, axis=0)
        return F[:n], np.diff(F, axis=0) / dt

    def _state_from_parameters(self, data, init_cond, q, kappa_r, o_scale=1.0, full=False):
        n, dt = self.model.t_steps, self.model.dt
        q = np.asarray(q, dtype=float)
        q_features = np.column_stack((np.ones(n), data["U"]))
        q_bar, q_f = q[[0, 2]], float(q[1])

        F_b, R_b = self._homogeneous_state(n, dt, q_f, kappa_r, init_cond[0], init_cond[1])
        forced_F, forced_R = self._forced_state_map(q_features, dt, q_f, kappa_r)
        F, R = F_b + forced_F @ q_bar, R_b + forced_R @ q_bar
        Q = q_features @ q_bar + q_f * F
        O_raw = data["O"].copy()
        O = float(o_scale) * O_raw
        I = O + R
        R_dot = kappa_r * (Q - R)

        state = {
            "t": self.model.t.copy(), "F": F, "R": R, "R_dot": R_dot,
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(),
            "O": O, "O_raw": O_raw, "o_scale": float(o_scale), "Q": Q, "I": I, "q": q.copy(),
        }
        if not full: return state
        state.update({
            "P_Q": np.column_stack((np.ones(n), F, data["U"])),
            "P_A": np.full((n, 4), np.nan), "A": np.full(n, np.nan),
        })
        return state

    def _solve_inner(self, data, initial_conditions, q_f, kappa_r, fit_initial_state, r0_range):
        n, dt = self.model.t_steps, self.model.dt
        obs = np.asarray(data["observation_mask"], dtype=bool)
        n_var = 5 if fit_initial_state else 3
        i_q, i_o = slice(0, 2), 2
        i_f0, i_r0 = (3, 4) if fit_initial_state else (None, None)
        q_features = np.column_stack((np.ones(n), data["U"]))

        F_X, R_X = np.zeros((n, n_var), dtype=float), np.zeros((n, n_var), dtype=float)
        forced_F, forced_R = self._forced_state_map(q_features, dt, q_f, kappa_r)
        F_X[:, i_q], R_X[:, i_q] = forced_F, forced_R
        F_b, R_b = np.zeros(n, dtype=float), np.zeros(n, dtype=float)

        if fit_initial_state:
            F_X[:, i_f0], R_X[:, i_f0] = self._homogeneous_state(n, dt, q_f, kappa_r, 1.0, 0.0)
            F_X[:, i_r0], R_X[:, i_r0] = self._homogeneous_state(n, dt, q_f, kappa_r, 0.0, 1.0)
        else:
            F_b, R_b = self._homogeneous_state(n, dt, q_f, kappa_r, initial_conditions[0], initial_conditions[1])

        I_X = R_X.copy()
        I_X[:, i_o] = data["O"]
        X = I_X[obs]
        y = data["I"][obs] - R_b[obs]

        structural_rows, structural_rhs = [], []
        for U in (0.0, 1.0):
            f = np.array([1.0, U], dtype=float)
            row = np.zeros(n_var); row[i_q] = -f
            structural_rows.append(row); structural_rhs.append(0.0)
            row = np.zeros(n_var); row[i_q] = f
            structural_rows.append(row); structural_rhs.append(-q_f)

        A_ub, b_ub = np.vstack(structural_rows), np.asarray(structural_rhs, dtype=float)
        lb, ub = np.full(n_var, -np.inf), np.full(n_var, np.inf)
        ub[1] = 0.0
        lb[i_o], ub[i_o] = 0.0, 1.0
        if fit_initial_state:
            lb[i_f0], ub[i_f0] = 0.0, 1.0
            lb[i_r0], ub[i_r0] = map(float, r0_range)

        solution_start = perf_counter()
        solution = self.model._least_squares_qp(X, y, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE)
        qp_solve_time = perf_counter() - solution_start

        q_bar = solution[i_q]
        q = np.array([q_bar[0], q_f, q_bar[1]], dtype=float)
        q = wm_4_validation.recovered_structural_coefficients(q)
        o_scale = float(solution[i_o])
        estimated_init = np.array([solution[i_f0], solution[i_r0]], dtype=float) if fit_initial_state else np.asarray(initial_conditions, dtype=float).copy()
        state = self._state_from_parameters(data, estimated_init, q, kappa_r, o_scale=o_scale, full=False)
        loss = float(np.mean((data["I"][obs] - state["I"][obs]) ** 2))

        return {
            "init_cond": estimated_init, "initial_condition_seed": np.asarray(initial_conditions, dtype=float).copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(),
            "O": data["O"].copy(), "observation_mask": obs.copy(),
            "q": q, "o_scale": o_scale, "kappa_R": float(kappa_r), "deterministic_state": state,
            "recovered_model_loss": loss, "structural_viable": True,
            "brute_q_F": float(q_f), "brute_kappa_R": float(kappa_r), "qp_solve_time": qp_solve_time,
        }

    def _fit_uncertainty_params(self, result, data):
        state, obs = result["deterministic_state"], np.asarray(data["observation_mask"], dtype=bool)
        index = np.flatnonzero(obs)
        if index.size < 3: raise ValueError("At least three observations are required to estimate wm_4 diffusion.")
        observed_I = np.asarray(data["I"][index], dtype=float)
        observed_O = float(result["o_scale"]) * np.asarray(data["O"][index], dtype=float)
        if np.any(observed_I < 0.0) or np.any(state["I"] < 0.0): raise ValueError("wm_4 produced negative I; input-dependent diffusion requires I >= 0.")

        observed_R = observed_I - observed_O
        dt = np.diff(state["t"][index])
        if np.any(dt <= 0.0): raise ValueError("wm_4 diffusion observations must have strictly increasing model time.")
        observed_F = np.empty(index.size, dtype=float)
        observed_F[0] = state["F"][index[0]]
        observed_F[1:] = observed_F[0] + np.cumsum(observed_R[:-1] * dt)
        q = np.asarray(result["q"], dtype=float)
        Q = q[0] + q[1] * observed_F[:-1] + q[2] * data["U"][index[:-1]]
        innovation = np.diff(observed_R) - result["kappa_R"] * (Q - observed_R[:-1]) * dt
        diffusion_I = observed_I[:-1]

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
        residual = data["I"][obs] - state["I"][obs]
        return {
            **result, "sigma_0": sigma_0, "sigma_I": sigma_I, "residual": residual,
            "diffusion_innovation": innovation, "diffusion_dt": dt, "diffusion_I": diffusion_I,
            "diffusion_F": observed_F, "diffusion_Q": Q, "diffusion_nll": float(fit.fun),
        }

    @staticmethod
    def _refined_interval(values, best, global_lo, global_hi):
        values = np.unique(np.asarray(values, dtype=float))
        width = 0.1 * max(1.0, global_hi - global_lo) if values.size <= 1 else float(np.max(np.diff(values)))
        lo, hi = max(global_lo, best - width), min(global_hi, best + width)
        if hi <= lo: hi = min(global_hi, lo + max(width, 1e-8))
        return lo, hi

    def _search_branch(self, branch, data, initial_conditions, grid_points, refinements, q_f_range, ratio_eps, ratio_max, fit_initial_state, r0_range, verbose, progress_every):
        q_bounds = tuple(map(float, q_f_range))
        rho_global = {"under": (ratio_eps, 1.0 - ratio_eps), "critical": (1.0, 1.0), "over": (1.0 + ratio_eps, ratio_max)}[branch]
        q_local, rho_local = q_bounds, rho_global
        best, history, branch_start = None, [], perf_counter()

        for level in range(refinements):
            q_grid = np.linspace(*q_local, grid_points)
            rho_grid = np.array([1.0]) if branch == "critical" else np.linspace(*rho_local, grid_points)
            level_best, tested, pruned, qp_times, level_start = None, 0, 0, [], perf_counter()
            if verbose: print(f"[brute] branch={branch} refinement={level + 1}/{refinements} candidates={len(q_grid) * len(rho_grid)}")

            for q_f in q_grid:
                if q_f >= -self.model._FIT_EPS:
                    pruned += len(rho_grid)
                    continue
                for rho in rho_grid:
                    kappa_r = 4.0 * float(rho) * abs(float(q_f))
                    if kappa_r <= 0.0 or not self._transition_stable(self.model.dt, q_f, kappa_r):
                        pruned += 1
                        continue

                    tested, candidate_start = tested + 1, perf_counter()
                    try: candidate = self._solve_inner(data, initial_conditions, q_f, kappa_r, fit_initial_state, r0_range)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        if verbose and tested % progress_every == 0: print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={perf_counter() - level_start:.2f}s last_candidate={perf_counter() - candidate_start:.4f}s status=failed")
                        continue

                    qp_times.append(float(candidate["qp_solve_time"]))
                    if verbose and (tested == 1 or tested % progress_every == 0): print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={perf_counter() - level_start:.2f}s last_qp={qp_times[-1]:.4f}s mean_qp={np.mean(qp_times):.4f}s")

                    score = float(candidate["recovered_model_loss"])
                    if level_best is None or score < level_best["score"]: level_best = {"score": score, "q_f": float(q_f), "rho": float(rho), "kappa_R": float(kappa_r), "q": candidate["q"].copy(), "result": candidate}

            level_elapsed = perf_counter() - level_start
            mean_qp = float(np.mean(qp_times)) if qp_times else np.nan
            history.append({"level": level, "tested": tested, "pruned": pruned, "best_loss": np.inf if level_best is None else level_best["score"], "elapsed": level_elapsed, "mean_qp_solve_time": mean_qp, "qp_solves": len(qp_times)})
            if verbose: print(f"[brute] branch={branch} refinement={level + 1} complete tested={tested} pruned={pruned} qp_solves={len(qp_times)} elapsed={level_elapsed:.2f}s mean_qp={mean_qp:.4f}s")
            if level_best is None: break

            best = level_best
            q_local = self._refined_interval(q_grid, best["q_f"], *q_bounds)
            if branch != "critical": rho_local = self._refined_interval(rho_grid, best["rho"], *rho_global)

        if best is not None: best.update({"history": history, "branch": branch, "elapsed": perf_counter() - branch_start})
        return best

    def _search_all(self, data, initial_conditions, grid_points, refinements, q_f_range, ratio_eps, ratio_max, fit_initial_state, r0_range, verbose, progress_every):
        branches = {
            branch: self._search_branch(branch, data, initial_conditions, grid_points, refinements, q_f_range, ratio_eps, ratio_max, fit_initial_state, r0_range, verbose, progress_every)
            for branch in ("under", "critical", "over")
        }
        valid = [item for item in branches.values() if item is not None]
        if not valid: raise RuntimeError("wm_4 brute-force parameter search found no structurally admissible damping branch candidate.")
        return min(valid, key=lambda item: item["score"]), branches

    def fit(self, input_data, initial_conditions, *, grid_points=4, refinements=2, q_f_range=(-50.0, -1e-4), ratio_eps=1e-4, ratio_max=10.0, fit_initial_state=True, r0_range=(-1.0, 1.0), fit_diffusion=True, verbose=True, progress_every=25):
        data = self._prepare_data(input_data)
        initial_conditions = wm_4_validation.initial_condition(initial_conditions)
        grid_points, refinements, progress_every = int(grid_points), int(refinements), int(progress_every)
        q_f_range, r0_range = tuple(map(float, q_f_range)), tuple(map(float, r0_range))

        if grid_points < 2 or refinements < 1: raise ValueError("grid_points >= 2 and refinements >= 1 are required.")
        if progress_every < 1: raise ValueError("progress_every must be at least 1.")
        if not q_f_range[0] < q_f_range[1] < 0.0: raise ValueError("q_f_range must be increasing and strictly negative.")
        if not 0.0 < ratio_eps < 0.5: raise ValueError("ratio_eps must lie in (0, 0.5).")
        ratio_max = float(ratio_max)
        if ratio_max <= 1.0 + ratio_eps: raise ValueError("ratio_max must exceed 1 + ratio_eps for the overdamped branch.")
        if len(r0_range) != 2 or not r0_range[0] < r0_range[1]: raise ValueError("r0_range must contain increasing lower/upper bounds.")

        fit_start = perf_counter()
        best, branches = self._search_all(data, initial_conditions, grid_points, refinements, q_f_range, ratio_eps, ratio_max, bool(fit_initial_state), r0_range, verbose, progress_every)
        result = {**best["result"], "q": best["q"].copy()}
        result["deterministic_state"] = self._state_from_parameters(data, result["init_cond"], result["q"], result["kappa_R"], o_scale=result["o_scale"], full=True)
        result = self._fit_uncertainty_params(result, data) if fit_diffusion else {**result, "sigma_0": self.model._FIT_EPS, "sigma_I": 0.0}
        result.update({
            "fit_method": "brute", "brute_branch": best["branch"], "brute_score": float(best["score"]),
            "brute_grid_points": grid_points, "brute_refinements": refinements, "brute_search_history": branches,
            "brute_fit_time": perf_counter() - fit_start, "fit_initial_state": bool(fit_initial_state),
            "deterministic_loss": float(result["recovered_model_loss"]),
        })
        return result

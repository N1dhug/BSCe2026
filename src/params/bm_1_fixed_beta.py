"""Joint stochastic BM1 MLE with independent initial beta-mixture densities.

Uses Nielsen (2026) fixed-basis Fokker--Planck QPs. Historical observations
are filtered; target predictions are open-loop with a separate oracle initial
distribution. No old point-state deterministic MLE initialization is used.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax

from src.method.bm_1_fixed_beta import bm_1_fixed_beta
from src.params.bm_1_beta import bm_1_beta_params


class bm_1_fixed_beta_params:
    NAMES = ("alpha", "beta", "k", "Xi", "C", "sigma_x", "sigma_y")
    BOUNDS = bm_1_beta_params.BOUNDS

    def __init__(self, degree=6, quadrature=48, steps_per_hour=2, model=None):
        self.solver = bm_1_fixed_beta(degree, quadrature, steps_per_hour, model=model)
        self.degree = degree

    def _weights(self, values):
        return softmax(np.r_[np.asarray(values, dtype=float), 0.0])

    def _decode(self, values, intervals):
        p = {name: float(np.exp(values[i])) for i, name in enumerate(self.NAMES)}
        start = len(self.NAMES)
        initials = np.array([self._weights(values[start+n*self.degree:start+(n+1)*self.degree])
                             for n in range(intervals)])
        return p, initials

    @staticmethod
    def _validate(datasets):
        if not isinstance(datasets,(list,tuple)) or not datasets: raise ValueError("Provide one or more historical intervals.")
        for data in datasets:
            for name in ("I","B","U"):
                arr = np.asarray(data[name],dtype=float)
                if arr.ndim != 1 or len(arr)<2 or not np.isfinite(arr).all() or np.any((arr<0)|(arr>1)):
                    raise ValueError(f"Every {name} interval needs finite hourly values in [0,1].")
            if len({len(data[name]) for name in ("I","B","U")}) != 1:
                raise ValueError("Inconsistent historical input lengths.")

    def fit(self, datasets, *, maxiter=12, maxfun=400, progress=True, seed=None):
        self._validate(datasets)
        n, m = len(datasets), self.degree
        starting = np.array([2.,2.,1.,.5,2.,.15,.075])
        if seed is not None: starting = np.array([seed[name] for name in self.NAMES], dtype=float)
        limits = [self.BOUNDS[key] for key in self.NAMES]
        starting = np.array([np.clip(value, lower*1.001, upper*.999)
                             for value,(lower,upper) in zip(starting,limits)])
        # A centered but nondegenerate distribution; each historical interval is free to move.
        seed_weights = np.exp(-.5*((np.arange(m+1)-m/2.)/max(1.,m/5.))**2)
        seed_weights /= seed_weights.sum()
        logits = np.log(seed_weights[:-1]/seed_weights[-1])
        x0 = np.r_[np.log(starting), np.tile(logits,n)]
        bounds = [(np.log(lo),np.log(hi)) for lo,hi in limits] + [(-10.,10.)]*(n*m)
        evaluations, invalid, best, started = 0, 0, [np.inf,None], perf_counter()

        def objective(vector):
            nonlocal evaluations, invalid
            evaluations += 1
            p, initials = self._decode(vector,n)
            try:
                total = 0.
                for data,c0 in zip(datasets,initials):
                    total += self.solver.simulate(data,p,c0,observations=True)["nll"]
                if not np.isfinite(total): raise FloatingPointError("Nonfinite likelihood.")
                if total < best[0]: best[:] = [float(total), np.asarray(vector).copy()]
                if progress and evaluations % 20 == 0:
                    print(f"[BM1 fixed beta] evaluations={evaluations} best_nll={best[0]:.6f}",flush=True)
                return float(total)
            except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
                invalid += 1
                return 1e9

        initial_nll = objective(x0)
        result = minimize(objective,x0,method="L-BFGS-B",bounds=bounds,
                          options={"maxiter":int(maxiter),"maxfun":int(maxfun),"ftol":1e-6,
                                   "finite_diff_rel_step":1e-4})
        if best[1] is None: raise RuntimeError("No admissible stochastic likelihood evaluation.")
        p, initials = self._decode(best[1],n)
        return {**p, "initial_distributions":initials, "nll":float(best[0]),
                "initial_nll":float(initial_nll), "converged":bool(result.success),
                "message":str(result.message), "evaluations":int(evaluations),
                "invalid_evaluations":int(invalid),"runtime_seconds":float(perf_counter()-started),
                "degree":self.degree,"fit_method":"fixed_beta_fokker_planck_joint_mle"}

    def oracle(self, target, fitted, *, maxiter=35):
        """Optimize only target initial distribution location against target demand.

        Historical dynamics and noise parameters are frozen. Target observations
        are used solely for this retrospective oracle choice, never for filtering
        the open-loop trajectory or re-estimating shared parameters.
        """
        self._validate([target])
        initial = np.mean(np.asarray(fitted["initial_distributions"]),axis=0)
        m = self.degree
        support = np.arange(m+1,dtype=float)
        center = float(np.dot(support,initial))
        def shifted(location):
            shifted_values = np.interp(support-(location-center),support,initial,left=0.,right=0.)
            if shifted_values.sum() < 1e-14:
                shifted_values = np.zeros(m+1); shifted_values[int(np.clip(round(location),0,m))] = 1.
            return shifted_values/shifted_values.sum()
        evaluations = {}
        def loss(location):
            location = float(location)
            if location not in evaluations:
                c = shifted(location)
                pred = self.solver.simulate(target,fitted,c,observations=False)
                evaluations[location] = float(np.mean((pred["mean"]-target["I"])**2))
            return evaluations[location]
        grid = np.linspace(0,m,2*m+1)
        best = min(grid,key=loss)
        opt = minimize_scalar(loss,bounds=(max(0.,best-.5),min(float(m),best+.5)),
                              method="bounded",options={"maxiter":int(maxiter),"xatol":1e-3})
        best = min((float(best),float(opt.x),0.,float(m)),key=loss)
        coefficients = shifted(best)
        forecast = self.solver.simulate(target,fitted,coefficients,observations=False)
        lower,upper = self.solver.intervals(forecast,fitted["sigma_y"])
        residual = forecast["mean"] - np.asarray(target["I"],dtype=float)
        coverage = (target["I"] >= lower) & (target["I"] <= upper)
        return {"initial_distribution":coefficients,"prediction":forecast,
                "target_marginal_nll":float(forecast["nll"]),
                "target_mean_nll":float(forecast["nll"]/len(residual)),
                "expected_demand_rmse":float(np.sqrt(np.mean(residual**2))),
                "empirical_95_coverage":float(np.mean(coverage)),
                "lower_95":lower,"upper_95":upper,
                "oracle_location":float(best),"oracle_evaluations":len(evaluations)}

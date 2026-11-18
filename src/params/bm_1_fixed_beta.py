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


class bm_1_fixed_beta_params:
    NAMES = ("alpha", "beta", "k", "Xi", "C", "sigma_x", "sigma_y")
    BOUNDS = {"alpha":(.05,20.), "beta":(.05,20.), "k":(.05,12.), "Xi":(1e-4,1.),
              "C":(.05,100.), "sigma_x":(1e-4,5.), "sigma_y":(.001,.5)}

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

    def initial_guess(self, datasets):
        """Historical-data initial density guesses; no deterministic-state MLE.

        Invert the BM1 first-hour demand response approximately on the fixed
        spatial grid, then distribute mass among nearby beta basis means.
        The seven shared parameters remain ordinary admissible seed values.
        """
        self._validate(datasets)
        seed = dict(zip(self.NAMES, (2., 2., 1., .5, 2., .15, .075)))
        centers = (np.arange(self.degree+1,dtype=float)+1.)/(self.degree+2.)
        sigma = max(1./(self.degree+2.), .075)
        initial = []
        for data in datasets:
            b,u,y = (float(data[key][0]) for key in ("B","U","I"))
            response = self.solver.demand(self.solver.x,b,u,seed)
            center = float(self.solver.x[np.argmin(np.abs(response-y))])
            weights = np.exp(-.5*((centers-center)/sigma)**2)
            weights += 1e-6
            initial.append(weights/weights.sum())
        return {"parameters":seed,"initial_distributions":np.array(initial,dtype=float),
                "method":"first-hour inversion with smooth fixed-beta coefficient seed"}

    def fit(self, datasets, *, maxiter=12, maxfun=400, progress=True, seed=None,
            initial_distributions=None):
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
        if initial_distributions is None: initial_distributions = np.tile(seed_weights,(n,1))
        initial_distributions = np.asarray(initial_distributions,dtype=float)
        if (initial_distributions.shape != (n,m+1) or not np.isfinite(initial_distributions).all()
                or np.any(initial_distributions <= 0.) or not np.allclose(initial_distributions.sum(axis=1),1.,atol=1e-9)):
            raise ValueError("Provide N strictly positive normalized initial beta coefficient vectors.")
        logits = np.log(initial_distributions[:,:-1]/initial_distributions[:,-1,None])
        x0 = np.r_[np.log(starting), logits.ravel()]
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
        result = minimize(objective,x0,method="L-BFGS-B",jac="2-point",bounds=bounds,
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
        """Retrospective simplex-constrained oracle initial density.

        Shared physical/noise parameters stay fixed. All target observations
        enter only this initial-density optimization; the final density
        propagation is open-loop without observation updates.
        """
        self._validate([target])
        initial = np.mean(np.asarray(fitted["initial_distributions"]),axis=0)
        initial /= initial.sum()
        count = [0]
        def objective(c):
            count[0] += 1
            if np.any(c < -1e-8) or not np.isfinite(c).all() or np.sum(c) <= 0.: return 1e9
            weights = np.maximum(np.asarray(c,dtype=float),0.)
            weights /= weights.sum()
            forecast = self.solver.simulate(target,fitted,weights,observations=False)
            return float(np.mean((forecast["mean"]-target["I"])**2))
        fit = minimize(objective,initial,method="SLSQP",bounds=[(0.,1.)]*(self.degree+1),
                       constraints=[{"type":"eq","fun":lambda c: np.sum(c)-1.}],
                       options={"maxiter":int(maxiter),"ftol":1e-8,"eps":1e-4})
        candidates = [initial,np.maximum(np.asarray(fit.x,dtype=float),0.)]
        coefficients = min((c/c.sum() for c in candidates),key=objective)
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
                "oracle_converged":bool(fit.success),"oracle_message":str(fit.message),
                "oracle_evaluations":count[0]}

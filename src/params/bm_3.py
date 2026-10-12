"""Reuse BM1's alternating demand fit; attach BM3's fixed soft-boundary dynamics."""
import math
import numpy as np
from src.model.bm_3 import bm_3
from src.params.bm_1 import lsq_params as BM1Fit
from src.validation.params.bm_1 import bm_1_params_validation as checks


class lsq_params(BM1Fit):
    def _model_parameters(self, parameters):
        return {**parameters, 'restoring_strength': self.model.restoring_strength}

    @staticmethod
    def logistic(x, slope):
        # Continue the state RESPONSE constantly outside [0,1]; never clip X.
        values = np.clip(BM1Fit.logistic(x, slope), -1.0, 1.0)
        return float(values) if np.ndim(values) == 0 else values

    @classmethod
    def _rate(cls, x, b, u, p):
        restoring = p['restoring_strength']*((max(-x, 0.0))**2-(max(x-1, 0.0))**2)
        return (cls._demand(x, b, u, p)-b)/p['C'] + restoring

    @classmethod
    def _advance(cls, x, dt, b, u, p, substeps):
        # Bound the restoring slope on the interval containing the current state.
        # Its endpoints have inward drift; h*L<=1 also controls RK4 stages.
        excursion = max(0.0, -x, x-1)
        slope = (0.25*p['k']*p['Xi']*p['alpha']/math.tanh(p['alpha']/4)/p['C']
                 + 2*p['restoring_strength']*excursion)
        steps = max(substeps, int(math.ceil(dt*slope)))
        h = dt/steps
        for _ in range(steps):
            a = cls._rate(x, b, u, p)
            c = cls._rate(x+h*a/2, b, u, p)
            e = cls._rate(x+h*c/2, b, u, p)
            g = cls._rate(x+h*e, b, u, p)
            x += h*(a+2*c+2*e+g)/6
        if not np.isfinite(x): raise FloatingPointError('Nonfinite BM3 state.')
        return x

    def fit(self, *args, **kwargs):
        if kwargs.get('estimate_diffusion', False):
            raise ValueError('BM3 uses historical_diffusion_params for additive-noise calibration.')
        return super().fit(*args, **kwargs)

    def fit_many(self, *args, **kwargs):
        result = super().fit_many(*args, **kwargs)
        for name in ('message', 'warning'):
            if result.get(name): result[name] = result[name].replace('BM1', 'BM3')
        result['model'] = 'BM3'
        return result

    def refine_joint(self, datasets, fitted, *, C=None, Xi=None, fit_bounds=None,
                     max_nfev=80, max_seconds=20.0, gtol=1e-5, substeps=2):
        from src.params.bm_1_joint import refine_joint
        bounds = {**self.BOUNDS, **(fit_bounds or {})}
        fixed = {key: float(value) for key, value in (('C', C), ('Xi', Xi)) if value is not None}
        names = [name for name in self.NAMES if name != 'X0' and name not in fixed]
        def unpack(values):
            p = self._model_parameters({**fixed, **dict(zip(names, values))})
            p.update(f=self.response(p['alpha']), g=self.response(p['beta']))
            return p
        return refine_joint(datasets, fitted, predict=self._predict, unpack=unpack, names=names,
            lower=[bounds[n][0] for n in names], upper=[bounds[n][1] for n in names],
            max_nfev=max_nfev, max_seconds=max_seconds, gtol=gtol, substeps=substeps)


class historical_diffusion_params:
    """Same historical residual regression as BM1, with unit additive-noise shape.

    Constant sigma_x has different meaning/scale than BM1 sigma_x*X*(1-X).
    This remains a process-variance proxy, not an identified SDE likelihood fit.
    """
    def fit_many(self, datasets, fit, *, substeps=2, derivative_step=1e-4,
                 minimum_observation_sd=0.005):
        states = checks.historical_diffusion_data(datasets, fit['initial_states'], fit)
        kernels, squares, initial, errors = [], [], [], []
        for data, x0 in zip(datasets, states):
            X, D = lsq_params._predict(data, float(x0), fit, substeps)
            error = data['I']-D[:len(data['I'])]
            initial.append(float(error[0])); errors.extend(error.tolist())
            variance = 0.0
            for k in range(len(error)-1):
                x = float(X[k]); dt = float(data['Time'][k+1]-data['Time'][k])
                args = (dt, float(data['B'][k]), float(data['U'][k]), fit, substeps)
                A = (lsq_params._advance(x+derivative_step, *args)
                     -lsq_params._advance(x-derivative_step, *args))/(2*derivative_step)
                variance = A*A*variance + dt
                j = k+1; args = (float(data['B'][j]), float(data['U'][j]), fit)
                J = (lsq_params._demand(float(X[j])+derivative_step, *args)
                     -lsq_params._demand(float(X[j])-derivative_step, *args))/(2*derivative_step)
                kernels.append(J*J*variance); squares.append(float(error[j]**2))
        sd = max(minimum_observation_sd, float(np.sqrt(np.mean(np.square(initial)))))
        K, E = checks.diffusion_variance_fit(kernels, squares, sd*sd)
        target = np.maximum(E-sd*sd, 0.0)
        q = max(0.0, float(K @ target/(K @ K)))
        return {'sigma_x': math.sqrt(q), 'q': q, 'observation_sd': sd,
            'historical_rmse': float(np.sqrt(np.mean(np.square(errors)))),
            'variance_rmse': float(np.sqrt(np.mean((K*q-target)**2))), 'samples': len(K),
            'method': 'historical linear-noise nonnegative variance regression; additive BM3 noise'}

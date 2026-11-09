"""BM1 finite-volume reference for independent beta-mixture numerical validation.

This is a reference, not the beta fitting method. It implements the same
hourly state/observation order and conditional Gaussian likelihood, while
delegating one-hour propagation to the existing conservative FV solver.
"""
import numpy as np
from scipy.special import logsumexp

from src.solver.finite_volume import finite_volume_1d
from src.method.bm_1_beta import bm_1_beta


class bm_1_fv_reference:
    def __init__(self, cells=1631, max_step=1.0/32.0):
        self.cells, self.max_step = int(cells), float(max_step)
        if self.cells < 5 or not 0.0 < self.max_step <= 1.0:
            raise ValueError("Invalid finite-volume reference resolution.")
        self.solver = finite_volume_1d()
        _, self.x, self.dx = self.solver.grid(self.cells)

    def initial_mass(self, x0):
        if not 0.0 <= x0 <= 1.0: raise ValueError("Initial BM1 state must be in [0,1].")
        mass = np.zeros(self.cells)
        if x0 <= self.x[0]: mass[0] = 1.0
        elif x0 >= self.x[-1]: mass[-1] = 1.0
        else:
            right = int(np.searchsorted(self.x, x0, side="right"))
            left = right-1
            w = (x0-self.x[left])/self.dx
            mass[left], mass[right] = 1.0-w, w
        return mass

    def simulate(self, data, params, x0, observations=None):
        B, U = np.asarray(data["B"],dtype=float), np.asarray(data["U"],dtype=float)
        obs = None if observations is None else np.asarray(observations,dtype=float)
        if B.shape != U.shape or B.ndim != 1 or not B.size:
            raise ValueError("Finite-volume reference requires matching hourly baseline and price.")
        if obs is not None and (obs.shape != B.shape or not np.isfinite(obs).all()):
            raise ValueError("Finite-volume observations must match the time grid.")
        mass = self.initial_mass(float(x0))
        means, supports, weights, total_nll = [], [], [], 0.0
        for j in range(len(B)):
            demand = bm_1_beta._demand(self.x, float(B[j]), float(U[j]), params)
            means.append(float(mass @ demand))
            supports.append(demand)
            weights.append(mass.copy())
            if obs is not None:
                sigma = params["sigma_y"]
                z = (float(obs[j])-demand)/sigma
                with np.errstate(divide="ignore"):\n                    logweights = np.log(mass)-0.5*z*z
                log_evidence = float(logsumexp(logweights))
                total_nll += -log_evidence + np.log(sigma) + 0.5*np.log(2*np.pi)
                mass = np.exp(logweights-log_evidence)
            if j == len(B)-1: break
            b0, b1, u = float(B[j]), float(B[j+1]), float(U[j])
            def drift(t, x):
                b = b0+(b1-b0)*float(t)
                return (bm_1_beta._demand(x,b,u,params)-b)/params["C"]
            def diffusion(t,x):
                return params["sigma_x"]*x*(1.0-x)
            result = self.solver.solve([0.0,1.0],mass,drift,diffusion,
                                       max_step=self.max_step)
            mass = np.asarray(result["cell_mass"][-1],dtype=float)
        return {"mean":np.asarray(means), "supports":supports,
                "weights":weights, "nll":float(total_nll)}

    def intervals(self, result, observation_sd):
        return bm_1_beta().intervals(result, observation_sd)

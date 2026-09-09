import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator


class Simulate:
    def energy(self, energy_data):
        t_e, e = energy_data
        return interp1d(t_e, e, kind="linear", bounds_error=False, fill_value="extrapolate")

    def price(self, price_data):
        t_p, p = price_data
        return interp1d(t_p, p, kind="previous", bounds_error=False, fill_value="extrapolate")

    def md(self, poi_x, pr_x, y_interval=(1, -1)):
        interp_y_vals = np.full(len(pr_x), y_interval[0]) - np.cumsum(pr_x)
        return PchipInterpolator(poi_x, interp_y_vals)

    @staticmethod
    def md_vals(kappa, n):
        poi_x = np.linspace(0.0, 1.0, n)
        F = 0.5 * (1.0 + np.tanh(kappa * (poi_x - 0.5)) / np.tanh(kappa / 2.0))
        F[0] = 0.0
        F[-1] = 1.0

        pr_x = np.zeros(n)
        pr_x[1:] = 2.0 * np.diff(F)
        pr_x[1:] *= 2.0 / np.sum(pr_x[1:])
        pr_x[0] = 0.0
        return poi_x, pr_x

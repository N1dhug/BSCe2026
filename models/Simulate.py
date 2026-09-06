import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

class Simulate:
    def energy(self, energy_data):
        t_e, e = energy_data
        return interp1d(t_e, e, kind = 'linear', bounds_error=False, fill_value="extrapolate")

    def price(self, price_data):
        t_p, p = price_data
        return interp1d(t_p, p, kind = 'previous', bounds_error=False, fill_value="extrapolate")
    
    #Hermitic interpolation to get 'monotonally decreasing (md)' functions: f(Xt; A) or g(u;B). (Eq. 2.1 && 2.2). 
    def md(self, poi_x, pr_x, y_interval=(1, -1)):
        interp_y_vals = np.full(len(pr_x), y_interval[0])-np.cumsum(pr_x)
        pchip = PchipInterpolator(poi_x, interp_y_vals)
        return pchip

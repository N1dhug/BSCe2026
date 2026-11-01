"""Five fixed centered, strictly increasing, normalized response functions.

Every basis maps [-1,1] to [-1,1], has f(0)=0 and f(±1)=±1.
The cubic's maximum central slope is 1.5 under global monotonicity.
"""
import numpy as np

from src.model.wm_5 import response_function


def fixed_responses():
    k_exp, k_log = 2.0, 4.0
    exp_den, log_den = np.expm1(k_exp), np.log1p(k_log)

    exponential = response_function(
        "exponential",
        lambda x: np.sign(x)*np.expm1(k_exp*abs(x))/exp_den,
        lambda x: k_exp*np.exp(k_exp*abs(x))/exp_den,
        (float(k_exp/exp_den), float(k_exp*np.exp(k_exp)/exp_den)),
    )
    logarithmic = response_function(
        "logarithmic",
        lambda x: np.sign(x)*np.log1p(k_log*abs(x))/log_den,
        lambda x: k_log/((1.0+k_log*abs(x))*log_den),
        (float(k_log/((1.0+k_log)*log_den)), float(k_log/log_den)),
    )
    functions = [exponential, logarithmic]
    for name, b in (("polynomial_steep", 0.15),
                    ("polynomial_very_steep", 0.35),
                    ("polynomial_extremely_steep", 0.49)):
        a = 1.0+b
        functions.append(response_function(
            name,
            lambda x, a=a, b=b: a*x-b*x**3,
            lambda x, a=a, b=b: a-3.0*b*x*x,
            (1.0-2.0*b, a),
        ))
    return {function.name: function for function in functions}


def ordered_response_pairs():
    functions = fixed_responses()
    return [(g, h) for g in functions for h in functions if g != h]

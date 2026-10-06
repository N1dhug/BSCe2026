"""Supervisor Bernstein family, explicitly represented by exact cell averages.

This G-only adapter preserves the legacy midpoint path and shared solvers.
Degree elevation changes coordinates, not the original Beta(20,20) prior.
"""
import numpy as np
from src.method.beta_mixture import beta_cell_average_basis

BASIS_KIND = 'g8-bernstein-m254-cell-average-v1'
DEGREE = 254


def bernstein_basis(edges):
    m = np.arange(DEGREE + 1, dtype=float)
    return beta_cell_average_basis(edges, m + 1., DEGREE - m + 1.)


def prior_coefficients():
    """Elevate l_(19,38)=Beta(20,20); no renormalization or clipping."""
    c = np.zeros(39)
    c[19] = 1.
    for n in range(38, DEGREE):
        m = np.arange(n + 1, dtype=float)
        higher = np.zeros(n + 2)
        higher[:-1] += c * (n + 1. - m) / (n + 2.)
        higher[1:] += c * (m + 1.) / (n + 2.)
        c = higher
    return c

"""G8 development bases with explicit, distinct dictionary identities.

The 41 narrow components plus the exact Beta(20,20) prior are declared before
the replay. This is not an adaptive basis, a posterior fit or a new FP solver.
"""
import numpy as np

from src.method.beta_mixture import beta_cell_average_basis

BASIS_KIND = 'g8-local-beta-41-k256-prior20-v1'


def dictionary_shapes():
    """Fresh shape vectors; fixed mean spacing 1/41 and concentration 256."""
    means = (np.arange(41, dtype=float) + .5) / 41.
    return np.r_[20., 256. * means], np.r_[20., 256. * (1. - means)]


def localized_beta_basis(edges):
    """Exact cell integrals using the shared stable CDF/complementary-CDF API.

    Basis construction applies no target-dependent normalization or clipping.
    The first component represents the original prior to roundoff on any grid.
    """
    alpha, beta = dictionary_shapes()
    return beta_cell_average_basis(edges, alpha, beta)


BERNSTEIN_BASIS_KIND = 'g8-bernstein-m254-cell-average-v1'
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

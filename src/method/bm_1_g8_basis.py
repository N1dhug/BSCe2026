"""One fixed G8 development dictionary: locations independent of concentration.

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

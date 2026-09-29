"""Reproduce F4-0 arithmetic and quadrature checks; not a fitted F4 model.

Run from the repository root:
    python -B root/f4/verify_preparation.py
    python -B root/f4/verify_preparation.py --output <new-result.json>
No output files are written unless --output is explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln

# Direct-script convenience; keep this in the experiment script, not core code.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.params.bm_1 import bayes_update, gaussian_log_likelihood, mass_to_log
from src.method.beta_mixture import PreparedBetaProjector, beta_cell_average_basis
from src.method.fokker_planc import bm_1_fokker_planc
from src.diagnostics.bm_1 import beta_mixture_variance_floor


def compute() -> dict:
    """Independent continuous quadrature plus static discrete representation checks."""
    def prior_pdf(x):
        if not 0. < x < 1.:
            return 0.
        return np.exp(19*np.log(x)+19*np.log1p(-x)-betaln(20, 20))

    def demand(x):
        # Independent formula for linear f/g, B=U=1/2, k=2, Xi=1.
        return .5+.5*np.tanh(2*(.5-x))

    def posterior_kernel(x):
        return prior_pdf(x)*np.exp(-.5*((.5-demand(x))/.01)**2)

    z_scaled, z_error = quad(posterior_kernel, 0., 1., epsabs=1e-13, epsrel=1e-12, points=[.5])
    v_numerator, v_error = quad(lambda x: (x-.5)**2*posterior_kernel(x), 0., 1.,
                                 epsabs=1e-14, epsrel=1e-11, points=[.5])
    variance = v_numerator/z_scaled
    example = bayes_update(mass_to_log([.6, .4]), gaussian_log_likelihood(.65, [.3, .7], .1))
    bank_example = bayes_update(mass_to_log([.6, .4]), np.log([.2, .8]))

    # Nonmonotone uncertainty: an admissible BM1 example with negative initial
    # variance derivative, obtained from Ito's formula. This is not a forecast.
    def wide_pdf(x):
        return 6*x*(1-x)
    covariance = quad(lambda x: (x-.5)*(demand(x)-.5)*wide_pdf(x), 0, 1,
                      epsabs=1e-13)[0]
    noise_contribution = quad(lambda x: (.1*x*(1-x))**2*wide_pdf(x), 0, 1,
                             epsabs=1e-13)[0]
    projections = []
    for cells in (201, 801):
        dx = 1./cells
        edges = np.linspace(0, 1, cells+1)
        centers = (edges[:-1]+edges[1:])/2
        # Integrate the continuous posterior in each cell, not just its PDF.
        target_mass = np.array([quad(posterior_kernel, a, b, epsabs=1e-15)[0]
                                for a,b in zip(edges[:-1], edges[1:])])/z_scaled
        target = target_mass/dx
        for degree in (30, 38):
            method = bm_1_fokker_planc()
            basis = method._normalise_basis_discretely(method.beta_basis(centers, degree), dx)
            result = PreparedBetaProjector(basis, dx).project(target)
            mass = result['density']*dx
            projections.append({'cells':cells, 'basis':f'legacy_{degree}',
                'mean':float(mass@centers), 'center_variance':float(mass@(centers-.5)**2),
                **result['diagnostics']})
        # Illustrative, fixed dictionary only; no final F4-D basis selection.
        shapes = np.array([20., 200., 1000., 2000.])
        basis = beta_cell_average_basis(edges, shapes, shapes)
        result = PreparedBetaProjector(basis, dx).project(target)
        mass = result['density']*dx
        projections.append({'cells':cells, 'basis':'illustration_symmetric_20_200_1000_2000',
            'mean':float(mass@centers), 'center_variance':float(mass@(centers-.5)**2),
            **result['diagnostics']})
    return {
        'scope':'F4-0 static/preparatory checks only; no recursive filter, parameter recovery or week forecast',
        'source_base':'5a4793cd57d70a46beae2a6c758b4cd4e0556536',
        'single_measurement':{
            'prior':'Beta(20,20)', 'B':.5, 'U':.5, 'Y':.5, 'k':2., 'Xi':1., 'sigma_y':.01,
            'continuous_posterior_mean':.5, 'continuous_posterior_variance':variance,
            'gaussian_predictive_density':float(z_scaled/(np.sqrt(2*np.pi)*.01)),
            'quad_absolute_error_estimates':[z_error, v_error],
            'continuous_variance_floor_30':float(beta_mixture_variance_floor(.5,30)),
            'continuous_variance_floor_38':float(beta_mixture_variance_floor(.5,38)),
        },
        'two_node_bayes_example':{'posterior':example.mass.tolist(), 'log_evidence':example.log_evidence},
        'two_hypothesis_algebra_example':{'posterior':bank_example.mass.tolist(),
                                         'log_evidence':bank_example.log_evidence},
        'nonmonotone_variance_example':{'prior':'Beta(2,2)', 'rho':1.,'B':.5,'U':.5,
            'k':2.,'Xi':1.,'sigma_x':.1,'initial_variance':.05,
            'two_covariance':2*covariance,'expected_diffusion_squared':noise_contribution,
            'initial_variance_derivative':2*covariance+noise_contribution},
        'static_projection':projections,
        'dictionary_warning':'Symmetric four-component illustration only; shifted, bimodal and boundary posteriors unvalidated.',
        'variance_warning':'Continuous beta variance floor is not an exact finite-grid theorem; cell-center variances are labeled separately.',
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args(argv)
    report=compute()
    text=json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

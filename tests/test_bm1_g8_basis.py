"""Two focused G8 risks: dictionary mass/prior and continuation identity."""
import json
import unittest
import numpy as np

from src.method.bm_1_g8 import context, known_filter, bank_filter, G8BankOrigin
from src.method.bm_1_g8_basis import BASIS_KIND, dictionary_shapes
from src.params.bm_1 import BM1ObservationSeries


class G8BasisTests(unittest.TestCase):
    def test_cell_integrals_prior_and_shape_variance(self):
        _,_,_,edges,prior,projector=context(129,basis_kind=BASIS_KIND)
        basis=projector.basis;dx=1/129
        self.assertEqual(basis.shape,(42,129))
        self.assertGreaterEqual(basis.min(),0.)
        np.testing.assert_allclose(basis.sum(axis=1)*dx,1.,atol=2e-14,rtol=0.)
        np.testing.assert_allclose(basis[0]*dx,prior,atol=2e-15,rtol=0.)
        alpha,beta=dictionary_shapes();mu=alpha/(alpha+beta)
        self.assertAlmostEqual(mu[21],.5)
        self.assertAlmostEqual(mu[21]*(1-mu[21])/(alpha[21]+beta[21]+1),.25/257)
        with self.assertRaises(ValueError):context(129,38,basis_kind=BASIS_KIND)

    def test_one_node_and_resume_preserve_custom_basis(self):
        obs=BM1ObservationSeries([.02,.04],[.54,np.nan],'hour',[True,False])
        full,_,_,_=bank_filter(65,obs,.01,nodes=[.25],basis_kind=BASIS_KIND)
        known,_,_,_=known_filter(65,obs,.01,basis_kind=BASIS_KIND)
        np.testing.assert_allclose(full[0]['posterior_mass'],known.posterior_mass[0],atol=1e-13,rtol=0.)
        restored=G8BankOrigin.from_record(json.loads(json.dumps(full[0]['origin'].to_record())))
        tail=BM1ObservationSeries([.04],[np.nan],'hour',[False])
        resumed,_,_,_=bank_filter(65,tail,.01,nodes=[.25],basis_kind=BASIS_KIND,origin=restored)
        np.testing.assert_array_equal(resumed[-1]['conditional_coefficients'],full[-1]['conditional_coefficients'])
        np.testing.assert_array_equal(resumed[-1]['conditional_mass'],full[-1]['conditional_mass'])
        np.testing.assert_array_equal(resumed[-1]['prior_log_weights'],resumed[-1]['log_weights'])
        with self.assertRaises(ValueError):bank_filter(65,tail,.01,nodes=[.25],degree=38,origin=restored)


if __name__=='__main__':unittest.main()

"""Fundamental piecewise response geometry, bounds and serialization."""
import unittest
import numpy as np
from src.params.bm_1_response import BM1PiecewiseResponse

class PiecewiseResponseTests(unittest.TestCase):
    def test_linear_reference_and_fitted_midpoint(self):
        x=np.linspace(0.,1.,1001)
        curve=BM1PiecewiseResponse.linear()
        np.testing.assert_allclose(curve(x),1.-2.*x,atol=2e-16,rtol=0.)
        curved=BM1PiecewiseResponse.from_inner_values([.2],pieces=2)
        np.testing.assert_allclose(curved([0.,.25,.5,.75,1.]),[1.,.6,.2,-.4,-1.],atol=1e-15,rtol=0.)

    def test_serialized_geometry_is_owned_and_continuous(self):
        values=np.array([1.,.2,-1.]);curve=BM1PiecewiseResponse([0.,.5,1.],values)
        values[1]=.9
        self.assertEqual(float(curve(.5)),.2)
        self.assertFalse(curve.values.flags.writeable)
        self.assertTrue(np.all(np.diff(curve(np.linspace(0.,1.,1000)))<=0.))
        self.assertAlmostEqual(float(curve(.5-1e-9)),float(curve(.5+1e-9)),places=7)
        np.testing.assert_array_equal(BM1PiecewiseResponse.from_record(curve.to_record()).values,curve.values)

    def test_five_piece_drop_fractions_preserve_monotonicity_and_endpoints(self):
        for fractions in ([.2,.25,1/3,.5], [0,1,0,1], [.1,.7,.2,.4]):
            curve = BM1PiecewiseResponse.from_drop_fractions(fractions)
            np.testing.assert_allclose(curve.knots, [0,.2,.4,.6,.8,1])
            self.assertEqual(curve.values[0], 1)
            self.assertEqual(curve.values[-1], -1)
            self.assertTrue(np.all(np.diff(curve.values) <= 0))
            np.testing.assert_array_equal(BM1PiecewiseResponse.from_record(curve.to_record()).values, curve.values)
        curve = BM1PiecewiseResponse.from_drop_fractions([.2,.25,1/3,.5])
        np.testing.assert_allclose(curve(np.linspace(0,1,101)), 1-2*np.linspace(0,1,101), atol=1e-15)

    def test_invalid_geometry_counts_and_extrapolation_are_rejected(self):
        for values in ([1.,1.1,-1.],[1.,np.nan,-1.],[1.,.2,-.9]):
            with self.assertRaises(ValueError):BM1PiecewiseResponse([0.,.5,1.],values)
        for inner,pieces in (([0.],4),([0.],True),([0.],3),([1.1],2)):
            with self.assertRaises((ValueError,TypeError)):
                BM1PiecewiseResponse.from_inner_values(inner,pieces=pieces)
        curve=BM1PiecewiseResponse.linear()
        for x in (-.01,1.01,np.nan,1j):
            with self.assertRaises(ValueError):curve(x)

if __name__=='__main__':unittest.main()

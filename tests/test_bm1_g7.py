"""One focused contract risk: state reset, node dependence and missing update."""
import unittest
from unittest.mock import patch
import numpy as np
from src.control.bm_1 import BM1ControlNumerics
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_g8 import context,G8BankOrigin
from src.method.bm_1_g8_bernstein import BASIS_KIND,prior_coefficients
from src.method.bm_1_distribution_panel import baseline,target


class G7OwnedTests(unittest.TestCase):
    def test_candidate_reset_own_nodes_and_missing_update(self):
        _,fixed,_,edges,_,projector=context(257,basis_kind=BASIS_KIND)
        c=np.stack([prior_coefficients(),np.roll(prior_coefficients(),20)])
        origin=G8BankOrigin(12.,fixed,[.125,.5],np.log([.25,.75]),edges,
            c@projector.basis*np.diff(edges),backend='mixture',basis=projector.basis,
            conditional_coefficients=c,basis_kind=BASIS_KIND)
        numerics=BM1ControlNumerics(x_steps=257,objective_max_step=.1,solver_max_step=.01)
        evaluator=OwnedBankPriceEvaluator(origin,baseline=baseline,target=target,horizon=(12.,12.2),
                                          price_edges=[12.,12.1,12.2],numerics=numerics)
        starts=[]
        def transport(**kwargs):
            initial=kwargs['initial_coefficients'];starts.append(initial.copy())
            coeff=np.tile(initial,(len(kwargs['t_eval']),1));coeff[-1]=np.roll(initial,1)
            return {'coefficients':coeff,'density':coeff@kwargs['basis'],'diagnostics':{}}
        with patch('src.method.bm_1_g7.evolve_projected_density',side_effect=transport), \
             patch.object(evaluator.projector,'project',side_effect=AssertionError('Hidden initial projection')):
            first=evaluator.evaluate([.3,.7]);second=evaluator.evaluate([.7,.3])
            missing=evaluator.assimilate(second,np.nan,observed=False)
        for i,start in enumerate(starts):np.testing.assert_array_equal(start,c[i%2])
        np.testing.assert_array_equal(origin.conditional_coefficients,c)
        self.assertAlmostEqual(second['total'],origin.weights@second['node_costs'],places=12)
        np.testing.assert_allclose(second['demand_variance'],second['within_variance']+second['between_variance'],atol=1e-12)
        self.assertGreater(second['between_parameter_variance'],0.)
        np.testing.assert_array_equal(missing['origin'].log_weights,origin.log_weights)
        np.testing.assert_array_equal(missing['origin'].conditional_coefficients,second['predicted_origin'].conditional_coefficients)
        np.testing.assert_array_equal(missing['origin'].conditional_mass,second['predicted_origin'].conditional_mass)
        self.assertNotEqual(first['total'],second['total'])
        with self.assertRaises(ValueError):evaluator.assimilate(dict(second,_owner=None),.5)
        with self.assertRaises(ValueError):OwnedBankPriceEvaluator(origin,baseline=baseline,target=target,
            horizon=(12.1,12.2),price_edges=[12.1,12.2],numerics=numerics)


if __name__=='__main__':unittest.main()

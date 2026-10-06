"""Three G8 checks for concrete chronology, joint-law and checkpoint risks."""
import json
import unittest
import numpy as np
from src.control.bm_1 import BM1ControlNumerics
from src.method.bm_1_distribution_panel import baseline, target
from src.method.bm_1_g8 import G8BankOrigin, context, bank_filter, known_filter, weighted_control_evaluation
from src.params.bm_1 import BM1ObservationSeries


class G8Tests(unittest.TestCase):
    def test_one_node_matches_known_filter_and_missing_prediction(self):
        observations = BM1ObservationSeries([.25,.5],[.54,np.nan],'hour',[True,False])
        known,_,_,_ = known_filter(17,observations,.125)
        records,_,_,_ = bank_filter(17,observations,.125,nodes=[.25])
        np.testing.assert_allclose(np.stack([r['posterior_mass'] for r in records]),known.posterior_mass,atol=1e-13)
        np.testing.assert_allclose(records[0]['log_evidence'],known.log_evidence[0],atol=1e-13)
        np.testing.assert_array_equal(records[1]['predicted_mass'],records[1]['posterior_mass'])
        np.testing.assert_array_equal(records[1]['prior_log_weights'],records[1]['log_weights'])
        self.assertTrue(np.isnan(records[1]['log_evidence']))
        with self.assertRaises(ValueError):
            bank_filter(17,observations,.125,nodes=[.25],origin=records[-1]['origin'])

    def test_two_node_full_objective_and_map_keep_conditional_laws(self):
        _,fixed,_,edges,_,_ = context(17)
        mass = np.zeros((2,17));mass[0,4]=1.;mass[1,12]=1.
        original = mass.copy()
        origin = G8BankOrigin(0.,fixed,[.125,.5],np.log([.3,.7]),edges,mass)
        mass[:]=0.
        np.testing.assert_array_equal(origin.conditional_mass,original)
        restored = G8BankOrigin.from_record(json.loads(json.dumps(origin.to_record())))
        np.testing.assert_array_equal(restored.conditional_mass,original)
        chosen = restored.map_origin()
        self.assertEqual(chosen.selected_node,1);self.assertEqual(chosen.parameters.C,2.)
        np.testing.assert_array_equal(chosen.posterior_mass,original[1])
        numerics=BM1ControlNumerics(x_steps=17,objective_max_step=.125,solver_max_step=.125)
        options=dict(baseline=baseline,target=target,horizon=(0.,.5),price_edges=[0.,.5])
        evaluation=weighted_control_evaluation(restored,prices=[.3],numerics=numerics,**options)
        direct=[]
        for node in restored.node_origins():
            problem=node.control_problem(**options)
            direct.append(problem.build_evaluator(numerics,fast=False).evaluate([.3]).total)
        self.assertAlmostEqual(evaluation['total'],.3*direct[0]+.7*direct[1],places=13)
        self.assertGreater(evaluation['between_parameter_variance'],0.)
        self.assertAlmostEqual(evaluation['variance'],evaluation['within_parameter_variance']+
                               evaluation['between_parameter_variance'],places=13)
        one=G8BankOrigin(0.,fixed,[.5],[0.],edges,original[1:])
        value=weighted_control_evaluation(one,prices=[.3],numerics=numerics,**options)
        self.assertAlmostEqual(value['total'],direct[1],places=13)
        with self.assertRaises(ValueError):
            weighted_control_evaluation(one,prices=[.3],numerics=numerics,
                                       **dict(options,horizon=(.1,.5),price_edges=[.1,.5]))

    def test_mixture_origin_roundtrip_and_mismatched_checkpoint_rejected(self):
        _,fixed,_,edges,_,projector=context(17,4)
        coefficients=np.array([[0.,0.,1.,0.,0.]])
        mass=coefficients @ projector.basis * projector.dx
        origin=G8BankOrigin(.25,fixed,[.25],[0.],edges,mass,backend='mixture',
                           conditional_coefficients=coefficients,basis=projector.basis)
        record=origin.to_record();restored=G8BankOrigin.from_record(json.loads(json.dumps(record)))
        np.testing.assert_array_equal(restored.conditional_coefficients,coefficients)
        self.assertFalse(restored.conditional_mass.flags.writeable)
        record['conditional_mass'][0][0] += .01
        with self.assertRaises(ValueError):G8BankOrigin.from_record(record)
        obs=BM1ObservationSeries([.5],[.5],'hour')
        with self.assertRaises(ValueError):
            bank_filter(17,obs,.125,nodes=[.25],degree=6,origin=restored)


if __name__=='__main__':unittest.main()

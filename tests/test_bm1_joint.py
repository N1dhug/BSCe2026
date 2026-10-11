"""Small checks of joint-fit acceptance, state fitting and beta step bounds."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
from src.params.bm_1_joint import refine_joint
from src.params.bm_1_response import BM1BetaResponse
from src.params.bm_1_families import FamilyFit,curves,shape_config
from src.model.bm_1 import bm_1


class JointFitTests(unittest.TestCase):
    def fixture(self):
        times=np.linspace(0,1,24)
        states=np.linspace(.1,.5,5)
        data=[{'Time':times,'I':.7+x*times} for x in states]
        fitted={'a':.2,'Xi':1.,'C':1.,'initial_states':np.full(5,.8),
                'converged':False,'stop_reason':'max_iterations'}
        def unpack(v):return {'a':float(v[0]),'Xi':1.,'C':1.}
        def predict(d,x,p,substeps):return np.zeros(24),p['a']+x*d['Time']
        return data,fitted,dict(predict=predict,unpack=unpack,names=['a'],lower=[0.],upper=[1.],max_seconds=60.)

    def test_joint_solves_shared_parameter_and_five_states(self):
        data,fitted,options=self.fixture()
        result=refine_joint(data,fitted,**options)
        self.assertTrue(result['converged'])
        self.assertLess(result['joint_fit']['final_sse'],result['joint_fit']['reference_sse'])
        self.assertAlmostEqual(result['a'],.7,places=5)
        np.testing.assert_allclose(result['initial_states'],np.linspace(.1,.5,5),atol=1e-5)

    def test_worse_candidate_cannot_replace_saved_fit(self):
        data,fitted,options=self.fixture()
        # An optimizer claiming success must still pass the actual SSE guard.
        proposed=OptimizeResult(x=np.r_[.99,np.ones(5)],status=1,success=True,optimality=0.,nfev=1)
        with patch('src.params.bm_1_joint.least_squares',return_value=proposed):
            result=refine_joint(data,fitted,**options)
        self.assertEqual(result['joint_fit']['final_sse'],result['joint_fit']['reference_sse'])
        self.assertFalse(result['converged'])
        np.testing.assert_array_equal(result['initial_states'],fitted['initial_states'])

    def test_linear_candidate_is_kept_at_joint_time_limit(self):
        t=np.arange(25,dtype=float);b=np.full(25,.45);u=.5+.3*np.sin(t/4)
        model=bm_1(lambda time:.45,lambda time:.5,(0.,24.),25)
        f,g=curves('linear',[])
        linear={'C':1.,'k':1.,'Xi':.2,'f':f,'g':g,'scalar_f':f.scalar_evaluator(),
                'initial_states':np.linspace(.2,.6,5),'converged':False,'stop_reason':'max_iterations'}
        data=[]
        for x in linear['initial_states']:
            d={'Time':t,'B':b,'U':u};d['I']=FamilyFit._predict(d,x,linear,2)[1][:24];data.append(d)
        names,_,_,_=shape_config('bernstein');values=[.9]*len(names)
        f,g=curves('bernstein',values)
        bad={**linear,**dict(zip(names,values)),'f':f,'g':g,'scalar_f':f.scalar_evaluator(),
             'family':'bernstein','shape_parameters':dict(zip(names,values))}
        result=FamilyFit(model,'bernstein').refine_joint(data,bad,linear_fit=linear,max_seconds=1e-12)
        self.assertEqual(result['joint_fit']['seed_source'],'linear_reference')
        self.assertEqual(result['joint_fit']['stop_reason'],'time_limit')
        self.assertLess(result['mse'],1e-25)
        self.assertFalse(result['converged'])

    def test_beta_slope_near_shape_one_stays_positive(self):
        for a,b in ((8.193721555566785,np.nextafter(1.,2.)),(np.nextafter(1.,2.),8.193721555566785)):
            self.assertAlmostEqual(BM1BetaResponse(a,b).max_abs_slope,2*max(a,b),places=10)


if __name__=='__main__':unittest.main()

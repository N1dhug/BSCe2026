"""G-only owned Bernstein posterior -> price forward -> point update.

No new FP/LS/Bayes kernel. Every price candidate starts from the same frozen
conditional coefficients. Geometry caches never carry a candidate's state.
"""
import numpy as np
from src.control.bm_1 import BM1ControlNumerics
from src.control import objective as tracking
from src.method.beta_mixture import PreparedBetaProjector, evolve_projected_density
from src.method.bm_1_estimation import bm1_observation_update, parameter_bank_weights
from src.method.bm_1_g8 import G8BankOrigin
from src.method.bm_1_g8_bernstein import BASIS_KIND


class OwnedBankPriceEvaluator:
    """Fixed-weight single-shooting evaluation from an explicit M254 origin.

    The original DynamicMixturePriceEvaluator retains its legacy basis.
    This adapter supplies owned basis/c to the same dynamic transport kernel.
    Future observations are accepted only by the separate assimilate method.
    """
    def __init__(self,origin,*,baseline,target,horizon,price_edges,numerics,
                 backend='mixture',configure_projector=None):
        if not isinstance(origin,G8BankOrigin) or origin.time_unit!='hour':
            raise TypeError('An owned G8 hourly bank origin is required.')
        if backend not in ('mixture','fv'):
            raise ValueError('Declare mixture or independent FV reference.')
        if not isinstance(numerics,BM1ControlNumerics) or numerics.x_steps!=origin.conditional_mass.shape[1]:
            raise ValueError('Explicit same-grid numerics required; no implicit regridding.')
        if backend=='mixture' and (origin.backend!='mixture' or origin.basis_kind!=BASIS_KIND):
            raise ValueError('Primary G7 requires the typed cell-integrated supervisor M254 basis.')
        self.origin=origin;self.backend=backend
        self.supports=tuple(node.control_problem(baseline=baseline,target=target,
             horizon=horizon,price_edges=price_edges).build_evaluator(numerics,fast=False)
             for node in origin.node_origins())
        if any(not np.array_equal(s.model.fokker_planc_reference.grid(numerics.x_steps)[0],origin.x_edges)
               for s in self.supports):
            raise ValueError('Control grid differs from the owned checkpoint.')
        self.projector=None
        if backend=='mixture':
            self.projector=PreparedBetaProjector(origin.basis,np.diff(origin.x_edges)[0])
            if configure_projector is not None:configure_projector(self.projector)
        elif configure_projector is not None:
            raise ValueError('FV has no mixture projector.')

    def evaluate(self,prices):
        prices=self.supports[0].validate_prices(prices).copy()
        masses=[];coefficients=[];objectives=[];diagnostics=[]
        for j,s in enumerate(self.supports):
            parameters=s.parameters_for_prices(prices)
            if self.backend=='mixture':
                evolved=evolve_projected_density(method=s.model.fokker_planc,
                    t_eval=s.solver_t_eval,breakpoints=s.breakpoints,x=s.x,dx=s.dx,
                    basis=self.origin.basis,initial_coefficients=self.origin.conditional_coefficients[j].copy(),
                    drift=lambda t,x:s.model._drift(t,x,parameters),
                    diffusion=lambda t,x:s.model.diffusion_coefficient(x,s.sigma_x),
                    max_step=s.solver_max_step,cfl=.4,
                    projection_data=self.projector.projection_data,
                    projection_solver=self.projector.projection_solver)
                mass=evolved['density']*s.dx
                coefficients.append(evolved['coefficients']);diagnostics.append(evolved['diagnostics'])
                demand=tracking.demand_grid(s.model,parameters,s.midpoints,s.x)
                mean,second,_=tracking.discrete_demand_moments(demand,mass[1:-1])
                obj=tracking.stochastic_tracking_objective(s.widths,mean,second,s.target)
            else:
                result=s.evaluate(prices);mass=result.cell_mass;obj=result.objective
                diagnostics.append(result.diagnostics)
            masses.append(mass);objectives.append(obj)
        weights=self.origin.weights
        means=np.stack([o.expected_demand for o in objectives])
        second=np.stack([o.second_moment for o in objectives])
        s=self.supports[0]
        objective=tracking.stochastic_tracking_objective(s.widths,weights@means,weights@second,s.target)
        within=weights@np.stack([o.demand_variance for o in objectives])
        between=np.sum(weights[:,None]*(means-objective.expected_demand)**2,axis=0)
        costs=np.array([o.total for o in objectives])
        if abs(objective.total-float(weights@costs))>1e-10:
            raise RuntimeError('Bank objective differs from weighted node costs.')
        if np.max(abs(objective.demand_variance-within-between))>1e-10:
            raise RuntimeError('Law of total variance failed.')
        mass=np.stack(masses);coefficients=np.stack(coefficients) if coefficients else None
        predicted=G8BankOrigin(s.model.T[1],self.origin.fixed_parameters,self.origin.rho_values,
            self.origin.log_weights,self.origin.x_edges,mass[:,-1],sigma_y=self.origin.sigma_y,
            backend=self.backend,conditional_coefficients=None if coefficients is None else coefficients[:,-1],
            basis=self.origin.basis if self.backend=='mixture' else None,
            price_side=self.origin.price_side,basis_kind=BASIS_KIND if self.backend=='mixture' else None)
        return dict(total=objective.total,bias=objective.bias,variance=objective.variance,
            within_parameter_variance=float(s.widths@within),between_parameter_variance=float(s.widths@between),
            expected_demand=objective.expected_demand,demand_variance=objective.demand_variance,
            within_variance=within,between_variance=between,node_costs=costs,weights=weights.copy(),
            times=s.solver_t_eval,midpoints=s.midpoints,widths=s.widths,target=s.target,
            conditional_mass=mass,conditional_coefficients=coefficients,prices=prices,
            diagnostics=diagnostics,predicted_origin=predicted,backend=self.backend,_owner=self)

    def __call__(self,prices):return self.evaluate(prices)['total']

    def assimilate(self,evaluation,value,*,observed=True):
        """Point observation at the control endpoint, under its actual price.

        No second propagation, no historical future y, no original fixture-price
        restart. State evidence is computed before its mixture LS projection.
        """
        if evaluation.get('_owner') is not self:
            raise ValueError('Assimilation requires this evaluator\'s own price forward.')
        predicted=evaluation['predicted_origin']
        if (predicted.backend!=self.backend or predicted.time!=self.supports[0].model.T[1]
            or not np.array_equal(predicted.log_weights,self.origin.log_weights)):
            raise ValueError('Evaluation does not match this frozen bank/horizon.')
        if not isinstance(observed,(bool,np.bool_)):
            raise ValueError('observed must be boolean.')
        scores=[];mass=[];coefficients=[];projections=[]
        for j,s in enumerate(self.supports):
            c=None if predicted.conditional_coefficients is None else predicted.conditional_coefficients[j]
            if observed:
                parameters=s.parameters_for_prices(evaluation['prices'])
                update=bm1_observation_update(s.model,parameters,self.origin.sigma_y,
                    predicted.conditional_mass[j],self.origin.time,predicted.time,float(value),
                    price_side=self.origin.price_side,likelihood_quadrature_order=3)
                scores.append(update.log_evidence);m=update.mass
                if self.backend=='mixture':
                    projection=self.projector.project(m/s.dx,initial_coefficients=c)
                    c=projection['coefficients'];m=projection['density']*s.dx
                    projections.append(projection['diagnostics'])
            else:
                scores.append(np.nan);m=predicted.conditional_mass[j].copy()
            mass.append(m)
            if c is not None:coefficients.append(c)
        weight_update=parameter_bank_weights(np.array([scores]),self.origin.log_weights,np.array([observed]))
        posterior=G8BankOrigin(predicted.time,predicted.fixed_parameters,predicted.rho_values,
            weight_update['log_weights'][0],predicted.x_edges,np.array(mass),sigma_y=predicted.sigma_y,
            backend=self.backend,conditional_coefficients=np.array(coefficients) if coefficients else None,
            basis=predicted.basis,price_side=predicted.price_side,basis_kind=predicted.basis_kind)
        return dict(origin=posterior,conditional_log_evidence=np.array(scores),
                    log_evidence=float(weight_update['log_evidence'][0]),projections=projections)

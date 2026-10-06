"""G6b thin adapters: existing BM FV, dynamic Beta projection and raw EM.

Hours are canonical. No fit, price optimization, solver rewrite or FV restart
of the evolving mixture. Initial distribution is Beta(20,20), not a point.
"""
import numpy as np
from scipy.integrate import solve_ivp
from src.control.bm_1 import BM1ControlProblem, BM1ControlNumerics, BM1Parameters, beta_initial_cell_mass, sample_beta_initial_states
from src.control.objective import discrete_demand_moments, stochastic_tracking_objective, monte_carlo_tracking_objective
from src.method.beta_mixture import evolve_projected_density, PreparedBetaProjector, _SimplexProjector
from src.method.bm_1_linear_noise import demand_state_derivative
from src.params.bm_1_response import BM1PiecewiseResponse

PRICES=np.array([.3,.7,.3,.7])
EDGES=np.array([0.,6.,12.,18.,24.])


class ProjectionObserver(_SimplexProjector):
    """Observe the final guard evaluation of each step, without another QP.

    The shared routine checks optimality twice on its primary path and once
    after a fallback. Retain the last check and flush it at the next drift
    evaluation (or after evolution); provisional checks are never accumulated.
    """
    def __init__(self, matrix, dx):
        super().__init__(matrix)
        self.dx = dx
        self.pending = None
        self.losses = []

    def optimality_gap(self, target, coefficients):
        result = super().optimality_gap(target, coefficients)
        residual = self.matrix @ coefficients - target
        self.pending = (float(np.sqrt(self.dx)*np.abs(residual).sum()),
                        float(np.linalg.norm(residual)), float(result[0]))
        return result

    def flush(self):
        if self.pending is not None:
            self.losses.append(self.pending)
            self.pending = None


def baseline(time):
    return .5+.1*np.sin(2*np.pi*np.asarray(time)/24)


def target(time):
    return np.full_like(np.asarray(time,dtype=float),.5)


def build_problem(*, time_unit='hours'):
    """Explicit hours/days adapter; both diffusion and drift scales transform."""
    if time_unit not in ('hours','days'):raise ValueError('Declare hours or days.')
    multiplier=1. if time_unit=='hours' else 24.
    curve=BM1PiecewiseResponse.linear(pieces=2)
    parameters=BM1Parameters(curve,curve,4/multiplier,2.,.4,.2*np.sqrt(multiplier))
    return BM1ControlProblem(lambda t:baseline(multiplier*np.asarray(t)),target,
        (0.,24/multiplier),EDGES/multiplier,parameters,beta_initial_cell_mass,
        deterministic_initial_state=.5,observed_prices=PRICES)


def make_support(cells, step):
    problem=build_problem()
    numerics=BM1ControlNumerics(x_steps=cells,objective_max_step=step,solver_max_step=step)
    support=problem.build_evaluator(numerics,fast=False)
    # Add hourly reporting without changing the common midpoint objective.
    report=np.arange(25.)
    times=np.unique(np.r_[support.solver_t_eval,report])
    return support,times,np.searchsorted(times,report),np.searchsorted(times,support.midpoints)


def uniform_grid_indices(grid, times):
    """Nearest exact knot, tolerant of different floating-point constructions.

    searchsorted alone can select the following knot when quadrature produces
    an otherwise identical midpoint a few ulps above the integration knot.
    No interpolation or acceptance of a missing time is allowed.
    """
    grid=np.asarray(grid)
    indices=np.rint((np.asarray(times)-grid[0])/(grid[1]-grid[0])).astype(int)
    if np.any(indices<0) or np.any(indices>=len(grid)) or not np.allclose(
            grid[indices],times,rtol=0,atol=1e-12):
        raise ValueError('Integration grid must contain every requested time.')
    return indices


def distribution_summary(model, parameters, time, x, mass, edges):
    """Midpoint state moments; uniform-cell CDF/quantiles; nonlinear demand integral."""
    mass=np.asarray(mass,float)
    dx=float(edges[1]-edges[0]);cdf_edges=np.c_[np.zeros(len(time)),np.cumsum(mass,axis=1)]
    d=np.array([model.eval(t,x,*parameters[:3],parameters[4],parameters[5])[-1] for t in time])
    mean=mass@x;variance=(mass@x**2)-mean**2
    dm,second,dv=discrete_demand_moments(d,mass)
    state_cdf=np.array([[np.interp(q,edges,row) for q in (.25,.5,.75)] for row in cdf_edges])
    state_quantile=np.array([[np.interp(q,row,edges) for q in (.05,.5,.95)] for row in cdf_edges])
    demand_cdf=np.array([[float(row[values<=q].sum()) for q in (.4,.5,.6)] for row,values in zip(mass,d)])
    demand_quantile=[]
    for row,values in zip(mass,d):
        sort=np.argsort(values);cumulative=np.cumsum(row[sort])
        demand_quantile.append(np.interp([.05,.5,.95],cumulative,values[sort]))
    return {'state_mean':mean,'state_variance':variance,'demand_mean':dm,'demand_variance':dv,
        'state_cdf':state_cdf,'state_quantile':state_quantile,
        'demand_cdf':demand_cdf,'demand_quantile':np.asarray(demand_quantile),
        'demand_at_mean_state':np.array([model.eval(t,v,*parameters[:3],parameters[4],parameters[5])[-1] for t,v in zip(time,mean)])}


def solve_distribution(method, cells, degree, step, *, deadline=None):
    from time import perf_counter
    support,times,hourly,objective_indices=make_support(cells,step)
    parameters=support.parameters_for_prices(PRICES)
    model=support.model
    def guard():
        if deadline is not None and perf_counter()>=deadline:raise RuntimeError('G6b wall-time budget stop.')
    initial_projection={}
    if method=='fv':
        guard()
        result=model.fokker_planc_reference_solver(support.initial_mass,parameters,.2,
            t_eval=times,breakpoints=EDGES[1:-1],max_step=step)
        mass=result[3];diagnostics=result[-1]
        coefficients=None
    elif method=='mixture':
        basis=model.fokker_planc._normalise_basis_discretely(model.fokker_planc.beta_basis(support.x,degree),support.dx)
        projector=PreparedBetaProjector(basis,support.dx)
        projected=projector.project(support.initial_mass/support.dx)
        coefficients=projected['coefficients']
        initial_mass=(coefficients@basis)*support.dx
        initial_projection={'l1':float(np.sum(abs(initial_mass-support.initial_mass))),
            'state_mean_error':float((initial_mass-support.initial_mass)@support.x),
            'state_variance_error':float(initial_mass@support.x**2-(initial_mass@support.x)**2-(support.initial_mass@support.x**2-(support.initial_mass@support.x)**2)),
            'diagnostics':projected['diagnostics']}
        observer=ProjectionObserver(projector.projection_data[0],support.dx)
        def drift(t,x):
            observer.flush()
            guard()
            return model._drift(t,x,parameters)
        def diffusion(t,x):return model.diffusion_coefficient(x,.2)
        evolved=evolve_projected_density(method=model.fokker_planc,t_eval=times,
            breakpoints=EDGES[1:-1],x=support.x,dx=support.dx,basis=basis,
            initial_coefficients=coefficients,drift=drift,diffusion=diffusion,
            max_step=step,cfl=.4,projection_data=projector.projection_data,
            projection_solver=observer)
        observer.flush()
        projection_losses=np.asarray(observer.losses)
        mass=evolved['density']*support.dx;diagnostics=evolved['diagnostics'];coefficients=evolved['coefficients']
        if len(projection_losses)!=diagnostics['n_steps']:
            raise RuntimeError('Projection observation count does not match transport steps.')
        diagnostics['maximum_step_projection_l1']=float(projection_losses[:,0].max())
        diagnostics['sum_step_projection_l1']=float(projection_losses[:,0].sum())
        diagnostics['maximum_step_projection_l2']=float(projection_losses[:,1].max())
        diagnostics['maximum_projection_optimality_gap']=float(projection_losses[:,2].max())
    else:raise ValueError('Declare fv or mixture.')
    guard()
    d=np.array([model.eval(t,support.x,*parameters[:3],parameters[4],parameters[5])[-1] for t in support.midpoints])
    dm,second,_=discrete_demand_moments(d,mass[objective_indices])
    objective=stochastic_tracking_objective(support.widths,dm,second,support.target)
    summary=distribution_summary(model,parameters,times[hourly],support.x,mass[hourly],support.edges)
    # Continuous Beta-mixture lower bound; not an exact discrete-grid bound.
    if method=='mixture':
        diagnostics['max_continuous_beta_variance_floor_at_hourly_mean']=float(np.max(summary['state_mean']*(1-summary['state_mean'])/(degree+3)))
    return {'hourly':summary,'objective':float(objective.total),'diagnostics':diagnostics,
        'initial_projection':initial_projection,'arrays':{'time':times,'x':support.x,'edges':support.edges,'cell_mass':mass,
        **({'coefficients':coefficients,'step_projection_losses':projection_losses} if coefficients is not None else {})}}


def solve_em(step, initial, increments, *, deadline=None):
    from time import perf_counter
    support,times,hourly,objective_indices=make_support(129,step)
    parameters=support.parameters_for_prices(PRICES);model=support.model
    integration=np.arange(round(48/step)+1)*(step/2)
    # Controlled integration is on the exactly nested grid; report includes quadrature midpoints.
    output=model.euler_maruyama_solver(initial,parameters,.2,n_paths=len(initial),boundary='none',
        # Every integration step is already an output knot. Avoid inserting
        # spurious subdivisions when a binary float width exceeds step/2.
        t_eval=integration,breakpoints=EDGES[1:-1],max_step=np.inf,
        brownian_increments=increments,return_diagnostics=True)
    states=np.asarray(output[1]);demands=np.asarray(output[7]);diagnostics=output[-1]
    if diagnostics['total_violations']:raise RuntimeError('Raw EM left [0,1]; no clipping or boundary correction.')
    if deadline is not None and perf_counter()>=deadline:raise RuntimeError('G6b wall-time budget stop.')
    # EM trajectories needed at midpoint objective times: use a half-step grid.
    hidx=uniform_grid_indices(integration,np.arange(25.))
    midpoint_index=uniform_grid_indices(integration,support.midpoints)
    objective=monte_carlo_tracking_objective(support.widths,demands[:,midpoint_index],support.target)
    xs=states[:,hidx];ds=demands[:,hidx]
    se=lambda values:np.std(values,axis=0,ddof=1)/np.sqrt(len(initial))
    state_mean=xs.mean(axis=0);demand_mean=ds.mean(axis=0)
    state_var=xs.var(axis=0);demand_var=ds.var(axis=0)
    state_cdf=np.array([(xs<=q).mean(axis=0) for q in (.25,.5,.75)]).T
    demand_cdf=np.array([(ds<=q).mean(axis=0) for q in (.4,.5,.6)]).T
    result={'state_mean':state_mean,'state_variance':state_var,'demand_mean':demand_mean,'demand_variance':demand_var,
        'state_cdf':state_cdf,'demand_cdf':demand_cdf,
        'state_quantile':np.quantile(xs,[.05,.5,.95],axis=0).T,'demand_quantile':np.quantile(ds,[.05,.5,.95],axis=0).T,
        'state_mean_se':se(xs),'demand_mean_se':se(ds),
        'state_variance_se':se((xs-state_mean)**2),'demand_variance_se':se((ds-demand_mean)**2),
        'state_cdf_se':np.sqrt(state_cdf*(1-state_cdf)/len(initial)),
        'demand_cdf_se':np.sqrt(demand_cdf*(1-demand_cdf)/len(initial))}
    return {'hourly':result,'objective':float(objective.objective.total),
        'objective_se':float(np.std(objective.path_objectives,ddof=1)/np.sqrt(len(initial))),
        'diagnostics':diagnostics,'arrays':{'integration_time':integration,'initial_states':initial,
        'states_hourly':xs,'demand_hourly':ds,'path_objectives':objective.path_objectives}}


def linear_noise_reference():
    """Same initial mean AND variance; local Gaussian approximation, no fitted noise."""
    support=make_support(129,1/48)[0];model=support.model;parameters=support.parameters_for_prices(PRICES)
    values=np.empty((25,2));start=np.array([.5,1/164])
    for left,right in zip(EDGES[:-1],EDGES[1:]):
        def rhs(t,y):
            time=np.nextafter(right,left) if t>=right and right<24 else t
            x,P=y;B=baseline(time);gv=1-2*parameters[0](time)
            derivative=float(demand_state_derivative(x,gv,B,2.,.4))
            return [float(model._drift(time,x,parameters)),2*derivative/4*P+.2**2*(x*(1-x))**2]
        solution=solve_ivp(rhs,(left,right),start,rtol=1e-9,atol=1e-11,max_step=1/48,dense_output=True)
        if not solution.success:raise RuntimeError(solution.message)
        indices=np.arange(int(left),int(right)+1)
        values[indices]=solution.sol(indices).T;start=solution.y[:,-1]
    x,P=values.T
    derivative=demand_state_derivative(x,1-2*parameters[0](np.arange(25.)),baseline(np.arange(25.)),2.,.4)
    d=np.array([model.eval(t,v,*parameters[:3],parameters[4],parameters[5])[-1] for t,v in zip(np.arange(25.),x)])
    return {'state_mean':x,'state_variance':P,'demand_mean':d,'demand_variance':derivative**2*P,
            'initial_variance':1/164,'scope':'same prior first two moments; Gaussian/delta approximation, not Beta distribution evolution'}

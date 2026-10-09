"""Frozen F4 origin shared by open forecasts and the existing F3 evaluator.

The plug-in approximation discards parameter uncertainty explicitly. No truth,
future observations, curve fitting or new time-stepping kernel belongs here.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.control.bm_1 import BM1ControlProblem, BM1Parameters
from src.method.bm_1_estimation import BM1FVFilterStep, BM1FVRhoBankStep
from src.method.bm_1_observation import conservative_mass_transfer
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation import estimation as checks


@dataclass(frozen=True)
class BM1ForecastOrigin:
    """Owned posterior at one assimilation cutoff, in an explicit time unit.

    Parameters and pure response callables are held fixed during forward use.
    sigma_y is an observation parameter; it is never substituted for sigma_x.
    MAP means maximum discrete posterior node weight, with first-node tie break.
    """

    time: float
    time_unit: str
    parameters: BM1Parameters
    sigma_y: float
    x_edges: object
    posterior_mass: object
    parameter_source: str = 'known'
    selected_node: int | None = None

    def __post_init__(self):
        time = checks.finite_scalar(self.time, 'origin time')
        if self.time_unit not in ('second', 'hour', 'day'):
            raise ValueError('Declare the origin time unit; no implicit conversion is performed.')
        if not isinstance(self.parameters, BM1Parameters):
            raise TypeError('parameters must be BM1Parameters.')
        sigma_y = checks.finite_scalar(self.sigma_y, 'sigma_y', positive=True)
        edges = checks.increasing_grid(self.x_edges, 'x_edges').copy()
        mass = checks.probability_mass(self.posterior_mass, 'posterior_mass').copy()
        if edges.size != mass.size + 1 or edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError('Posterior cells must cover exactly [0,1].')
        if self.parameter_source not in ('known', 'rho_map'):
            raise ValueError('parameter_source must be known or rho_map.')
        if self.parameter_source == 'rho_map':
            if (isinstance(self.selected_node, (bool, np.bool_))
                    or not isinstance(self.selected_node, (int, np.integer))
                    or self.selected_node < 0):
                raise ValueError('A rho MAP origin needs its selected node index.')
        elif self.selected_node is not None:
            raise ValueError('Known parameters have no selected bank node.')
        edges.setflags(write=False)
        mass.setflags(write=False)
        object.__setattr__(self, 'time', time)
        object.__setattr__(self, 'sigma_y', sigma_y)
        object.__setattr__(self, 'x_edges', edges)
        object.__setattr__(self, 'posterior_mass', mass)

    @classmethod
    def from_filter_step(cls, step, parameters, sigma_y, *, time_unit):
        if not isinstance(step, BM1FVFilterStep):
            raise TypeError('step must be BM1FVFilterStep.')
        return cls(step.time, time_unit, parameters, sigma_y,
                   step.x_edges, step.posterior_mass)

    @classmethod
    def from_rho_bank_step(cls, step, fixed_parameters, sigma_y, *, time_unit):
        if not isinstance(step, BM1FVRhoBankStep):
            raise TypeError('step must be BM1FVRhoBankStep.')
        if not isinstance(fixed_parameters, BM1Parameters):
            raise TypeError('fixed_parameters must be BM1Parameters.')
        index = int(np.argmax(step.log_weights))
        conditional = step.conditional_steps[index]
        parameters = replace(fixed_parameters, C=1.0 / step.rho_values[index])
        return cls(step.time, time_unit, parameters, sigma_y,
                   conditional.x_edges, conditional.posterior_mass, 'rho_map', index)

    def initial_mass(self, target_edges):
        """F3's builder contract: exact overlaps of the represented cell law."""
        return conservative_mass_transfer(self.x_edges, self.posterior_mass, target_edges)

    def control_problem(self, *, baseline, target, horizon, price_edges, **settings):
        """Build the existing F3 problem at exactly this information cutoff."""
        problem = BM1ControlProblem(
            baseline=baseline, target=target, horizon=horizon, price_edges=price_edges,
            parameters=self.parameters, initial_mass_builder=self.initial_mass, **settings)
        if problem.horizon[0] != self.time:
            raise ValueError('F3 horizon must start at the origin; no unmodelled time gap.')
        return problem


def forecast_fv(origin, *, baseline, price_edges, prices, times, max_step,
                x_steps=None, breakpoints=None):
    """One open propagation supplies every requested absolute forecast time.

    All times use origin.time_unit. Future B/U are fixed caller inputs, not
    inferred by this helper. There is no observations argument or assimilation.
    Returns the cell law and centered state/demand moments at cell centers;
    Y variance additionally includes sigma_y**2. These are plug-in marginals,
    not a path law, simultaneous interval or an accepted calibration study.
    """
    if not isinstance(origin, BM1ForecastOrigin):
        raise TypeError('origin must be BM1ForecastOrigin.')
    times = checks.increasing_grid(times, 'forecast times', minimum_size=1)
    if times[0] <= origin.time:
        raise ValueError('Forecast times must strictly follow the origin.')
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    cells = origin.posterior_mass.size if x_steps is None else checks.positive_integer(
        x_steps, 'x_steps', minimum=3)
    edges = checks.increasing_grid(price_edges, 'price_edges')
    problem = origin.control_problem(
        baseline=baseline, target=baseline, horizon=(origin.time, edges[-1]),
        price_edges=edges, breakpoints=breakpoints, observed_prices=prices)
    if times[-1] > problem.horizon[1]:
        raise ValueError('Future price plan must cover every forecast time.')
    from src.model.bm_1 import bm_1
    model = bm_1(baseline, baseline, problem.horizon, 2)
    model.fokker_planc_reference = finite_volume_1d_cached()
    x_edges, x, _ = model.fokker_planc_reference.grid(cells, bounds=(0., 1.))
    initial = origin.initial_mass(x_edges)
    parameters = origin.parameters.parameter_template(edges)
    from src.tools.input_generator.input_builder import input_builder
    parameters[0] = input_builder.price_intervals((edges, problem.observed_prices))
    all_breakpoints = np.unique(np.r_[edges[1:-1], problem.breakpoints])
    forecast_breakpoints = all_breakpoints[(all_breakpoints > origin.time)
                                           & (all_breakpoints < times[-1])]
    output = model.fokker_planc_reference_solver(
        initial, parameters, origin.parameters.sigma_x,
        t_eval=np.r_[origin.time, times],
        breakpoints=forecast_breakpoints, max_step=max_step)
    mass = output[3][1:]
    diagnostics = output[-1]
    if (diagnostics['maximum_abs_mass_error'] > 1e-10
            or diagnostics['minimum_cell_mass'] < 0.0
            or diagnostics['used_clipping'] or diagnostics['used_renormalization']):
        raise RuntimeError('Forecast probability control failed; no repair applied.')
    demand = np.asarray(model.eval(times[:, None], x[None, :], *(
        parameters[:3] + parameters[4:]))[-1], dtype=float)
    demand = np.broadcast_to(demand, mass.shape)
    state_mean = mass @ x
    demand_mean = np.sum(mass * demand, axis=1)
    state_variance = np.sum(mass * (x[None, :] - state_mean[:, None])**2, axis=1)
    demand_variance = np.sum(mass * (demand - demand_mean[:, None])**2, axis=1)
    return dict(time_unit=origin.time_unit, origin_time=origin.time, times=times.copy(),
                x_edges=x_edges.copy(), cell_mass=mass.copy(),
                state_mean=state_mean, state_variance=state_variance,
                demand_mean=demand_mean, demand_variance=demand_variance,
                observation_variance=demand_variance + origin.sigma_y**2,
                parameters=origin.parameters, parameter_source=origin.parameter_source,
                includes_parameter_uncertainty=False, diagnostics=diagnostics)

"""Small F-based I integration, separate from population/forecast acceptance."""

from time import perf_counter

import numpy as np

from src.control.bm_1 import BM1ControlNumerics, BM1Parameters, multistart_powell
from src.control.bm_1_handoff import BM1ForecastOrigin, forecast_fv
from src.method.bm_1_estimation import run_fv_filter, run_fv_rho_bank
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation.bm_1 import estimation as synthetic


HORIZONS_HOURS = (1, 6, 12, 24, 48, 72, 168)
DEVELOPMENT_SEQUENCE_ID = 1326  # Previously used F3 development record, not holdout.
# Approximate relative phase times from the first local implementation pilot.
# Used only for progress, never optimizer stopping or scientific acceptance.
PHASE_WORK = dict(known_filter=1., rho_bank=15., open_forecasts=1.5,
                  forecast_refinement=5., f3_optimization=41., fixed_policy_comparison=1.)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _origin_record(origin):
    p = origin.parameters
    return dict(time=origin.time, time_unit=origin.time_unit,
                information_cutoff=origin.time, parameter_source=origin.parameter_source,
                selected_node=origin.selected_node,
                parameters=dict(C=p.C, rho=1. / p.C, k=p.k, Xi=p.Xi,
                                sigma_x=p.sigma_x, sigma_y=origin.sigma_y,
                                state_response='1-2*x', price_response='1-2*u'),
                x_edges=origin.x_edges.tolist(), posterior_mass=origin.posterior_mass.tolist(),
                includes_parameter_uncertainty=False)


def _forecast_record(result):
    return _plain({key: value for key, value in result.items() if key != 'parameters'})


def _control_record(result, evaluator):
    return dict(prices=result.prices.tolist(), total=result.total,
                tracking=result.objective.total, bias=result.objective.bias,
                variance=result.objective.variance,
                terminal=result.penalties.terminal, smoothing=result.penalties.smoothing,
                midpoints=evaluator.midpoints.tolist(), widths=evaluator.widths.tolist(),
                target=evaluator.target.tolist(), x_grid=evaluator.x.tolist(),
                expected_demand=result.objective.expected_demand.tolist(),
                second_moment=result.objective.second_moment.tolist(),
                terminal_mass=result.cell_mass[-1].tolist(),
                diagnostics=_plain(result.diagnostics))


def run_i_baseline(*, progress=None):
    """Fixed development design; no G2 dependencies and no future measurements.

    Known and estimated cases filter identical Y_1..Y_24. The rho MAP uses its
    own conditional posterior. Future synthetic B is 0.5 and the locked daily
    U pattern repeats; these inputs are declared before filtering. Two bounded
    F3 prices are optimized over the next day using the existing objective.
    This is one old-record integration demonstration, not recovery or calibration.
    """
    started = perf_counter()
    phase_seconds = {}

    def mark(name, phase_started):
        phase_seconds[name] = perf_counter() - phase_started
        if progress is not None:
            progress(name, len(phase_seconds), 6,
                     sum(PHASE_WORK[key] for key in phase_seconds), sum(PHASE_WORK.values()))

    case = synthetic.build_f4_b_case()
    origin_time = case.horizon[1]
    cells, step, order = 401, 1. / 768., 3
    horizons = np.asarray(HORIZONS_HOURS)
    future_times = origin_time + horizons / 24.
    # Declare the entire input plan before observing/estimating this record.
    daily_edges = np.asarray(case.price_edges)
    future_edges = np.concatenate([
        origin_time + day + daily_edges[:-1] for day in range(7)] + [[origin_time + 7.]])
    future_prices = np.tile(case.price_values, 7)

    def baseline(times):
        return np.full(np.asarray(times).shape, case.baseline, dtype=float)

    begun = perf_counter()
    sequence = synthetic.generate_synthetic_sequence(case, sequence_id=DEVELOPMENT_SEQUENCE_ID)
    if not sequence.truth.valid:
        raise RuntimeError('Development truth failed physical controls; no repair/drop.')
    observations = sequence.observations.prefix(origin_time)
    model, template = case.build_model()
    model.fokker_planc_reference = finite_volume_1d_cached()
    initial = synthetic.beta_initial_mass(case, model, cells)
    options = dict(breakpoints=case.breakpoints, max_step=step,
                   likelihood_quadrature_order=order)
    fixed = BM1Parameters(template[1], template[2], case.C, case.k, case.Xi, case.sigma_x)
    known_filter = run_fv_filter(model, template, case.sigma_x, case.sigma_y,
                                 initial, case.horizon[0], observations, keep_history=False, **options)
    known = BM1ForecastOrigin.from_filter_step(
        known_filter.final_step, fixed, case.sigma_y, time_unit=observations.time_unit)
    mark('known_filter', begun)

    begun = perf_counter()
    rho = np.asarray(synthetic.F4_E_PARAMETER_GRIDS['rho'])
    prior = synthetic.f4_f_parameter_prior(rho, 'discrete_uniform', 'rho')
    bank = run_fv_rho_bank(model, template, case.sigma_x, case.sigma_y,
                           initial, case.horizon[0], observations,
                           rho_values=rho, initial_log_weights=np.log(prior), **options)
    estimated = BM1ForecastOrigin.from_rho_bank_step(
        bank.steps[-1], fixed, case.sigma_y, time_unit=observations.time_unit)
    mark('rho_bank', begun)

    begun = perf_counter()
    origins = dict(known=known, rho_map=estimated)
    forecasts = {name: forecast_fv(
        origin, baseline=baseline, price_edges=future_edges, prices=future_prices,
        times=future_times, max_step=step) for name, origin in origins.items()}
    mark('open_forecasts', begun)

    begun = perf_counter()
    refined = {}
    for name, origin in origins.items():
        time_fine = forecast_fv(origin, baseline=baseline, price_edges=future_edges,
                               prices=future_prices, times=future_times, max_step=step / 2.)
        spatial_fine = forecast_fv(origin, baseline=baseline, price_edges=future_edges,
                                  prices=future_prices, times=future_times, max_step=step,
                                  x_steps=801)
        refined[name] = dict(time=_forecast_record(time_fine), spatial=_forecast_record(spatial_fine),
            differences={axis: dict(
                maximum_abs_demand_mean=float(np.max(np.abs(other['demand_mean']
                                                            - forecasts[name]['demand_mean']))),
                maximum_abs_state_mean=float(np.max(np.abs(other['state_mean']
                                                           - forecasts[name]['state_mean']))))
                for axis, other in (('time', time_fine), ('spatial', spatial_fine))})
    mark('forecast_refinement', begun)

    begun = perf_counter()
    price_edges = np.asarray([origin_time, origin_time + .5, origin_time + 1.])
    numerics = BM1ControlNumerics(x_steps=cells, objective_max_step=1/96,
                                solver_max_step=step)
    settings = dict(baseline=baseline, target=baseline, horizon=(origin_time, origin_time + 1.),
                    price_edges=price_edges, terminal_reference=.5, terminal_weight=.01,
                    price_smoothing_weight=.001)
    evaluators = {name: origin.control_problem(**settings).build_evaluator(numerics)
                  for name, origin in origins.items()}
    starts = np.asarray([[.5, .5], [.7, .3]])
    optimizer_options = dict(xtol=.002, ftol=1e-7, maxiter=30, maxfev=1500)
    policies = {'neutral': starts[0]}
    searches = {}
    for name, evaluator in evaluators.items():
        search = multistart_powell(evaluator, starts, options=optimizer_options)
        policies[name] = search.best.x.copy()
        searches[name] = dict(prices=search.best.x.tolist(), objective=float(search.best.fun),
                             objective_evaluations=search.objective_evaluations,
                             cache_hits=search.cache_hits, elapsed_seconds=search.elapsed_seconds,
                             starts=[dict(success=bool(r.success), nfev=int(r.nfev),
                                          message=str(r.message), objective=float(r.fun))
                                     for r in search.results])
    mark('f3_optimization', begun)

    begun = perf_counter()
    # Each fixed policy is assessed under both frozen beliefs; policies are not averaged.
    control = {belief: {name: _control_record(evaluator.evaluate(prices), evaluator)
                        for name, prices in policies.items()}
               for belief, evaluator in evaluators.items()}
    probability_checks = [known_filter.maximum_abs_mass_error]
    probability_checks += [item.diagnostics['maximum_abs_mass_error'] for s in bank.steps
                           for item in s.conditional_steps]
    probability_checks += [result['diagnostics']['maximum_abs_mass_error']
                           for result in forecasts.values()]
    probability_checks += [result[axis]['diagnostics']['maximum_abs_mass_error']
                           for result in refined.values() for axis in ('time', 'spatial')]
    probability_checks += [item['diagnostics']['maximum_abs_mass_error']
                           for results in control.values() for item in results.values()]
    if max(probability_checks) > 1e-10:
        raise RuntimeError('I integration probability control failed; no repair.')
    mark('fixed_policy_comparison', begun)
    return dict(stage='I-baseline-development-v1', registered_design=False,
        interpretation='One previously used F3 development record. Integration and numerical diagnostics only; no population, recovery, operational forecast or H-f/g acceptance.',
        integration_passes=True, forecast_scientific_acceptance=False,
        sequence_id=DEVELOPMENT_SEQUENCE_ID,
        observations=dict(times=observations.times.tolist(), values=observations.values.tolist(),
                          observed=observations.observed.tolist(),
                          time_unit=observations.time_unit, price_side=observations.price_side),
        past_inputs=dict(baseline=case.baseline, price_edges=case.price_edges.tolist(),
                         prices=case.price_values.tolist(), initial_beta=[case.initial_alpha, case.initial_beta],
                         seed_roots=[case.initial_seed_root, case.brownian_seed_root, case.measurement_seed_root],
                         truth_max_step=case.truth_max_step),
        estimation=dict(rho_nodes=rho.tolist(), prior=prior.tolist(), prior_name='discrete_uniform',
                        log_weights=bank.log_weights.tolist(), conditional_log_evidence=bank.conditional_log_evidence.tolist(),
                        marginal_log_evidence=bank.log_evidence.tolist(),
                        known_log_evidence=known_filter.log_evidence.tolist(),
                        final_conditional_posterior_mass=bank.steps[-1].conditional_posterior_mass.tolist(),
                        map_definition='Largest discrete posterior node weight; first-node tie break; own conditional posterior.',
                        includes_parameter_uncertainty=False),
        origins={name: _origin_record(origin) for name, origin in origins.items()},
        future_inputs=dict(kind='declared synthetic input plan', baseline=case.baseline,
                           price_edges=future_edges.tolist(), prices=future_prices.tolist(),
                           horizons_hours=list(HORIZONS_HOURS), future_observations_assimilated=0),
        forecast={name: _forecast_record(result) for name, result in forecasts.items()},
        refinement=refined,
        control_design=dict(horizon=list(settings['horizon']), price_edges=price_edges.tolist(),
                            target='constant baseline 0.5', terminal_reference=.5, terminal_weight=.01,
                            price_smoothing_weight=.001, starts=starts.tolist(), optimizer='bounded Powell',
                            optimizer_options=optimizer_options,
                            evaluation='Same Y1..Y24 cutoff; known conditional state law, not true X24.'),
        optimization=searches, fixed_policy_costs=control,
        numerics=dict(cells=cells, max_step=step, likelihood_quadrature_order=order,
                      objective_max_step=1/96, spatial_refinement_cells=801, time_refinement_step=step/2),
        maximum_abs_mass_error=max(probability_checks), phase_seconds=phase_seconds,
        eta_phase_work=PHASE_WORK.copy(),
        elapsed_seconds=perf_counter() - started)

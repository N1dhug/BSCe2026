"""Stdlib frozen G7 day contract; safe before numerical imports."""
from integrated_budget import validate_plan


def validate_design(plan):
    """The orchestration implements this exact proposal, not arbitrary labels."""
    validate_plan(plan)
    expected = dict(coarse_cells=257, fine_cells=513, mixture_degree=254,
        coarse_max_step_hours=1/96, fine_max_step_hours=1/192,
        cost_step_hours=1/96, refined_cost_step_hours=1/192,
        em_paths_per_node=256, em_coarse_step_hours=1/192,
        em_fine_step_hours=1/384, entropy=20261008,
        separate_role_node_streams=True, shared_initial_and_brownian_per_policy=True)
    gates = dict(conditional_mass_l1=.05, marginal_mass_l1=.05, projection_l1=.05,
        bank_weight=.03, log_evidence=.05, state_mean=.005, demand_mean=.005,
        state_variance=.002, demand_variance=.002, state_cdf=.03, demand_cdf=.03,
        state_quantile=.02, demand_quantile=.02, mass=1e-9, checkpoint=1e-10,
        coupled_em_cost_difference=.001, raw_em_boundary_violations=0)
    f = plan['feedback']; p = plan['problem']; o = plan['optimizer']
    if (plan['numerics'] != expected
            or any(plan['tolerances'].get(k) != v for k, v in gates.items())
            or plan['tolerances'].get('objective') != 'max(.001,.02*abs(fine_J))'
            or plan['tolerances'].get('em_fp') != 'max(.001,4*stratified_SE)'
            or p['target'] != 'native analytic constant A=.5'
            or p['baseline'] != 'unchanged native bm_1_distribution_panel.baseline'
            or plan['policies'] != ['neutral', 'deterministic plug-in reference', 'stochastic M254 primary']
            or o['method'] != 'existing bounded Powell single shooting'
            or o['old_max_seconds_total'] != 300
            or f['truth_node_index'] != 2 or f['truth_fine_path_index'] != 0
            or f['truth_policy'] != 'stochastic' or f['point_observation'] is not True
            or f['independent_observation_stream'] is not True
            or f['backend_own_posteriors'] is not True or f['update_receipts_required'] is not True
            or f['selection'] != 'complete updated coarse M254 bank J; tie candidate order'):
        raise ValueError('Driver does not implement a changed numerical/scientific contract.')

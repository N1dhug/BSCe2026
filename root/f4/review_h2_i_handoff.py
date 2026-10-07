"""Read-only review of the actual frozen H origin through the existing I/F3 API.

No forward, optimization, refit, measurement decoding or population claim.
"""
from src.exe.run_model.bm_1_h2_continuation import c, verify_origin


def review(directory):
    manifest, _, _ = verify_origin(directory)
    artifact = c._read_array_checkpoint((directory/'conditional_handoff.npz').read_bytes())
    frozen = c._read_array_checkpoint((directory/'forecast_frozen.npz').read_bytes())
    selection = c.load(directory/'selection.json')
    packet = c.load(directory/'packet.json')
    for value in (artifact, frozen):
        c.require(value['identity'] == manifest['identity'] and value['checksum'] ==
            c.fingerprint({k:v for k,v in value.items() if k!='checksum'}), 'Frozen artifact identity/checksum mismatch.')
    c.require(artifact['frozen_forecast_sha256'] == c.sha(directory/'forecast_frozen.npz')
        and frozen['selection_sha256'] == c.sha(directory/'selection.json')
        and artifact['origin_timestamp'] == selection['cutoff'] == frozen['origin']
        == packet['policy']['split']['holdout_start'], 'Origin or frozen dependency mismatch.')
    c.require(frozen['assimilated_observations'] == 0 and frozen['final_observations_decoded'] == 0
        and artifact['parameter_uncertainty'] is False and artifact['input_uncertainty'] is False,
        'Unexpected future information or uncertainty interpretation.')
    points = c.np.linspace(0., 1., 65); rows = []
    for key, model in artifact['origins'].items():
        expected_f, expected_g = c.curve(model['candidate'])
        for grid, state in model['grids'].items():
            origin = c.as_forecast_origin(artifact, model_key=key, grid=grid)
            c.require(origin.time == artifact['origin_time'] == 4. and origin.time_unit == 'day'
                and c.np.array_equal(origin.x_edges, state['cell_edges'])
                and c.np.array_equal(origin.posterior_mass, c.np.exp(state['log_state']))
                and c.np.array_equal(state['log_state'], frozen['origins'][key][grid]),
                'Origin lost its own posterior/time/grid.')
            for name in ('C','k','Xi','sigma_x'):
                c.require(getattr(origin.parameters, name) == model['physical'][name], 'Physical condition changed.')
            c.require(origin.sigma_y == model['physical']['sigma_y']
                and c.np.array_equal(origin.parameters.state_response(points), expected_f(points))
                and c.np.array_equal(origin.parameters.price_response(points), expected_g(points)),
                'Observation noise or shared response changed.')
            transfers = {}
            for cells in (401, 801):
                edges = c.np.linspace(0.,1.,cells+1); mass = origin.initial_mass(edges)
                c.require(abs(float(mass.sum())-1.) <= 1e-10 and float(mass.min()) >= 0.,
                    'I/F3 conservative transfer failed.')
                # This is an adapter view of the represented cell law. The
                # original full log state remains immutable in the artifact.
                transfers[str(cells)] = dict(edges=edges, mass=mass)
            rows.append(dict(model_key=key, grid=grid, origin_time=origin.time,
                origin_timestamp=artifact['origin_timestamp'], original_log_state_fingerprint=c.fingerprint(state['log_state']),
                parameter_source='fixed nominal physical condition with fitted shared response',
                curves=model['curves'], transfers=transfers))
    result = dict(schema='f4-actual-H-to-I-origin-review-v1', passed=True,
        manifest_identity=manifest['identity'], handoff_sha256=c.sha(directory/'conditional_handoff.npz'),
        as_forecast_origin_source_sha256=c.sha(c.ROOT/'src/exe/run_model/bm_1_h2_compact.py'),
        I_F3_interface_source_sha256=c.sha(c.ROOT/'src/control/bm_1_handoff.py'), rows=rows,
        model_propagations=0, future_observations_decoded=0,
        interpretation='Actual origin/interface evidence only; independent I forecast and F3 policy evidence remain separate.',
        scientific_acceptance=False, population_calibration=False)
    result['checksum'] = c.fingerprint(result)
    return result

"""G-only comparison algebra; no propagation, fitting or price selection."""
import numpy as np
from src.exe.run_model.bm_1.g7 import compare_forward
from src.params.bm_1 import conservative_mass_transfer
from src.method.bm_1_g7_jobs import digest


def compare_laws(a, b, *, tolerances, scores_a=None, scores_b=None):
    """Compare owned laws after conservative transfer; never overwrite either."""
    if (a.time != b.time or not np.array_equal(a.rho_values, b.rho_values)
            or digest(a.to_record()['parameters']) != digest(b.to_record()['parameters'])
            or a.sigma_y != b.sigma_y
            or a.price_side != b.price_side):
        raise ValueError('Different model, information cutoff or observation contract.')
    transferred = np.array([conservative_mass_transfer(a.x_edges, m, b.x_edges)
                            for m in a.conditional_mass])
    errors = dict(
        conditional_mass_l1=float(np.max(np.sum(abs(transferred-b.conditional_mass), axis=1))),
        marginal_mass_l1=float(np.sum(abs(a.weights@transferred-b.weights@b.conditional_mass))),
        bank_weight=float(np.max(abs(a.weights-b.weights))))
    if scores_a is not None or scores_b is not None:
        if scores_a is None or scores_b is None:
            raise ValueError('Both saved update evidence arrays are required.')
        errors['log_evidence'] = float(np.max(abs(np.asarray(scores_a)-np.asarray(scores_b))))
    if not np.isfinite(list(errors.values())).all():
        raise ValueError('Nonfinite law comparison.')
    return dict(errors=errors, passed=all(v <= tolerances[k] for k, v in errors.items()))


def forward_checks(values, controls, tolerances):
    """Own refinements, finest cross-backend and cost-partition refinement."""
    pairs = [('m254_refinement', 'm254_coarse', 'm254_fine'),
             ('fv_refinement', 'fv_coarse', 'fv_fine'),
             ('m254_vs_fv', 'm254_fine', 'fv_fine')]
    if 'quadrature' in values:
        pairs.append(('cost_partition_refinement', 'm254_fine', 'quadrature'))
    result = {}
    for name, left, right in pairs:
        check = compare_forward(values[left], values[right], controls[right].evaluator)
        weights = compare_laws(values[left]['predicted_origin'],
                               values[right]['predicted_origin'], tolerances=tolerances)
        check['law'] = weights
        check['passed'] = bool(check['passed'] and weights['passed'])
        result[name] = check
    return result


def stratified(values, weights):
    """Five conditional means with independent strata, not pooled iid paths."""
    a = np.asarray(values, float); w = np.asarray(weights, float)
    if (a.ndim != 2 or a.shape[0] != len(w) or a.shape[1] < 2
            or not np.isfinite(a).all() or not np.isfinite(w).all()
            or np.min(w) < 0 or abs(w.sum()-1) > 1e-9):
        raise ValueError('Finite stratified path costs and normalized weights required.')
    return dict(mean=float(w@a.mean(axis=1)),
                standard_error=float(np.sqrt(np.sum(w*w*a.var(axis=1, ddof=1)/a.shape[1]))))


def review_em(records, weights, fp_total):
    """Native cost tolerances; raw boundary failures cannot be corrected away."""
    if len(records) != len(weights):
        raise ValueError('Every conditional EM stratum is required.')
    if any(r['raw_boundary_violations'] != 0 for r in records):
        raise ValueError('Raw EM boundary violation; no corrected-path acceptance.')
    coarse = np.array([r['coarse_costs'] for r in records])
    fine = np.array([r['fine_costs'] for r in records])
    a, b = stratified(coarse, weights), stratified(fine, weights)
    error = abs(b['mean']-fp_total); coupled = stratified(fine-coarse, weights)
    gate = max(.001, 4*b['standard_error'])
    return dict(coarse=a, fine=b, fp_error=error, fp_gate=gate,
                coupled_difference=coupled,
                passed=bool(error <= gate and abs(coupled['mean']) <= .001))


def paired_improvement(reference, candidate, weights):
    """Positive means lower candidate cost; same draws must be authenticated."""
    if len(reference) != len(candidate):
        raise ValueError('Different EM strata.')
    for a, b in zip(reference, candidate):
        if (a['input_sha256'] != b['input_sha256'] or a['index'] != b['index']
                or a['raw_boundary_violations'] or b['raw_boundary_violations']):
            raise ValueError('Paired estimate requires the same valid initial/Brownian inputs.')
    return stratified(np.array([r['fine_costs'] for r in reference])
                      -np.array([r['fine_costs'] for r in candidate]), weights)

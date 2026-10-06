"""F4-H2 interval forecasts from the native positive split operator.

No observations, fitting, data loading or automatic runtime admission. Full
log state belongs to the caller's open branch. Legacy endpoint forecasts and
the native likelihood are unchanged. See root/H2_COMPACT_ESTIMATION_REVIEW.md.
"""

from dataclasses import dataclass
import math

import numpy as np
from numba import njit
from scipy.special import logsumexp, ndtr

from src.method.bm_1_interval_native import _mix, native_prediction
from src.validation.bm_1.interval_characteristic import CharacteristicNumericalError


@njit(cache=True, fastmath=False, inline='always')
def _moment_mix(a, b, ma, va, mb, vb):
    total, w0, w1 = _mix(a, b)
    return total, w0*ma+w1*mb, w0*va+w1*vb+w0*w1*(ma-mb)**2


@njit(cache=True, fastmath=False, nogil=True)
def _advance_moments(logs, mean, variance, neutral, step_ids, ed, el, eu,
                     lp, lbeta, lu, pivots, beta, upper, reward_half):
    logs, mean, variance, neutral = logs.copy(), mean.copy(), variance.copy(), neutral.copy()
    n = len(logs)
    explicit_logs, forward_logs = np.empty(n), np.empty(n)
    explicit_mean, forward_mean = np.empty(n), np.empty(n)
    explicit_var, forward_var = np.empty(n), np.empty(n)
    rhs = np.empty(n)
    e_diag, e_lower, e_upper = np.exp(ed), np.exp(el), np.exp(eu)
    maximum_error = 0.
    for profile in step_ids:
        for j in range(n):
            mean[j] += reward_half[profile, j]
        for j in range(n):
            a = logs[j]+ed[profile, j]
            b = logs[j-1]+el[profile, j-1] if j else -np.inf
            c = logs[j+1]+eu[profile, j] if j+1 < n else -np.inf
            ab, mab, vab = _moment_mix(a, b, mean[j], variance[j],
                mean[j-1] if j else 0., variance[j-1] if j else 0.)
            total, m, v = _moment_mix(ab, c, mab, vab,
                mean[j+1] if j+1 < n else 0., variance[j+1] if j+1 < n else 0.)
            explicit_logs[j], explicit_mean[j], explicit_var[j] = total, m, v
            value = e_diag[profile, j]*neutral[j]
            if j:
                value += e_lower[profile, j-1]*neutral[j-1]
            if j+1 < n:
                value += e_upper[profile, j]*neutral[j+1]
            rhs[j] = value
        forward_logs[0], forward_mean[0], forward_var[0] = explicit_logs[0], explicit_mean[0], explicit_var[0]
        for j in range(1, n):
            total, m, v = _moment_mix(explicit_logs[j], lbeta[profile, j-1]+forward_logs[j-1],
                explicit_mean[j], explicit_var[j], forward_mean[j-1], forward_var[j-1])
            forward_logs[j], forward_mean[j], forward_var[j] = total, m, v
            rhs[j] += beta[profile, j-1]*rhs[j-1]
        logs[-1] = forward_logs[-1]-lp[profile, -1]
        mean[-1], variance[-1] = forward_mean[-1], forward_var[-1]
        neutral[-1] = rhs[-1]/pivots[profile, -1]
        for j in range(n-2, -1, -1):
            total, m, v = _moment_mix(forward_logs[j], lu[profile, j]+logs[j+1],
                forward_mean[j], forward_var[j], mean[j+1], variance[j+1])
            logs[j], mean[j], variance[j] = total-lp[profile, j], m, v
            neutral[j] = (rhs[j]+upper[profile, j]*neutral[j+1])/pivots[profile, j]
        mass = 0.
        for j in range(n):
            mean[j] += reward_half[profile, j]
            if (not math.isfinite(neutral[j]) or neutral[j] < 0.
                    or math.isnan(logs[j]) or logs[j] == np.inf
                    or not math.isfinite(mean[j]) or not math.isfinite(variance[j])
                    or variance[j] < 0.):
                raise ValueError('Native interval-moment sign/finite guard failed.')
            mass += neutral[j]
        error = abs(mass-1.)
        maximum_error = max(maximum_error, error)
        if error > 1e-10:
            raise ValueError('Native interval-moment neutral mass guard failed.')
    return logs, mean, variance, neutral, maximum_error


def _log_state(logs):
    logs = np.asarray(logs, dtype=np.float64)
    if (logs.ndim != 1 or len(logs) < 2 or np.isnan(logs).any()
            or np.isposinf(logs).any() or abs(float(logsumexp(logs))) > 1e-10):
        raise CharacteristicNumericalError('Forecast prior must be a normalized full log-state.')
    return logs


def native_interval_moments(operator, logs, sigma_y, *, conditional_mean=None,
                            conditional_variance=None):
    """Moments of one interval, with optional accumulator continuation.

    Operators must retain the original Delta in reward_half across within-hour
    input changes. Reset accumulator fields only at a new observation interval.
    The returned log state is never reconstructed from ordinary probabilities.
    """
    logs = _log_state(logs)
    if not math.isfinite(sigma_y) or sigma_y <= 0.:
        raise ValueError('Declare a strictly positive finite observation noise.')
    if (conditional_mean is None) != (conditional_variance is None):
        raise ValueError('Continue both interval accumulator fields together.')
    m = np.zeros_like(logs) if conditional_mean is None else np.asarray(conditional_mean, dtype=np.float64)
    v = np.zeros_like(logs) if conditional_variance is None else np.asarray(conditional_variance, dtype=np.float64)
    if m.shape != logs.shape or v.shape != logs.shape or not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v < 0.):
        raise ValueError('Invalid conditional interval accumulator fields.')
    if operator['reward_half'].shape[1] != len(logs):
        raise ValueError('Operator and full log-state dimensions disagree.')
    args = [operator[k] for k in ('step_ids', 'ed', 'el', 'eu', 'lp', 'lbeta', 'lu',
                                  'pivots', 'beta', 'upper', 'reward_half')]
    try:
        state, m, v, neutral, mass_error = _advance_moments(logs, m, v, np.exp(logs), *args)
    except ValueError as exc:
        raise CharacteristicNumericalError(str(exc)) from exc
    _log_state(state)
    p = np.exp(state)
    if np.max(np.abs(p-neutral)) > 1e-10 or np.any(m < 0.) or np.any(m > 1.+1e-10):
        raise CharacteristicNumericalError('Forecast interval-state/moment guard failed.')
    mean = float(p@m)
    variance = float(p@(v+(m-mean)**2))
    if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0.:
        raise CharacteristicNumericalError('Invalid marginal interval moments.')
    return dict(log_state=state, conditional_mean=m, conditional_variance=v,
                interval_mean=mean, interval_variance=variance,
                observation_mean=mean, observation_variance=variance+sigma_y**2,
                maximum_abs_mass_error=mass_error)


@dataclass(frozen=True)
class IntervalCDFPlan:
    """Freeze noise/span/frequency work before exposing final observations."""
    sigma_y: float
    span: float
    frequencies: int
    alias_bound: float
    truncation_bound: float

    @property
    def left(self):
        return -(self.span-1.)/2.

    @property
    def right(self):
        return 1.+(self.span-1.)/2.


def interval_cdf_plan(sigma_y):
    if not math.isfinite(sigma_y) or sigma_y <= 0.:
        raise ValueError('Declare a strictly positive finite observation noise.')
    span = max(2., 1.+16.*sigma_y)
    beta = 2.*math.pi**2*(sigma_y/span)**2
    if not math.isfinite(span) or beta <= 0.:
        raise ValueError('CDF noise/span is outside the representable range.')
    # Solve in dimensionless q; no import/JIT or frequency-bank computation.
    low, high = 0., 64.
    for _ in range(80):
        q = (low+high)/2.
        if math.exp(-q)/(math.pi*q) > 1e-10:
            low = q
        else:
            high = q
    count = max(64, math.ceil(math.sqrt(high/beta)))
    def tail(n):
        q = beta*n*n
        return math.exp(-q)/(math.pi*q)
    while count > 64 and tail(count-1) <= 1e-10:
        count -= 1
    while tail(count) > 1e-10:
        count += 1
    alias = float(2.*ndtr(-(span-1.)/(2.*sigma_y)))
    if alias > 1e-10:
        raise CharacteristicNumericalError('CDF alias bound exceeds its declared tolerance.')
    return IntervalCDFPlan(float(sigma_y), span, count, alias, tail(count))


@dataclass(frozen=True)
class IntervalDistribution:
    plan: IntervalCDFPlan
    characteristic: np.ndarray

    def cdf(self, values):
        """Periodic-law CDF inside its support, extended by 0/mass outside.

        The extension defines the compact periodic approximation; it is not
        clipping an invalid computed CDF. No normalization or shape repair.
        Chunked evaluation limits memory for narrow-noise frequency banks.
        """
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('CDF evaluation points must be finite.')
        y = values.reshape(-1)
        out = np.empty_like(y)
        a, b, span = self.plan.left, self.plan.right, self.plan.span
        mass = self.characteristic[0].real
        out[y <= a], out[y >= b] = 0., mass
        inside = np.flatnonzero((y > a) & (y < b))
        omega = 2.*np.pi*np.arange(1, self.plan.frequencies+1)/span
        factors = self.characteristic[1:]/(1j*omega)
        left_phase = np.exp(-1j*omega*a)
        chunk = max(1, 65536//self.plan.frequencies)
        for start in range(0, len(inside), chunk):
            ids = inside[start:start+chunk]
            phase = np.exp(-1j*y[ids, None]*omega)
            out[ids] = mass*(y[ids]-a)/span + (2./span)*np.real((left_phase-phase)@factors)
        if not np.isfinite(out).all():
            raise CharacteristicNumericalError('Nonfinite interval CDF.')
        return out.reshape(values.shape)

    def audit(self, extra_points=()):
        points = np.unique(np.r_[np.linspace(self.plan.left, self.plan.right,
            8*self.plan.frequencies+1), np.asarray(extra_points, dtype=float)])
        cdf = self.cdf(points)
        # Analytic approximation allowances, plus the existing mass guard.
        # This empirical check is not a universal floating-point certificate.
        allowance = self.plan.alias_bound+self.plan.truncation_bound+1e-10
        lower = float(max(0., -cdf.min()))
        upper = float(max(0., cdf.max()-self.characteristic[0].real))
        decrease = float(max(0., -np.diff(cdf).min()))
        if max(lower, upper, decrease) > 2.*allowance:
            raise CharacteristicNumericalError('Interval CDF range/monotonicity guard failed.')
        return dict(points=len(points), lower_violation=lower, upper_violation=upper,
                    maximum_decrease=decrease, check_allowance=2.*allowance)

    def quantile(self, probability):
        if not math.isfinite(probability) or not 0. < probability < 1.:
            raise ValueError('Quantile probability must lie strictly between zero and one.')
        left, right = self.plan.left, self.plan.right
        fl, fr = float(self.cdf(left)), float(self.cdf(right))
        if not fl <= probability <= fr:
            raise CharacteristicNumericalError('Unresolved interval quantile bracket.')
        for _ in range(100):
            if right-left <= 0.0002:
                return dict(probability=probability, value=(left+right)/2., left=left,
                            right=right, resolved=True)
            middle = (left+right)/2.
            fm = float(self.cdf(middle))
            allowance = 2.*(self.plan.alias_bound+self.plan.truncation_bound+1e-10)
            if fm < fl-allowance or fm > fr+allowance:
                raise CharacteristicNumericalError('Interval quantile monotonicity guard failed.')
            if fm < probability:
                left, fl = middle, fm
            else:
                right, fr = middle, fm
        raise CharacteristicNumericalError('Interval quantile exceeded 100 iterations.')


def native_interval_distribution(operator, logs, plan, *, mapper=None):
    logs = _log_state(logs)
    if plan != interval_cdf_plan(plan.sigma_y):
        raise ValueError('CDF plan differs from the frozen analytic policy.')
    predicted = native_prediction(operator, logs, frequencies=plan.frequencies,
                                  span=plan.span, mapper=mapper)
    state = predicted['log_amplitude']
    _log_state(state)
    cf = np.exp(state)@predicted['cf']
    omega = predicted['frequencies']
    cf *= np.exp(-.5*(plan.sigma_y*omega)**2)
    if (not np.isfinite(cf).all() or abs(cf[0]-1.) > 1e-10
            or np.any(np.abs(cf) > 1.+1e-10)):
        raise CharacteristicNumericalError('Global interval characteristic/mass guard failed.')
    cf.setflags(write=False)
    distribution = IntervalDistribution(plan, cf)
    quantiles = [distribution.quantile(q) for q in (.05, .5, .95)]
    audit = distribution.audit([p[k] for p in quantiles for k in ('left', 'right')])
    return dict(log_state=state, distribution=distribution, quantiles=quantiles,
                cdf_audit=audit, maximum_abs_mass_error=predicted['maximum_abs_mass_error'])


def compare_interval_distributions(base, fine, *, observed_values=()):
    """Fine-grid checks; observed targets are allowed only after forecast freeze."""
    if base.plan != fine.plan:
        raise ValueError('Base/fine distributions must use the same frozen CDF plan.')
    qb = [base.quantile(q) for q in (.05, .5, .95)]
    qf = [fine.quantile(q) for q in (.05, .5, .95)]
    extra = [p[k] for p in qb+qf for k in ('left', 'right')]
    extra.extend(observed_values)
    base.audit(extra); fine.audit(extra)
    points = np.unique(np.r_[np.linspace(base.plan.left, base.plan.right,
        8*base.plan.frequencies+1), extra])
    cdf_error = float(np.max(np.abs(base.cdf(points)-fine.cdf(points))))
    quantile_error = max(abs(b['value']-f['value']) for b, f in zip(qb, qf))
    if cdf_error > .002 or quantile_error > .002:
        raise CharacteristicNumericalError('Interval forecast base/fine CDF/quantile gate failed.')
    return dict(maximum_cdf_difference=cdf_error, maximum_quantile_difference=quantile_error,
                points=len(points), passed=True)


def compare_interval_moments(base, fine):
    """Normalized interval moments only; retain caller's separate state gates."""
    errors = {name: abs(float(base[name])-float(fine[name]))
              for name in ('interval_mean', 'interval_variance')}
    if any(not math.isfinite(v) or v > .002 for v in errors.values()):
        raise CharacteristicNumericalError('Interval forecast base/fine moment gate failed.')
    return dict(errors, passed=True)


def forecast_open_intervals(operators, log_prior, plan, *, interval_count=168,
                            horizons=(1, 6, 12, 24, 48, 72, 168), mapper=None,
                            checkpoint=None, should_stop=None, completed_intervals=0):
    """One open prefix; no observation argument and no posterior updates.

    Caller freezes inputs, owns the deadline/ledger and supplies operators
    lazily in chronological order. A checkpoint callback receives each completed
    interval and its full log state. Returned accumulators reset each hour.
    On resume the caller must verify immutable source/input/plan identities,
    supply the saved boundary log state and only the remaining operators.
    Previously completed rows/distributions stay in their original checkpoints.
    """
    if (not isinstance(interval_count, int) or not isinstance(completed_intervals, int)
            or not 0 <= completed_intervals <= interval_count or interval_count < 1
            or sorted(set(horizons)) != list(horizons) or any(
            h < 1 or h > interval_count for h in horizons)):
        raise ValueError('Declare unique ordered horizons within the complete forecast.')
    logs = _log_state(log_prior).copy()
    rows, distributions = [], {}
    iterator = iter(operators)
    for hour in range(completed_intervals+1, interval_count+1):
        if should_stop is not None and should_stop():
            raise TimeoutError('Open forecast paused before the next interval.')
        try:
            operator = next(iterator)
        except StopIteration as exc:
            raise ValueError('Incomplete declared open forecast operators.') from exc
        moments = native_interval_moments(operator, logs, plan.sigma_y)
        row = dict(hour=hour, interval_mean=moments['interval_mean'],
            interval_variance=moments['interval_variance'],
            observation_variance=moments['observation_variance'],
            maximum_abs_mass_error=moments['maximum_abs_mass_error'])
        next_logs = moments['log_state']
        if hour in horizons:
            full = native_interval_distribution(operator, logs, plan, mapper=mapper)
            active = np.isfinite(next_logs)
            if (not np.array_equal(active, np.isfinite(full['log_state']))
                    or np.max(np.abs(next_logs[active]-full['log_state'][active])) > 1e-7):
                raise CharacteristicNumericalError('Moment/CF open state agreement failed.')
            # CF returns the same untilted state; never reconstruct from exp(logs).
            next_logs = full['log_state']
            distributions[hour] = full
            row['quantiles'] = full['quantiles']
        logs = next_logs
        rows.append(row)
        if checkpoint is not None:
            checkpoint(row, logs.copy(), distributions.get(hour))
    return dict(log_state=logs, rows=rows, distributions=distributions,
                assimilated_observations=0, completed_intervals=interval_count)


def interval_forecast_checkpoint(row, log_state, full_distribution, identity):
    """Payload for the existing real-array .npz checkpoint writer.

    The caller's immutable manifest digest covers model/input/grid/plan/source.
    Keep log tails/true zeros as a real array; split the CF into real/imaginary
    arrays instead of passing complex arrays or dataclasses to report I/O.
    """
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('Declare the immutable SHA256 forecast manifest identity.')
    state = _log_state(log_state).copy()
    distribution = None
    if full_distribution is not None:
        law = full_distribution['distribution']
        p = law.plan
        distribution = dict(plan=dict(sigma_y=p.sigma_y, span=p.span,
            frequencies=p.frequencies, alias_bound=p.alias_bound,
            truncation_bound=p.truncation_bound),
            characteristic_real=law.characteristic.real.copy(),
            characteristic_imag=law.characteristic.imag.copy(),
            quantiles=full_distribution['quantiles'], cdf_audit=full_distribution['cdf_audit'])
    return dict(schema='f4-h2-open-interval-checkpoint-v1', identity=identity,
                hour=row['hour'], row=row, log_state=state, distribution=distribution)

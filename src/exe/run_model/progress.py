"""Cheap, descriptive runtime estimates; never control scientific execution."""
from math import ceil, isfinite
from time import perf_counter


def estimate_seconds(elapsed, completed, total, *, initial_completed=0):
    """Return remaining and total seconds for this invocation, or two Nones.

    Work can be a task count or a consistent positive cost proxy. Previously
    completed work is excluded from the measured rate on resume. Setup and
    storage already inside elapsed time are included; later validation, cost
    variation and scheduling tails can still change this rough estimate.
    """
    elapsed, completed, total, initial_completed = map(
        float, (elapsed, completed, total, initial_completed))
    if not all(isfinite(v) for v in (elapsed, completed, total, initial_completed)):
        raise ValueError('Runtime inputs must be finite.')
    if elapsed < 0 or not 0 <= initial_completed <= completed <= total:
        raise ValueError('Invalid runtime progress or elapsed time.')
    if completed == total:
        return 0., elapsed
    measured = completed - initial_completed
    if measured == 0 or elapsed == 0:
        return None, None
    remaining = elapsed * ((total - completed) / measured)
    projected = elapsed + remaining
    if not isfinite(projected):
        return None, None
    return remaining, projected


def _duration(seconds):
    """Round upward so unfinished work never displays a zero-second ETA."""
    hours, rest = divmod(ceil(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f'{hours}h {minutes:02d}m {seconds:02d}s'
    if minutes:
        return f'{minutes}m {seconds:02d}s'
    return f'{seconds}s'


def runtime_text(started, completed, total, *, initial_completed=0, now=None):
    """Format elapsed, remaining and total time at an existing print boundary."""
    elapsed = (perf_counter() if now is None else now) - started
    remaining, projected = estimate_seconds(
        elapsed, completed, total, initial_completed=initial_completed)
    prefix = f'elapsed={_duration(elapsed)}'
    if remaining is None:
        return prefix + '; remaining=estimating; total=estimating'
    return prefix + f'; remaining~{_duration(remaining)}; total~{_duration(projected)}'


def fv_work(settings, days, observations):
    """Grid-weighted ETA proxy per replay, not an operation count or a gate.

    Linear solves scale with cells times time steps; observation quadrature
    scales with cells times observations times q. Coefficients, caching and
    other overhead differ, so extrapolation remains explicitly approximate.
    """
    return settings['x_steps'] * (
        days / settings['max_step'] + observations * settings['likelihood_quadrature_order'])

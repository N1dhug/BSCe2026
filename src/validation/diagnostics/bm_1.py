"""Validation for reusable deterministic bm_1 diagnostics."""
import numpy as np

class bm_1_diagnostics_validation:
    """Validate normalized bm_1 diagnostic inputs."""

    @staticmethod
    def _finite_scalar(value, name):
        values = np.asarray(value, dtype=float)
        if values.size != 1:
            raise ValueError(f'{name} must be a scalar.')
        value = float(values.reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite.')
        return value

    @staticmethod
    def response_inputs(state, demand, baseline, flexibility, steepness):
        """Validate inputs used to invert the linear benchmark responses."""
        state = np.asarray(state, dtype=float)
        demand = np.asarray(demand, dtype=float)
        try:
            state, demand = np.broadcast_arrays(state, demand)
        except ValueError as exc:
            raise ValueError('state and demand must be broadcast-compatible.') from exc
        if state.size == 0:
            raise ValueError('state and demand must not be empty.')
        if not np.all(np.isfinite(state)):
            raise ValueError('state must contain finite values.')
        if not np.all(np.isfinite(demand)):
            raise ValueError('demand must contain finite values.')
        baseline = bm_1_diagnostics_validation._finite_scalar(baseline, 'baseline')
        flexibility = bm_1_diagnostics_validation._finite_scalar(flexibility, 'flexibility')
        steepness = bm_1_diagnostics_validation._finite_scalar(steepness, 'steepness')
        if not 0.0 < baseline < 1.0:
            raise ValueError('baseline must lie strictly inside (0, 1).')
        if not 0.0 < flexibility <= 1.0:
            raise ValueError('flexibility must lie inside (0, 1].')
        if steepness <= 0.0:
            raise ValueError('steepness must be positive.')
        return (state, demand, baseline, flexibility, steepness)

    @staticmethod
    def envelope_inputs(state, baseline, flexibility, steepness):
        """Validate inputs for the linear-response demand envelope."""
        state = np.asarray(state, dtype=float)
        if state.size == 0:
            raise ValueError('state must not be empty.')
        if not np.all(np.isfinite(state)):
            raise ValueError('state must contain finite values.')
        baseline = bm_1_diagnostics_validation._finite_scalar(baseline, 'baseline')
        flexibility = bm_1_diagnostics_validation._finite_scalar(flexibility, 'flexibility')
        steepness = bm_1_diagnostics_validation._finite_scalar(steepness, 'steepness')
        if not 0.0 <= baseline <= 1.0:
            raise ValueError('baseline must lie inside [0, 1].')
        if not 0.0 <= flexibility <= 1.0:
            raise ValueError('flexibility must lie inside [0, 1].')
        if steepness <= 0.0:
            raise ValueError('steepness must be positive.')
        return (state, baseline, flexibility, steepness)

    @staticmethod
    def capacity_inputs(target_demand, baseline, capacity, initial_state, interval):
        """Validate a constant-demand state-capacity certificate."""
        target_demand = bm_1_diagnostics_validation._finite_scalar(target_demand, 'target_demand')
        baseline = bm_1_diagnostics_validation._finite_scalar(baseline, 'baseline')
        capacity = bm_1_diagnostics_validation._finite_scalar(capacity, 'capacity')
        initial_state = bm_1_diagnostics_validation._finite_scalar(initial_state, 'initial_state')
        if not 0.0 <= target_demand <= 1.0:
            raise ValueError('target_demand must lie inside [0, 1].')
        if not 0.0 <= baseline <= 1.0:
            raise ValueError('baseline must lie inside [0, 1].')
        if capacity <= 0.0:
            raise ValueError('capacity must be positive.')
        if not 0.0 <= initial_state <= 1.0:
            raise ValueError('initial_state must lie inside [0, 1].')
        interval = np.asarray(interval, dtype=float)
        if interval.shape != (2,) or not np.all(np.isfinite(interval)):
            raise ValueError('interval must contain two finite endpoints.')
        if interval[1] <= interval[0]:
            raise ValueError('interval end must be greater than its start.')
        return (target_demand, baseline, capacity, initial_state, (float(interval[0]), float(interval[1])))

    @staticmethod
    def implied_state_inputs(time, target_demand, baseline, capacity, initial_state, initial_time):
        """Validate evaluation inputs for a constant-demand implied state."""
        time = np.asarray(time, dtype=float)
        if time.size == 0:
            raise ValueError('time must not be empty.')
        if not np.all(np.isfinite(time)):
            raise ValueError('time must contain finite values.')
        initial_time = bm_1_diagnostics_validation._finite_scalar(initial_time, 'initial_time')
        target_demand, baseline, capacity, initial_state, _ = bm_1_diagnostics_validation.capacity_inputs(target_demand,
            baseline, capacity, initial_state, (initial_time, initial_time + 1.0))
        return (time, target_demand, baseline, capacity, initial_state, initial_time)

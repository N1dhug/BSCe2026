"""Frozen G response geometry, with an explicit capacity time unit."""
from dataclasses import dataclass, field
import numpy as np

from src.method.bm_1_deterministic_fit import _parameters
from src.params.bm_1_response import BM1PiecewiseResponse


@dataclass(frozen=True)
class BM1ResponseModel:
    C_hours: float
    k: float
    Xi: float
    state_response: BM1PiecewiseResponse = field(default_factory=lambda: BM1PiecewiseResponse.linear(pieces=2))
    price_response: BM1PiecewiseResponse = field(default_factory=lambda: BM1PiecewiseResponse.linear(pieces=2))

    def __post_init__(self):
        p = _parameters([self.C_hours, self.k, self.Xi, .5])
        for name, value in zip(('C_hours', 'k', 'Xi'), p[:3]):
            object.__setattr__(self, name, float(value))
        if not all(isinstance(curve, BM1PiecewiseResponse) for curve in (self.state_response, self.price_response)):
            raise ValueError('G response models require validated BM1PiecewiseResponse geometry.')

    def to_record(self):
        return {'schema': 'bm1-deterministic-response-v1', 'time_unit': 'hours',
                'parameter_order': ['C_hours', 'k', 'Xi'],
                'parameters': [self.C_hours, self.k, self.Xi],
                'f': self.state_response.to_record(), 'g': self.price_response.to_record()}

    @classmethod
    def from_record(cls, record):
        if (record.get('schema') != 'bm1-deterministic-response-v1' or record.get('time_unit') != 'hours'
                or record.get('parameter_order') != ['C_hours', 'k', 'Xi']):
            raise ValueError('Unsupported G response model or capacity time unit.')
        p = np.asarray(record['parameters'], dtype=float)
        if p.shape != (3,):
            raise ValueError('G response model needs C_hours, k, Xi.')
        return cls(*p, BM1PiecewiseResponse.from_record(record['f']), BM1PiecewiseResponse.from_record(record['g']))

    def as_bm1_parameters(self, *, time_unit='hours'):
        """Adapter to the existing F3 model interface; G4 is noise-free."""
        from src.control.bm_1 import BM1Parameters
        if time_unit not in ('hours', 'days'):
            raise ValueError('Declare hours or days for the receiving BM model.')
        return BM1Parameters(self.state_response, self.price_response,
                             self.C_hours if time_unit == 'hours' else self.C_hours/24,
                             self.k, self.Xi, 0.)

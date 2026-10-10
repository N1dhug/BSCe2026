"""Shared data and input-function utilities."""

from .baseline import output_callable, statistical_output
from .input_generator.input_builder import input_builder

__all__ = ["input_builder", "output_callable", "statistical_output"]

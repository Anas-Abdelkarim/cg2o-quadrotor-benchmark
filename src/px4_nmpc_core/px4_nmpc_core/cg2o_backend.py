"""Compatibility import for the cg2o backend.

The complete cg2o formulation now lives under ``px4_nmpc_controller``.
This module intentionally contains no optimization implementation.
"""
from px4_nmpc_controller.px4_nmpc_cg2o.solver import Cg2oOcpSolver, _load_extension

__all__ = ['Cg2oOcpSolver', '_load_extension']

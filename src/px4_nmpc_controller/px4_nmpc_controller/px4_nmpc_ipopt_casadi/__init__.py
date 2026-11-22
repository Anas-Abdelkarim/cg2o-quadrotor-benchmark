"""Explicit name for the existing CasADi/IPOPT backend.

The implementation remains in px4_nmpc_ipopt for backwards compatibility.
"""
from px4_nmpc_controller.px4_nmpc_ipopt.solver import NS, SolveResult, NmpcOcpSolver

__all__ = ['NS', 'SolveResult', 'NmpcOcpSolver']

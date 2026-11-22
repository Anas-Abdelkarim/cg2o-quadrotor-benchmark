"""Native C++ IPOPT backend (trajectory interface, no CasADi in solve path)."""
from .solver import IpoptCppOcpSolver

__all__ = ['IpoptCppOcpSolver']

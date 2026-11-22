"""Solver router.

Backends:
- ``ipopt_casadi``: CasADi/IPOPT reference implementation (all three interfaces).
- ``ipopt_cpp``: direct native C++ IPOPT TNLP implementation (trajectory interface).
- ``cg2o``: native C++ constrained factor-graph implementation (trajectory interface).
- ``ipopt``: backwards-compatible alias for ``ipopt_casadi``.
- ``fatrop``: optional legacy CasADi/FATROP path (falls back to ipopt_casadi when
  CasADi has no fatrop plugin).

The native backends need a compiled extension and support only the trajectory
interface.  When one of them cannot be used, the router falls back to
``ipopt_casadi`` and reports why in ``solver.backend_error`` (the controller
logs it), exactly like a missing fatrop plugin.  Benchmarks must not silently
measure a different solver, so they set ``solver_allow_fallback: false`` and get
the original exception instead.
"""
from px4_nmpc_controller.px4_nmpc_ipopt.solver import NS, SolveResult, NmpcOcpSolver

_NATIVE_BACKENDS = ('cg2o', 'ipopt_cpp')
_CASADI_BACKENDS = ('ipopt_casadi', 'ipopt', 'fatrop')


def _make_native(backend, cfg):
    if backend == 'cg2o':
        from px4_nmpc_controller.px4_nmpc_cg2o.solver import Cg2oOcpSolver
        return Cg2oOcpSolver(cfg)
    from px4_nmpc_controller.px4_nmpc_ipopt_cpp.solver import IpoptCppOcpSolver
    return IpoptCppOcpSolver(cfg)


def make_ocp_solver(cfg):
    backend = str(cfg.get("backend", "ipopt_casadi")).lower()
    if cfg.get('prediction_model', 'linear') == 'challenging':
        if backend not in _NATIVE_BACKENDS:
            raise ValueError('The challenging MPC requires native cg2o or ipopt_cpp')
        from px4_nmpc_controller.challenging_solver import ChallengingOcpSolver
        return ChallengingOcpSolver(cfg)
    if cfg.get('prediction_model', 'linear') == 'nonlinear':
        if backend not in _NATIVE_BACKENDS:
            raise ValueError('Nonlinear prediction requires native cg2o or ipopt_cpp')
        from px4_nmpc_controller.nonlinear_solver import NonlinearOcpSolver
        return NonlinearOcpSolver(cfg)
    if backend in _NATIVE_BACKENDS:
        try:
            return _make_native(backend, cfg)
        except (ImportError, ValueError, RuntimeError, FileNotFoundError) as exc:
            if not bool(cfg.get("allow_fallback", True)):
                raise
            solver = NmpcOcpSolver(dict(cfg, backend="ipopt_casadi"))
            solver.backend_error = f"{backend} backend unavailable ({exc}); using ipopt_casadi"
            return solver
    if backend in _CASADI_BACKENDS:
        return NmpcOcpSolver(dict(cfg, backend=backend))
    raise ValueError(
        f"unsupported solver backend {backend!r}; choose ipopt_casadi, ipopt_cpp, cg2o "
        "(or legacy ipopt/fatrop)"
    )

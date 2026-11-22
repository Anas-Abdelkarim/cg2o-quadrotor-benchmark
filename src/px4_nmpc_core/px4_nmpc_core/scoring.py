"""One objective and one feasibility measure for every solver backend.

Each backend reports its own cost (IPOPT's objective, the cg2o factor-graph chi2, ...).
For a fair comparison every returned solution ``(X, U, S)`` is also re-evaluated here
with the *reference* NLP of the CasADi formulation (``NmpcOcpSolver``): objective,
equality residual (dynamics + initial condition), inequality violation and bound
violation.  A solver that returns one point and reports the cost of another is caught by
``cost`` vs ``solver_cost``.
"""
from __future__ import annotations

import casadi as ca
import numpy as np


class CommonScorer:
    def __init__(self, solver_cfg):
        from px4_nmpc_controller.px4_nmpc_ipopt.solver import NmpcOcpSolver
        cfg = dict(solver_cfg, backend='ipopt_casadi', build_nlpsol=False)
        # only the symbolic NLP of the CasADi formulation (objective, constraints, bounds) is built
        self.ref = NmpcOcpSolver(cfg)
        nlp = self.ref._nlp
        self._f = ca.Function('score_f', [nlp['x'], nlp['p']], [nlp['f']])
        self._g = ca.Function('score_g', [nlp['x'], nlp['p']], [nlp['g']])
        self._eq = self.ref.lbg == self.ref.ubg

    def score(self, X, U, S, x0, pref, vref, obstacles, arena_lb=None, arena_ub=None,
              d=None, uprev=None, yaw_ref=0.0, hdg_ref=None, hdg_w=None):
        """Return dict(cost, eq_residual, ineq_violation, bound_violation) (NaN if not evaluable)."""
        nan = {'cost': float('nan'), 'eq_residual': float('nan'),
               'ineq_violation': float('nan'), 'bound_violation': float('nan')}
        if X is None or U is None or S is None:
            return nan
        X, U, S = (np.asarray(a, float) for a in (X, U, S))
        ref = self.ref
        if X.shape != (ref.nx, ref.N + 1) or U.shape != (ref.nu, ref.N) or S.shape != (3, ref.N):
            return nan
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(U)) and np.all(np.isfinite(S))):
            return nan
        packed = ref.parameter_vector(x0, pref, vref, obstacles, d=d, uprev=uprev, yaw_ref=yaw_ref,
                                      arena_lb=arena_lb, arena_ub=arena_ub, hdg_ref=hdg_ref, hdg_w=hdg_w)
        if packed is None:
            return nan
        p = packed[0]
        z = ref._pack(X, U, S)
        f = float(self._f(z, p))
        g = np.asarray(self._g(z, p), float).reshape(-1)
        lbg, ubg = ref.lbg, ref.ubg
        eq = float(np.max(np.abs(g[self._eq] - lbg[self._eq]))) if np.any(self._eq) else 0.0
        ineq = ~self._eq
        viol = np.maximum(lbg[ineq] - g[ineq], g[ineq] - ubg[ineq])
        ineq_v = float(max(0.0, np.max(viol))) if viol.size else 0.0
        bnd = np.maximum(ref.lbw - z, z - ref.ubw)
        return {'cost': f, 'eq_residual': eq, 'ineq_violation': ineq_v,
                'bound_violation': float(max(0.0, np.max(bnd)))}

    def score_request(self, result, request, arena_lb=None, arena_ub=None):
        """Score a SolveResult for the SolveRequest that produced it."""
        if result is None or getattr(result, 'X', None) is None:
            return self.score(None, None, None, None, None, None, None)
        return self.score(result.X, result.U, result.S, request.x0, request.pref, request.vref,
                          request.obstacles, arena_lb=arena_lb, arena_ub=arena_ub, d=request.d,
                          uprev=request.uprev, yaw_ref=request.yaw_ref,
                          hdg_ref=getattr(request, 'hdg_ref', None), hdg_w=getattr(request, 'hdg_w', None))

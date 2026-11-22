"""Structure-exploiting multiple-shooting NMPC for obstacle avoidance.

The solver uses CasADi symbolic models, stage-wise state/control/slack variables,
soft state constraints, a fixed number of nearest-obstacle slots, warm starts,
and FATROP by default with IPOPT as a fallback.  Solutions are accepted only
when the shooting dynamics are sufficiently consistent and the safety-slack
checks pass.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from px4_nmpc_core.models import make_model
from px4_nmpc_core.solver_base import NmpcSolverBase


NS = 3  # slack per stage: [obstacle, arena, dynamic limits]


@dataclass
class SolveResult:
    X: np.ndarray | None
    U: np.ndarray | None
    S: np.ndarray | None
    success: bool          # optimiser reported convergence
    usable: bool           # prediction dynamically consistent -> may be applied
    status: str
    iterations: int
    solve_ms: float
    gap_residual: float = float('inf')
    max_obstacle_slack: float = float('inf')
    max_arena_slack: float = float('inf')
    max_dynamic_slack: float = float('inf')
    cost: float = float('nan')
    extra: dict = field(default_factory=dict)


class NmpcOcpSolver(NmpcSolverBase):
    """Generic OCP builder + solver for the three control interfaces."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.interface = cfg.get('interface', 'trajectory')
        self.N = int(cfg['N'])
        self.dt = float(cfg['dt'])
        self.K = int(cfg.get('max_obstacles', 8))
        requested_backend = str(cfg.get('backend', 'ipopt')).lower()
        if requested_backend not in ('ipopt', 'ipopt_casadi', 'fatrop'):
            raise ValueError(f'CasADi backend must be ipopt_casadi/ipopt or fatrop, got {requested_backend!r}')
        # `ipopt` is retained as a backwards-compatible alias.  The explicit
        # `ipopt_casadi` label lets experiments distinguish this implementation
        # from the native C++ IPOPT TNLP backend.
        self.backend = requested_backend
        self._plugin_backend = 'ipopt' if requested_backend in ('ipopt', 'ipopt_casadi') else 'fatrop'
        self.backend_error = ''
        if self._plugin_backend == 'fatrop' and not ca.has_nlpsol('fatrop'):
            self.backend_error = 'CasADi has no fatrop plugin; using ipopt_casadi'
            self.backend = 'ipopt_casadi'
            self._plugin_backend = 'ipopt'
        self.model = make_model(self.interface, cfg['model'])
        self.model.build_numeric(self.dt)
        self.nx, self.nu = self.model.nx, self.model.nu
        self.w = dict(cfg.get('weights', {}))
        if self.N < 2 or self.dt <= 0.0:
            raise ValueError('N >= 2 and dt > 0 are required')
        self._build()
        self._warm = None

    # ------------------------------------------------------------------ build
    def _build(self):
        N, nx, nu, K = self.N, self.nx, self.nu, self.K
        m, w = self.model, self.w
        nuu = nu + NS

        P_x0 = ca.SX.sym('x0', nx)
        P_pref = ca.SX.sym('pref', 3, N + 1)
        P_vref = ca.SX.sym('vref', 3, N + 1)
        P_obs = ca.SX.sym('obs', 4, K)
        P_d = ca.SX.sym('d', 3)
        P_uprev = ca.SX.sym('uprev', nu)
        P_yaw = ca.SX.sym('yaw')
        P_lb = ca.SX.sym('arena_lb', 3)
        P_ub = ca.SX.sym('arena_ub', 3)
        params = [P_x0, ca.vec(P_pref), ca.vec(P_vref), ca.vec(P_obs), P_d, P_uprev, P_yaw, P_lb, P_ub]
        self.has_yaw = bool(getattr(m, 'has_yaw', False))
        if self.has_yaw:
            # per-stage heading reference [rad, unwrapped] and its weight (see PointMassJerkYawModel)
            P_href = ca.SX.sym('hdg_ref', N + 1)
            P_hw = ca.SX.sym('hdg_w', N + 1)
            params += [P_href, P_hw]
        self._param_sizes = [int(p.numel()) for p in params]

        X = [ca.SX.sym(f'x{k}', nx) for k in range(N + 1)]
        UU = [ca.SX.sym(f'u{k}', nuu) for k in range(N)]

        q_pos = np.asarray(w.get('pos', [12.0, 12.0, 16.0]), float)
        q_vel = np.asarray(w.get('vel', [2.0, 2.0, 3.0]), float)
        q_pos_N = np.asarray(w.get('pos_terminal', (q_pos * 6.0).tolist()), float)
        q_vel_N = np.asarray(w.get('vel_terminal', (q_vel * 4.0).tolist()), float)
        r_u = m.input_weight(w)
        r_du0 = np.asarray(w.get('delta_input_first', [0.0] * nu), float)
        if r_du0.shape != (nu,):
            r_du0 = np.full(nu, float(r_du0.reshape(-1)[0]))
        so_l1, so_l2 = float(w.get('slack_obstacle_l1', 2000.0)), float(w.get('slack_obstacle_l2', 4000.0))
        sa_l1, sa_l2 = float(w.get('slack_arena_l1', 1000.0)), float(w.get('slack_arena_l2', 2000.0))
        sd_l1, sd_l2 = float(w.get('slack_dynamic_l1', 200.0)), float(w.get('slack_dynamic_l2', 400.0))
        w_term_pen = float(w.get('terminal_penalty', 4000.0))
        eps = 0.05

        def obstacle_h(p):
            out = []
            for j in range(K):
                dist = ca.sqrt(ca.sumsqr(p - P_obs[0:3, j]) + eps ** 2)
                out.append(dist - P_obs[3, j])
            return out

        def arena_h(p):
            return [p[i] - P_lb[i] for i in range(3)] + [P_ub[i] - p[i] for i in range(3)]

        u_ref = m.u_ref(P_d)
        wv, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []
        J = 0
        u_lb, u_ub = m.u_bounds()
        n_dyn = None

        for k in range(N + 1):
            xk = X[k]
            wv.append(xk)
            lbw += [-ca.inf] * nx
            ubw += [ca.inf] * nx
            if k == N:
                break
            uk = UU[k]
            u, s = uk[0:nu], uk[nu:nu + NS]
            wv.append(uk)
            lbw += list(u_lb) + [0.0] * NS
            # Stage-0 slacks are unused (x0 is fixed); pin them to zero.
            ubw += list(u_ub) + ([0.0] * NS if k == 0 else [ca.inf] * NS)

            # -- constraints of stage k: gap closing first (FATROP layout) --
            g.append(X[k + 1] - m.discrete(xk, u, P_d, P_yaw, self.dt))
            lbg += [0.0] * nx
            ubg += [0.0] * nx
            if k == 0:
                g.append(xk - P_x0)
                lbg += [0.0] * nx
                ubg += [0.0] * nx
            else:
                for h in obstacle_h(xk[0:3]):
                    g.append(h + s[0]); lbg.append(0.0); ubg.append(ca.inf)
                for h in arena_h(xk[0:3]):
                    g.append(h + s[1]); lbg.append(0.0); ubg.append(ca.inf)
                dyn = m.soft_path_constraints(xk, u)
                n_dyn = len(dyn)
                for e in dyn:
                    g.append(e - s[2]); lbg.append(-ca.inf); ubg.append(0.0)

            # -- stage cost --
            ep = xk[0:3] - P_pref[:, k]
            ev = xk[3:6] - P_vref[:, k]
            eu = u - u_ref
            J += ca.dot(q_pos * ep, ep) + ca.dot(q_vel * ev, ev) + ca.dot(r_u * eu, eu)
            J += m.extra_stage_cost(xk, u, P_d, P_yaw, w)
            if self.has_yaw:
                J += m.heading_cost(xk, P_href[k], P_hw[k])
            if k == 0:
                du = u - P_uprev
                J += ca.dot(r_du0 * du, du)
            else:
                J += so_l1 * s[0] + so_l2 * s[0] ** 2
                J += sa_l1 * s[1] + sa_l2 * s[1] ** 2
                J += sd_l1 * s[2] + sd_l2 * s[2] ** 2

        # terminal cost + smooth exterior penalty for the terminal state
        xN = X[N]
        ep = xN[0:3] - P_pref[:, N]
        ev = xN[3:6] - P_vref[:, N]
        J += ca.dot(q_pos_N * ep, ep) + ca.dot(q_vel_N * ev, ev)
        J += m.extra_stage_cost(xN, ca.DM.zeros(nu), P_d, P_yaw, w)
        if self.has_yaw:
            J += m.heading_cost(xN, P_href[N], P_hw[N])
        for h in obstacle_h(xN[0:3]) + arena_h(xN[0:3]):
            J += w_term_pen * ca.fmin(h, 0.0) ** 2

        self.n_dyn = int(n_dyn or 0)
        z = ca.vertcat(*wv)
        gg = ca.vertcat(*g)
        pvec = ca.vertcat(*params)
        nlp = {'x': z, 'f': J, 'g': gg, 'p': pvec}
        self.lbw = np.asarray(lbw, float)
        self.ubw = np.asarray(ubw, float)
        self.lbg = np.asarray(lbg, float)
        self.ubg = np.asarray(ubg, float)
        self.nz = z.numel()
        self.ng = gg.numel()

        # index helpers for packing / warm-start shifting
        self._stage_w = []
        off = 0
        for k in range(N + 1):
            n = nx + (nuu if k < N else 0)
            self._stage_w.append(np.arange(off, off + n))
            off += n
        self._stage_g = []
        counts = []
        for k in range(N):
            if k == 0:
                counts.append(nx + nx)
            else:
                counts.append(nx + K + 6 + self.n_dyn)
        off = 0
        for c in counts:
            self._stage_g.append(np.arange(off, off + c))
            off += c
        assert off == self.ng, (off, self.ng)

        self._nlp = nlp
        self._counts = counts
        t0 = time.perf_counter()
        if not bool(self.cfg.get('build_nlpsol', True)):
            # symbolic NLP only (CommonScorer): never load CasADi's bundled IPOPT, which shares the
            # soname libipopt.so.3 with the native ipopt_cpp backend's IPOPT
            self.solver = None
        else:
          try:
            self.solver = ca.nlpsol('nmpc', self._plugin_backend, nlp, self._solver_options(self._plugin_backend))
          except Exception as exc:  # noqa: BLE001 - e.g. an old CasADi without the fatrop plugin
            if self._plugin_backend == 'ipopt':
                raise
            self.backend_error = f'{self.backend} unavailable ({exc}); falling back to ipopt_casadi'
            self.backend = 'ipopt_casadi'
            self._plugin_backend = 'ipopt'
            self.solver = ca.nlpsol('nmpc', 'ipopt', nlp, self._solver_options('ipopt'))
        self.build_s = time.perf_counter() - t0

        # vectorised helpers for checking the returned prediction
        xs = ca.SX.sym('xs', nx)
        us = ca.SX.sym('us', nu)
        ds = ca.SX.sym('ds', 3)
        ys = ca.SX.sym('ys')
        F1 = ca.Function('F1', [xs, us, ds, ys], [m.discrete(xs, us, ds, ys, self.dt)])
        self._Fmap = F1.map(N)

    def _solver_options(self, backend):
        N, nx, nu = self.N, self.nx, self.nu
        nuu = nu + NS
        counts, lbg, ubg = self._counts, self.lbg, self.ubg
        max_iter = int(self.cfg.get('max_iter', 150))
        tol = float(self.cfg.get('tol', 1e-4))
        if backend == 'fatrop':
            opts = {
                'record_time': True,
                'structure_detection': 'manual',
                'N': N,
                'nx': [nx] * (N + 1),
                'nu': [nuu] * N + [0],
                'ng': [c - nx for c in counts] + [0],
                'equality': [bool(lo == hi) for lo, hi in zip(lbg, ubg)],
                'expand': True,
                'print_time': False,
                'fatrop': {
                    'print_level': 0,
                    'max_iter': max_iter,
                    'tol': tol,
                    'mu_init': float(self.cfg.get('mu_init', 1e-1)),
                },
            }
            opts['fatrop'].update(self.cfg.get('fatrop_options', {}))
        else:
            ipopt = {
                'print_level': 0,
                'sb': 'yes',
                'max_iter': max_iter,
                'tol': tol,
                'acceptable_tol': float(self.cfg.get('acceptable_tol', 1e-3)),
                'acceptable_iter': 3,
                'mu_strategy': 'adaptive',
                'max_wall_time': float(self.cfg.get('max_wall_time_s', 0.5)),
            }
            ipopt.update(self.cfg.get('ipopt_options', {}))
            opts = {'expand': True, 'print_time': False, 'record_time': True, 'ipopt': ipopt}
        return opts

    # ------------------------------------------------------------- utilities
    def _unpack(self, z):
        N, nx, nu = self.N, self.nx, self.nu
        X = np.zeros((nx, N + 1))
        U = np.zeros((nu, N))
        S = np.zeros((NS, N))
        for k in range(N + 1):
            blk = z[self._stage_w[k]]
            X[:, k] = blk[:nx]
            if k < N:
                U[:, k] = blk[nx:nx + nu]
                S[:, k] = blk[nx + nu:]
        return X, U, S

    def _pack(self, X, U, S):
        z = np.zeros(self.nz)
        for k in range(self.N + 1):
            idx = self._stage_w[k]
            if k < self.N:
                z[idx] = np.r_[X[:, k], U[:, k], S[:, k]]
            else:
                z[idx] = X[:, k]
        return z

    def reset_warm_start(self):
        self._warm = None

    def _initial_guess(self, x0, pref, vref, d, yaw, shift, hdg_ref=None):
        N, nx, nu = self.N, self.nx, self.nu
        if self._warm is not None and shift is not None and 0 <= shift < N:
            zprev, lam_g, lam_x = self._warm
            Xo, Uo, So = self._unpack(zprev)
            idx = [min(k + shift, N) for k in range(N + 1)]
            X = Xo[:, idx]
            U = Uo[:, [min(k + shift, N - 1) for k in range(N)]]
            S = So[:, [min(k + shift, N - 1) for k in range(N)]]
            X[:, 0] = x0
            S[:, 0] = 0.0
            z = self._pack(X, U, S)
            lg = np.zeros(self.ng)
            lx = np.zeros(self.nz)
            for k in range(N):
                src = min(k + shift, N - 1)
                if k == 0:
                    # stage 0 = [gap_0, initial condition]; later stages carry
                    # path constraints instead, so only the gap duals transfer.
                    n_copy = len(self._stage_g[0]) if src == 0 else nx
                    lg[self._stage_g[0][:n_copy]] = lam_g[self._stage_g[src][:n_copy]]
                else:
                    lg[self._stage_g[k]] = lam_g[self._stage_g[src]]
            for k in range(N + 1):
                src = min(k + shift, N)
                if len(self._stage_w[k]) == len(self._stage_w[src]):
                    lx[self._stage_w[k]] = lam_x[self._stage_w[src]]
            return z, lg, lx
        # Cold start: roll the model out under a simple feedback law that
        # tracks the (A*-guided) reference.  The guess is dynamically
        # consistent (zero shooting gaps) and already close to the reference,
        # which matters for extreme initial states (large tilt, outside arena).
        X = np.zeros((nx, N + 1))
        U = np.zeros((nu, N))
        X[:, 0] = x0
        lb, ub = (np.asarray(b, float) for b in self.model.u_bounds())
        for k in range(N):
            yaw_k = yaw if hdg_ref is None else hdg_ref[k + 1]
            u = np.clip(self.model.guess_input(X[:, k], pref[:, k + 1], vref[:, k + 1], d, yaw_k), lb, ub)
            U[:, k] = u
            X[:, k + 1] = self.model.step(X[:, k], u, d, yaw, self.dt)
        S = np.zeros((NS, N))
        S[:, 1:] = 0.01
        return self._pack(X, U, S), None, None

    def _gap_residual(self, X, U, d, yaw, x0):
        Xn = np.asarray(self._Fmap(X[:, :-1], U, np.tile(np.asarray(d, float).reshape(3, 1), (1, self.N)),
                                   np.full((1, self.N), float(yaw))))
        gap = float(np.max(np.abs(X[:, 1:] - Xn))) if self.N else 0.0
        return max(gap, float(np.max(np.abs(X[:, 0] - x0))))

    # ------------------------------------------------------------ parameters
    def parameter_vector(self, x0, pref, vref, obstacles, d=None, uprev=None, yaw_ref=0.0,
                         arena_lb=None, arena_ub=None, hdg_ref=None, hdg_w=None):
        """NLP parameter vector of one OCP (same padding/defaults as every backend).

        Returns ``(p, x0, pref, vref, d, hdg_ref)`` or ``None`` for non-finite input.
        Also used by :class:`px4_nmpc_core.scoring.CommonScorer` to evaluate any backend's
        solution with this reference objective/constraints.
        """
        N, nx, K = self.N, self.nx, self.K
        x0 = np.asarray(x0, float).reshape(nx)
        pref = np.asarray(pref, float).reshape(3, N + 1)
        vref = np.asarray(vref, float).reshape(3, N + 1)
        d = np.zeros(3) if d is None else np.asarray(d, float).reshape(3)
        uprev = self.model.hover_input(d) if uprev is None else np.asarray(uprev, float).reshape(-1)[:self.nu]
        obs = np.zeros((4, K))
        obs[0:3, :] = x0[0:3].reshape(3, 1) + 1.0e3
        obs[3, :] = 0.1
        if obstacles is not None and len(obstacles):
            o = np.asarray(obstacles, float).reshape(-1, 4)[:K]
            obs[:, :len(o)] = o.T
        lb = np.full(3, -1.0e4) if arena_lb is None else np.asarray(arena_lb, float).reshape(3)
        ub = np.full(3, 1.0e4) if arena_ub is None else np.asarray(arena_ub, float).reshape(3)
        inputs = [x0, pref, vref, obs, d, uprev, np.array([yaw_ref]), lb, ub]
        extra = []
        if self.has_yaw:
            hdg_ref = np.full(N + 1, float(x0[9])) if hdg_ref is None else np.asarray(hdg_ref, float).reshape(N + 1)
            hdg_w = np.zeros(N + 1) if hdg_w is None else np.asarray(hdg_w, float).reshape(N + 1)
            inputs += [hdg_ref, hdg_w]
            extra = [hdg_ref, hdg_w]
        else:
            hdg_ref = None
        if not all(np.all(np.isfinite(a)) for a in inputs):
            return None
        p = np.concatenate([x0, pref.reshape(-1, order='F'), vref.reshape(-1, order='F'),
                            obs.reshape(-1, order='F'), d, uprev, [float(yaw_ref)], lb, ub] + extra)
        return p, x0, pref, vref, d, hdg_ref

    # ------------------------------------------------------------------ solve
    def solve_ocp(self, x0, pref, vref, obstacles, d=None, uprev=None, yaw_ref=0.0,
                  arena_lb=None, arena_ub=None, shift=None, hdg_ref=None, hdg_w=None):
        """Solve one NMPC problem.

        ``obstacles``: array (K', 4) of [N, E, D, effective_radius]; padded or
        truncated to K slots.  ``shift``: number of stages between the previous
        plan's start time and this one (for warm start); ``None`` = cold start.
        ``hdg_ref`` / ``hdg_w`` (length N+1): heading reference [rad, unwrapped near ``x0[9]``]
        and weight per stage; only used by models with an optimised yaw (``has_yaw``).
        """
        N = self.N
        packed = self.parameter_vector(x0, pref, vref, obstacles, d=d, uprev=uprev, yaw_ref=yaw_ref,
                                       arena_lb=arena_lb, arena_ub=arena_ub, hdg_ref=hdg_ref, hdg_w=hdg_w)
        if packed is None:
            return SolveResult(None, None, None, False, False, 'non-finite NMPC input', 0, 0.0)
        p, x0, pref, vref, d, hdg_ref = packed
        z0, lg0, lx0 = self._initial_guess(x0, pref, vref, d, yaw_ref, shift, hdg_ref)
        args = dict(x0=z0, p=p, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        if lg0 is not None and self._plugin_backend == 'ipopt' and self.cfg.get('ipopt_warm_duals', False):
            args['lam_g0'] = lg0
            args['lam_x0'] = lx0
        t0 = time.perf_counter()
        try:
            sol = self.solver(**args)
            stats = self.solver.stats()
        except Exception as exc:  # noqa: BLE001 - report every solver error
            return SolveResult(None, None, None, False, False, f'solver exception: {exc}', 0,
                               (time.perf_counter() - t0) * 1e3)
        call_wall_ms = (time.perf_counter() - t0) * 1e3
        # CasADi's recorded wall time measures the nlpsol execution itself and
        # avoids attributing the surrounding Python call overhead to IPOPT.
        # Keep the full wrapper wall time separately for transparency.
        recorded_s = stats.get('t_wall_total', None)
        try:
            solve_ms = float(recorded_s) * 1e3 if recorded_s is not None else call_wall_ms
            if not np.isfinite(solve_ms) or solve_ms <= 0.0:
                solve_ms = call_wall_ms
        except Exception:
            solve_ms = call_wall_ms
        z = np.asarray(sol['x'], float).reshape(-1)
        status = str(stats.get('return_status', stats.get('unified_return_status', '')))
        success = bool(stats.get('success', False))
        iters = int(stats.get('iter_count', -1)) if stats.get('iter_count') is not None else -1
        if self._plugin_backend == 'fatrop':
            # FATROP reports an integer code; show IPOPT-style text in diagnostics
            if success:
                status = 'Solve_Succeeded'
            elif iters >= int(self.cfg.get('max_iter', 150)):
                status = 'Maximum_Iterations_Exceeded'
            else:
                status = f'fatrop_status_{status}'
        if not np.all(np.isfinite(z)):
            return SolveResult(None, None, None, False, False, f'non-finite solution ({status})', iters, solve_ms)
        X, U, S = self._unpack(z)
        gap = self._gap_residual(X, U, d, yaw_ref, x0)
        usable = gap < float(self.cfg.get('gap_tolerance', 2e-3))
        if usable:
            self._warm = (z, np.asarray(sol['lam_g'], float).reshape(-1), np.asarray(sol['lam_x'], float).reshape(-1))
        else:
            self._warm = None
        return SolveResult(
            X=X, U=U, S=S, success=success, usable=usable, status=status, iterations=iters,
            solve_ms=solve_ms, gap_residual=gap,
            max_obstacle_slack=float(np.max(S[0, 1:])) if N > 1 else 0.0,
            max_arena_slack=float(np.max(S[1, 1:])) if N > 1 else 0.0,
            max_dynamic_slack=float(np.max(S[2, 1:])) if N > 1 else 0.0,
            cost=float(sol['f']), extra={'backend_call_wall_ms': call_wall_ms},
        )

    # NmpcSolverBase contract (kept for backwards compatibility)
    def solve(self, x0, xref, uprev):
        xref = np.asarray(xref, float)
        r = self.solve_ocp(x0, xref[0:3], xref[3:6], None, uprev=uprev)
        return r.U, r.X, r.usable, r.solve_ms, r.status



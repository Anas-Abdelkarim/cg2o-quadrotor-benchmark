"""cg2o backend for the trajectory-interface NMPC OCP.

The complete factor-graph formulation is implemented in C++. Solver-specific
settings are loaded from ``px4_nmpc_config/config/cg2o.yaml`` and can be
overridden per solver instance with ``cfg['cg2o_overrides']`` (experiments,
tuning). The constrained optimizer is ISPD (infeasible-start primal-dual
interior point); the linear solver and the warm-start strategy are selected at
run time.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

from px4_nmpc_core.models import make_model
from px4_nmpc_core.paths import repo_root
from px4_nmpc_controller.px4_nmpc_ipopt.solver import NS, SolveResult
from px4_nmpc_core.solver_base import NmpcSolverBase
from .config import load_cg2o_config

_EXTENSION_ENV_VAR = 'PX4_NMPC_CG2O_BUILD_DIR'


def _candidate_extension_dirs():
    """Yield possible locations of the compiled cg2o pybind extension."""
    env = os.environ.get(_EXTENSION_ENV_VAR)
    if env:
        yield Path(env).expanduser().resolve()

    here = Path(__file__).resolve()

    # Source-tree execution: locate the workspace/repository root.
    try:
        root = repo_root(here)
    except FileNotFoundError:
        pass
    else:
        yield root / 'build' / 'cg2o_backend'

    # Legacy source-tree build location, retained for compatibility.
    yield here.parent / 'cpp' / 'build'

    # Colcon-installed execution:
    #   <workspace>/install/px4_nmpc_controller/.../solver.py
    # Find the install ancestor and recover <workspace>.
    for parent in here.parents:
        if parent.name == 'install':
            yield parent.parent / 'build' / 'cg2o_backend'
            break


def _load_extension():
    """Load the compiled cg2o pybind extension."""
    try:
        import cpp_px4_nmpc_cg2o as ext  # noqa: F401
        return ext
    except ImportError:
        pass

    tried = []
    for d in _candidate_extension_dirs():
        d = Path(d)
        tried.append(str(d))

        if not d.is_dir():
            continue
        if not any(d.glob('cpp_px4_nmpc_cg2o*.so')):
            continue

        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

        try:
            import cpp_px4_nmpc_cg2o as ext
            return ext
        except ImportError as exc:
            raise ImportError(
                f'Found cg2o extension in {d}, but Python could not load it: {exc}'
            ) from exc

    raise ImportError(
        'compiled cpp_px4_nmpc_cg2o extension not found; '
        f'searched: {tried}. '
        f'Set {_EXTENSION_ENV_VAR} to override or run '
        './scripts/build_cg2o_backend.sh'
    )


def _construct_extension(cls, args, yaw_args, has_yaw, name):
    """Create the native solver. The yaw arguments exist only in extensions built from 3.1.0 or later; they are
    passed only when the optimised yaw is requested, so a fixed-yaw flight still works with an older build."""
    try:
        return cls(*args, *yaw_args)
    except TypeError as exc:
        if has_yaw:
            raise RuntimeError(f'the compiled {name} extension is older than this source (no yaw support): '
                               f'rebuild it with ./scripts/build_{name}_backend.sh') from exc
        raise


def deep_merge(base, override):
    """Recursive dict merge (override wins); used for cfg['cg2o_overrides']."""
    out = copy.deepcopy(dict(base))
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Cg2oOcpSolver(NmpcSolverBase):
    """Drop-in replacement for :class:`NmpcOcpSolver` for the trajectory
    interface only, backed by the cg2o factor-graph solver."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.interface = cfg.get('interface', 'trajectory')
        if self.interface != 'trajectory':
            raise ValueError(
                f"cg2o backend only supports the 'trajectory' interface "
                f"(got {self.interface!r}); the attitude/body_rate g2o "
                f"formulations are not implemented"
            )
        self.N = int(cfg['N'])
        self.dt = float(cfg['dt'])
        self.K = int(cfg.get('max_obstacles', 8))
        self.backend = 'cg2o'
        self.backend_error = ''
        if self.N < 2 or self.dt <= 0.0:
            raise ValueError('N >= 2 and dt > 0 are required')

        ext = _load_extension()
        self._ext = ext
        self.cg2o_config = deep_merge(load_cg2o_config(cfg.get('cg2o_config_path')),
                                      cfg.get('cg2o_overrides') or {})
        if int(getattr(ext, 'api_version', 1)) < 2:
            raise RuntimeError('the compiled cg2o extension is older than this source (no run-time optimizer '
                               'selection): rebuild it with ./scripts/build_backends/cg2o.sh')
        optimizer = str(self.cg2o_config.get('optimizer', 'ispd')).strip().lower()
        if optimizer in ('', 'auto'):
            optimizer = 'ispd'
        if optimizer != 'ispd':
            raise ValueError(f'the cg2o backend uses ISPD only (optimizer: ispd), got {optimizer!r}')
        self.cg2o_config['optimizer'] = optimizer
        self.cg2o_optimizer = optimizer
        warm = dict(self.cg2o_config.get('warm_start') or {})
        self.warm_strategy = int(warm.get('strategy', 0))

        self.model = make_model(self.interface, cfg['model'])
        self.model.build_numeric(self.dt)
        self.nx, self.nu = self.model.nx, self.model.nu
        self.has_yaw = bool(getattr(self.model, 'has_yaw', False))

        m = dict(cfg.get('model', {}))
        w = dict(cfg.get('weights', {}))
        max_iter = int(cfg.get('max_iter', 150))
        self._max_iter = int(cfg.get('cg2o_max_iterations') or self.cg2o_config.get('max_iterations', max_iter))
        r_delta = np.asarray(w.get('delta_input_first', [0.0, 0.0, 0.0]), float).reshape(-1)
        if r_delta.size == 1:
            r_delta = np.repeat(r_delta, 3)
        r_delta = r_delta[:3].reshape(3)  # the yaw acceleration input (4th entry, optimised yaw) has no first-input-delta cost


        yaw_args = ()
        if self.has_yaw:
            yaw_args = (True,
                        float(getattr(self.model, 'yaw_rate_max', 0.8)), float(getattr(self.model, 'yaw_acc_max', 2.0)),
                        float(getattr(self.model, 'w_lat', 1.0)), float(getattr(self.model, 'w_rate', 0.05)),
                        float(getattr(self.model, 'w_alpha', 0.05)))
        cpp_args = (
            self.N, self.dt, self.K,
            float(m.get('g', 9.81)),
            float(m.get('vmax', 3.0)),
            float(m.get('tilt_max', 0.5236)),
            np.asarray(m.get('jerk_max', [12.0, 12.0, 8.0]), float).reshape(3),
            float(m.get('acc_up_max', 4.0)),
            float(m.get('acc_down_max', 3.0)),
            np.asarray(w.get('pos', [12.0, 12.0, 16.0]), float).reshape(3),
            np.asarray(w.get('vel', [2.0, 2.0, 3.0]), float).reshape(3),
            np.asarray(w.get('pos_terminal', [72.0, 72.0, 96.0]), float).reshape(3),
            np.asarray(w.get('vel_terminal', [8.0, 8.0, 12.0]), float).reshape(3),
            float(w.get('jerk', 0.02)),
            float(w.get('acc', 0.3)),
            float(w.get('slack_obstacle_l1', 2000.0)),
            float(w.get('slack_obstacle_l2', 4000.0)),
            float(w.get('slack_arena_l1', 1000.0)),
            float(w.get('slack_arena_l2', 2000.0)),
            float(w.get('slack_dynamic_l1', 200.0)),
            float(w.get('slack_dynamic_l2', 400.0)),
            float(w.get('terminal_penalty', 4000.0)),
            r_delta,
            int(self._max_iter),
            {k: v for k, v in self.cg2o_config.items() if not k.startswith('_')},
        )
        self._cpp = _construct_extension(ext.Cg2oTrajectorySolver, cpp_args, yaw_args, self.has_yaw, 'cg2o')

        # No CasADi NLP here; these exist only so controller_node.py's
        # unconditional diagnostics logging (self.solver.build_s / .nz /
        # .ng) has something sane to report. nz/ng mirror what an
        # equivalent CasADi-style problem would be: nz counts (p,v,a) at
        # every stage plus (jerk, slack) at every input stage; ng counts
        # the per-stage equality/inequality rows the graph actually builds
        # (dynamics + initial condition + obstacle/arena/dynamic-limit +
        # slack-nonneg, matching NmpcOcpSolver's own stage-row accounting).
        self.build_s = 0.0
        nyx = 2 if self.has_yaw else 0      # yaw states (psi, omega)
        nyu = 1 if self.has_yaw else 0      # yaw acceleration input
        self.nz = (9 + nyx) * (self.N + 1) + (3 + nyu + NS) * self.N
        self.ng = (9 + nyx) * 2 + (self.N - 1) * (9 + nyx + self.K + 6 + 4 + NS + (1 if self.has_yaw else 0))

        self._warm_started = False

    # ------------------------------------------------------------------ solve
    def solve_ocp(self, x0, pref, vref, obstacles, d=None, uprev=None, yaw_ref=0.0,
                  arena_lb=None, arena_ub=None, shift=None, hdg_ref=None, hdg_w=None):
        """Solve one NMPC problem. Signature matches
        :meth:`NmpcOcpSolver.solve_ocp`; ``d`` and ``yaw_ref`` do not enter the
        trajectory triple-integrator, while ``uprev`` is used by the optional
        first-input-change cost. ``shift`` = number of stages between the
        previous solve and this one (warm start, shifted by that many stages,
        exactly like the IPOPT backends); ``None`` forces a cold start.
        With ``warm_start.strategy`` 1/2 (ISPD) the multipliers, slacks and the
        barrier parameter are warm-started as well (cg2o's own MPC shift).
        """
        N, nx, K = self.N, self.nx, self.K
        x0 = np.asarray(x0, float).reshape(nx)
        pref = np.asarray(pref, float).reshape(3, N + 1)
        vref = np.asarray(vref, float).reshape(3, N + 1)
        obs = np.zeros((4, K))
        obs[0:3, :] = x0[0:3].reshape(3, 1) + 1.0e3
        obs[3, :] = 0.1
        if obstacles is not None and len(obstacles):
            o = np.asarray(obstacles, float).reshape(-1, 4)[:K]
            obs[:, :len(o)] = o.T
        lb = np.full(3, -1.0e4) if arena_lb is None else np.asarray(arena_lb, float).reshape(3)
        ub = np.full(3, 1.0e4) if arena_ub is None else np.asarray(arena_ub, float).reshape(3)
        up = np.zeros(3) if uprev is None else np.asarray(uprev, float).reshape(-1)[:3]
        if self.has_yaw:   # optimised yaw: heading reference / weight per stage
            hdg_ref = np.full(N + 1, float(x0[9])) if hdg_ref is None else np.asarray(hdg_ref, float).reshape(N + 1)
            hdg_w = np.zeros(N + 1) if hdg_w is None else np.asarray(hdg_w, float).reshape(N + 1)
        else:
            hdg_ref = hdg_w = np.zeros(0)
        inputs = [x0, pref, vref, obs, lb, ub, up, hdg_ref, hdg_w]
        if not all(np.all(np.isfinite(a)) for a in inputs):
            return SolveResult(None, None, None, False, False, 'non-finite NMPC input', 0, 0.0)

        if shift is None:
            if self._warm_started:
                self._cpp.reset_warm_start()
                self._warm_started = False
            shift_steps = -1
        else:
            shift_steps = max(0, int(shift))

        t0 = time.perf_counter()
        attempt = self._solve_once(x0, pref, vref, obs, lb, ub, up, t0, hdg_ref, hdg_w, shift_steps)
        if attempt.usable or not self._warm_started_before_attempt:
            return attempt
        # A warm-started solve that is not usable (dynamics gap) is repeated once from a fresh
        # cold start of the same problem in this cycle (ISPD strategies 1/2 additionally retry
        # with cold duals inside cg2o after a breakdown). Both attempts count in solve_ms.
        self._cpp.reset_warm_start()
        self._warm_started = False
        retry = self._solve_once(x0, pref, vref, obs, lb, ub, up, t0, hdg_ref, hdg_w, -1)
        retry.solve_ms += attempt.solve_ms
        retry.iterations += attempt.iterations
        retry.extra['python_cold_retry'] = True
        if retry.usable:
            retry.status = 'Solve_Succeeded (cold-start retry after warm-start divergence)'
        return retry

    def _solve_once(self, x0, pref, vref, obs, lb, ub, uprev, t0, hdg_ref, hdg_w, shift_steps):
        self._warm_started_before_attempt = self._warm_started and shift_steps >= 0
        try:
            X, U, S, iters, cost, solve_ms, info = self._cpp.solve(
                x0, pref, vref, obs, lb, ub, uprev, shift_steps=int(shift_steps),
                **self._yaw_kwargs(hdg_ref, hdg_w))
        except Exception as exc:  # noqa: BLE001 - report every solver error
            return SolveResult(None, None, None, False, False, f'solver exception: {exc}', 0,
                               (time.perf_counter() - t0) * 1e3)
        self._warm_started = True
        info = dict(info)
        X = np.asarray(X, float)
        U = np.asarray(U, float)
        S = np.asarray(S, float)
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(U)) and np.all(np.isfinite(S))):
            return SolveResult(None, None, None, False, False, 'non-finite solution (cg2o)', int(iters), solve_ms,
                               extra=info)

        gap = self._gap_residual(X, U, x0)
        gap_tol = float(self.cfg.get('gap_tolerance', 2e-3))
        usable = gap < gap_tol
        # ISPD's own convergence test (lastConverged())
        success = bool(info.get('converged', False))
        if usable:
            status = 'Solve_Succeeded' if success else 'cg2o_not_converged_usable'
        else:
            status = 'cg2o_gap_exceeded'
        info['hit_max_iter'] = int(iters) >= self._max_iter
        return SolveResult(
            X=X, U=U, S=S, success=success, usable=usable, status=status, iterations=int(iters),
            solve_ms=float(solve_ms), gap_residual=gap,
            max_obstacle_slack=float(np.max(S[0, 1:])) if self.N > 1 else 0.0,
            max_arena_slack=float(np.max(S[1, 1:])) if self.N > 1 else 0.0,
            max_dynamic_slack=float(np.max(S[2, 1:])) if self.N > 1 else 0.0,
            cost=float(cost), extra=info,
        )

    def _yaw_kwargs(self, hdg_ref, hdg_w):
        return {'hdg_ref': hdg_ref, 'hdg_w': hdg_w} if self.has_yaw else {}

    def _gap_residual(self, X, U, x0):
        """Max shooting-gap residual, computed the same way as
        NmpcOcpSolver._gap_residual but via a plain Python loop over
        model.step (no CasADi .map to build here)."""
        gap = float(np.max(np.abs(X[:, 0] - x0)))
        d0 = np.zeros(3)
        for k in range(self.N):
            xn = self.model.step(X[:, k], U[:, k], d0, 0.0, self.dt)
            gap = max(gap, float(np.max(np.abs(X[:, k + 1] - xn))))
        return gap

    def reset_warm_start(self):
        self._cpp.reset_warm_start()
        self._warm_started = False

    # NmpcSolverBase contract (kept for backwards compatibility)
    def solve(self, x0, xref, uprev):
        xref = np.asarray(xref, float)
        r = self.solve_ocp(x0, xref[0:3], xref[3:6], None, uprev=uprev)
        return r.U, r.X, r.usable, r.solve_ms, r.status

"""Native C++ IPOPT backend for the trajectory NMPC formulation.

This backend solves the same trajectory-interface OCP as the CasADi/IPOPT
reference, but evaluates the objective, gradient, constraints and sparse
Jacobian directly in C++. CasADi is not used in the timed optimizer call.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

from px4_nmpc_core.models import make_model
from px4_nmpc_core.paths import repo_root
from px4_nmpc_core.solver_base import NmpcSolverBase
from px4_nmpc_controller.px4_nmpc_ipopt.solver import NS, SolveResult
from .config import load_ipopt_cpp_config

_EXTENSION_ENV_VAR = 'PX4_NMPC_IPOPT_CPP_BUILD_DIR'


def _candidate_extension_dirs():
    env = os.environ.get(_EXTENSION_ENV_VAR)
    if env:
        yield Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == 'install':
            yield parent.parent / 'build' / 'ipopt_cpp_backend'
            break
    try:
        root = repo_root(here)
    except FileNotFoundError:
        pass
    else:
        yield root / 'build' / 'ipopt_cpp_backend'
    yield here.parent / 'cpp' / 'build'


def _load_extension():
    try:
        import cpp_px4_nmpc_ipopt_cpp as ext
        return ext
    except ImportError:
        pass
    tried = []
    load_errors = []
    for base in _candidate_extension_dirs():
        # also accept an older build layout where the module sits in bindings/
        for d in (Path(base), Path(base) / 'bindings'):
            if str(d) in tried:
                continue
            tried.append(str(d))
            if not d.is_dir() or not any(d.glob('cpp_px4_nmpc_ipopt_cpp*.so')):
                continue
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            try:
                import cpp_px4_nmpc_ipopt_cpp as ext
                return ext
            except ImportError as exc:
                load_errors.append(f'{d}: {exc}')
    extra = f'; load errors: {load_errors}' if load_errors else ''
    raise ImportError(
        'compiled cpp_px4_nmpc_ipopt_cpp extension not found '
        f'(searched: {tried}{extra}); run ./scripts/build_ipopt_cpp_backend.sh '
        f'or set {_EXTENSION_ENV_VAR}'
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


class IpoptCppOcpSolver(NmpcSolverBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.interface = cfg.get('interface', 'trajectory')
        if self.interface != 'trajectory':
            raise ValueError("ipopt_cpp currently supports only control_interface='trajectory'")
        self.N = int(cfg['N'])
        self.dt = float(cfg['dt'])
        self.K = int(cfg.get('max_obstacles', 8))
        self.backend = 'ipopt_cpp'
        self.backend_error = ''
        self.build_s = 0.0
        self.model = make_model(self.interface, cfg['model'])
        self.model.build_numeric(self.dt)
        self.nx, self.nu = self.model.nx, self.model.nu
        self.has_yaw = bool(getattr(self.model, 'has_yaw', False))
        from px4_nmpc_controller.px4_nmpc_cg2o.solver import deep_merge
        self.ipopt_cpp_config = deep_merge(load_ipopt_cpp_config(cfg.get('ipopt_cpp_config_path')),
                                           cfg.get('ipopt_cpp_overrides') or {})
        self.warm_start_mode = str(self.ipopt_cpp_config.get('warm_start', 'primal'))

        ext = _load_extension()
        m = dict(cfg.get('model', {})); w = dict(cfg.get('weights', {}))
        r_delta = np.asarray(w.get('delta_input_first', [0.0, 0.0, 0.0]), float).reshape(-1)
        if r_delta.size == 1:
            r_delta = np.repeat(r_delta, 3)
        r_delta = r_delta[:3]            # the yaw acceleration input (4th entry, optimised yaw) has no first-input-delta cost
        opts = {k: v for k, v in self.ipopt_cpp_config.items() if not k.startswith('_')}

        yaw_args = ()
        if self.has_yaw:
            yaw_args = (True,
                        float(getattr(self.model, 'yaw_rate_max', 0.8)), float(getattr(self.model, 'yaw_acc_max', 2.0)),
                        float(getattr(self.model, 'w_lat', 1.0)), float(getattr(self.model, 'w_rate', 0.05)),
                        float(getattr(self.model, 'w_alpha', 0.05)))
        cpp_args = (
            self.N, self.dt, self.K,
            float(m.get('g', 9.81)), float(m.get('vmax', 3.0)), float(m.get('tilt_max', 0.5236)),
            np.asarray(m.get('jerk_max', [12., 12., 8.]), float).reshape(3),
            float(m.get('acc_up_max', 4.0)), float(m.get('acc_down_max', 3.0)),
            np.asarray(w.get('pos', [12.,12.,16.]), float).reshape(3),
            np.asarray(w.get('vel', [2.,2.,3.]), float).reshape(3),
            np.asarray(w.get('pos_terminal', [72.,72.,96.]), float).reshape(3),
            np.asarray(w.get('vel_terminal', [8.,8.,12.]), float).reshape(3),
            float(w.get('jerk', 0.02)), float(w.get('acc', 0.3)),
            float(w.get('slack_obstacle_l1', 2000.0)), float(w.get('slack_obstacle_l2', 4000.0)),
            float(w.get('slack_arena_l1', 1000.0)), float(w.get('slack_arena_l2', 2000.0)),
            float(w.get('slack_dynamic_l1', 200.0)), float(w.get('slack_dynamic_l2', 400.0)),
            float(w.get('terminal_penalty', 4000.0)), r_delta.reshape(3),
            int(cfg.get('max_iter', 150)), float(cfg.get('tol', 1e-4)),
            float(cfg.get('max_wall_time_s', 0.25)), opts,
        )
        self._cpp = _construct_extension(ext.NativeIpoptTrajectorySolver, cpp_args, yaw_args, self.has_yaw, 'ipopt_cpp')
        self.nz = int(self._cpp.n_vars)
        self.ng = int(self._cpp.n_cons)

    def reset_warm_start(self):
        self._cpp.reset_warm_start()

    def solve_ocp(self, x0, pref, vref, obstacles, d=None, uprev=None, yaw_ref=0.0,
                  arena_lb=None, arena_ub=None, shift=None, hdg_ref=None, hdg_w=None):
        N, nx, K = self.N, self.nx, self.K
        x0 = np.asarray(x0, float).reshape(nx)
        pref = np.asarray(pref, float).reshape(3, N + 1)
        vref = np.asarray(vref, float).reshape(3, N + 1)
        # the trajectory model does not use d or yaw_ref; uprev (jerk part) enters the first-input delta cost;
        # hdg_ref / hdg_w: heading reference and weight per stage (optimised yaw only)
        up = np.zeros(3) if uprev is None else np.asarray(uprev, float).reshape(-1)[:3]
        if self.has_yaw:
            hdg_ref = np.full(N + 1, float(x0[9])) if hdg_ref is None else np.asarray(hdg_ref, float).reshape(N + 1)
            hdg_w = np.zeros(N + 1) if hdg_w is None else np.asarray(hdg_w, float).reshape(N + 1)
        else:
            hdg_ref = hdg_w = np.zeros(0)
        obs = np.zeros((4, K)); obs[0:3, :] = x0[0:3, None] + 1e3; obs[3, :] = 0.1
        if obstacles is not None and len(obstacles):
            o = np.asarray(obstacles, float).reshape(-1, 4)[:K]
            obs[:, :len(o)] = o.T
        lb = np.full(3, -1e4) if arena_lb is None else np.asarray(arena_lb, float).reshape(3)
        ub = np.full(3, 1e4) if arena_ub is None else np.asarray(arena_ub, float).reshape(3)
        if not all(np.all(np.isfinite(a)) for a in (x0,pref,vref,obs,lb,ub,up,hdg_ref,hdg_w)):
            return SolveResult(None,None,None,False,False,'non-finite NMPC input',0,0.0)
        warm_shift = -1 if shift is None else int(shift)
        try:
            out = self._cpp.solve(x0,pref,vref,obs,lb,ub,up,warm_shift=warm_shift,
                                  **({'hdg_ref': hdg_ref, 'hdg_w': hdg_w} if self.has_yaw else {}))
            X,U,S,iters,cost,solve_ms,status,success = out[:8]
            info = dict(out[8]) if len(out) > 8 else {}
        except Exception as exc:
            return SolveResult(None,None,None,False,False,f'solver exception: {exc}',0,0.0)
        X=np.asarray(X,float);U=np.asarray(U,float);S=np.asarray(S,float)
        if X.size == 0 or not (np.all(np.isfinite(X)) and np.all(np.isfinite(U)) and np.all(np.isfinite(S))):
            self.reset_warm_start()
            return SolveResult(None,None,None,bool(success),False,f'non-finite/empty solution ({status})',int(iters),float(solve_ms))
        gap=self._gap_residual(X,U,x0)
        usable=gap < float(self.cfg.get('gap_tolerance',2e-3))
        if not usable:
            self.reset_warm_start()
        return SolveResult(
            X=X,U=U,S=S,success=bool(success),usable=usable,status=str(status),iterations=int(iters),
            solve_ms=float(solve_ms),gap_residual=gap,
            max_obstacle_slack=float(np.max(S[0,1:])) if N>1 else 0.0,
            max_arena_slack=float(np.max(S[1,1:])) if N>1 else 0.0,
            max_dynamic_slack=float(np.max(S[2,1:])) if N>1 else 0.0,
            cost=float(cost), extra=dict(info, hit_max_iter=status == 'Maximum_Iterations_Exceeded'),
        )

    def _gap_residual(self, X, U, x0):
        gap=float(np.max(np.abs(X[:,0]-x0)))
        d0=np.zeros(3)
        for k in range(self.N):
            xn=self.model.step(X[:,k],U[:,k],d0,0.0,self.dt)
            gap=max(gap,float(np.max(np.abs(X[:,k+1]-xn))))
        return gap

    def solve(self, x0, xref, uprev):
        xref=np.asarray(xref,float)
        r=self.solve_ocp(x0,xref[0:3],xref[3:6],None,uprev=uprev)
        return r.U,r.X,r.usable,r.solve_ms,r.status

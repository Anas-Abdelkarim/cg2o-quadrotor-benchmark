"""Runtime adapter of the challenging nonlinear MPC (benchmark/model).

The PX4 runtime works in the controller frame (PX4 local NED, origin at the
take-off point). The course, guide line and obstacles are defined in the hall
frame; ``origin`` is the controller origin expressed in the hall frame.

PX4 receives U[0] (collective thrust) and U[1:4] (body rates) exactly as
optimized. U[4] is the guide progress speed and is not sent anywhere.

Every solve appends its complete problem data (x0, previous input) and the
SHA-256 problem hash to PX4_NMPC_CHALLENGING_TRACE, so that IPOPT can solve the
identical problems after the flight (``px4_nmpc_compare.challenging_replay``).
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

from px4_nmpc_compare import challenging as C
from px4_nmpc_controller.px4_nmpc_ipopt.solver import SolveResult


class ChallengingModel:
    name = 'challenging_thrust_rate_lag_progress'
    nx = 13
    nu = 5
    has_yaw = True
    nonlinear_trajectory = True
    state_names = ('pN', 'pE', 'pD', 'vN', 'vE', 'vD', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 's')
    input_names = ('thrust_N', 'p_setpoint', 'q_setpoint', 'r_setpoint', 'progress_speed')

    def __init__(self, p, origin):
        self.p = p
        self.origin = np.asarray(origin, float)
        self.ext = C.extension()
        self.g = p.gravity
        self.dt = p.dt
        self.drag = np.zeros(3)

    def state_from_measurement(self, pos, vel, rpy, accel=None, omega=None, **kwargs):
        s = self.ext.project(np.asarray(pos, float) + self.origin, self.p)
        w = np.zeros(3) if omega is None else np.asarray(omega, float)
        return np.r_[pos, vel, rpy, w, s]

    def clip_state(self, x):
        return np.array(x, float, copy=True)

    def hover_input(self, d):
        return np.array([self.p.mass * self.p.gravity, 0., 0., 0., 0.])

    def command_thrust_accel(self, force_newton):
        return float(force_newton) / self.p.mass

    def step(self, x, u, d, yaw_ref, h):
        return self.ext.step(np.asarray(x, float), np.asarray(u, float), self.p, float(h))

    def state_at(self, plan, t):
        if t > plan.t_end:
            return plan.X[:, -1].copy()
        k, tau = plan.stage(t)
        if t < plan.t0:
            tau = 0.
        return self.step(plan.X[:, k], plan.U[:, k], None, 0., tau) if tau > 0 else plan.X[:, k].copy()

    def trajectory_at(self, plan, t):
        x = self.state_at(plan, t)
        if t > plan.t_end:
            return x[:3], np.zeros(3), np.zeros(3)
        k, _ = plan.stage(t)
        a = self.ext.rhs(x, plan.U[:, k], self.p)[3:6]
        return x[:3], x[3:6], a

    def yaw_at(self, plan, t):
        x = self.state_at(plan, t)
        if t > plan.t_end:
            return float(x[8]), 0.
        k, _ = plan.stage(t)
        return float(x[8]), float(self.ext.rhs(x, plan.U[:, k], self.p)[8])


class ChallengingOcpSolver:
    logs_phase = True

    def __init__(self, cfg):
        start = time.perf_counter()
        self.cfg = cfg
        self.ext = C.extension()
        self.interface = cfg['interface']
        if self.interface != 'body_rate':
            raise ValueError('The challenging MPC sends thrust and body rates: control_interface must be body_rate')
        self.backend = cfg['backend']
        if self.backend not in ('cg2o', 'ipopt_cpp'):
            raise ValueError('challenging MPC backends: cg2o or ipopt_cpp')
        course = C.load_course(cfg.get('challenging_course') or C.COURSE)
        overrides = dict(vs_max=float(cfg.get('challenging_vs_max', course['mpc']['vs_max'])))
        self.p = C.make_params(course, overrides)
        self.N = self.p.N
        self.dt = self.p.dt
        if abs(self.dt - float(cfg['dt'])) > 1e-12 or self.N != int(cfg['N']):
            raise ValueError(f'control_dt/horizon_steps must match the course ({self.dt}, {self.N})')
        self.K = int(cfg.get('max_obstacles', 10))
        self.origin = np.asarray(cfg.get('challenging_origin_ned', [-14., 0., -0.15]), float)
        self.model = ChallengingModel(self.p, self.origin)
        self.budget = int(cfg['max_iter'])
        self.tol = float(cfg['tol'])
        self.force_cold = bool(cfg.get('force_cold_start', False))
        self.settings = C.cg2o_settings()
        # Frozen tuned cg2o settings (selected_parameters.json of a tuning study) for repeat flights.
        tuned = os.environ.get('PX4_NMPC_CHALLENGING_CG2O')
        if tuned:
            with open(tuned) as f:
                C.apply_dotted(self.settings, json.load(f))
        # Optional cold-start tuned values: cold solves of a warm flight (navigation start, recovery after an
        # unusable solve) use them on top of the warm settings; warm solves keep the warm settings.
        self.cold_settings = None
        tuned_cold = os.environ.get('PX4_NMPC_CHALLENGING_CG2O_COLD')
        if tuned_cold:
            with open(tuned_cold) as f:
                self.cold_settings = C.apply_dotted(copy.deepcopy(self.settings), json.load(f))
        self.previous = None
        self.session = None
        self._reset_session()
        trace = os.environ.get('PX4_NMPC_CHALLENGING_TRACE')
        self.trace = open(trace, 'a', buffering=1) if trace else None
        if self.trace:
            meta = dict(kind='metadata', course=str(cfg.get('challenging_course') or C.COURSE),
                        overrides=overrides, origin=self.origin.tolist(), backend=self.backend,
                        budget=self.budget, tol=self.tol, force_cold=self.force_cold, cg2o_settings=self.settings,
                        cg2o_cold_settings=self.cold_settings)
            self.trace.write(json.dumps(meta) + '\n')
        self.rows = 0
        self.backend_error = ''
        # NLP size (reported by the node): states, inputs and one slack per stage;
        # dynamics, state/input bounds and 5 body rows per obstacle per stage.
        n_obs = len(np.asarray(self.p.obstacles))
        self.nz = (self.ext.NX + self.ext.NU + 1) * self.N
        self.ng = (self.ext.NX + 9 + 10 + 5 * n_obs + 1) * self.N
        self.build_s = time.perf_counter() - start

    def _reset_session(self):
        if self.backend == 'cg2o':
            self.session = self.ext.CGSession()
            self.session.configure(self.settings)
            self._configured = self.settings
        else:
            self.session = self.ext.IPOPTSession()
            self.session.set_options(dict(hessian_approximation='exact'))

    def reset_warm_start(self):
        # Invalidate the stored iterate; keep the graph (a rebuild costs ~1.5 s).
        self.previous = None
        self.session.reset()

    def solve_ocp(self, x0, pref, vref, obstacles, d=None, uprev=None, yaw_ref=0., arena_lb=None,
                  arena_ub=None, shift=None, phase=None, **kwargs):
        x0 = np.asarray(x0, float)
        hall = C.measured_state(self.p, x0[:3] + self.origin, x0[3:6], x0[6:9], x0[9:12])
        u_prev = self.model.hover_input(d) if uprev is None else np.asarray(uprev, float)
        if u_prev.size != 5:
            u_prev = self.model.hover_input(d)
        b = C.make_problem(self.p, hall, u_prev)
        t_init = time.perf_counter()
        cold = b.cold_start()
        warm = (not self.force_cold) and self.previous is not None and shift is not None
        n = 0 if shift is None else max(0, min(self.N - 1, int(shift)))
        if self.backend == 'cg2o':
            initial = cold
        else:
            initial = C.shift_primal(b, self.previous, n) if warm else cold
        init_ms = (time.perf_counter() - t_init) * 1000.
        if self.backend == 'cg2o':
            # configure() only parses settings (no graph rebuild, stored iterate kept)
            wanted = self.cold_settings if (self.cold_settings is not None and not warm) else self.settings
            if wanted is not self._configured:
                self.session.configure(wanted)
                self._configured = wanted
            r = self.session.solve(b, initial, self.budget, self.tol, warm=warm, shift=n)
        else:
            r = self.session.solve(b, initial, self.budget, self.tol, warm=warm, shift=n, dual=True)
        z = np.asarray(r['z'])
        usable = bool(np.isfinite(z).all() and r['final_max_eq_residual'] <= 10 * self.tol
                      and r['final_max_ineq_violation'] <= 10 * self.tol)
        if usable:
            self.previous = z.copy()
        else:
            # Next solve starts cold. Invalidate the stored point instead of
            # rebuilding the graph (a rebuild costs ~1.3 s with ~2000 edges).
            self.previous = None
            self.session.reset()
        X = b.states(z)
        X[:3, :] -= self.origin[:, None]
        U = b.inputs(z)
        wall = init_ms + float(r['solve_time_ms'])
        penetration = max(0., -float(r['min_body_clearance_m']))
        row = dict(kind='problem', problem_index=self.rows, phase=phase, problem_sha256=C.problem_sha256(b),
                   cold_sha256=C.array_sha256(cold), x0_hall=hall.tolist(), uprev=u_prev.tolist(),
                   shift=-1 if shift is None else int(shift), warm=bool(warm), usable=usable,
                   iterations=int(r['iterations']), solve_ms=wall, init_ms=init_ms,
                   converged=bool(r['converged']), final_cost=float(r['final_cost']),
                   eq=float(r['final_max_eq_residual']), ineq=float(r['final_max_ineq_violation']),
                   min_body_clearance_m=float(r['min_body_clearance_m']), max_slack=float(r['max_slack']))
        self.rows += 1
        if self.trace:
            self.trace.write(json.dumps(row) + '\n')
        S = np.zeros((3, self.N))
        S[0] = b.slacks(z)
        return SolveResult(X=X, U=U, S=S, success=bool(r['converged']), usable=usable,
                           status='converged' if r['converged'] else str(r['status']),
                           iterations=int(r['iterations']), solve_ms=wall,
                           gap_residual=float(r['final_max_eq_residual']),
                           max_obstacle_slack=penetration if usable else float('inf'),
                           max_arena_slack=0. if usable else float('inf'),
                           max_dynamic_slack=0. if usable else float('inf'),
                           cost=float(r['final_cost']), extra=row)

    def solve(self, x0, xref, uprev):
        r = self.solve_ocp(x0, xref[:3], xref[3:6], None, uprev=uprev)
        return r.U, r.X, r.success, r.solve_ms, r.status

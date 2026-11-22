"""Closed-loop episode: NmpcRuntime + solver + PX4-like simulator.

The solver is executed for real (wall-clock time is measured) and its result
is delivered to the runtime ``latency_scale * wall_time`` later in simulated
time, which emulates an asynchronous worker thread on a machine that is
``latency_scale`` times slower (e.g. a laptop that also runs Gazebo + RViz).
"""
from __future__ import annotations

import time

import numpy as np

from ..config import build_configs
from ..constraint_activity import evaluate_constraint_activity
from ..ocp import make_ocp_solver
from ..runtime import NmpcRuntime
from .px4_like import Px4LikeQuad, SimParams


_SOLVER_CACHE = {}


def get_solver(solver_cfg):
    """Solvers only depend on structure/weights (obstacles are parameters)."""
    key = repr({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in sorted(solver_cfg.items())
                if k not in ('model', 'weights')}) + repr(solver_cfg['model']) + repr(solver_cfg['weights'])
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = make_ocp_solver(solver_cfg)
    s = _SOLVER_CACHE[key]
    s.reset_warm_start()
    return s


def run_episode(params, sim_params=None, latency_scale=1.0, t_max=60.0, seed=0, record=False,
                start_delay=0.5, sim_dt=0.004, odom_period=0.01, stream_period=0.02, success_radius=0.35,
                target_schedule=None, constraint_active_tolerance=1.0e-3, on_solve=None):
    """Closed-loop flight. ``on_solve(request, result, t, runtime)`` (optional) is called after every
    solve, e.g. to record the exact OCP sequence for paired solver comparisons (px4_nmpc_compare.trace)."""
    solver_cfg, runtime_cfg, p = build_configs(params)
    solver = get_solver(solver_cfg)
    rt = NmpcRuntime(runtime_cfg, solver)
    rng = np.random.default_rng(seed)
    sp = sim_params or SimParams()
    quad = Px4LikeQuad(sp, rng=rng, start=(0.0, 0.0, 0.0), yaw=0.0)

    obstacles = runtime_cfg['obstacles']
    arena_o = np.asarray(p['arena_origin_local_ned'], float)
    half = np.array([p['arena_north_m'] / 2.0, p['arena_east_m'] / 2.0])

    t = 0.0
    next_odom = 0.0
    next_stream = 0.0
    started = False
    arm_requested = False
    inflight = None
    outcome = 'timeout'
    t_goal = None
    min_phys_clear = float('inf')
    log = [] if record else None
    solve_ms = []
    wall_ms = []
    constraint_activity = []
    target_events = sorted(list(target_schedule or []), key=lambda x: float(x.get('time_s', 0.0)))
    next_target_event = 0
    target_updates_applied = 0
    stream_ticks = 0
    t_airborne = None
    while t < t_max:
        quad.step(sim_dt)
        t = quad.t
        while next_target_event < len(target_events) and t >= float(target_events[next_target_event].get('time_s', 0.0)):
            ev = target_events[next_target_event]
            goal = np.asarray(ev['goal_ned'], float).reshape(3)
            ok, msg = rt.set_goal(goal, t, goal_yaw=ev.get('goal_yaw'), resume=True)
            if not ok:
                outcome = 'target_update_rejected'
                break
            target_updates_applied += 1
            next_target_event += 1
        if outcome == 'target_update_rejected':
            break
        if t >= next_odom:
            next_odom += odom_period
            od = quad.odometry()
            if od is not None:
                rt.on_odometry(t, od[1], od[2], od[3], omega=od[4] if len(od) > 4 else None)
        if not started and t >= start_delay:
            ok, _ = rt.start(t)
            started = ok
        if t >= next_stream:
            next_stream += stream_period
            rt.set_vehicle_status(t, quad.armed, quad.offboard)
            sp_msg = rt.setpoint(t)
            if sp_msg is not None:
                quad.set_setpoint(sp_msg)
                stream_ticks += 1
            if started and not arm_requested and rt.phase == 'ARMING' and stream_ticks >= int(1.0 / stream_period):
                quad.request_offboard_and_arm()
                arm_requested = True
            if record:
                log.append((t, quad.pos.copy(), quad.vel.copy(), quad.rpy().copy(), rt.phase,
                            None if sp_msg is None else sp_msg.source))
        # asynchronous solver emulation
        if inflight is None:
            req = rt.make_request(t)
            if req is not None:
                w0 = time.perf_counter()
                res = rt.run_solver(req)
                wall = time.perf_counter() - w0
                solve_ms.append(float(res.solve_ms))
                wall_ms.append(wall * 1e3)
                if on_solve is not None:
                    on_solve(req, res, t, rt)
                act = evaluate_constraint_activity(res, req, rt, tolerance=constraint_active_tolerance)
                if act.get('constraint_activity_available'):
                    constraint_activity.append(act)
                inflight = (req, res, t + latency_scale * wall)
        elif t >= inflight[2]:
            rt.accept(inflight[0], inflight[1], t)
            inflight = None

        # safety bookkeeping (physical collision with the real airframe)
        pos = quad.pos
        if not quad.on_ground and t_airborne is None:
            t_airborne = t
        for c, r in obstacles:
            clear = float(np.linalg.norm(pos - c) - r - sp.phys_radius)
            min_phys_clear = min(min_phys_clear, clear)
            if clear < 0.0:
                outcome = 'collision'
        rel = pos[:2] - arena_o[:2]
        if np.any(np.abs(rel) > half - sp.phys_radius * 0.5) or pos[2] < arena_o[2] - p['arena_height_m'] + 0.1:
            outcome = 'wall_collision'
        if quad.failsafe:
            outcome = 'offboard_failsafe'
        if outcome not in ('timeout',):
            break
        if rt.phase == 'GOAL':
            if t_goal is None:
                t_goal = t
            if t - t_goal >= 3.0:
                # judged on the *true* position after the terminal hold
                ok = np.linalg.norm(quad.pos - runtime_cfg['goal']) < success_radius
                outcome = 'success' if ok else 'goal_hold_error'
                break
        elif rt.phase == 'ABORTED':
            outcome = 'aborted'
            break

    diag = rt.diagnostics(t)
    goal = runtime_cfg['goal']
    res = {
        'outcome': outcome,
        'success': outcome == 'success',
        't_end': t,
        'time_to_goal': (t_goal - t_airborne) if (t_goal is not None and t_airborne is not None) else float('nan'),
        'final_goal_distance': float(np.linalg.norm(quad.pos - goal)),
        'min_physical_clearance': min_phys_clear,
        'min_effective_clearance': diag['min_clearance_seen_m'],
        'solve_ms_median': float(np.median(solve_ms)) if solve_ms else float('nan'),
        'solve_ms_p95': float(np.percentile(solve_ms, 95)) if solve_ms else float('nan'),
        'solve_ms_max': float(np.max(solve_ms)) if solve_ms else float('nan'),
        'wall_ms_median': float(np.median(wall_ms)) if wall_ms else float('nan'),
        'wall_ms_p95': float(np.percentile(wall_ms, 95)) if wall_ms else float('nan'),
        'wall_ms_max': float(np.max(wall_ms)) if wall_ms else float('nan'),
        'iterations_median': float(np.median(rt.stats.iterations_hist)) if rt.stats.iterations_hist else float('nan'),
        'iterations_p95': float(np.percentile(rt.stats.iterations_hist, 95)) if rt.stats.iterations_hist else float('nan'),
        'iterations_max': int(np.max(rt.stats.iterations_hist)) if rt.stats.iterations_hist else -1,
        'solves': diag['solves'],
        'usable_rate': diag['usable_solves'] / max(1, diag['solves']),
        'converged_rate': diag['converged_solves'] / max(1, diag['solves']),
        'hold_events': diag['hold_events'],
        'rejected_unsafe': diag['rejected_unsafe'],
        'reseeds': diag['reseeds'],
        'disturbance_estimate': diag['disturbance_estimate'].tolist(),
        'phase': rt.phase,
        'target_updates_applied': target_updates_applied,
    }
    if constraint_activity:
        def _amedian(key):
            vals = [float(a[key]) for a in constraint_activity if key in a and np.isfinite(float(a[key]))]
            return float(np.median(vals)) if vals else float('nan')

        def _amax(key):
            vals = [float(a[key]) for a in constraint_activity if key in a and np.isfinite(float(a[key]))]
            return float(np.max(vals)) if vals else float('nan')

        def _amin(key):
            vals = [float(a[key]) for a in constraint_activity if key in a and np.isfinite(float(a[key]))]
            return float(np.min(vals)) if vals else float('nan')

        res.update({
            'active_constraint_tolerance': float(constraint_active_tolerance),
            'active_constraints_median': _amedian('active_constraints_total'),
            'active_constraints_max': _amax('active_constraints_total'),
            'active_constraint_fraction_median': _amedian('active_constraint_fraction'),
            'active_obstacle_constraints_median': _amedian('active_obstacle_constraints'),
            'active_obstacle_constraints_max': _amax('active_obstacle_constraints'),
            'solves_with_active_obstacle_fraction': float(np.mean([
                float(a.get('active_obstacle_constraints', 0)) > 0 for a in constraint_activity
            ])),
            'active_velocity_constraints_median': _amedian('active_velocity_constraints'),
            'active_tilt_constraints_median': _amedian('active_tilt_constraints'),
            'active_vertical_acc_constraints_median': _amedian('active_vertical_acc_constraints'),
            'active_yaw_rate_constraints_median': _amedian('active_yaw_rate_constraints'),
            'active_yaw_rate_constraints_max': _amax('active_yaw_rate_constraints'),
            'solves_with_active_yaw_rate_fraction': float(np.mean([
                float(a.get('active_yaw_rate_constraints', 0)) > 0 for a in constraint_activity
            ])),
            'active_input_bound_constraints_median': _amedian('active_input_bound_constraints'),
            'violated_constraints_max': _amax('violated_constraints_total'),
            'min_predicted_obstacle_margin_m': _amin('min_obstacle_margin_m'),
        })
    else:
        res.update({
            'active_constraint_tolerance': float(constraint_active_tolerance),
            'active_constraints_median': float('nan'),
            'active_constraints_max': float('nan'),
            'active_constraint_fraction_median': float('nan'),
            'active_obstacle_constraints_median': float('nan'),
            'active_obstacle_constraints_max': float('nan'),
            'solves_with_active_obstacle_fraction': float('nan'),
            'active_velocity_constraints_median': float('nan'),
            'active_tilt_constraints_median': float('nan'),
            'active_vertical_acc_constraints_median': float('nan'),
            'active_yaw_rate_constraints_median': float('nan'),
            'active_yaw_rate_constraints_max': float('nan'),
            'solves_with_active_yaw_rate_fraction': float('nan'),
            'active_input_bound_constraints_median': float('nan'),
            'violated_constraints_max': float('nan'),
            'min_predicted_obstacle_margin_m': float('nan'),
        })
    if record:
        res['log'] = log
        res['runtime'] = rt
    return res

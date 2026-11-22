"""Controller diagnostics shared by the monitor and RViz visualizer.

The diagnostic mapping contains mission state, solver timing/status, safety
clearance, control-source statistics, current goal/start, and PX4 state.
"""
from __future__ import annotations

import math

import numpy as np


def _f(x, fmt='{:.3f}', none='nan'):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return none
    if math.isnan(x):
        return 'nan'
    if math.isinf(x):
        return 'inf' if x > 0 else '-inf'
    return fmt.format(x)


def _vec(v):
    if v is None:
        return 'none'
    v = np.asarray(v, float).reshape(-1)
    return '[' + ', '.join(f'{x:.3f}' for x in v) + ']'


FLYING_PHASES = ('ARMING', 'TAKEOFF', 'NMPC', 'GOAL', 'LANDING')


def build_diagnostics(rt, t, params, odom_fresh=True, px4=None):
    """Return (level, message, warnings, values) for the current runtime state.

    ``level`` is 'OK', 'WARN' or 'ERROR'.  ``values`` maps key -> str.
    ``px4`` is an optional mapping with PX4-side facts reported by the ROS
    node (vehicle-status topic, nav_state, failsafe flags, ...).  It is merged
    into ``values`` and can add warnings/errors of its own.
    """
    d = rt.diagnostics(t)
    state = rt.state
    px4 = dict(px4 or {})
    if state is None:
        values = {'phase': d['phase'], 'mission_state': d['phase'],
                  'warnings': 'WAITING_FOR_ODOMETRY'}
        values.update({k: str(v) for k, v in px4.items() if not k.startswith('_')})
        return 'WARN', 'waiting for odometry', ['WAITING_FOR_ODOMETRY'], values

    pos, vel, rpy = state['pos'], state['vel'], state['rpy']
    warnings, errors = [], []
    if not odom_fresh:
        # Only an error while the controller is (or may be) flying the vehicle.
        (errors if d['phase'] in FLYING_PHASES else warnings).append('ODOM_STALE')
    errors.extend(px4.pop('_errors', []))
    warnings.extend(px4.pop('_warnings', []))
    ema = d['solve_ms_ema']
    if np.isfinite(ema) and ema * 1e-3 > float(params['solve_period_s']):
        warnings.append('SOLVE_TOO_SLOW_FOR_DT')
    holding = d['phase'] == 'NMPC' and d['applied_source'] == 'hold'
    if holding:
        warnings.append('POSITION_HOLD_FALLBACK')
    if d.get('min_safety_clearance_m', 1.0) < 0.0:
        errors.append('SAFETY_REGION_VIOLATED')
    if d['consecutive_failures'] > 0:
        warnings.append('LAST_SOLVE_UNUSABLE')
    if d['phase'] == 'ABORTED':
        errors.append('ABORTED:' + (d.get('abort_reason') or 'offboard lost').replace(' ', '_'))
    tilt_lim = math.radians(float(params['tilt_max_deg']))
    if abs(rpy[0]) > tilt_lim + 0.1 or abs(rpy[1]) > tilt_lim + 0.1:
        warnings.append('TILT_ABOVE_LIMIT')
    lb, ub = rt.arena_lb, rt.arena_ub
    if lb is not None and d['phase'] == 'NMPC' and (np.any(pos < lb - 0.05) or np.any(pos > ub + 0.05)):
        warnings.append('ARENA_SOFT_LIMIT')

    level = 'ERROR' if errors else ('WARN' if warnings else 'OK')
    all_msgs = errors + warnings
    message = ', '.join(all_msgs) if all_msgs else 'OK'

    nav_total = d['nmpc_control_s'] + d['hold_control_s']
    nmpc_pct = 100.0 * d['nmpc_control_s'] / nav_total if nav_total > 1e-9 else 0.0
    hold_pct = 100.0 * d['hold_control_s'] / nav_total if nav_total > 1e-9 else 0.0
    takeoff = d.get('takeoff_target')
    takeoff_err = float(np.linalg.norm(pos - takeoff)) if takeoff is not None else float('nan')
    dh = d['disturbance_estimate']
    accepted = d["usable_solves"]   # installed as the active plan
    values = {
        # --- controller/runtime keys ----------------------------------------
        'phase': d['phase'],
        'interface': d['interface'],
        'backend': d['backend'],
        'experiment_active': d['phase'] not in ('IDLE', 'ABORTED'),
        'done': d['phase'] in ('GOAL', 'LANDING'),
        'armed': bool(d.get('armed', False)),
        'offboard': bool(d.get('offboard', False)),
        'solver_status': d['solver_status'],
        'last_solver_success': d['consecutive_failures'] == 0 and d['solves'] > 0,
        'last_solve_ms': _f(d['last_solve_ms']),
        'solve_ms_ema': _f(ema),
        'solve_ms_max': _f(d['solve_ms_max']),
        'iterations': d['iterations'],
        'deadline_ratio': _f(d['deadline_ratio']),
        'control_dt_ms': _f(1000.0 * float(params['control_dt'])),
        'latency_compensation_s': _f(d['latency_compensation_s']),
        'prediction_age_s': _f(d['prediction_age_s']),
        'sequence_age_s': _f(d['plan_age_s']),
        'applied_source': d['applied_source'],
        'applied_stage': d['applied_stage'],
        'solves': d['solves'],
        'usable_solves': d['usable_solves'],
        'rejected_unsafe': d['rejected_unsafe'],
        'hold_events': d['hold_events'],
        'reseeds': d['reseeds'],
        'deadline_misses': d['deadline_misses'],
        'distance_to_goal_m': _f(d.get('distance_to_goal_m')),
        'speed_mps': _f(d.get('speed_mps')),
        'vN_mps': _f(vel[0]), 'vE_mps': _f(vel[1]), 'vD_mps': _f(vel[2]),
        'roll_deg': _f(math.degrees(rpy[0])),
        'pitch_deg': _f(math.degrees(rpy[1])),
        'yaw_deg': _f(math.degrees(rpy[2])),
        'yaw_mode': d.get('yaw_mode', 'fixed'),
        'yaw_cmd_deg': _f(d.get('yaw_cmd_deg')),
        'yaw_rate_cmd_dps': _f(d.get('yaw_rate_cmd_dps')),
        'tilt_limit_deg': _f(float(params['tilt_max_deg'])),
        'min_safety_clearance_m': _f(d.get('min_safety_clearance_m', float('inf'))),
        'min_clearance_seen_m': _f(d['min_clearance_seen_m']),
        'disturbance_estimate': f'[{dh[0]:.2f}, {dh[1]:.2f}, {dh[2]:.2f}]',
        'position_ned': _vec(pos),
        'mission_start_ned': _vec(d.get('mission_start')),
        'goal_ned': _vec(d.get('goal')),
        'leg_index': d.get('leg_index', 0),
        'goals_completed': d.get('goals_completed', 0),
        'takeoff_target_ned': _vec(takeoff),
        'hold_point_ned': _vec(d.get('hold_point')),
        # --- visualization/report aliases ---------------------------------
        'mission_state': 'NMPC_HOLD' if holding else d['phase'],
        'experiment_result': d['experiment_result'],
        'abort_reason': d.get('abort_reason', ''),
        'navigation_elapsed_s': _f(d['navigation_elapsed_s']),
        'last_solver_iterations': d['iterations'],
        'solve_total': d['solves'],
        'solve_accepted': accepted,
        'solve_failed': d['failed_solves'],
        'solve_rejected': d['rejected_unsafe'],
        'recovery_entries': d['hold_events'],
        'nmpc_control_percent': f'{nmpc_pct:.1f}',
        'recovery_control_percent': f'{hold_pct:.1f}',
        'takeoff_target_error_m': _f(takeoff_err),
        'safety_override_reason': 'none',
        'warnings': '|'.join(all_msgs) if all_msgs else 'none',
    }
    values.update(px4)
    values = {k: (v if isinstance(v, str) else str(v)) for k, v in values.items()}
    return level, message, all_msgs, values

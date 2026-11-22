"""Single source of truth for turning flat (ROS-style) parameters into the
nested configuration used by :class:`NmpcOcpSolver` and :class:`NmpcRuntime`.

The ROS node, the MAVLink runner, the offline simulator and the tests all
call :func:`build_configs`, so a parameter means the same thing everywhere.
"""
from __future__ import annotations

import math

import numpy as np


DEFAULTS = {
    # --- interface / solver -------------------------------------------------
    'control_interface': 'trajectory',     # trajectory | attitude | body_rate
    'solver_backend': 'ipopt_casadi',      # ipopt_casadi | ipopt_cpp | cg2o (legacy: ipopt | fatrop)
    'solver_allow_fallback': True,         # native backend unavailable -> use ipopt_casadi (false: raise; used by benchmarks)
    'prediction_model': 'linear',         # unchanged trajectory planner, or nonlinear thrust/body-rate prediction
    'nonlinear_profile': 'matched',        # unchanged formulation defaults, or old-envelope matched settings
    'nonlinear_backtracking_min': 0.03,
    'nonlinear_mass': 2.0643076923,
    # Additional NMPC-only obstacle reserve for direct-rate tracking error.
    # Runtime collision/effective-clearance metrics retain the declared safety margin.
    'nonlinear_tracking_margin': 0.0,
    'nonlinear_margin_adaptive': True,
    'nonlinear_recovery_buffer': 0.02,
    # prediction_model: challenging -> benchmark/model (course file defines the NLP)
    'challenging_course': '',                       # '' = experiments/challenging_course/course.yaml
    'challenging_vs_max': 2.5,                      # guide progress speed cap [m/s]
    'challenging_origin_ned': [-14.0, 0.0, -0.15],  # controller origin in the hall frame
    'control_dt': 0.10,
    'horizon_steps': 25,
    'max_obstacles': 8,
    'solver_max_iter': 150,
    'solver_tol': 1.0e-4,
    'solver_mu_init': 0.1,
    'solver_max_wall_time_s': 0.25,        # IPOPT only
    'gap_tolerance': 2.0e-3,
    # keep an obstacle in the same OCP slot while it stays among the K nearest (needed so that
    # warm-started multipliers belong to the same obstacle in consecutive solves)
    'stable_obstacle_slots': True,
    'gravity': 9.81,
    # --- timing / real-time ------------------------------------------------
    'solve_period_s': 0.10,
    'force_cold_start': False,            # evaluation mode: discard solver history on every MPC update
    'setpoint_stream_hz': 20.0,
    'diagnostics_hz': 5.0,
    'odom_timeout_s': 0.5,
    'plan_timeout_s': -1.0,                # <0: 1.5 s (trajectory) / 0.5 s (attitude, body_rate)
    'max_consecutive_failures': 5,
    'latency_margin_s': 0.01,
    'max_latency_compensation_s': 0.30,
    'reseed_position_error_m': 0.5,
    'reseed_velocity_error_mps': 0.8,
    'accept_obstacle_slack_m': 0.05,
    'offboard_loss_grace_s': 1.5,
    'disarm_loss_grace_s': 0.5,
    # --- arena -------------------------------------------------------------
    'enforce_arena_bounds': True,
    'arena_north_m': 14.0,
    'arena_east_m': 10.0,
    'arena_height_m': 5.0,
    'arena_state_margin_m': 0.40,
    'arena_floor_clearance_m': 0.30,
    'arena_origin_local_ned': [0.0, 0.0, 0.0],
    # --- mission -----------------------------------------------------------
    'goal_ned': [5.0, -2.0, -3.0],
    'goal_yaw': 0.0,                       # heading held at the goal [rad, NED, 0 = north]; NaN = keep the travel heading
    'goal_tolerance': 0.35,
    'goal_speed_tolerance': 0.30,
    'takeoff_height_m': 1.0,
    'takeoff_altitude_m': -1.0,            # >0: take-off altitude above the arena floor
    'goal_capture_radius_m': 1.0,          # dual-mode terminal set (0 disables)
    'goal_capture_speed_mps': 0.4,
    'auto_land_at_goal': False,
    # --- reference / guidance ------------------------------------------------
    'desired_speed': 1.0,
    'reference_accel_max': 0.6,
    'reference_decel_max': 0.8,
    'guidance_mode': 'auto',               # auto | planner | direct | fixed_line
    'fixed_line_start_ned': [math.nan, math.nan, math.nan],  # optional local-NED anchor; NaNs = capture first update
    'planner_enabled': True,
    'planner_resolution': 0.25,
    'planner_margin': 0.05,
    'planner_xy_min': [-6.5, -4.5],
    'planner_xy_max': [6.5, 4.5],
    'planner_replan_period_s': 1.0,
    'altitude_blend_fraction': 0.5,
    'altitude_blend_min_m': 2.0,
    # --- yaw (trajectory interface only) -------------------------------------
    # fixed: yaw is the constant `goal_yaw` (PX4 holds it).  optimized: the NMPC plans the heading together with the
    # path (states psi, omega, input yaw acceleration) and sends yaw + yawspeed to PX4.
    'yaw_mode': 'fixed',                   # fixed | optimized
    'yaw_rate_max': 0.8,                   # [rad/s] soft limit (PX4 MPC_YAWRAUTO_MAX default is 0.79)
    'yaw_acc_max': 2.0,                    # [rad/s^2] hard limit on the yaw acceleration input
    'w_yaw_travel': 2.0,                   # weight: nose along the planned direction of travel
    'w_yaw_goal': 3.0,                     # weight: nose to `goal_yaw` when near the goal
    'w_yaw_lateral': 1.0,                  # weight: sideways (body-frame) velocity^2, couples heading and path
    'w_yaw_rate': 0.05,
    'w_yaw_acc': 0.05,
    'yaw_goal_blend_m': 1.5,               # blend from travel heading to goal_yaw within this distance of the goal
    'yaw_min_speed_mps': 0.25,             # the travel heading is only trusted above this reference speed
    # --- vehicle limits ------------------------------------------------------
    'velocity_max': 3.0,
    'tilt_max_deg': 30.0,
    'thrust_accel_min': 2.0,
    'thrust_accel_max': 17.0,
    'body_rate_max': [2.5, 2.5, 1.5],
    'jerk_max': [12.0, 12.0, 8.0],
    'acc_up_max': 4.0,
    'acc_down_max': 3.0,
    'linear_drag': [0.0, 0.0, 0.0],
    'attitude_tau_s': 0.18,
    'attitude_gain': 1.0,
    # --- weights -------------------------------------------------------------
    'w_pos': [12.0, 12.0, 16.0],
    'w_vel': [2.0, 2.0, 3.0],
    'w_pos_terminal': [72.0, 72.0, 96.0],
    'w_vel_terminal': [8.0, 8.0, 12.0],
    'w_acc': 0.3,
    'w_jerk': 0.02,
    'w_tilt': 1.0,
    'w_yaw': 2.0,
    'w_attitude_input': [2.0, 2.0, 0.05],
    'w_rate_input': [0.05, 0.3, 0.3, 0.3],
    'w_delta_input_first': [0.0],
    'nonlinear_rate_slew_weight': -1.0,  # <0 keeps the matched native default; >=0 overrides wdu[1:4]
    'w_slack_obstacle_l1': 2000.0,
    'w_slack_obstacle_l2': 4000.0,
    'w_slack_arena_l1': 1000.0,
    'w_slack_arena_l2': 2000.0,
    'w_slack_dynamic_l1': 200.0,
    'w_slack_dynamic_l2': 400.0,
    'w_terminal_penalty': 4000.0,
    # --- obstacles -----------------------------------------------------------
    'uav_radius': 0.35,
    'safety_margin': 0.45,
    'obstacle_centers_flat': [3.8, 0.0, -3.0, 6.4, 1.2, -3.0, 8.0, -1.0, -3.0],
    'obstacle_radii': [0.75, 0.65, 0.60],
    # --- thrust mapping / offset-free ----------------------------------------
    'hover_throttle': 0.729,               # gz_x500 real hover throttle (MPC_THR_HOVER=0.6 is only PX4's first guess)
    'thrust_model': 'quadratic_esc',       # quadratic_esc (gz ESC 150..1000 rad/s) | linear
    'esc_omega_min': 150.0,
    'esc_omega_max': 1000.0,
    'disturbance_observer': True,
    'observer_omega': 1.5,
    'observer_zeta': 0.9,
}


def _vec(v, n, name):
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 1 and n > 1:
        a = np.full(n, float(a[0]))
    if a.shape != (n,):
        raise ValueError(f'{name} must contain {n} values')
    return a


def arena_bounds(p):
    """Soft NMPC position bounds [lb, ub] in PX4-local NED (or (None, None))."""
    if not bool(p['enforce_arena_bounds']):
        return None, None
    o = _vec(p['arena_origin_local_ned'], 3, 'arena_origin_local_ned')
    m = float(p['arena_state_margin_m'])
    lb = np.array([o[0] - p['arena_north_m'] / 2 + m, o[1] - p['arena_east_m'] / 2 + m, o[2] - p['arena_height_m'] + m])
    ub = np.array([o[0] + p['arena_north_m'] / 2 - m, o[1] + p['arena_east_m'] / 2 - m, o[2] - float(p['arena_floor_clearance_m'])])
    if np.any(lb >= ub):
        raise ValueError('arena bounds are empty after margins')
    return lb, ub


def build_configs(params):
    """Return (solver_cfg, runtime_cfg, p) from a flat parameter mapping."""
    p = dict(DEFAULTS)
    p.update({k: v for k, v in dict(params).items() if v is not None})
    interface = str(p['control_interface'])
    if interface not in ('trajectory', 'attitude', 'body_rate'):
        raise ValueError('control_interface must be trajectory, attitude or body_rate')

    flat = np.asarray(p['obstacle_centers_flat'], dtype=float).reshape(-1)
    radii = np.asarray(p['obstacle_radii'], dtype=float).reshape(-1)
    if flat.size % 3 != 0 or radii.size != flat.size // 3:
        raise ValueError('obstacle_centers_flat and obstacle_radii sizes disagree')
    if np.any(radii <= 0.0):
        raise ValueError('all obstacle radii must be positive')
    obstacles = [(flat[3 * i:3 * i + 3].copy(), float(radii[i])) for i in range(radii.size)]

    g = float(p['gravity'])
    tilt = math.radians(float(p['tilt_max_deg']))
    if not 0.0 < tilt < math.radians(60.0):
        raise ValueError('tilt_max_deg must be in (0, 60)')
    model_cfg = {
        'g': g,
        'vmax': float(p['velocity_max']),
        'tilt_max': tilt,
        'drag': _vec(p['linear_drag'], 3, 'linear_drag'),
        'T_min': float(p['thrust_accel_min']),
        'T_max': float(p['thrust_accel_max']),
        'jerk_max': _vec(p['jerk_max'], 3, 'jerk_max'),
        'acc_up_max': float(p['acc_up_max']),
        'acc_down_max': float(p['acc_down_max']),
        'tau_roll': float(p['attitude_tau_s']),
        'tau_pitch': float(p['attitude_tau_s']),
        'k_roll': float(p['attitude_gain']),
        'k_pitch': float(p['attitude_gain']),
        'rate_max': _vec(p['body_rate_max'], 3, 'body_rate_max'),
    }
    yaw_mode = str(p['yaw_mode']).lower()
    if yaw_mode not in ('fixed', 'optimized'):
        raise ValueError('yaw_mode must be fixed or optimized')
    yaw_opt = yaw_mode == 'optimized'
    if yaw_opt and interface != 'trajectory' and str(p['prediction_model']) not in ('nonlinear', 'challenging'):
        raise ValueError("yaw_mode: optimized needs control_interface: trajectory "
                         "(attitude/body_rate hold the yaw at goal_yaw)")
    if yaw_opt:
        model_cfg['yaw'] = {
            'enabled': True,
            'rate_max': float(p['yaw_rate_max']),
            'acc_max': float(p['yaw_acc_max']),
            'w_lateral': float(p['w_yaw_lateral']),
            'w_rate': float(p['w_yaw_rate']),
            'w_acc': float(p['w_yaw_acc']),
        }
    if not yaw_opt and not math.isfinite(float(p['goal_yaw'])):
        raise ValueError('goal_yaw: NaN (keep the travel heading) needs yaw_mode: optimized')
    nu = {'trajectory': 4 if yaw_opt else 3, 'attitude': 3, 'body_rate': 4}[interface]
    weights = {
        'pos': _vec(p['w_pos'], 3, 'w_pos'),
        'vel': _vec(p['w_vel'], 3, 'w_vel'),
        'pos_terminal': _vec(p['w_pos_terminal'], 3, 'w_pos_terminal'),
        'vel_terminal': _vec(p['w_vel_terminal'], 3, 'w_vel_terminal'),
        'acc': float(p['w_acc']),
        'jerk': float(p['w_jerk']),
        'tilt': float(p['w_tilt']),
        'yaw': float(p['w_yaw']),
        'attitude_input': _vec(p['w_attitude_input'], 3, 'w_attitude_input'),
        'rate_input': _vec(p['w_rate_input'], 4, 'w_rate_input'),
        'delta_input_first': _vec(p['w_delta_input_first'], nu, 'w_delta_input_first'),
        'slack_obstacle_l1': float(p['w_slack_obstacle_l1']),
        'slack_obstacle_l2': float(p['w_slack_obstacle_l2']),
        'slack_arena_l1': float(p['w_slack_arena_l1']),
        'slack_arena_l2': float(p['w_slack_arena_l2']),
        'slack_dynamic_l1': float(p['w_slack_dynamic_l1']),
        'slack_dynamic_l2': float(p['w_slack_dynamic_l2']),
        'terminal_penalty': float(p['w_terminal_penalty']),
    }
    solver_cfg = {
        'prediction_model': str(p['prediction_model']),
        'nonlinear_profile': str(p['nonlinear_profile']),
        'nonlinear_backtracking_min': float(p['nonlinear_backtracking_min']),
        'nonlinear_mass': float(p['nonlinear_mass']),
        'nonlinear_tracking_margin': float(p['nonlinear_tracking_margin']),
        'nonlinear_margin_adaptive': bool(p['nonlinear_margin_adaptive']),
        'nonlinear_recovery_buffer': float(p['nonlinear_recovery_buffer']),
        'nonlinear_rate_slew_weight': float(p['nonlinear_rate_slew_weight']),
        'challenging_course': str(p['challenging_course']),
        'challenging_vs_max': float(p['challenging_vs_max']),
        'challenging_origin_ned': [float(v) for v in p['challenging_origin_ned']],
        'interface': interface,
        'backend': str(p['solver_backend']),
        'allow_fallback': bool(p['solver_allow_fallback']),
        'N': int(p['horizon_steps']),
        'dt': float(p['control_dt']),
        'max_obstacles': int(p['max_obstacles']),
        'max_iter': int(p['solver_max_iter']),
        'tol': float(p['solver_tol']),
        'mu_init': float(p['solver_mu_init']),
        'max_wall_time_s': float(p['solver_max_wall_time_s']),
        'gap_tolerance': float(p['gap_tolerance']),
        'model': model_cfg,
        'weights': weights,
    }
    # programmatic (non-ROS) overrides used by experiments and tuning; dicts cannot be ROS parameters
    for key in ('cg2o_overrides', 'ipopt_cpp_overrides', 'ipopt_options', 'cg2o_max_iterations',
                'cg2o_config_path', 'ipopt_cpp_config_path', 'ipopt_warm_duals'):
        if p.get(key) is not None:
            solver_cfg[key] = p[key]

    lb, ub = arena_bounds(p)
    takeoff_height = float(p['takeoff_height_m'])
    if float(p.get('takeoff_altitude_m', -1.0)) > 0.0:
        # altitude above the floor -> climb above the start (start is origin_D above the floor)
        floor_below_start = float(_vec(p['arena_origin_local_ned'], 3, 'arena_origin_local_ned')[2])
        takeoff_height = max(0.3, float(p['takeoff_altitude_m']) - floor_below_start)
    uav_r, margin = float(p['uav_radius']), float(p['safety_margin'])
    plan_timeout = float(p['plan_timeout_s'])
    if plan_timeout <= 0.0:
        plan_timeout = 1.5 if interface == 'trajectory' else 0.5
    fixed_line_start = _vec(p['fixed_line_start_ned'], 3, 'fixed_line_start_ned')
    if np.all(np.isnan(fixed_line_start)):
        fixed_line_start = None
    elif not np.all(np.isfinite(fixed_line_start)):
        raise ValueError('fixed_line_start_ned must contain either 3 finite values or 3 NaNs')

    runtime_cfg = {
        'g': g,
        'goal': _vec(p['goal_ned'], 3, 'goal_ned'),
        'goal_yaw': float(p['goal_yaw']),
        'yaw': {
            'enabled': yaw_opt,
            'w_travel': float(p['w_yaw_travel']),
            'w_goal': float(p['w_yaw_goal']),
            'goal_blend_m': float(p['yaw_goal_blend_m']),
            'min_speed_mps': float(p['yaw_min_speed_mps']),
        },
        'obstacles': obstacles,
        'stable_obstacle_slots': bool(p['stable_obstacle_slots']),
        'uav_radius': uav_r,
        'safety_margin': margin,
        'arena_lb': lb,
        'arena_ub': ub,
        'guidance': {
            'mode': str(p['guidance_mode']),
            'fixed_line_start_ned': (None if fixed_line_start is None else fixed_line_start.copy()),
            'desired_speed': float(p['desired_speed']),
            'reference_accel_max': float(p['reference_accel_max']),
            'reference_decel_max': float(p['reference_decel_max']),
            'planner_enabled': bool(p['planner_enabled']),
            'planner_resolution': float(p['planner_resolution']),
            'planner_xy_min': _vec(p['planner_xy_min'], 2, 'planner_xy_min'),
            'planner_xy_max': _vec(p['planner_xy_max'], 2, 'planner_xy_max'),
            'planner_inflation': uav_r + margin + float(p['planner_margin']),
            'planner_replan_period_s': float(p['planner_replan_period_s']),
            'altitude_blend_fraction': float(p['altitude_blend_fraction']),
            'altitude_blend_min_m': float(p['altitude_blend_min_m']),
        },
        'thrust_map': {
            'g': g,
            'hover_throttle': float(p['hover_throttle']),
            'model': str(p['thrust_model']),
            'omega_min': float(p['esc_omega_min']),
            'omega_max': float(p['esc_omega_max']),
        },
        'observer': {
            'enabled': bool(p['disturbance_observer']) and interface != 'trajectory',
            'omega': float(p['observer_omega']),
            'zeta': float(p['observer_zeta']),
        },
        'timing': {
            'solve_period_s': float(p['solve_period_s']),
            'force_cold_start': bool(p['force_cold_start']),
            'latency_margin_s': float(p['latency_margin_s']),
            'max_latency_compensation_s': float(p['max_latency_compensation_s']),
            'plan_timeout_s': plan_timeout,
            'max_consecutive_failures': int(p['max_consecutive_failures']),
            'reseed_position_error_m': float(p['reseed_position_error_m']),
            'reseed_velocity_error_mps': float(p['reseed_velocity_error_mps']),
            'accept_obstacle_slack_m': float(p['accept_obstacle_slack_m']),
            'offboard_loss_grace_s': float(p['offboard_loss_grace_s']),
            'disarm_loss_grace_s': float(p['disarm_loss_grace_s']),
        },
        'mission': {
            'goal_tolerance': float(p['goal_tolerance']),
            'goal_speed_tolerance': float(p['goal_speed_tolerance']),
            'takeoff_height_m': takeoff_height,
            'goal_capture_radius_m': float(p['goal_capture_radius_m']),
            'goal_capture_speed_mps': float(p['goal_capture_speed_mps']),
            'auto_land_at_goal': bool(p['auto_land_at_goal']),
            'ground_down_m': None,
        },
    }
    if lb is not None:
        goal = runtime_cfg['goal']
        if np.any(goal < lb - 1e-6) or np.any(goal > ub + 1e-6):
            raise ValueError(f'goal_ned {goal.tolist()} lies outside the arena bounds (after margins)')
    return solver_cfg, runtime_cfg, p


def build_runtime(params):
    """Convenience: construct (solver, runtime, flat_params)."""
    from .ocp import make_ocp_solver
    from .runtime import NmpcRuntime
    solver_cfg, runtime_cfg, p = build_configs(params)
    solver = make_ocp_solver(solver_cfg)
    return solver, NmpcRuntime(runtime_cfg, solver), p

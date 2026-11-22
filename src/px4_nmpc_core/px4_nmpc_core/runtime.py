"""ROS-free NMPC runtime: timing, plan management, fallbacks, state machine.

The same class is driven by

* the ROS 2 node ``px4_nmpc_controller`` (PX4 over uXRCE-DDS),
* the MAVLink runner ``nmpc_mavlink_runner`` (PX4 SITL without ROS 2), and
* the offline PX4-like simulator used by the Monte-Carlo benchmark,

so every closed-loop test exercises exactly the code that flies.

Real-time design:

* The solver is *never* in the setpoint path.  Setpoints are produced at the
  stream rate by interpolating the most recent accepted plan.
* **Delay compensation** (Findeisen & Allgoewer 2004): each OCP is started
  from the state predicted at ``t_request + expected_latency``, so the plan's
  stage 0 lines up with the time at which it becomes available.
* Trajectory interface: the plan is re-seeded from its own predicted state
  while PX4 tracks it well (no feedback fight between MPC and PX4 loops) and
  from the measurement when tracking error exceeds a threshold.
* **Fallback = PX4 position hold** (not attitude levelling).  If no usable
  plan is available the vehicle is held at a fixed point by PX4's own
  position controller, so it cannot drift out of the arena while the NMPC
  recovers.
* **Terminal (dual-mode) controller** at the goal: once inside the goal
  tolerance the goal is held by PX4 position control.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from .estimator import DisturbanceObserver, ThrustMap
from .guidance import (
    GuidanceReference,
    StableObstacleSlots,
    effective_obstacles,
    min_clearance,
    safe_takeoff_height,
    select_obstacles,
)
from .math_utils import euler_to_quat_wxyz, wrap_pi


PHASES = ('IDLE', 'ARMING', 'TAKEOFF', 'NMPC', 'GOAL', 'LANDING', 'ABORTED')


@dataclass
class Setpoint:
    kind: str                       # 'position' | 'trajectory' | 'attitude' | 'body_rate'
    source: str                     # 'arming' | 'takeoff' | 'nmpc' | 'hold' | 'goal'
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    yaw: float = float('nan')
    q_d: np.ndarray | None = None   # Hamilton (w, x, y, z), body FRD -> NED
    rates: np.ndarray | None = None  # body FRD [rad/s]
    thrust_accel: float = float('nan')
    thrust_normalized: float = float('nan')
    stage: int = -1
    yawspeed: float = float('nan')   # [rad/s] feed-forward for ``yaw`` (trajectory interface, optimised yaw)


@dataclass
class Plan:
    t0: float
    dt: float
    X: np.ndarray
    U: np.ndarray
    pref: np.ndarray
    installed_t: float
    solve_ms: float
    from_measurement: bool
    prediction_model: object = None   # nonlinear interpolation; None preserves the original jerk trajectory

    @property
    def t_end(self):
        return self.t0 + self.dt * self.U.shape[1]

    def stage(self, t):
        k = int(math.floor((t - self.t0) / self.dt + 1e-9))
        return int(np.clip(k, 0, self.U.shape[1] - 1)), float(np.clip(t - self.t0 - k * self.dt, 0.0, self.dt))


@dataclass
class SolveRequest:
    t_request: float
    t0: float
    x0: np.ndarray
    pref: np.ndarray
    vref: np.ndarray
    obstacles: np.ndarray
    d: np.ndarray
    uprev: np.ndarray
    yaw_ref: float
    shift: int | None
    from_measurement: bool
    phase: str
    hdg_ref: np.ndarray | None = None    # heading reference per stage (optimised yaw only)
    hdg_w: np.ndarray | None = None      # its weight per stage


@dataclass
class RuntimeStats:
    solves: int = 0
    usable: int = 0
    converged: int = 0
    rejected_unsafe: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    deadline_misses: int = 0
    solve_ms_last: float = float('nan')
    solve_ms_ema: float = float('nan')
    solve_ms_max: float = 0.0
    iterations_last: int = -1
    status_last: str = 'waiting for first solve'
    hold_events: int = 0
    reseeds: int = 0
    solve_ms_hist: list = field(default_factory=list)
    iterations_hist: list = field(default_factory=list)


class NmpcRuntime:
    def __init__(self, cfg, solver):
        self.cfg = cfg
        self.solver = solver
        self.interface = solver.interface
        self.model = solver.model
        self.nonlinear_trajectory = bool(getattr(self.model, 'nonlinear_trajectory', False))
        self.N = solver.N
        self.dt = solver.dt
        self.g = float(cfg.get('g', 9.81))
        self.lock = threading.RLock()

        self.goal = np.asarray(cfg['goal'], float).reshape(3)
        gy = float(cfg.get('goal_yaw', 0.0))
        self.goal_yaw = wrap_pi(gy) if np.isfinite(gy) else float('nan')   # NaN: no goal heading (optimised yaw only)
        ycfg = dict(cfg.get('yaw', {}))
        self.yaw_opt = bool(getattr(self.model, 'has_yaw', False))
        self.yaw_w_travel = float(ycfg.get('w_travel', 2.0))
        self.yaw_w_goal = float(ycfg.get('w_goal', 3.0))
        self.yaw_blend_m = max(1e-3, float(ycfg.get('goal_blend_m', 1.5)))
        self.yaw_min_speed = max(1e-3, float(ycfg.get('min_speed_mps', 0.25)))
        self.obstacles_phys = [(np.asarray(c, float).reshape(3), float(r)) for c, r in cfg['obstacles']]
        self.eff_obs = effective_obstacles(self.obstacles_phys, cfg['uav_radius'], cfg['safety_margin'])
        # stable slot assignment of the K nearest obstacles (see guidance.StableObstacleSlots)
        self.obstacle_slots = (StableObstacleSlots(self.solver.K)
                               if bool(cfg.get('stable_obstacle_slots', True)) else None)
        self.arena_lb = np.asarray(cfg['arena_lb'], float) if cfg.get('arena_lb') is not None else None
        self.arena_ub = np.asarray(cfg['arena_ub'], float) if cfg.get('arena_ub') is not None else None

        self._guidance_cfg = dict(cfg['guidance'])
        self.guidance = self._new_guidance()

        self.thrust_map = ThrustMap(**cfg.get('thrust_map', {}))
        ocfg = dict(cfg.get('observer', {}))
        self.observer_enabled = bool(ocfg.pop('enabled', self.interface != 'trajectory'))
        self.observer = DisturbanceObserver(g=self.g, drag=self.model.drag, **ocfg)

        t = cfg['timing']
        self.solve_period = float(t.get('solve_period_s', self.dt))
        self.force_cold_start = bool(t.get('force_cold_start', False))
        self.latency_margin = float(t.get('latency_margin_s', 0.01))
        self.latency_max = float(t.get('max_latency_compensation_s', 0.3))
        self.plan_timeout = float(t.get('plan_timeout_s', 0.5 if self.interface != 'trajectory' else 1.5))
        self.max_failures = int(t.get('max_consecutive_failures', 5))
        self.reseed_pos = float(t.get('reseed_position_error_m', 0.5))
        self.reseed_vel = float(t.get('reseed_velocity_error_mps', 0.8))
        self.reseed_yaw = float(t.get('reseed_yaw_error_rad', 0.6))
        self.slack_tol = float(t.get('accept_obstacle_slack_m', 0.05))
        self.offboard_loss_grace = max(0.0, float(t.get('offboard_loss_grace_s', 1.5)))
        self.disarm_loss_grace = max(0.0, float(t.get('disarm_loss_grace_s', 0.5)))

        m = cfg['mission']
        self.goal_tol = float(m.get('goal_tolerance', 0.35))
        self.goal_speed_tol = float(m.get('goal_speed_tolerance', 0.30))
        self.takeoff_height = float(m.get('takeoff_height_m', 1.0))
        self.takeoff_tol = float(m.get('takeoff_tolerance_m', 0.25))
        # The first OCP of the navigation phase is solved cold (from the reference rollout), not warm-started from
        # the plans pre-solved during the take-off. A slow climb (PX4) would otherwise carry a plan made near the
        # ground (e.g. passing under every obstacle) into the flight.
        self.cold_start_at_navigation = bool(m.get('cold_start_at_navigation', True))
        self._cold_next = False
        self.airborne_height = float(m.get('airborne_height_m', 0.4))
        self.auto_land = bool(m.get('auto_land_at_goal', False))
        self.capture_radius = float(m.get('goal_capture_radius_m', 1.0))
        self.capture_speed = float(m.get('goal_capture_speed_mps', 0.4))
        self.ground_d = m.get('ground_down_m')  # local D of the floor (None = PX4 local origin, D = 0)

        self.reset()

    def _new_guidance(self):
        gcfg = dict(self._guidance_cfg)
        gcfg.update(goal=self.goal, N=self.N, dt=self.dt, obstacles=self.obstacles_phys)
        return GuidanceReference(gcfg)

    def _validate_goal(self, goal):
        g = np.asarray(goal, float).reshape(3)
        if not np.all(np.isfinite(g)):
            return False, 'goal contains non-finite values'
        if self.arena_lb is not None and (np.any(g < self.arena_lb - 1e-6) or np.any(g > self.arena_ub + 1e-6)):
            return False, f'goal {g.tolist()} lies outside the configured arena bounds'
        clearance = min_clearance(self.eff_obs, g)
        if clearance < 0.0:
            return False, f'goal lies inside an obstacle safety shell (clearance {clearance:.3f} m)'
        return True, 'ok'

    def _reset_plan_for_new_leg(self, t, resume):
        self.plan = None
        self.pending = None
        self.last_request_t = -float('inf')
        self.hold_point = None
        # Terminal PX4 hold point is intentionally independent of self.goal.
        # This lets a new environment/goal be staged while the vehicle keeps
        # holding the completed leg until the physical Gazebo scene is ready.
        if resume:
            self.goal_hold_point = None
            self.t_goal = None
        self.guidance.reset()
        if self.obstacle_slots is not None:
            self.obstacle_slots.reset()
        if hasattr(self.solver, 'reset_warm_start'):
            self.solver.reset_warm_start()
        if self.state is not None:
            self.mission_start = self.state['pos'].copy()
            self.min_clearance_seen = min_clearance(self.eff_obs, self.state['pos'])
        if resume and self.phase in ('NMPC', 'GOAL'):
            self.phase = 'NMPC'
            self.t_phase = float(t)
            self.t_nav_start = float(t)
            self.leg_index += 1

    def set_goal(self, goal, t, goal_yaw=None, resume=True):
        """Change the navigation target without restarting PX4/Gazebo.

        If the vehicle is currently holding a completed goal, a valid new goal
        immediately starts a new NMPC leg from the measured current position.
        During ARMING/TAKEOFF the new target is simply used once navigation starts.
        A goal is deliberately rejected during LANDING: accepting it there would
        make the command appear successful while PX4 continues the land mode.
        """
        with self.lock:
            if self.phase == 'LANDING':
                return False, 'cannot accept a new goal while landing; wait for IDLE after touchdown'
            g = np.asarray(goal, float).reshape(3)
            ok, msg = self._validate_goal(g)
            if not ok:
                return False, msg
            self.goal = g.copy()
            if goal_yaw is not None and np.isfinite(float(goal_yaw)):
                self.goal_yaw = wrap_pi(float(goal_yaw))
            self.guidance.goal = self.goal.copy()
            self.guidance.reset()
            active_resume = bool(resume and self.phase in ('NMPC', 'GOAL'))
            self._reset_plan_for_new_leg(t, active_resume)
            return True, ('new goal accepted; NMPC resumed' if active_resume else 'new goal accepted')

    def set_environment(self, obstacles, arena_lb=None, arena_ub=None, guidance_overrides=None,
                        goal=None, t=0.0, resume=True):
        """Atomically replace obstacle / arena geometry for a new mission leg.

        The NLP structure is unchanged: obstacles and arena bounds are already
        solver parameters, so no CasADi/FATROP rebuild is required.
        """
        with self.lock:
            new_obs = [(np.asarray(c, float).reshape(3), float(r)) for c, r in obstacles]
            if any((not np.all(np.isfinite(c)) or not np.isfinite(r) or r <= 0.0) for c, r in new_obs):
                return False, 'environment contains an invalid obstacle'
            new_lb = None if arena_lb is None else np.asarray(arena_lb, float).reshape(3)
            new_ub = None if arena_ub is None else np.asarray(arena_ub, float).reshape(3)
            if (new_lb is None) != (new_ub is None):
                return False, 'arena_lb and arena_ub must both be set or both be None'
            if new_lb is not None and (np.any(~np.isfinite(new_lb)) or np.any(~np.isfinite(new_ub)) or np.any(new_lb >= new_ub)):
                return False, 'invalid arena bounds'

            old = (self.obstacles_phys, self.eff_obs, self.arena_lb, self.arena_ub, dict(self._guidance_cfg), self.goal.copy())
            self.obstacles_phys = new_obs
            self.eff_obs = effective_obstacles(new_obs, self.cfg['uav_radius'], self.cfg['safety_margin'])
            self.arena_lb, self.arena_ub = new_lb, new_ub
            if guidance_overrides:
                self._guidance_cfg.update(dict(guidance_overrides))
            if goal is not None:
                self.goal = np.asarray(goal, float).reshape(3)
            ok, msg = self._validate_goal(self.goal)
            if self.state is not None and min_clearance(self.eff_obs, self.state['pos']) < 0.0:
                ok, msg = False, 'current vehicle position lies inside a new obstacle safety shell'
            if not ok:
                self.obstacles_phys, self.eff_obs, self.arena_lb, self.arena_ub, self._guidance_cfg, self.goal = old
                self.guidance = self._new_guidance()
                return False, msg
            self.guidance = self._new_guidance()
            active_resume = bool(resume and self.phase in ('NMPC', 'GOAL'))
            self._reset_plan_for_new_leg(t, active_resume)
            return True, ('environment loaded; NMPC resumed' if active_resume else 'environment loaded')

    # ------------------------------------------------------------ lifecycle
    def reset(self):
        with self.lock:
            self.phase = 'IDLE'
            self.t_phase = 0.0
            self.state = None           # dict(t, pos, vel, rpy)
            self.plan = None
            self.pending = None
            self.last_request_t = -float('inf')
            self.stats = RuntimeStats()
            self.hold_point = None
            self.goal_hold_point = None
            self.start_pos = None
            self.mission_start = None
            self.takeoff_target = None
            self.applied_u = None
            self.applied_thrust_accel = self.g
            self.armed = False
            self.offboard = False
            self.t_start = None
            self.t_goal = None
            self.leg_index = 0
            self.goals_completed = 0
            self.last_setpoint = None
            self.guidance.reset()
            self.observer.reset()
            if getattr(self, 'obstacle_slots', None) is not None:
                self.obstacle_slots.reset()
            if hasattr(self.solver, 'reset_warm_start'):
                self.solver.reset_warm_start()
            self.min_clearance_seen = float('inf')
            # experiment bookkeeping for the diagnostics / visualization tools
            self.t_nav_start = None
            self.t_last_setpoint = None
            self.nmpc_control_s = 0.0
            self.hold_control_s = 0.0
            self.abort_reason = ''
            self.offboard_lost_since = None
            self.disarmed_since = None
            self.px4_failsafe = False
            self.px4_mode = ''
            self.yaw_last_raw = None      # last trusted travel heading (carried through slow segments)
            self.hold_yaw = None          # heading kept while PX4 holds a point
            self.goal_hold_yaw = None     # heading kept at the goal when goal_yaw is NaN
            self.takeoff_yaw = None

    def start(self, t):
        with self.lock:
            if self.state is None:
                return False, 'no odometry yet'
            if self.phase not in ('IDLE', 'ABORTED'):
                return False, (
                    f'cannot start from phase {self.phase}; use a live goal after GOAL, or stop/land first')
            self.reset_keep_state()
            self.phase = 'ARMING'
            self.t_phase = t
            self.t_start = t
            self.start_pos = self.state['pos'].copy()
            self.mission_start = self.start_pos.copy()
            self.leg_index = 1
            return True, 'started'

    def reset_keep_state(self):
        st = self.state
        px4 = (self.armed, self.offboard, self.px4_failsafe, self.px4_mode)
        self.reset()
        self.state = st
        # PX4's arming/mode state is a fact about the vehicle, not controller state.
        self.armed, self.offboard, self.px4_failsafe, self.px4_mode = px4

    def abort(self, t, reason):
        """Enter ABORTED (no further setpoints are generated)."""
        with self.lock:
            self.abort_reason = str(reason)
            self.phase = 'ABORTED'
            self.t_phase = float(t)
            self.plan = None
            self.pending = None

    def stop(self, t):
        """Stop controller activity for an already-grounded/non-flying vehicle."""
        with self.lock:
            self.phase = 'IDLE'
            self.t_phase = t
            self.plan = None
            self.pending = None

    def begin_landing(self, t):
        """Enter LANDING while PX4 performs the physical landing.

        Keeping a distinct LANDING phase prevents mission/environment commands
        from treating an aircraft that is still descending as safely IDLE.
        """
        with self.lock:
            self.phase = 'LANDING'
            self.t_phase = float(t)
            self.plan = None
            self.pending = None
            if self.goal_hold_point is None and self.state is not None:
                self.goal_hold_point = self.state['pos'].copy()

    def set_vehicle_status(self, t, armed, offboard, failsafe=False, mode=''):
        """Update PX4 mode state.

        ``mode`` is PX4's current navigation state name (for messages only).
        A PX4 failsafe that takes the vehicle out of OFFBOARD aborts the
        experiment immediately: the controller must never fight PX4's own
        safety logic by re-requesting OFFBOARD.  Any other loss of OFFBOARD
        or arming (e.g. the pilot switched mode in QGroundControl) aborts after
        a short grace period that absorbs out-of-order status samples.
        """
        with self.lock:
            t = float(t)
            self.armed = bool(armed)
            self.offboard = bool(offboard)
            self.px4_failsafe = bool(failsafe)
            self.px4_mode = str(mode or '')
            active = self.phase in ('TAKEOFF', 'NMPC', 'GOAL')

            if self.armed:
                self.disarmed_since = None
            elif active and self.disarmed_since is None:
                self.disarmed_since = t

            if self.offboard:
                self.offboard_lost_since = None
            elif active and self.offboard_lost_since is None:
                self.offboard_lost_since = t

            if self.phase == 'ARMING' and self.armed and self.offboard:
                self.disarmed_since = None
                self.offboard_lost_since = None
                self._enter_takeoff(t)
            elif self.phase == 'LANDING' and not self.armed:
                # Touchdown/disarm completes a requested or automatic landing.
                self.phase = 'IDLE'
                self.t_phase = t
                self.plan = None
                self.pending = None
                self.goal_hold_point = None
                self.disarmed_since = None
                self.offboard_lost_since = None
            elif active:
                failsafe_exit = self.px4_failsafe and not self.offboard
                disarmed_long = (not self.armed and self.disarmed_since is not None and
                                 t - self.disarmed_since >= self.disarm_loss_grace)
                offboard_long = (self.armed and not self.offboard and self.offboard_lost_since is not None and
                                 t - self.offboard_lost_since >= self.offboard_loss_grace)
                if failsafe_exit or disarmed_long or offboard_long:
                    now = f' (PX4 mode {self.px4_mode})' if self.px4_mode else ''
                    if failsafe_exit:
                        why = 'PX4 failsafe'
                    elif disarmed_long:
                        why = 'PX4 disarmed'
                    else:
                        why = 'PX4 left OFFBOARD'
                    self.abort_reason = f'{why} during {self.phase}{now}'
                    self.phase = 'ABORTED'
                    self.t_phase = t

    def _enter_takeoff(self, t):
        p = self.state['pos']
        # PX4's local origin is the EKF origin, normally on the ground at the
        # take-off point; an already airborne vehicle does not climb further.
        ground = 0.0 if self.ground_d is None else float(self.ground_d)
        height_now = ground - p[2]
        if height_now >= self.airborne_height:
            self.takeoff_target = p.copy()
        else:
            top = None
            if self.arena_lb is not None:
                top = self.arena_lb[2]
            h = safe_takeoff_height(p, self.takeoff_height, self.eff_obs, top)
            self.takeoff_target = p + np.array([0.0, 0.0, -max(h, min(0.3, self.takeoff_height))])
        self.takeoff_yaw = float(self.state['rpy'][2])
        self.phase = 'TAKEOFF'
        self.t_phase = t

    # ------------------------------------------------------------ odometry
    def on_odometry(self, t, pos, vel, rpy, omega=None):
        with self.lock:
            pos = np.asarray(pos, float).reshape(3)
            vel = np.asarray(vel, float).reshape(3)
            rpy = np.asarray(rpy, float).reshape(3)
            if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(rpy))):
                return
            self.state = {'t': float(t), 'pos': pos, 'vel': vel, 'rpy': rpy}
            if omega is not None and np.all(np.isfinite(omega)):
                # measured body rates (FRD); used by models with rate-loop states
                self.state['omega'] = np.asarray(omega, float).reshape(3)
            # The observer only runs while the NMPC itself commands thrust
            # (attitude / body-rate interfaces).  Under PX4 position control
            # (takeoff, hold, goal) the applied thrust is not known here.
            own_thrust = (self.phase in ('NMPC', 'GOAL') and self.last_setpoint is not None
                          and self.last_setpoint.kind in ('attitude', 'body_rate'))
            self.observer.update(t, vel, rpy, self.applied_thrust_accel,
                                 active=self.observer_enabled and own_thrust)
            if self.phase in ('NMPC', 'TAKEOFF', 'GOAL'):
                self.min_clearance_seen = min(self.min_clearance_seen, min_clearance(self.eff_obs, pos))
            self._transitions(t)

    def _transitions(self, t):
        p, v = self.state['pos'], self.state['vel']
        if self.phase == 'TAKEOFF':
            # Only the altitude matters: the NMPC plans from wherever the
            # vehicle is.  Horizontal hold precision is limited by the state
            # estimate (GPS-level noise in SITL), so it must not gate the start.
            climb = self.start_pos[2] - self.takeoff_target[2]
            alt_err = abs(p[2] - self.takeoff_target[2])
            settled = alt_err < self.takeoff_tol and abs(v[2]) < 0.3
            timed_out = (t - self.t_phase) > 8.0 and (self.start_pos[2] - p[2]) > 0.6 * climb
            if (settled or timed_out) and self._plan_valid(t):
                self.phase = 'NMPC'
                self.t_phase = t
                self.t_nav_start = t
                self._cold_next = self.cold_start_at_navigation
        elif self.phase == 'NMPC':
            dist = float(np.linalg.norm(p - self.goal))
            speed = float(np.linalg.norm(v))
            reached = dist < self.goal_tol and speed < self.goal_speed_tol
            # Dual-mode MPC (Michalska & Mayne 1993): once the vehicle and the
            # NMPC prediction are inside a small obstacle-free terminal set
            # around the goal, hand over to the local terminal controller
            # (PX4 position hold at the goal).  This removes the endless
            # "0.5 m from the goal" creeping caused by estimate noise /
            # residual model error, without ever leaving the safe region.
            captured = (self.capture_radius > 0.0 and dist < self.capture_radius
                        and speed < self.capture_speed and self._plan_ends_at_goal()
                        and self._segment_clear(p, self.goal))
            if reached or captured:
                self.phase = 'GOAL'
                self.t_phase = t
                self.t_goal = t
                self.goal_hold_point = self.goal.copy()
                self.goal_hold_yaw = self._yaw_command(t)[0] if self.yaw_opt else None
                self.goals_completed += 1

    def _solving_phases(self):
        # attitude / body-rate: the NMPC (with its disturbance observer) also
        # holds the goal, because PX4's hover-thrust estimate is not updated
        # while PX4 is in attitude/rate offboard mode and a hand-over to PX4
        # position control can sag until PX4 re-converges.
        if self.interface == 'trajectory':
            return ('TAKEOFF', 'NMPC')
        return ('TAKEOFF', 'NMPC', 'GOAL')

    def _plan_ends_at_goal(self):
        if self.plan is None:
            return False
        return float(np.linalg.norm(self.plan.X[0:3, -1] - self.goal)) < 0.15

    def _segment_clear(self, a, b, n=20):
        if len(self.eff_obs) == 0:
            return True
        for s in np.linspace(0.0, 1.0, n):
            if min_clearance(self.eff_obs, a + s * (b - a)) < 0.0:
                return False
        return True

    # ------------------------------------------------------------ solving
    def _latency(self):
        ema = self.stats.solve_ms_ema
        lat = (0.0 if not np.isfinite(ema) else 1.2 * ema * 1e-3) + self.latency_margin
        return float(np.clip(lat, 0.0, self.latency_max))

    def _plan_valid(self, t):
        if self.plan is None:
            return False
        if t > self.plan.t_end:
            return False
        return (t - self.plan.t0) <= self.plan_timeout

    def _predict_measured(self, t_target):
        """Propagate the measured state to t_target with the prediction model."""
        s = self.state
        if self.nonlinear_trajectory:
            x = self.model.state_from_measurement(s['pos'], s['vel'], s['rpy'], omega=s.get('omega'))
            if self.interface == 'body_rate' and self.last_setpoint is not None \
                    and self.last_setpoint.kind == 'body_rate' \
                    and self.last_setpoint.source in ('nmpc', 'goal') \
                    and self._plan_valid(s['t']):
                # During solve/communication latency the vehicle continues
                # applying the previous thrust and body-rate plan.  Advancing
                # position alone leaves the optimizer an attitude and velocity
                # that already belong to the past.
                t = s['t']
                while t < t_target - 1e-9:
                    h = min(.02, t_target - t)
                    x = self.model.step(x, self._input_at(t), None, self.goal_yaw, h)
                    t += h
            else:
                # PX4's position controller owns takeoff and fallback; its
                # applied thrust/rates are not available to this predictor.
                x[:3] += x[3:6] * max(0.0, t_target - s['t'])
            return x
        if self.interface == 'trajectory':
            a = np.zeros(3)
            om = 0.0
            if self.plan is not None and self._plan_valid(s['t']):
                a = self._traj_eval(self.plan, s['t'])[2]
                if self.yaw_opt:
                    om = self._yaw_eval(self.plan, s['t'])[1]
            if self.yaw_opt:
                x = self.model.state_from_measurement(s['pos'], s['vel'], s['rpy'], a, yaw_rate=om)
            else:
                x = self.model.state_from_measurement(s['pos'], s['vel'], s['rpy'], a)
            x = self.model.clip_state(x)
            h = max(0.0, t_target - s['t'])
            x[0:3] = x[0:3] + x[3:6] * h
            if self.yaw_opt:
                x[9] = x[9] + x[10] * h
            return x
        x = self.model.state_from_measurement(s['pos'], s['vel'], s['rpy'])
        d = self.observer.d_hat
        t = s['t']
        h_max = 0.02
        while t < t_target - 1e-9:
            h = min(h_max, t_target - t)
            u = self._input_at(t)
            x = self.model.step(x, u, d, self.goal_yaw, h)
            t += h
        return x

    def _input_at(self, t):
        if self._plan_valid(t) and self.phase in ('NMPC', 'GOAL'):
            k, _ = self.plan.stage(t)
            return self.plan.U[:, k]
        return self.model.hover_input(self.observer.d_hat)

    def make_request(self, t):
        """Return a SolveRequest if a new solve should start now."""
        with self.lock:
            if self.state is None or self.phase not in self._solving_phases():
                return None
            if self.pending is not None:
                return None
            if t - self.last_request_t < self.solve_period - 1e-6:
                return None
            lat = self._latency()
            t0 = t + lat
            s = self.state
            from_meas = True
            if self.interface == 'trajectory' and self.plan is not None and self._plan_valid(t0) \
                    and self.phase == 'NMPC':
                pp, pv, _ = self._traj_eval(self.plan, s['t'])
                yaw_ok = True
                if self.yaw_opt:
                    yaw_ok = abs(wrap_pi(self._yaw_eval(self.plan, s['t'])[0] - s['rpy'][2])) < self.reseed_yaw
                if (np.linalg.norm(pp - s['pos']) < self.reseed_pos and
                        np.linalg.norm(pv - s['vel']) < self.reseed_vel and yaw_ok):
                    x0 = self._traj_state(self.plan, t0)
                    if self.yaw_opt and not self.nonlinear_trajectory:
                        x0 = np.r_[x0, self._yaw_eval(self.plan, t0)]
                    from_meas = False
                else:
                    self.stats.reseeds += 1
                    x0 = self._predict_measured(t0)
            elif self.phase == 'TAKEOFF' and self.interface == 'trajectory':
                x0 = self._predict_measured(t0)
            else:
                x0 = self._predict_measured(t0)

            self.guidance.update(t, x0[0:3])
            pref, vref, _ = self.guidance.horizon(x0[0:3], x0[3:6])
            if self.obstacle_slots is not None:
                obs = self.obstacle_slots.assign(self.eff_obs, x0[0:3], pref.T)
            else:
                obs = select_obstacles(self.eff_obs, x0[0:3], pref.T, self.solver.K)
            d = self.observer.d_hat.copy() if self.observer_enabled else np.zeros(3)
            uprev = self._input_at(t0) if self.plan is not None else self.model.hover_input(d)
            shift = None
            if self.plan is not None:
                shift = int(round((t0 - self.plan.t0) / self.dt))
                if shift < 0 or shift >= self.N:
                    shift = None
            if self._cold_next and self.phase == 'NMPC':
                shift = None                     # cold start (see cold_start_at_navigation)
                self._cold_next = False
            if self.force_cold_start:
                shift = None
            hdg_ref = hdg_w = None
            if self.yaw_opt and not self.nonlinear_trajectory:
                hdg_ref, hdg_w = self._heading_reference(x0, pref, vref)
            yaw_param = self.goal_yaw if np.isfinite(self.goal_yaw) else 0.0
            req = SolveRequest(t_request=t, t0=t0, x0=x0, pref=pref, vref=vref, obstacles=obs, d=d,
                               uprev=uprev, yaw_ref=yaw_param, shift=shift,
                               from_measurement=from_meas, phase=self.phase,
                               hdg_ref=hdg_ref, hdg_w=hdg_w)
            self.pending = req
            self.last_request_t = t
            return req

    def run_solver(self, req):
        """Blocking solve (call from a worker thread)."""
        extra = {}
        if req.hdg_ref is not None:
            extra = dict(hdg_ref=req.hdg_ref, hdg_w=req.hdg_w)
        if getattr(self.solver, 'logs_phase', False):
            extra['phase'] = req.phase           # recorded in the problem trace (replay starts at NMPC)
        return self.solver.solve_ocp(req.x0, req.pref, req.vref, req.obstacles, d=req.d, uprev=req.uprev,
                                     yaw_ref=req.yaw_ref, arena_lb=self.arena_lb, arena_ub=self.arena_ub,
                                     shift=req.shift, **extra)

    def accept(self, req, res, t_done):
        with self.lock:
            self.pending = None
            st = self.stats
            st.solves += 1
            st.solve_ms_last = float(res.solve_ms)
            st.solve_ms_hist.append(float(res.solve_ms))
            if len(st.solve_ms_hist) > 5000:
                st.solve_ms_hist = st.solve_ms_hist[-5000:]
            st.solve_ms_ema = res.solve_ms if not np.isfinite(st.solve_ms_ema) else 0.8 * st.solve_ms_ema + 0.2 * res.solve_ms
            st.solve_ms_max = max(st.solve_ms_max, float(res.solve_ms))
            st.iterations_last = int(res.iterations)
            st.iterations_hist.append(int(res.iterations))
            if len(st.iterations_hist) > 5000:
                st.iterations_hist = st.iterations_hist[-5000:]
            st.status_last = str(res.status)
            if res.success:
                st.converged += 1
            if t_done > req.t0 + 1e-3:
                st.deadline_misses += 1
            if self.phase not in self._solving_phases():
                return False
            if not res.usable:
                st.failures += 1
                st.consecutive_failures += 1
                return False
            # Safety gate: a plan must not *create* an obstacle violation.
            x0_clear = min_clearance(req.obstacles, req.x0[0:3]) if len(req.obstacles) else float('inf')
            if res.max_obstacle_slack > self.slack_tol and x0_clear > -self.slack_tol:
                st.rejected_unsafe += 1
                st.consecutive_failures += 1
                return False
            st.usable += 1
            st.consecutive_failures = 0
            self.plan = Plan(t0=req.t0, dt=self.dt, X=res.X, U=res.U, pref=req.pref,
                             installed_t=t_done, solve_ms=res.solve_ms, from_measurement=req.from_measurement,
                             prediction_model=self.model if self.nonlinear_trajectory else None)
            self.hold_point = None
            return True

    # ------------------------------------------------------------ setpoints
    @staticmethod
    def _traj_eval(plan, t):
        if plan.prediction_model is not None:
            return plan.prediction_model.trajectory_at(plan, t)
        k, tau = plan.stage(t)
        if t < plan.t0:
            k, tau = 0, 0.0
        x = plan.X[:, k]
        j = plan.U[0:3, k]      # (the yaw acceleration, if any, is row 3)
        p, v, a = x[0:3], x[3:6], x[6:9]
        pos = p + v * tau + a * tau ** 2 / 2.0 + j * tau ** 3 / 6.0
        vel = v + a * tau + j * tau ** 2 / 2.0
        acc = a + j * tau
        if t > plan.t_end:
            pos, vel, acc = plan.X[0:3, -1], plan.X[3:6, -1] * 0.0, plan.X[6:9, -1] * 0.0
        return pos, vel, acc

    def _traj_state(self, plan, t):
        if plan.prediction_model is not None:
            return plan.prediction_model.state_at(plan, t)
        p, v, a = self._traj_eval(plan, t)
        return np.r_[p, v, a]

    # ------------------------------------------------------------ yaw (optimised heading)
    @staticmethod
    def _smoothstep(x):
        x = min(1.0, max(0.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def _heading_reference(self, x0, pref, vref):
        """Per-stage heading reference and weight for the optimised yaw.

        Far from the goal the nose follows the direction of the reference velocity (A* guided,
        obstacle-aware) with a weight that fades out below ``min_speed`` (a hovering vehicle keeps
        its heading instead of jittering).  Within ``goal_blend_m`` of the goal the reference turns
        to ``goal_yaw`` (skipped when it is NaN).  The reference is *unwrapped* along the horizon and
        relative to ``x0[9]``, so the NLP never sees an angle jump.
        """
        N = self.N
        psi0 = float(x0[9])
        ref = np.zeros(N + 1)
        wgt = np.zeros(N + 1)
        last = psi0 if self.yaw_last_raw is None else float(self.yaw_last_raw)
        prev = psi0
        for k in range(N + 1):
            vh = vref[0:2, k]
            speed = float(np.hypot(vh[0], vh[1]))
            gate = self._smoothstep(speed / self.yaw_min_speed - 0.5)   # 0 below half the minimum speed, 1 at 1.5x
            if speed > 0.5 * self.yaw_min_speed:
                last = math.atan2(vh[1], vh[0])
            target = last
            w = self.yaw_w_travel * gate
            if np.isfinite(self.goal_yaw):
                dist = float(np.linalg.norm(pref[:, k] - self.goal))
                beta = self._smoothstep(1.0 - dist / self.yaw_blend_m)
                target = target + beta * wrap_pi(self.goal_yaw - target)
                w = (1.0 - beta) * w + beta * self.yaw_w_goal
            ref[k] = prev + wrap_pi(target - prev)
            prev = ref[k]
            wgt[k] = w
        self.yaw_last_raw = wrap_pi(last)
        return ref, wgt

    @staticmethod
    def _yaw_eval(plan, t):
        """Planned (psi, omega) at time t (psi unwrapped)."""
        if plan.prediction_model is not None:
            return plan.prediction_model.yaw_at(plan, t)
        k, tau = plan.stage(t)
        if t < plan.t0:
            k, tau = 0, 0.0
        psi, om, al = plan.X[9, k], plan.X[10, k], plan.U[3, k]
        if t > plan.t_end:
            return float(plan.X[9, -1]), 0.0
        return float(psi + om * tau + al * tau ** 2 / 2.0), float(om + al * tau)

    def _yaw_command(self, t):
        """(yaw [rad, wrapped], yawspeed) currently commanded by the plan, else the measured yaw."""
        if self.plan is not None and self._plan_valid(t):
            psi, om = self._yaw_eval(self.plan, t)
            return wrap_pi(psi), om
        if self.state is not None:
            return float(self.state['rpy'][2]), 0.0
        return 0.0, 0.0

    def _goal_heading(self):
        """Yaw sent with the position hold at the goal."""
        if not self.yaw_opt:
            return self.goal_yaw
        if np.isfinite(self.goal_yaw):
            return self.goal_yaw
        return self.goal_hold_yaw if self.goal_hold_yaw is not None else self._yaw_command(0.0)[0]

    def _hold(self, t, reason):
        if self.hold_point is None:
            base = self.state['pos'].copy() if self.state is not None else self.goal.copy()
            self.hold_point = base
            self.stats.hold_events += 1
            self.hold_yaw = float(self.state['rpy'][2]) if self.state is not None else self.goal_yaw
        yaw = self.hold_yaw if self.yaw_opt else self.goal_yaw
        return Setpoint(kind='position', source='hold', position=self.hold_point.copy(),
                        velocity=np.zeros(3), acceleration=np.full(3, np.nan), yaw=yaw)

    def setpoint(self, t, odom_fresh=True):
        """Setpoint to stream to PX4 at time t (None = do not stream).

        With stale odometry the NMPC cannot close the loop, so PX4's own
        position controller holds the last known position instead.
        """
        with self.lock:
            if not odom_fresh and self.phase in ('TAKEOFF', 'NMPC', 'GOAL'):
                sp = self._hold(t, 'stale odometry')
            else:
                sp = self._setpoint(t)
            if sp is None or sp.source != 'hold':
                self.hold_point = None
            # control-time split during navigation (NMPC vs PX4 position-hold fallback)
            if self.t_last_setpoint is not None and self.phase == 'NMPC' and sp is not None:
                dt = float(np.clip(t - self.t_last_setpoint, 0.0, 0.2))
                if sp.source == 'hold':
                    self.hold_control_s += dt
                else:
                    self.nmpc_control_s += dt
            self.t_last_setpoint = t
            if sp is not None:
                if sp.kind in ('attitude', 'body_rate'):
                    self.applied_thrust_accel = float(sp.thrust_accel)
                else:
                    # PX4 position control: estimate thrust from attitude and
                    # hover assumption (observer inactive in this mode anyway)
                    self.applied_thrust_accel = self.g
            self.last_setpoint = sp
            return sp

    def _setpoint(self, t):
        if self.state is None or self.phase in ('IDLE', 'ABORTED'):
            return None
        if self.phase == 'ARMING':
            p = self.state['pos'].copy()
            return Setpoint(kind='position', source='arming', position=p, velocity=np.zeros(3),
                            acceleration=np.full(3, np.nan), yaw=self.state['rpy'][2])
        if self.phase == 'TAKEOFF':
            yaw = self.takeoff_yaw if (self.yaw_opt and self.takeoff_yaw is not None) else self.goal_yaw
            return Setpoint(kind='position', source='takeoff', position=self.takeoff_target.copy(),
                            velocity=np.full(3, np.nan), acceleration=np.full(3, np.nan), yaw=yaw)
        if self.phase == 'GOAL' and self.interface != 'trajectory' and self._plan_valid(t) \
                and self.stats.consecutive_failures < self.max_failures:
            sp = self._nmpc_setpoint(t)
            sp.source = 'goal'
            return sp
        if self.phase in ('GOAL', 'LANDING'):
            target = self.goal_hold_point if self.goal_hold_point is not None else self.goal
            return Setpoint(kind='position', source='goal', position=target.copy(), velocity=np.zeros(3),
                            acceleration=np.full(3, np.nan), yaw=self._goal_heading())
        # NMPC phase
        if not self._plan_valid(t) or self.stats.consecutive_failures >= self.max_failures:
            # PX4 brakes and holds the point where the plan became unusable.
            return self._hold(t, 'no valid plan')
        return self._nmpc_setpoint(t)

    def _nmpc_setpoint(self, t):
        k, tau = self.plan.stage(t)
        if self.interface == 'trajectory':
            pos, vel, acc = self._traj_eval(self.plan, t)
            if self.yaw_opt:
                psi, om = self._yaw_eval(self.plan, t)
                return Setpoint(kind='trajectory', source='nmpc', position=pos, velocity=vel, acceleration=acc,
                                yaw=wrap_pi(psi), yawspeed=om, stage=k)
            return Setpoint(kind='trajectory', source='nmpc', position=pos, velocity=vel, acceleration=acc,
                            yaw=self.goal_yaw, stage=k)
        u = self.plan.U[:, k]
        if self.interface == 'attitude':
            q = euler_to_quat_wxyz(u[0], u[1], self.goal_yaw)
            T = float(u[2])
            return Setpoint(kind='attitude', source='nmpc', q_d=q, thrust_accel=T,
                            thrust_normalized=self.thrust_map.to_throttle(T), yaw=self.goal_yaw, stage=k)
        T = float(u[0])
        thrust_accel = (self.model.command_thrust_accel(T)
                        if hasattr(self.model, 'command_thrust_accel') else T)
        return Setpoint(kind='body_rate', source='nmpc', rates=np.asarray(u[1:4], float),
                        thrust_accel=thrust_accel,
                        thrust_normalized=self.thrust_map.to_throttle(thrust_accel), stage=k)

    # ------------------------------------------------------------ status
    def goal_reached(self):
        return self.phase in ('GOAL', 'LANDING')

    def experiment_result(self):
        if self.phase in ('GOAL', 'LANDING'):
            return 'success'
        if self.phase == 'ABORTED':
            return 'aborted'
        if self.phase == 'IDLE':
            return 'idle'
        return 'running'

    def navigation_elapsed(self, t):
        """Time since the NMPC took over after take-off (frozen at the goal)."""
        if self.t_nav_start is None:
            return 0.0
        end = self.t_goal if self.t_goal is not None else t
        return max(0.0, end - self.t_nav_start)

    def diagnostics(self, t):
        with self.lock:
            st = self.stats
            s = self.state
            out = {
                'phase': self.phase,
                'interface': self.interface,
                'backend': getattr(self.solver, 'backend', 'n/a'),
                'solver_status': st.status_last,
                'last_solve_ms': st.solve_ms_last,
                'solve_ms_ema': st.solve_ms_ema,
                'solve_ms_max': st.solve_ms_max,
                'iterations': st.iterations_last,
                'deadline_ratio': (st.solve_ms_last * 1e-3 / self.solve_period) if np.isfinite(st.solve_ms_last) else float('nan'),
                'latency_compensation_s': self._latency(),
                'solves': st.solves,
                'usable_solves': st.usable,
                'converged_solves': st.converged,
                'rejected_unsafe': st.rejected_unsafe,
                'failed_solves': st.failures,
                'consecutive_failures': st.consecutive_failures,
                'hold_events': st.hold_events,
                'reseeds': st.reseeds,
                'plan_age_s': (t - self.plan.installed_t) if self.plan is not None else float('inf'),
                'prediction_age_s': (t - self.plan.t0) if self.plan is not None else float('inf'),
                'applied_source': self.last_setpoint.source if self.last_setpoint is not None else 'none',
                'applied_stage': self.last_setpoint.stage if self.last_setpoint is not None else -1,
                'disturbance_estimate': self.observer.d_hat.copy(),
                'min_clearance_seen_m': self.min_clearance_seen,
                'deadline_misses': st.deadline_misses,
                'experiment_result': self.experiment_result(),
                'abort_reason': self.abort_reason,
                'navigation_elapsed_s': self.navigation_elapsed(t),
                'nmpc_control_s': self.nmpc_control_s,
                'hold_control_s': self.hold_control_s,
                'takeoff_target': None if self.takeoff_target is None else self.takeoff_target.copy(),
                'hold_point': None if self.hold_point is None else self.hold_point.copy(),
                'mission_start': None if self.mission_start is None else self.mission_start.copy(),
                'leg_index': self.leg_index,
                'goals_completed': self.goals_completed,
                'goal': self.goal.copy(),
                'armed': bool(self.armed),
                'offboard': bool(self.offboard),
                'px4_failsafe': bool(self.px4_failsafe),
                'px4_mode': self.px4_mode,
                'yaw_mode': 'optimized' if self.yaw_opt else 'fixed',
            }
            if self.yaw_opt and self.plan is not None:
                psi, om = self._yaw_eval(self.plan, t)
                out['yaw_cmd_deg'] = math.degrees(wrap_pi(psi))
                out['yaw_rate_cmd_dps'] = math.degrees(om)
            if s is not None:
                out['yaw_deg'] = math.degrees(float(s['rpy'][2]))
                out['distance_to_goal_m'] = float(np.linalg.norm(s['pos'] - self.goal))
                out['speed_mps'] = float(np.linalg.norm(s['vel']))
                out['min_safety_clearance_m'] = min_clearance(self.eff_obs, s['pos'])
            return out

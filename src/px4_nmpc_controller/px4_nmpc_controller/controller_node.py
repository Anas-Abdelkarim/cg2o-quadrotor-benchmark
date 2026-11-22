"""ROS 2 / PX4 adapter for the NMPC runtime.

All control logic lives in ``px4_nmpc_core.runtime.NmpcRuntime`` (ROS-free and
tested offline).  This node only

* converts PX4 ``VehicleOdometry`` / ``VehicleStatus`` into runtime calls,
* runs the NMPC solver in a dedicated worker thread (never in the setpoint
  path),
* streams ``OffboardControlMode`` + the interface-specific setpoint at a fixed
  rate (``TrajectorySetpoint`` / ``VehicleAttitudeSetpoint`` /
  ``VehicleRatesSetpoint``; PX4 position hold as fallback),
* requests OFFBOARD + ARM, reports PX4's command acknowledgements and the
  reason of any PX4 failsafe, and
* publishes paths and ``/nmpc/diagnostics`` for the visualizer and monitor.

PX4 topic names follow PX4's own rule (PX4 >= v1.16): versioned messages carry
a ``_v<N>`` suffix, e.g. ``/fmu/out/vehicle_status_v1``.  See
``px4_nmpc_core.px4_interface``.
"""
from __future__ import annotations

import pathlib
import threading

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from px4_msgs.msg import (
    FailsafeFlags,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitudeSetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleStatus,
)
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from px4_nmpc_core import __version__ as CORE_VERSION
from px4_nmpc_core.config import DEFAULTS, arena_bounds, build_configs
from px4_nmpc_core.diagnostics import build_diagnostics
from px4_nmpc_core.math_utils import enu_to_ned, ned_to_enu, quat_wxyz_to_euler, wrap_pi
from px4_nmpc_core.ocp import make_ocp_solver
from px4_nmpc_core.px4_interface import active_flag_names, constant_names, px4_topic, topic_version_mode
from px4_nmpc_core.runtime import NmpcRuntime
from px4_nmpc_core.scenario import (
    controller_geometry,
    load_scenario_json,
    scenario_parameters,
    scenario_wall_margin,
)


NODE_DEFAULTS = {
    'experiment_auto_start': False,
    'disarm_on_stop': True,
    'auto_arm': True,
    'offboard_warmup_s': 1.0,
    'actual_path_sample_period_s': 0.10,
    'px4_topic_prefix': '/fmu',
    # 'auto' = PX4's own naming rule (v1.16+): add _v<N> for MESSAGE_VERSION > 0.
    # 'never' only for PX4 <= v1.15 (which also needs that release's px4_msgs).
    'px4_versioned_topics': 'auto',
    'trajectory_acceleration_feedforward': False,
    'status_timeout_s': 8.0,          # no VehicleStatus after the first ARM request -> abort
    'arming_timeout_s': 15.0,         # not armed + OFFBOARD this long after the first request -> abort
    # Scenario geometry source.  A non-empty, existing scenario_json is
    # authoritative for goal/obstacles/arena (the same file Gazebo and RViz use).
    # '' or 'none' = fly the geometry given by the ROS parameters instead.
    'scenario_json': '',
    'fallback_scenario_json': '',
}

# VehicleCommandAck results / commands we care about (px4_msgs constants).
_ACK_RESULTS = constant_names(VehicleCommandAck, 'VEHICLE_CMD_RESULT_')
_COMMAND_NAMES = {
    VehicleCommand.VEHICLE_CMD_DO_SET_MODE: 'SET_MODE(OFFBOARD)',
    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM: 'ARM/DISARM',
    VehicleCommand.VEHICLE_CMD_NAV_LAND: 'LAND',
}
_NAV_STATES = constant_names(VehicleStatus, 'NAVIGATION_STATE_')


def _use_scenario_file(text):
    text = str(text or '').strip()
    return '' if text.lower() in ('', 'none', 'off', 'false') else text


class NmpcNode(Node):
    START_SOLVER_THREAD = True   # tests drive solve_once() synchronously

    def __init__(self):
        super().__init__('px4_nmpc_controller')
        self.params = self._declare_and_read()
        self.scenario_source = self._apply_startup_scenario(self.params)
        solver_cfg, runtime_cfg, self.p = build_configs(self.params)
        self.get_logger().info(f'Scenario geometry: {self.scenario_source}')
        self.get_logger().info(
            f'Building NMPC: interface={solver_cfg["interface"]}, backend={solver_cfg["backend"]}, '
            f'N={solver_cfg["N"]}, dt={solver_cfg["dt"]}, K={solver_cfg["max_obstacles"]} ...')
        self.solver = make_ocp_solver(solver_cfg)
        self.rt = NmpcRuntime(runtime_cfg, self.solver)
        if getattr(self.solver, 'backend_error', ''):
            self.get_logger().warning(self.solver.backend_error)
        self.get_logger().info(
            f'NMPC ready in {self.solver.build_s:.2f} s (backend actually used: {self.solver.backend}; '
            f'{self.solver.nz} variables, {self.solver.ng} constraints, '
            f'{len(runtime_cfg["obstacles"])} obstacles in scene).')

        self.last_odom_t = None
        self.first_odom_t = None
        self.path_actual = []
        self.last_actual_sample_t = None
        self.status_seen = False
        self.status_missing_reported = False
        self.last_cmd_t = -1e9
        self.first_cmd_t = None
        self.stream_count = 0
        self.arming_started_t = None
        self.land_sent = False
        self.last_solution_seen = None
        self.px4_nav_state = None
        self.px4_failsafe = False
        self.failsafe_flags = []
        self.last_ack = {}          # command -> result name
        self._failsafe_flags_logged = []
        self._abort_logged = False

        # Match PX4 uXRCE-DDS QoS in each direction: PX4 writers are
        # BEST_EFFORT/TRANSIENT_LOCAL, PX4 readers BEST_EFFORT/VOLATILE.
        qos_from_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_acks = QoSProfile(      # several acks can arrive in one burst (SET_MODE + ARM)
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_to_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        pre = str(self.p['px4_topic_prefix'])
        mode = topic_version_mode(self.p['px4_versioned_topics'])
        self.topics = {
            'mode': px4_topic(pre, 'in', 'offboard_control_mode', OffboardControlMode, mode),
            'traj': px4_topic(pre, 'in', 'trajectory_setpoint', TrajectorySetpoint, mode),
            'att': px4_topic(pre, 'in', 'vehicle_attitude_setpoint', VehicleAttitudeSetpoint, mode),
            'rates': px4_topic(pre, 'in', 'vehicle_rates_setpoint', VehicleRatesSetpoint, mode),
            'cmd': px4_topic(pre, 'in', 'vehicle_command', VehicleCommand, mode),
            'odom': px4_topic(pre, 'out', 'vehicle_odometry', VehicleOdometry, mode),
            'status': px4_topic(pre, 'out', 'vehicle_status', VehicleStatus, mode),
            'ack': px4_topic(pre, 'out', 'vehicle_command_ack', VehicleCommandAck, mode),
            'failsafe': px4_topic(pre, 'out', 'failsafe_flags', FailsafeFlags, mode),
        }
        tp = self.topics
        self.get_logger().info(
            f'PX4 topics ({mode}): in <- {tp["odom"]}, {tp["status"]}, {tp["ack"]}, {tp["failsafe"]}; '
            f'out -> {tp["mode"]}, {tp["traj"]}, {tp["att"]}, {tp["rates"]}, {tp["cmd"]}')

        self.pub_mode = self.create_publisher(OffboardControlMode, tp['mode'], qos_to_px4)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, tp['traj'], qos_to_px4)
        self.pub_att = self.create_publisher(VehicleAttitudeSetpoint, tp['att'], qos_to_px4)
        self.pub_rates = self.create_publisher(VehicleRatesSetpoint, tp['rates'], qos_to_px4)
        self.pub_cmd = self.create_publisher(VehicleCommand, tp['cmd'], qos_to_px4)
        self.create_subscription(VehicleOdometry, tp['odom'], self.odom_cb, qos_from_px4)
        self.create_subscription(VehicleStatus, tp['status'], self.status_cb, qos_from_px4)
        self.create_subscription(VehicleCommandAck, tp['ack'], self.ack_cb, qos_acks)
        self.create_subscription(FailsafeFlags, tp['failsafe'], self.failsafe_flags_cb, qos_from_px4)
        # Live mission goal. frame_id='px4_local_ned' uses x=N,y=E,z=D;
        # frame_id='map'/'enu' uses the RViz ENU convention and is converted.
        self.create_subscription(PoseStamped, '/nmpc/goal', self.goal_cb, 10)

        self.pub_pred = self.create_publisher(Path, '/nmpc/predicted_path', 10)
        self.pub_ref = self.create_publisher(Path, '/nmpc/reference_path', 10)
        self.pub_actual = self.create_publisher(Path, '/nmpc/actual_path', 10)
        # Global A* guide: canonical topic plus the alias used by older tools.
        self.pub_global = self.create_publisher(Path, '/nmpc/global_path', 10)
        self.pub_guide = self.create_publisher(Path, '/nmpc/guide_path', 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, '/nmpc/diagnostics', 10)

        self.create_service(Trigger, '/nmpc/start', self.start_cb)
        self.create_service(Trigger, '/nmpc/stop', self.stop_cb)
        self.create_service(Trigger, '/nmpc/reset_controller', self.reset_cb)
        self.create_service(Trigger, '/nmpc/status', self.status_srv_cb)
        self.create_service(Trigger, '/nmpc/reload_scenario', self.reload_scenario_cb)
        self.create_service(Trigger, '/nmpc/commit_staged_goal', self.commit_staged_goal_cb)

        self.create_timer(1.0 / float(self.p['setpoint_stream_hz']), self.stream_tick)
        self.create_timer(1.0 / float(self.p['diagnostics_hz']), self.publish_diagnostics)
        self.create_timer(0.2, self.publish_paths)
        self.create_timer(1.0, self.status_tick)

        self._stop_evt = threading.Event()
        self._solver_thread = None
        if self.START_SOLVER_THREAD:
            self._solver_thread = threading.Thread(target=self._solver_loop, name='nmpc_solver', daemon=True)
            self._solver_thread.start()
        self._auto_start_pending = bool(self.p['experiment_auto_start'])
        self.get_logger().info(
            f'px4_nmpc v{CORE_VERSION} ready ('
            f'{"auto-start" if self._auto_start_pending else "PAUSED; call /nmpc/start when ready"}).')

    # ------------------------------------------------------------ params
    def _declare_and_read(self):
        # dynamic_typing: YAML integers (e.g. ``takeoff_height_m: 1``) must not
        # abort start-up with a type mismatch; build_configs() casts values.
        desc = ParameterDescriptor(dynamic_typing=True)
        values = {}
        for key, default in list(DEFAULTS.items()) + list(NODE_DEFAULTS.items()):
            if isinstance(default, list):
                default = [float(v) for v in default]
            self.declare_parameter(key, default, desc)
            values[key] = self.get_parameter(key).value
        return values

    def _apply_startup_scenario(self, params):
        """Use scenario.json (same file as Gazebo/RViz) as the geometry source.

        Order: ``scenario_json`` -> ``fallback_scenario_json`` -> ROS parameters.
        Returns a human-readable description of the source actually used.
        """
        primary = _use_scenario_file(params.get('scenario_json'))
        if not primary:
            return 'ROS parameters (nmpc.yaml / scenario_file); scenario_json disabled'
        candidates = [(primary, False)]
        fallback = _use_scenario_file(params.get('fallback_scenario_json'))
        if fallback:
            candidates.append((fallback, True))
        problems = []
        for path_text, is_fallback in candidates:
            path = pathlib.Path(path_text).expanduser()
            try:
                scenario = load_scenario_json(path)
                params.update(scenario_parameters(scenario, scenario_wall_margin(scenario)))
            except Exception as exc:  # noqa: BLE001
                problems.append(f'{path}: {exc}')
                continue
            text = f'{path} (seed {scenario.get("seed")}, {len(scenario["obstacle_radii"])} obstacles)'
            if is_fallback:
                self.get_logger().warning(
                    'Runtime scenario unavailable (' + '; '.join(problems) + '). Using the PACKAGED example '
                    'scenario. Run `ros2 run px4_nmpc_tools nmpc_experiment new` + `apply` so that '
                    'Gazebo, RViz and the controller use the same obstacles.')
                text = 'PACKAGED EXAMPLE ' + text
            return text
        self.get_logger().error('No usable scenario JSON (' + '; '.join(problems) + '); '
                                'flying the geometry from the ROS parameters.')
        return 'ROS parameters (no usable scenario JSON)'

    # ------------------------------------------------------------ helpers
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def odom_fresh(self):
        return self.last_odom_t is not None and (self.now_s() - self.last_odom_t) <= float(self.p['odom_timeout_s'])

    def nav_state_name(self, value=None):
        value = self.px4_nav_state if value is None else value
        if value is None:
            return 'unknown'
        return _NAV_STATES.get(int(value), f'nav_state {int(value)}')

    def _status_topic_hint(self):
        """Explain a missing VehicleStatus using the live ROS graph."""
        try:
            names = sorted(n for n, _ in self.get_topic_names_and_types() if 'vehicle_status' in n)
        except Exception:  # noqa: BLE001
            names = []
        seen = ', '.join(names) if names else 'none'
        return (f'No PX4 VehicleStatus on {self.topics["status"]} (vehicle_status topics in the graph: {seen}). '
                'PX4 v1.16+ publishes /fmu/out/vehicle_status_v1; keep px4_versioned_topics:=auto and '
                'check MicroXRCEAgent / PX4 (ros2 run px4_nmpc_tools nmpc_doctor).')

    # ------------------------------------------------------------ callbacks
    def odom_cb(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED or msg.velocity_frame != VehicleOdometry.VELOCITY_FRAME_NED:
            self.get_logger().warning('Ignoring VehicleOdometry that is not NED position + NED velocity',
                                      throttle_duration_sec=5.0)
            return
        pos = np.asarray(msg.position, float)
        vel = np.asarray(msg.velocity, float)
        q = np.asarray(msg.q, float)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(q))):
            return
        rpy = quat_wxyz_to_euler(q)
        rpy[2] = wrap_pi(rpy[2])
        t = self.now_s()
        if self.first_odom_t is None:
            self.first_odom_t = t
        self.last_odom_t = t
        omega = np.asarray(msg.angular_velocity, float)
        self.rt.on_odometry(t, pos, vel, rpy, omega=omega if np.all(np.isfinite(omega)) else None)
        period = max(0.01, float(self.p['actual_path_sample_period_s']))
        if self.last_actual_sample_t is None or t - self.last_actual_sample_t >= period:
            self.path_actual.append(pos.copy())
            self.path_actual = self.path_actual[-3000:]
            self.last_actual_sample_t = t
        if self._auto_start_pending and self.status_seen:
            ok, _ = self.rt.start(t)
            if ok:
                self._auto_start_pending = False
                self.get_logger().info('experiment_auto_start: NMPC experiment started')

    def goal_cb(self, msg):
        """Accept a new target while paused, flying, or holding the old goal."""
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        frame = str(getattr(getattr(msg, 'header', None), 'frame_id', '') or '').strip().lower()
        if frame in ('px4_local_ned', 'ned', 'local_ned'):
            goal = p
        elif frame in ('map', 'enu', 'local_enu'):
            goal = enu_to_ned(p)
        else:
            self.get_logger().error(
                f'Rejected /nmpc/goal with frame_id={frame!r}; use px4_local_ned or map/enu.')
            return
        ok, text = self.rt.set_goal(goal, self.now_s(), resume=True)
        if ok:
            self.land_sent = False
            self.last_solution_seen = None
            self.get_logger().info(f'Live goal: {goal.tolist()} NED -> {text}')
        else:
            self.get_logger().error(f'Rejected live goal {goal.tolist()}: {text}')

    def status_cb(self, msg):
        first = not self.status_seen
        self.status_seen = True
        self.status_missing_reported = False
        armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        offboard = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        failsafe = bool(getattr(msg, 'failsafe', False))
        prev_nav, prev_failsafe = self.px4_nav_state, self.px4_failsafe
        self.px4_nav_state = int(msg.nav_state)
        self.px4_failsafe = failsafe
        if first:
            self.get_logger().info(
                f'PX4 VehicleStatus received on {self.topics["status"]}: '
                f'{"armed" if armed else "disarmed"}, mode {self.nav_state_name()}.')
        elif prev_nav is not None and prev_nav != self.px4_nav_state:
            self.get_logger().info(f'PX4 mode: {self.nav_state_name(prev_nav)} -> {self.nav_state_name()}'
                                   + (' (FAILSAFE)' if failsafe else ''))
        if failsafe and not prev_failsafe:
            flags = ', '.join(self.failsafe_flags) or 'reported on /fmu/out/failsafe_flags shortly'
            self._failsafe_flags_logged = list(self.failsafe_flags)
            self.get_logger().error(f'PX4 FAILSAFE active: mode {self.nav_state_name()}; active flags: {flags}')
        prev = self.rt.phase
        self.rt.set_vehicle_status(self.now_s(), armed, offboard, failsafe=failsafe, mode=self.nav_state_name())
        if prev != 'ABORTED' and self.rt.phase == 'ABORTED':
            self.get_logger().error(
                f'NMPC experiment aborted: {self.rt.abort_reason}. The controller no longer streams setpoints; '
                'PX4 keeps control. Call /nmpc/start to resume from the current position.')

    def ack_cb(self, msg):
        command = int(msg.command)
        if command not in _COMMAND_NAMES:
            return
        result = _ACK_RESULTS.get(int(msg.result), str(int(msg.result)))
        previous = self.last_ack.get(command)
        self.last_ack[command] = result
        if result not in ('ACCEPTED', 'IN_PROGRESS') and result != previous:
            self.get_logger().error(f'PX4 rejected {_COMMAND_NAMES[command]}: {result} '
                                    '(the PX4 console / QGroundControl shows the reason).')

    def failsafe_flags_cb(self, msg):
        flags = active_flag_names(msg)
        if self.px4_failsafe and flags and flags != self._failsafe_flags_logged:
            self.get_logger().error('PX4 failsafe flags: ' + ', '.join(flags))
            self._failsafe_flags_logged = list(flags)
        self.failsafe_flags = flags

    # ------------------------------------------------------------ solver thread
    def _solver_loop(self):
        while not self._stop_evt.is_set():
            if not self.solve_once():
                self._stop_evt.wait(0.003)

    def solve_once(self):
        """Run one solve if the runtime wants one. Returns True if a solve ran."""
        req = self.rt.make_request(self.now_s())
        if req is None:
            return False
        try:
            res = self.rt.run_solver(req)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'NMPC solver raised: {exc}', throttle_duration_sec=1.0)
            with self.rt.lock:
                self.rt.pending = None
            return True
        accepted = self.rt.accept(req, res, self.now_s())
        if not accepted and self.rt.phase in ('NMPC', 'TAKEOFF'):
            self.get_logger().warning(
                f'NMPC result not used: status={res.status}, usable={res.usable}, '
                f'obstacle_slack={res.max_obstacle_slack:.3f} m, solve={res.solve_ms:.1f} ms',
                throttle_duration_sec=1.0)
        return True

    # ------------------------------------------------------------ PX4 output
    def command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.now_us()
        msg.param1 = float(p1)
        msg.param2 = float(p2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.pub_cmd.publish(msg)

    def _publish_mode(self, kind):
        msg = OffboardControlMode()
        msg.timestamp = self.now_us()
        msg.position = kind in ('position', 'trajectory')
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = kind == 'attitude'
        msg.body_rate = kind == 'body_rate'
        if hasattr(msg, 'thrust_and_torque'):
            msg.thrust_and_torque = False
        if hasattr(msg, 'direct_actuator'):
            msg.direct_actuator = False
        self.pub_mode.publish(msg)

    @staticmethod
    def _f3(v):
        if v is None:
            return [float('nan')] * 3
        return [float(x) for x in np.asarray(v, float).reshape(3)]

    def _publish_setpoint(self, sp):
        self._publish_mode(sp.kind)
        if sp.kind in ('position', 'trajectory'):
            msg = TrajectorySetpoint()
            msg.timestamp = self.now_us()
            msg.position = self._f3(sp.position)
            msg.velocity = self._f3(sp.velocity)
            if sp.kind == 'trajectory' and not bool(self.p['trajectory_acceleration_feedforward']):
                msg.acceleration = [float('nan')] * 3
            else:
                msg.acceleration = self._f3(sp.acceleration)
            if hasattr(msg, 'jerk'):
                msg.jerk = [float('nan')] * 3
            msg.yaw = float(sp.yaw) if np.isfinite(sp.yaw) else float('nan')
            ys = getattr(sp, 'yawspeed', float('nan'))
            msg.yawspeed = float(ys) if ys is not None and np.isfinite(ys) else float('nan')
            self.pub_traj.publish(msg)
        elif sp.kind == 'attitude':
            msg = VehicleAttitudeSetpoint()
            msg.timestamp = self.now_us()
            msg.q_d = [float(v) for v in sp.q_d]
            msg.thrust_body = [0.0, 0.0, -float(sp.thrust_normalized)]
            if hasattr(msg, 'yaw_sp_move_rate'):
                msg.yaw_sp_move_rate = 0.0
            self.pub_att.publish(msg)
        elif sp.kind == 'body_rate':
            msg = VehicleRatesSetpoint()
            msg.timestamp = self.now_us()
            msg.roll, msg.pitch, msg.yaw = [float(v) for v in sp.rates]
            msg.thrust_body = [0.0, 0.0, -float(sp.thrust_normalized)]
            msg.reset_integral = False
            self.pub_rates.publish(msg)

    def _abort_arming(self, t, reason, land):
        """Leave ARMING safely: optionally ask PX4 to land, then stop streaming."""
        if land:
            # PX4 may be armed in OFFBOARD without us knowing it.  A LAND
            # command is deterministic; silently dropping the setpoint stream
            # would instead trigger PX4's offboard-loss failsafe.
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.rt.abort(t, reason)
        self.get_logger().error(f'NMPC experiment aborted: {reason}')

    def _may_inject_setpoints(self, phase):
        """True when a TrajectorySetpoint/attitude/rate setpoint may be published.

        PX4's uXRCE-DDS client writes incoming setpoints straight into the
        uORB topics that PX4's own flight tasks also publish
        (``trajectory_setpoint`` ...).  Outside OFFBOARD they would be mixed
        into AUTO_LAND / RTL / POSCTL setpoints (seen in PX4 v1.17 SITL as a
        landing vehicle climbing back up).  So setpoints are only sent while
        PX4 is in OFFBOARD, or while arming a vehicle that is still disarmed.
        """
        if phase == 'LANDING':
            return False
        if self.rt.offboard:
            return True
        return phase == 'ARMING' and not self.rt.armed

    def stream_tick(self):
        t = self.now_s()
        sp = self.rt.setpoint(t, odom_fresh=self.odom_fresh())
        phase = self.rt.phase
        if sp is None or phase == 'LANDING':
            return          # PX4 (AUTO_LAND/RTL/...) is in charge; publish nothing
        if self._may_inject_setpoints(phase):
            self._publish_setpoint(sp)
        else:
            # Heartbeat only: keeps OFFBOARD available for the pending mode
            # switch without touching PX4's current flight task.
            self._publish_mode(sp.kind)
        self.stream_count += 1
        if phase == 'ARMING':
            if self.arming_started_t is None:
                self.arming_started_t = t
            warm = (t - self.arming_started_t) >= float(self.p['offboard_warmup_s'])
            if warm and bool(self.p['auto_arm']) and t - self.last_cmd_t >= 1.0:
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                if not self.rt.armed:
                    self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
                self.last_cmd_t = t
                self.get_logger().info('Requested OFFBOARD + ARM', throttle_duration_sec=2.0)
                if self.first_cmd_t is None:
                    self.first_cmd_t = t
            if self.first_cmd_t is not None:
                waited = t - self.first_cmd_t
                if not self.status_seen and waited > float(self.p['status_timeout_s']):
                    self.status_missing_reported = True
                    self._abort_arming(t, 'VehicleStatus not received; ' + self._status_topic_hint(), land=True)
                elif self.status_seen and waited > float(self.p['arming_timeout_s']):
                    acks = ', '.join(f'{_COMMAND_NAMES[c]}={r}' for c, r in self.last_ack.items()) or 'no ACK'
                    self._abort_arming(
                        t, f'PX4 did not arm and enter OFFBOARD within {waited:.0f} s '
                           f'(armed={self.rt.armed}, mode={self.nav_state_name()}, {acks}). '
                           'Check the PX4 console / QGroundControl for pre-arm failures.',
                        land=self.rt.armed)
        else:
            self.arming_started_t = None
            self.first_cmd_t = None

        if phase == 'GOAL' and self.rt.auto_land and not self.land_sent:
            # Use the same landing state-machine path as an explicit stop so
            # mission/environment commands cannot see a descending vehicle as
            # IDLE and any stale NMPC plan is cleared consistently.
            self.rt.begin_landing(t)
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.land_sent = True
            self.get_logger().info('Goal reached at low speed; LAND command sent')

    # ------------------------------------------------------------ visualisation
    def _path_msg(self, points):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for point in points:
            enu = ned_to_enu(point)
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(enu[0])
            pose.pose.position.y = float(enu[1])
            pose.pose.position.z = float(enu[2])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def publish_paths(self):
        if len(self.path_actual) >= 2:
            self.pub_actual.publish(self._path_msg(self.path_actual))
        with self.rt.lock:
            plan = self.rt.plan
            guide = self.rt.guidance.path3
        if plan is not None and plan is not self.last_solution_seen:
            self.pub_pred.publish(self._path_msg(plan.X[0:3, :].T))
            self.pub_ref.publish(self._path_msg(plan.pref.T))
            self.last_solution_seen = plan
        if guide is not None:
            guide_msg = self._path_msg(guide)
            self.pub_global.publish(guide_msg)
            self.pub_guide.publish(guide_msg)

    @staticmethod
    def _kv(key, value):
        item = KeyValue()
        item.key = str(key)
        item.value = str(value)
        return item

    def _px4_facts(self):
        facts = {
            'px4_status_topic': self.topics['status'],
            'px4_status_received': self.status_seen,
            'px4_mode': self.nav_state_name(),
            'px4_failsafe': self.px4_failsafe,
            'px4_failsafe_flags': '|'.join(self.failsafe_flags) if self.failsafe_flags else 'none',
            'px4_last_ack': ', '.join(f'{_COMMAND_NAMES[c]}={r}' for c, r in self.last_ack.items()) or 'none',
            'scenario_source': self.scenario_source,
        }
        errors, warnings = [], []
        if self.px4_failsafe:
            errors.append('PX4_FAILSAFE')
        if not self.status_seen and self.first_odom_t is not None and self.now_s() - self.first_odom_t > 3.0:
            warnings.append('NO_PX4_VEHICLE_STATUS')
        if self.scenario_source.startswith('PACKAGED EXAMPLE'):
            warnings.append('PACKAGED_EXAMPLE_SCENARIO')
        facts['_errors'], facts['_warnings'] = errors, warnings
        return facts

    def publish_diagnostics(self):
        t = self.now_s()
        level, message, _warnings, values = build_diagnostics(
            self.rt, t, self.p, odom_fresh=self.odom_fresh(), px4=self._px4_facts())
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = 'px4_nmpc/controller'
        st.hardware_id = 'px4'
        st.level = {'OK': DiagnosticStatus.OK, 'WARN': DiagnosticStatus.WARN,
                    'ERROR': DiagnosticStatus.ERROR}[level]
        st.message = message
        st.values = [self._kv(k, v) for k, v in values.items()]
        msg.status = [st]
        self.pub_diag.publish(msg)

    def status_tick(self):
        d = self.rt.diagnostics(self.now_s())
        if self.rt.state is None:
            self.get_logger().info(f'Waiting for PX4 VehicleOdometry on {self.topics["odom"]} ...',
                                   throttle_duration_sec=5.0)
            return
        if not self.status_seen and not self.status_missing_reported and self.first_odom_t is not None \
                and self.now_s() - self.first_odom_t > 3.0:
            self.status_missing_reported = True
            self.get_logger().error(self._status_topic_hint() + ' /nmpc/start is refused until it arrives.')
        ema = d['solve_ms_ema']
        self.get_logger().info(
            f"[{d['phase']}] d_goal={d.get('distance_to_goal_m', float('nan')):.2f} m, "
            f"solve={d['last_solve_ms']:.1f} ms (ema {ema:.1f}, it={d['iterations']}), "
            f"cmd={d['applied_source']}[{d['applied_stage']}], status={d['solver_status']}, "
            f"px4={'armed' if self.rt.armed else 'disarmed'}/{self.nav_state_name()}"
            + (' FAILSAFE' if self.px4_failsafe else ''))

    # ------------------------------------------------------------ services
    def start_cb(self, request, response):
        del request
        if self.rt.state is None or not self.odom_fresh():
            response.success = False
            response.message = f'Cannot start: waiting for fresh PX4 odometry on {self.topics["odom"]}.'
            return response
        if not self.status_seen:
            response.success = False
            response.message = 'Cannot start: ' + self._status_topic_hint()
            self.get_logger().error(response.message)
            return response
        if self.px4_failsafe:
            response.success = False
            response.message = (f'Cannot start: PX4 failsafe is active (mode {self.nav_state_name()}, flags: '
                                f'{", ".join(self.failsafe_flags) or "unknown"}). Resolve it first.')
            self.get_logger().error(response.message)
            return response
        ok, msg = self.rt.start(self.now_s())
        self.land_sent = False
        self.last_cmd_t = -1e9
        self.first_cmd_t = None
        self.arming_started_t = None
        self.last_ack = {}
        self.path_actual = []
        response.success = bool(ok)
        response.message = ('NMPC experiment started: streaming offboard setpoints, then OFFBOARD+ARM, '
                            'takeoff, NMPC.') if ok else msg
        self.get_logger().info(response.message)
        return response

    def _stop_vehicle(self):
        """Land if flying/possibly flying; only disarm when safely non-airborne.

        In particular an ``ABORTED`` runtime can still correspond to an armed
        vehicle after PX4 left OFFBOARD.  Falling through to an immediate
        disarm in that state would be unsafe, so an armed ABORTED vehicle is
        conservatively sent a LAND command.
        """
        phase = self.rt.phase
        flying_phase = phase in ('TAKEOFF', 'NMPC', 'GOAL', 'LANDING')
        armed_abort = bool(self.rt.armed and phase == 'ABORTED')
        airborne_by_height = False
        if self.rt.armed and self.rt.state is not None:
            ground = 0.0 if self.rt.ground_d is None else float(self.rt.ground_d)
            height = ground - float(self.rt.state['pos'][2])
            airborne_by_height = height > max(0.20, 0.5 * float(self.rt.airborne_height))
        if flying_phase or armed_abort or airborne_by_height:
            self.rt.begin_landing(self.now_s())
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.land_sent = True
            return 'LAND commanded; controller remains in LANDING until PX4 reports disarmed'
        self.rt.stop(self.now_s())
        if bool(self.p['disarm_on_stop']) and self.rt.armed:
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 0.0)
            return 'disarm commanded'
        return 'no vehicle command sent'

    def stop_cb(self, request, response):
        del request
        action = self._stop_vehicle()
        response.success = True
        response.message = f'NMPC experiment stopped: {action}.'
        self.get_logger().info(response.message)
        return response

    def reset_cb(self, request, response):
        del request
        if self.rt.phase in ('ARMING', 'TAKEOFF', 'NMPC', 'GOAL', 'LANDING'):
            response.success = False
            response.message = (
                f'Controller reset rejected in {self.rt.phase}: call /nmpc/stop, wait for PX4 to disarm/IDLE, '
                'then reset. This prevents an airborne vehicle from being exposed as IDLE.')
            self.get_logger().warning(response.message)
            return response
        if self.rt.armed:
            # e.g. ABORTED while PX4 flies its own RTL/Land: a reset must neither
            # override PX4's failsafe with a LAND command nor report the vehicle IDLE.
            response.success = False
            response.message = (
                f'Controller reset rejected: PX4 is still armed (mode {self.nav_state_name()}). '
                'Wait until PX4 has landed and disarmed (or call /nmpc/stop), then reset.')
            self.get_logger().warning(response.message)
            return response
        self._stop_vehicle()
        self.rt.reset_keep_state()
        self.path_actual = []
        response.success = True
        response.message = ('Controller state reset. Physical Gazebo/PX4 pose is unchanged; restart SITL for a '
                            'full experiment reset to the randomized start pose.')
        self.get_logger().info(response.message)
        return response

    def _load_scenario_geometry(self):
        path_text = _use_scenario_file(self.p.get('scenario_json', ''))
        if not path_text:
            raise RuntimeError('scenario_json parameter is empty/none')
        scenario = load_scenario_json(pathlib.Path(path_text).expanduser())
        sp = scenario_parameters(scenario, scenario_wall_margin(scenario, self.p.get('planner_margin', 0.45)))
        merged = dict(self.p)
        merged.update(sp)
        lb, ub = arena_bounds(merged)
        geom = controller_geometry(scenario)
        obstacles = [(np.asarray(c, float), float(r)) for c, r in zip(
            geom['obstacle_centers_local_ned'], scenario['obstacle_radii'])]
        guidance = {
            'planner_xy_min': np.asarray(sp['planner_xy_min'], float),
            'planner_xy_max': np.asarray(sp['planner_xy_max'], float),
        }
        return scenario, sp, obstacles, lb, ub, guidance, np.asarray(geom['goal_local_ned'], float)

    def reload_scenario_cb(self, request, response):
        del request
        if self.rt.phase not in ('IDLE', 'GOAL', 'ABORTED'):
            response.success = False
            response.message = (
                f'Environment reload is allowed only in IDLE, GOAL or ABORTED, not {self.rt.phase}. '
                'Use /nmpc/goal for an in-flight retarget without changing obstacles.')
            return response
        try:
            scenario, sp, obstacles, lb, ub, guidance, goal = self._load_scenario_geometry()
            ok, text = self.rt.set_environment(
                obstacles, arena_lb=lb, arena_ub=ub, guidance_overrides=guidance,
                goal=goal, t=self.now_s(), resume=False)
            if not ok:
                raise RuntimeError(text)
            self.p.update(sp)
            self.scenario_source = (f'{_use_scenario_file(self.p.get("scenario_json"))} '
                                    f'(seed {scenario.get("seed")}, {len(obstacles)} obstacles)')
            self.land_sent = False
            self.last_solution_seen = None
            response.success = True
            response.message = text + '; staged while holding current setpoint'
            self.get_logger().info(
                f'Reloaded scenario: {len(obstacles)} obstacles, goal={goal.tolist()} NED; {response.message}')
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f'Could not reload scenario: {exc}'
            self.get_logger().error(response.message)
        return response

    def commit_staged_goal_cb(self, request, response):
        """Resume a GOAL-held vehicle toward the goal already staged by reload_scenario."""
        del request
        if self.rt.phase != 'GOAL':
            response.success = False
            response.message = f'No staged GOAL mission to commit (phase={self.rt.phase}).'
            return response
        goal = self.rt.goal.copy()
        ok, text = self.rt.set_goal(goal, self.now_s(), resume=True)
        if ok:
            self.land_sent = False
            self.last_solution_seen = None
            response.success = True
            response.message = text
            self.get_logger().info(f'Committed staged goal {goal.tolist()} NED: {text}')
        else:
            response.success = False
            response.message = text
            self.get_logger().error(f'Could not commit staged goal {goal.tolist()}: {text}')
        return response

    def status_srv_cb(self, request, response):
        del request
        if self.rt.state is None:
            response.success = False
            response.message = f'No PX4 odometry received yet on {self.topics["odom"]}.'
            return response
        d = self.rt.diagnostics(self.now_s())
        response.success = True
        response.message = (
            f"phase={d['phase']}, leg={d.get('leg_index', 0)}, goals={d.get('goals_completed', 0)}, "
            f"interface={d['interface']}, backend={d['backend']}, "
            f"d_goal={d.get('distance_to_goal_m', float('nan')):.2f} m, solver={d['solver_status']}, "
            f"solve={d['last_solve_ms']:.1f} ms, cmd={d['applied_source']}[{d['applied_stage']}], "
            f"usable={d['usable_solves']}/{d['solves']}, holds={d['hold_events']}, "
            f"px4={'armed' if self.rt.armed else 'disarmed'}/{self.nav_state_name()}"
            + (f", abort_reason={d['abort_reason']}" if d.get('abort_reason') else ''))
        return response

    def destroy_node(self):
        self._stop_evt.set()
        if self._solver_thread is not None:
            self._solver_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NmpcNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

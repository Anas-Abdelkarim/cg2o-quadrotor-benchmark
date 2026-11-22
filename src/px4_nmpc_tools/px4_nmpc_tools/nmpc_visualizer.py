"""RViz visualization for PX4 NMPC missions.

The node is intentionally outside the control loop. It renders the arena,
physical obstacles and safety shells, live vehicle pose, global A* guide,
local NMPC reference, predicted trajectory, actual trajectory, hold point,
current leg start/goal, and controller diagnostics on ``/nmpc/visualization``.

Visualization must never prevent the controller from starting. If the runtime
scenario file does not exist yet, the visualizer uses the launch-provided
packaged scenario, or finally a geometry snapshot constructed from nmpc.yaml.
When a runtime scenario later appears or changes it is hot-reloaded.
"""
import math
import pathlib

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from px4_nmpc_core.config import DEFAULTS, arena_bounds
from px4_nmpc_core.math_utils import ned_to_enu, quat_wxyz_to_euler
from px4_nmpc_core.px4_interface import diagnostic_level, px4_topic, topic_version_mode
from px4_nmpc_core.scenario import controller_geometry, load_scenario_json

try:  # optional: publish map -> nmpc_vehicle so RViz's fixed frame 'map' exists in TF
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster
except ImportError:  # pragma: no cover - tf2_ros is part of every ROS 2 desktop install
    TransformBroadcaster = None


def _file_param(text):
    text = str(text or '').strip()
    return '' if text.lower() in ('', 'none', 'off', 'false') else text


def parse_vector(text):
    """'[1.0, 2.0, 3.0]' -> np.array, 'none'/'' -> None."""
    if text is None:
        return None
    text = str(text).strip()
    if not text or text.lower() in ('none', 'nan', 'n/a'):
        return None
    try:
        v = np.asarray([float(x) for x in text.strip('[]').split(',')], dtype=float)
    except ValueError:
        return None
    return v if v.shape == (3,) and np.all(np.isfinite(v)) else None


def first_key(diag, *keys, default='n/a'):
    """Return the first present diagnostics value."""
    for k in keys:
        if k in diag and diag[k] not in ('', None):
            return diag[k]
    return default


def status_lines(diag, pred_age):
    """Multi-line overlay text (pure function, unit-tested)."""
    mission = first_key(diag, 'mission_state', 'phase')
    result = first_key(diag, 'experiment_result')
    iface = first_key(diag, 'interface', default='')
    backend = first_key(diag, 'backend', default='')
    dgoal = first_key(diag, 'distance_to_goal_m')
    nav_t = first_key(diag, 'navigation_elapsed_s')
    solve = first_key(diag, 'last_solve_ms')
    ratio = first_key(diag, 'deadline_ratio')
    iterations = first_key(diag, 'last_solver_iterations', 'iterations')
    source = first_key(diag, 'applied_source')
    accepted = first_key(diag, 'solve_accepted', 'usable_solves', default='0')
    total = first_key(diag, 'solve_total', 'solves', default='0')
    failed = first_key(diag, 'solve_failed', 'failed_solves', default='0')
    rejected = first_key(diag, 'solve_rejected', 'rejected_unsafe', default='0')
    nmpc_pct = first_key(diag, 'nmpc_control_percent', default='0')
    hold_pct = first_key(diag, 'recovery_control_percent', default='0')
    clearance = first_key(diag, 'min_safety_clearance_m')
    warnings = first_key(diag, 'warnings', default='none')
    warning_lines = '\n'.join(warnings.split('|')) if warnings not in ('none', 'n/a') else 'OK'
    leg = first_key(diag, 'leg_index', default='0')
    done = first_key(diag, 'goals_completed', default='0')
    goal = first_key(diag, 'goal_ned', default='n/a')
    px4_mode = first_key(diag, 'px4_mode', default='')
    armed = 'armed' if str(diag.get('armed', '')).lower() == 'true' else 'disarmed'
    # RViz draws every space two characters wide: keep the overlay compact.
    head = f'{mission}/{result} leg={leg} goals={done}'
    if iface:
        head += f' [{iface}{"/" + backend if backend else ""}]'
    pa = f'{pred_age:.2f}s' if math.isfinite(pred_age) else 'n/a'
    lines = [
        head,
        f'goal={str(goal).replace(", ", ",")}',
        f'd_goal={dgoal}m nav_t={nav_t}s clear={clearance}m',
        f'solve={solve}ms ({ratio}x) iter={iterations} age={pa}',
        f'cmd={source} ok={accepted}/{total} fail={failed} rej={rejected}',
        f'NMPC={nmpc_pct}% hold={hold_pct}%',
    ]
    if px4_mode:
        fs = ' FAILSAFE' if str(diag.get('px4_failsafe', '')).lower() == 'true' else ''
        lines.append(f'px4={armed}/{px4_mode}{fs}')
    lines.append(warning_lines)
    return '\n'.join(lines)


class NmpcVisualizer(Node):
    """RViz visualization and diagnosis kept completely outside the control loop."""

    def __init__(self):
        super().__init__('nmpc_visualizer')
        self.declare_parameter('scenario_json', str(pathlib.Path.home() / '.ros/px4_nmpc/scenario.json'))
        self.declare_parameter('nmpc_config_yaml', '')
        self.declare_parameter('scenario_yaml', '')
        self.declare_parameter('fallback_scenario_json', '')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('uav_radius', float(DEFAULTS['uav_radius']))
        self.declare_parameter('safety_margin', float(DEFAULTS['safety_margin']))
        self.declare_parameter('prediction_warn_after_s', 0.45)
        self.declare_parameter('prediction_stale_after_s', 0.90)
        self.declare_parameter('prediction_hide_after_s', 2.40)
        self.declare_parameter('actual_path_sample_period_s', 0.10)
        self.declare_parameter('actual_path_history_s', 60.0)
        self.declare_parameter('actual_path_recent_s', 15.0)
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('text_size', 0.16)

        # controller parameters: DEFAULTS <- nmpc.yaml (same precedence as the node)
        self.params = dict(DEFAULTS)
        self.params['uav_radius'] = float(self.get_parameter('uav_radius').value)
        self.params['safety_margin'] = float(self.get_parameter('safety_margin').value)
        for pname in ('nmpc_config_yaml', 'scenario_yaml'):
            config_yaml = _file_param(self.get_parameter(pname).value)
            if not config_yaml:
                continue
            cfg_path = pathlib.Path(config_yaml).expanduser()
            if cfg_path.exists():
                raw = yaml.safe_load(cfg_path.read_text()) or {}
                self.params.update(raw.get('px4_nmpc_controller', {}).get('ros__parameters', {}) or {})
            else:
                self.get_logger().warning(f'{pname} not found: {cfg_path}; ignored')
        self.uav_radius = float(self.params['uav_radius'])
        self.safety_margin = float(self.params['safety_margin'])

        self.warn_after = float(self.get_parameter('prediction_warn_after_s').value)
        self.stale_after = float(self.get_parameter('prediction_stale_after_s').value)
        self.hide_after = float(self.get_parameter('prediction_hide_after_s').value)
        self.actual_period = float(self.get_parameter('actual_path_sample_period_s').value)
        self.actual_history_s = float(self.get_parameter('actual_path_history_s').value)
        self.actual_recent_s = float(self.get_parameter('actual_path_recent_s').value)
        self.text_size = float(self.get_parameter('text_size').value)

        scenario_text = _file_param(self.get_parameter('scenario_json').value)
        self.scenario_path = pathlib.Path(scenario_text).expanduser() if scenario_text else None
        fallback_text = _file_param(self.get_parameter('fallback_scenario_json').value)
        self.fallback_scenario_path = pathlib.Path(fallback_text).expanduser() if fallback_text else None
        if self.scenario_path is None:
            self.fallback_scenario_path = None      # scenario_json:=none -> ROS parameter geometry
        self.scenario_mtime_ns = None
        try:
            if self.scenario_path is None:
                raise FileNotFoundError('scenario_json disabled')
            scenario = self._load_scenario(self.scenario_path)
            source = str(self.scenario_path)
        except Exception as primary_exc:
            scenario = None
            source = 'nmpc.yaml geometry'
            if self.fallback_scenario_path is not None:
                try:
                    scenario = self._load_scenario(self.fallback_scenario_path)
                    source = str(self.fallback_scenario_path)
                except Exception as fallback_exc:
                    self.get_logger().warning(
                        f'Runtime scenario unavailable ({primary_exc}); packaged fallback unavailable '
                        f'({fallback_exc}); using nmpc.yaml geometry.')
            if scenario is None:
                scenario = self._scenario_from_params()
            else:
                self.get_logger().warning(
                    f'Runtime scenario unavailable ({primary_exc}); visualizing packaged scenario until it appears.')
        self._configure_scenario(scenario)
        self.get_logger().info(f'Visualizer initial geometry source: {source}')

        self.state = None
        self.actual = []  # list[(stamp_ns, local_ned_xyz)]
        self.last_actual_ns = None
        self.predicted = None
        self.reference = None
        self.global_path = None
        self.prediction_received_ns = None
        self.diag = {}

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        odom_topic = px4_topic(
            self.params.get('px4_topic_prefix', '/fmu'), 'out', 'vehicle_odometry', VehicleOdometry,
            topic_version_mode(self.params.get('px4_versioned_topics', 'auto')))
        self.create_subscription(VehicleOdometry, odom_topic, self.odom_cb, px4_qos)
        self.get_logger().info(f'Visualizer PX4 odometry topic: {odom_topic}')
        self.create_subscription(Path, '/nmpc/predicted_path', self.pred_cb, 10)
        self.create_subscription(Path, '/nmpc/reference_path', self.ref_cb, 10)
        self.create_subscription(Path, '/nmpc/global_path', self.global_cb, 10)
        self.create_subscription(Path, '/nmpc/guide_path', self.global_cb, 10)
        self.create_subscription(DiagnosticArray, '/nmpc/diagnostics', self.diag_cb, 10)
        self.pub = self.create_publisher(MarkerArray, '/nmpc/visualization', 10)
        self.tf_broadcaster = None
        self._last_tf_ns = 0
        if bool(self.get_parameter('publish_tf').value) and TransformBroadcaster is not None:
            self.tf_broadcaster = TransformBroadcaster(self)

        hz = float(self.get_parameter('publish_hz').value)
        self.timer = self.create_timer(1.0 / max(hz, 1.0), self.publish)
        self.get_logger().info('NMPC visualizer ready on /nmpc/visualization. RViz Fixed Frame must be map.')

    # ------------------------------------------------------------------ setup
    def _scenario_from_params(self):
        flat = np.asarray(self.params.get('obstacle_centers_flat', []), dtype=float).reshape(-1)
        if flat.size % 3 != 0:
            flat = np.zeros(0, dtype=float)
        centers = flat.reshape((-1, 3)).tolist() if flat.size else []
        radii = [float(v) for v in self.params.get('obstacle_radii', [])]
        if len(radii) != len(centers):
            n = min(len(radii), len(centers))
            centers, radii = centers[:n], radii[:n]
        return {
            'schema_version': 5,
            'coordinate_frame': 'arena_ned_floor_center_origin',
            'controller_frame': 'px4_local_ned_persistent_origin',
            'start_ned': [0.0, 0.0, 0.0],
            'controller_origin_ned': [0.0, 0.0, 0.0],
            'goal_ned': [float(v) for v in self.params.get('goal_ned', [0.0, 0.0, -1.0])],
            'obstacle_centers_ned': centers,
            'obstacle_radii': radii,
            'arena_north_m': float(self.params.get('arena_north_m', 14.0)),
            'arena_east_m': float(self.params.get('arena_east_m', 10.0)),
            'arena_height_m': float(self.params.get('arena_height_m', 5.0)),
            'takeoff_local_ned': [0.0, 0.0, -float(self.params.get('takeoff_height_m', 1.0))],
        }

    @staticmethod
    def _load_scenario(path):
        return load_scenario_json(path)

    def _configure_scenario(self, scenario):
        self.scenario = scenario
        geometry = controller_geometry(self.scenario)
        self.start = np.asarray(geometry['start_local_ned'], dtype=float)
        self.goal = np.asarray(geometry['goal_local_ned'], dtype=float)
        self.obstacles = [
            (np.asarray(c, dtype=float), float(r))
            for c, r in zip(geometry['obstacle_centers_local_ned'], self.scenario['obstacle_radii'])
        ]
        self.arena_origin = np.asarray(geometry['arena_origin_local_ned'], dtype=float)
        # Challenging course: posts (1) and bars (2) in the controller frame, and its guide line.
        self.typed = []
        for kind, a, b, c, radius in self.scenario.get('typed_obstacles', []):
            if int(kind) in (1, 2):
                self.typed.append((int(kind), np.array([a, b, c], float) + self.arena_origin, float(radius)))
        self.guide_line = []
        guide = self.scenario.get('guide')
        if guide:
            n0, n1 = float(self.scenario['start_ned'][0]), float(self.scenario['goal_ned'][0])
            for n in np.linspace(n0, n1, 113):
                e = guide['e_amp'] * np.sin(guide['e_freq'] * n + guide['e_phase'])
                alt = guide['alt0'] + guide['alt_amp'] * np.sin(guide['alt_freq'] * n + guide['alt_phase'])
                self.guide_line.append(ned_to_enu(np.array([n, e, -alt]) + self.arena_origin))
        arena_p = dict(self.params)
        arena_p.update({
            'arena_north_m': float(self.scenario['arena_north_m']),
            'arena_east_m': float(self.scenario['arena_east_m']),
            'arena_height_m': float(self.scenario['arena_height_m']),
            'arena_origin_local_ned': self.arena_origin.tolist(),
            'enforce_arena_bounds': True,
        })
        self.arena_lb, self.arena_ub = arena_bounds(arena_p)
        self.takeoff_planned = self._planned_takeoff(geometry)
        try:
            self.scenario_mtime_ns = self.scenario_path.stat().st_mtime_ns if self.scenario_path else None
        except OSError:
            self.scenario_mtime_ns = None

    def _maybe_reload_scenario(self):
        """Hot-reload geometry when nmpc_mission next replaces scenario.json."""
        if self.scenario_path is None:
            return
        try:
            mtime = self.scenario_path.stat().st_mtime_ns
        except OSError:
            return
        if self.scenario_mtime_ns is not None and mtime == self.scenario_mtime_ns:
            return
        try:
            self._configure_scenario(self._load_scenario(self.scenario_path))
            # Do not render paths from the previous obstacle layout against the
            # new geometry while waiting for fresh controller publications.
            self.predicted = None
            self.reference = None
            self.global_path = None
            self.prediction_received_ns = None
            self.get_logger().info('Visualizer hot-reloaded scenario geometry and cleared stale planned paths.')
        except Exception as exc:
            self.get_logger().warning(f'Could not hot-reload scenario geometry: {exc}')

    def _planned_takeoff(self, geometry):
        """Take-off point before the controller reports its actual target."""
        if 'takeoff_local_ned' in geometry:
            return np.asarray(geometry['takeoff_local_ned'], dtype=float)
        # A schema-5 continuation leg deliberately has no takeoff corridor: it
        # starts at the already-airborne current vehicle pose.
        if np.linalg.norm(self.start) > 1e-6:
            return self.start.copy()
        if 'takeoff_local_ned' in self.scenario:
            return np.asarray(self.scenario['takeoff_local_ned'], dtype=float)
        alt = float(self.params.get('takeoff_altitude_m', -1.0) or -1.0)
        if alt > 0.0:                                             # explicit altitude above floor
            return np.array([0.0, 0.0, self.arena_origin[2] - alt])
        return np.array([0.0, 0.0, -float(self.params.get('takeoff_height_m', 1.0))])  # height above start

    # ------------------------------------------------------------------ callbacks
    def odom_cb(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED or msg.velocity_frame != VehicleOdometry.VELOCITY_FRAME_NED:
            return
        raw = np.r_[np.asarray(msg.position, dtype=float), np.asarray(msg.velocity, dtype=float), np.asarray(msg.q, dtype=float)]
        if not np.all(np.isfinite(raw)):
            return
        euler = quat_wxyz_to_euler(msg.q)
        self.state = np.r_[np.asarray(msg.position, dtype=float), np.asarray(msg.velocity, dtype=float), euler]
        self._broadcast_tf(np.asarray(msg.position, dtype=float), np.asarray(msg.q, dtype=float))
        now = self.get_clock().now().nanoseconds
        if self.last_actual_ns is None or (now - self.last_actual_ns) * 1e-9 >= self.actual_period:
            self.actual.append((now, self.state[:3].copy()))
            cutoff = now - int(max(1.0, self.actual_history_s) * 1e9)
            while self.actual and self.actual[0][0] < cutoff:
                self.actual.pop(0)
            self.last_actual_ns = now

    def _broadcast_tf(self, pos_ned, q_ned_frd):
        """map (ENU) -> nmpc_vehicle (FLU), so RViz knows the 'map' frame."""
        if self.tf_broadcaster is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_tf_ns < 20_000_000:          # <= 50 Hz is plenty for RViz
            return
        self._last_tf_ns = now_ns
        # q_ENU_FLU = q_NED->ENU * q_NED_FRD * q_FRD->FLU  (Hamilton, w-x-y-z)
        w, x, y, z = [float(v) for v in q_ned_frd]
        s2 = math.sqrt(0.5)
        a = (0.0, s2, s2, 0.0)                     # NED -> ENU
        def mul(p, q):
            pw, px, py, pz = p
            qw, qx, qy, qz = q
            return (pw*qw - px*qx - py*qy - pz*qz, pw*qx + px*qw + py*qz - pz*qy,
                    pw*qy - px*qz + py*qw + pz*qx, pw*qz + px*qy - py*qx + pz*qw)
        q = mul(mul(a, (w, x, y, z)), (0.0, 1.0, 0.0, 0.0))   # FRD -> FLU = 180 deg about x
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'nmpc_vehicle'
        enu = ned_to_enu(pos_ned)
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = [float(v) for v in enu]
        t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = q
        try:
            self.tf_broadcaster.sendTransform(t)
        except Exception:  # noqa: BLE001 - visualization only
            pass

    def pred_cb(self, msg):
        self.predicted = msg
        self.prediction_received_ns = self.get_clock().now().nanoseconds

    def ref_cb(self, msg):
        self.reference = msg

    def global_cb(self, msg):
        self.global_path = msg

    def diag_cb(self, msg):
        if not msg.status:
            return
        status = msg.status[0]
        values = {kv.key: kv.value for kv in status.values}
        values['_message'] = status.message
        values['_level'] = str(diagnostic_level(status.level))
        self.diag = values
        live_goal = parse_vector(values.get('goal_ned'))
        if live_goal is not None:
            self.goal = live_goal
        live_start = parse_vector(values.get('mission_start_ned'))
        if live_start is not None:
            self.start = live_start

    # ------------------------------------------------------------------ helpers
    def _age(self, stamp_ns):
        if stamp_ns is None:
            return float('inf')
        return max(0.0, (self.get_clock().now().nanoseconds - stamp_ns) * 1e-9)

    @staticmethod
    def _point(xyz):
        p = Point()
        p.x, p.y, p.z = [float(v) for v in xyz]
        return p

    def _base(self, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    @staticmethod
    def _set_color(m, rgba):
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]

    def _sphere(self, ns, mid, ned, radius, rgba, mtype=None):
        m = self._base(ns, mid, Marker.SPHERE if mtype is None else mtype)
        p = ned_to_enu(ned)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) for v in p]
        m.scale.x = m.scale.y = m.scale.z = 2.0 * float(radius)
        self._set_color(m, rgba)
        return m

    def _line(self, ns, mid, points_enu, width, rgba):
        m = self._base(ns, mid, Marker.LINE_STRIP)
        m.scale.x = float(width)
        m.points = [self._point(p) for p in points_enu]
        self._set_color(m, rgba)
        return m

    @staticmethod
    def _path_points(path_msg):
        if path_msg is None:
            return []
        return [
            np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z], dtype=float)
            for p in path_msg.poses
        ]

    def _arena_edges(self):
        lb, ub = self.arena_lb, self.arena_ub
        corners = [np.array([nn, ee, dd], float)
                   for dd in (lb[2], ub[2]) for nn in (lb[0], ub[0]) for ee in (lb[1], ub[1])]
        pairs = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        out = []
        for a, b in pairs:
            out.extend([ned_to_enu(corners[a]), ned_to_enu(corners[b])])
        return out

    @staticmethod
    def _safe_float(value, default=float('nan')):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def takeoff_point(self):
        actual = parse_vector(self.diag.get('takeoff_target_ned'))
        return actual if actual is not None else self.takeoff_planned

    # ------------------------------------------------------------------ publish
    def build_markers(self):
        self._maybe_reload_scenario()
        arr = MarkerArray()
        clear = self._base('reset', 0, Marker.SPHERE)
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # Obstacles: opaque physical sphere plus transparent optimization shell.
        for i, (center, radius) in enumerate(self.obstacles):
            arr.markers.append(self._sphere('obstacle_physical', i, center, radius, (0.95, 0.38, 0.08, 0.95)))
            effective = radius + self.uav_radius + self.safety_margin
            arr.markers.append(self._sphere('obstacle_safety', i, center, effective, (0.95, 0.20, 0.10, 0.14)))

        height = float(self.scenario.get('arena_height_m', 6.0)) if self.scenario else 6.0
        width = float(self.scenario.get('arena_east_m', 10.0)) if self.scenario else 10.0
        for i, (kind, center, radius) in enumerate(getattr(self, 'typed', [])):
            m = self._base('obstacle_rod', i, Marker.CYLINDER)
            c = np.array(center, float)
            if kind == 1:      # vertical post, floor to ceiling
                c[2] = self.arena_origin[2] - 0.5 * height
                m.scale.x = m.scale.y = 2.0 * radius
                m.scale.z = height
                rgba = (0.85, 0.55, 0.10, 0.95)
            else:              # bar along East across the hall
                c[1] = self.arena_origin[1]
                m.scale.x = m.scale.y = 2.0 * radius
                m.scale.z = width
                # 90 deg about ENU y: the cylinder axis (z) points East (x)
                m.pose.orientation.w = float(np.cos(np.pi / 4))
                m.pose.orientation.y = float(np.sin(np.pi / 4))
                rgba = (0.55, 0.20, 0.75, 0.95)
            p = ned_to_enu(c)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) for v in p]
            self._set_color(m, rgba)
            arr.markers.append(m)
        if getattr(self, 'guide_line', None):
            arr.markers.append(self._line('guide_line', 0, self.guide_line, 0.03, (0.95, 0.95, 0.95, 0.6)))

        takeoff = self.takeoff_point()
        arr.markers.append(self._sphere('goal', 0, self.goal, 0.24, (0.10, 0.90, 0.25, 0.95)))
        arr.markers.append(self._sphere('start', 0, self.start, 0.14, (0.70, 0.72, 0.78, 0.75)))
        arr.markers.append(self._sphere('takeoff_stage', 0, takeoff, 0.17, (0.10, 0.85, 0.95, 0.75)))
        for ns, mid, point, label, rgba in (
            ('labels', 10, self.start, 'LEG START', (0.82, 0.84, 0.88, 0.95)),
            ('labels', 11, self.goal, 'CURRENT GOAL', (0.20, 1.00, 0.35, 0.98)),
        ):
            m = self._base(ns, mid, Marker.TEXT_VIEW_FACING)
            q = ned_to_enu(point)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(q[0]), float(q[1]), float(q[2] + 0.35)
            m.scale.z = max(0.12, self.text_size * 0.85)
            self._set_color(m, rgba)
            m.text = label
            arr.markers.append(m)
        arr.markers.append(self._line(
            'takeoff_corridor', 0, [ned_to_enu(self.start), ned_to_enu(takeoff)],
            0.025, (0.10, 0.85, 0.95, 0.55)))

        arena = self._base('arena_bounds', 0, Marker.LINE_LIST)
        arena.scale.x = 0.025
        arena.points = [self._point(p) for p in self._arena_edges()]
        self._set_color(arena, (0.65, 0.70, 0.78, 0.85))
        arr.markers.append(arena)

        hold = parse_vector(self.diag.get('hold_point_ned'))
        if hold is not None:
            arr.markers.append(self._sphere('px4_hold_point', 0, hold, 0.13, (0.95, 0.20, 0.85, 0.95),
                                            mtype=Marker.CUBE))

        if self.state is not None:
            pos_enu = ned_to_enu(self.state[:3])
            arr.markers.append(self._sphere('vehicle', 0, self.state[:3], 0.18, (0.05, 0.85, 0.95, 0.95)))
            yaw = float(self.state[8])
            head_ned = self.state[:3] + np.array([math.cos(yaw), math.sin(yaw), 0.0]) * 0.8
            arrow = self._base('vehicle_heading', 0, Marker.ARROW)
            arrow.points = [self._point(pos_enu), self._point(ned_to_enu(head_ned))]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.06, 0.12, 0.16
            self._set_color(arrow, (0.05, 0.85, 0.95, 0.95))
            arr.markers.append(arrow)
            arr.markers.append(self._line(
                'goal_line', 0, [pos_enu, ned_to_enu(self.goal)], 0.018, (0.72, 0.72, 0.75, 0.55)))

        # The old trajectory remains visible but faded; the recent segment is emphasized.
        if len(self.actual) >= 2:
            now = self.get_clock().now().nanoseconds
            recent_cutoff = now - int(max(1.0, self.actual_recent_s) * 1e9)
            older = [ned_to_enu(p) for t, p in self.actual if t <= recent_cutoff]
            recent = [ned_to_enu(p) for t, p in self.actual if t >= recent_cutoff]
            if len(older) >= 2:
                arr.markers.append(self._line('actual_old', 0, older, 0.018, (0.50, 0.52, 0.56, 0.24)))
            if len(recent) >= 2:
                arr.markers.append(self._line('actual_recent', 0, recent, 0.035, (0.82, 0.84, 0.88, 0.92)))

        # Global A* guide and local NMPC reference are intentionally distinct.
        global_path = self._path_points(self.global_path)
        if len(global_path) >= 2:
            arr.markers.append(self._line('global_path', 0, global_path, 0.035, (0.70, 0.35, 0.95, 0.78)))

        ref = self._path_points(self.reference)
        if len(ref) >= 2:
            arr.markers.append(self._line('reference_path', 0, ref, 0.045, (0.95, 0.80, 0.10, 0.95)))
            dots = self._base('reference_points', 0, Marker.SPHERE_LIST)
            dots.scale.x = dots.scale.y = dots.scale.z = 0.09
            dots.points = [self._point(p) for p in ref]
            self._set_color(dots, (0.95, 0.80, 0.10, 0.95))
            arr.markers.append(dots)

        pred = self._path_points(self.predicted)
        # Controller-reported age (time since the plan's stage 0). It can be
        # slightly negative right after a solve (latency-compensated start).
        pred_age = self._safe_float(self.diag.get('prediction_age_s'))
        if not math.isfinite(pred_age):
            pred_age = self._age(self.prediction_received_ns)
        pred_age = max(0.0, pred_age) if math.isfinite(pred_age) else pred_age
        if len(pred) >= 2 and pred_age <= self.hide_after:
            if pred_age <= self.warn_after:
                color = (0.10, 0.45, 1.00, 0.98)
            elif pred_age <= self.stale_after:
                color = (1.00, 0.55, 0.05, 0.98)
            else:
                color = (0.95, 0.10, 0.10, 0.92)
            arr.markers.append(self._line('prediction_path', 0, pred, 0.055, color))
            dots = self._base('prediction_points', 0, Marker.SPHERE_LIST)
            dots.scale.x = dots.scale.y = dots.scale.z = 0.085
            dots.points = [self._point(p) for p in pred]
            self._set_color(dots, color)
            arr.markers.append(dots)

        if self.state is not None:
            text = self._base('status_text', 0, Marker.TEXT_VIEW_FACING)
            p = ned_to_enu(self.state[:3])
            text.pose.position.x, text.pose.position.y, text.pose.position.z = float(p[0]), float(p[1]), float(p[2] + 1.15)
            text.scale.z = self.text_size
            level = self.diag.get('_level', '0')
            color = {'0': (0.97, 0.97, 0.97, 0.98), '1': (1.00, 0.72, 0.25, 0.98),
                     '2': (1.00, 0.35, 0.30, 0.98)}.get(str(level), (0.97, 0.97, 0.97, 0.98))
            self._set_color(text, color)
            text.text = status_lines(self.diag, pred_age)
            arr.markers.append(text)
        return arr

    def publish(self):
        self.pub.publish(self.build_markers())


def main(args=None):
    rclpy.init(args=args)
    node = NmpcVisualizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

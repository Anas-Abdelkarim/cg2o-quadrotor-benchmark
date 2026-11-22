"""Visualization and mission-tool integration tests without a ROS installation."""
import json
import pathlib
import sys
import types

import numpy as np
import pytest
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / 'px4_nmpc_core'))
sys.path.insert(0, str(SRC / 'px4_nmpc_controller'))
sys.path.insert(0, str(SRC / 'px4_nmpc_tools'))
sys.path.insert(0, str(SRC / 'px4_nmpc_controller' / 'test'))

import fake_ros  # noqa: E402

CLOCK = fake_ros.install(SRC / 'px4_msgs' / 'msg')

# experiment_manager normally gets package-share paths from ament. Provide the
# source-tree equivalent so mission_cli pure helpers can also be unit-tested.
import types as _types  # noqa: E402
_ament = _types.ModuleType('ament_index_python')
_ament_pkgs = _types.ModuleType('ament_index_python.packages')
_ament_pkgs.get_package_share_directory = lambda name: str(SRC / name)
_ament.packages = _ament_pkgs
sys.modules['ament_index_python'] = _ament
sys.modules['ament_index_python.packages'] = _ament_pkgs

from px4_nmpc_controller import controller_node as cn  # noqa: E402
from px4_nmpc_core.math_utils import euler_to_quat_wxyz, ned_to_enu  # noqa: E402
from px4_nmpc_core.scenario import generate_scenario, scenario_parameters  # noqa: E402
from px4_nmpc_core.sim.px4_like import Px4LikeQuad, SimParams  # noqa: E402
from px4_nmpc_tools import mission_cli, nmpc_monitor, nmpc_visualizer  # noqa: E402

CONFIG = SRC / 'px4_nmpc_config' / 'config'
EXP = yaml.safe_load((CONFIG / 'experiment.yaml').read_text())['experiment']
NMPC_YAML = CONFIG / 'nmpc.yaml'


def _write_json(tmp_path, scenario_dict, name='scenario.json'):
    p = tmp_path / name
    p.write_text(json.dumps(scenario_dict))
    return p


def _markers(viz):
    arr = viz.build_markers()
    return {m.ns: m for m in arr.markers}


def _enu(m):
    return np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])


SAMPLE_DIAG = {
    'mission_state': 'TRACKING', 'experiment_result': 'running', 'navigation_elapsed_s': '4.200',
    'distance_to_goal_m': '3.100', 'speed_mps': '0.700', 'roll_deg': '1.0', 'pitch_deg': '-2.0',
    'yaw_deg': '0.0', 'min_safety_clearance_m': '0.800', 'last_solve_ms': '41.0', 'deadline_ratio': '0.14',
    'prediction_age_s': '0.20', 'last_solver_iterations': '9', 'solver_status': 'Solve_Succeeded',
    'solve_total': '30', 'solve_accepted': '28', 'solve_failed': '1', 'solve_rejected': '1',
    'deadline_rejected': '0', 'recovery_entries': '0', 'nmpc_control_percent': '97.0',
    'recovery_control_percent': '3.0', 'applied_source': 'nmpc_sequence', 'applied_stage': '1',
    'warnings': 'none',
}


def test_monitor_and_overlay_render_complete_diagnostics():
    text, warn = nmpc_monitor.format_report(SAMPLE_DIAG)
    assert '[TRACKING] result=running' in text and 'accepted=28/30' in text and not warn
    overlay = nmpc_visualizer.status_lines(SAMPLE_DIAG, 0.2)
    assert overlay.startswith('TRACKING/running') and 'iter=9' in overlay and 'NMPC=97.0%' in overlay


@pytest.mark.parametrize('interface', ['trajectory', 'attitude'])
def test_tools_show_live_controller_state(tmp_path, interface):
    scn = generate_scenario(EXP, 42)
    scenario_json = _write_json(tmp_path, scn.to_dict())
    params = yaml.safe_load(NMPC_YAML.read_text())['px4_nmpc_controller']['ros__parameters']
    params.update(scenario_parameters(scn, float(EXP['wall_margin_m'])))
    params['control_interface'] = interface
    if interface != 'trajectory':
        params['horizon_steps'] = 25
    params.update(scenario_json=str(scenario_json), nmpc_config_yaml=str(NMPC_YAML))
    CLOCK.t = 0.0
    fake_ros.Node.OVERRIDES = params
    cn.NmpcNode.START_SOLVER_THREAD = False
    node = cn.NmpcNode()
    viz = nmpc_visualizer.NmpcVisualizer()
    assert '/nmpc/global_path' in viz.subscriptions and '/nmpc/guide_path' in viz.subscriptions
    assert '/nmpc/global_path' in node.publishers

    # arena box drawn with exactly the NMPC soft bounds
    assert np.allclose(viz.arena_lb, node.rt.arena_lb) and np.allclose(viz.arena_ub, node.rt.arena_ub)
    # take-off corridor end point from the scenario = runtime target height
    assert np.allclose(viz.takeoff_planned, [0.0, 0.0, -1.35])

    sim = Px4LikeQuad(SimParams(), rng=np.random.default_rng(3))
    forwarded = {k: 0 for k in ('/nmpc/predicted_path', '/nmpc/reference_path', '/nmpc/global_path', '/nmpc/diagnostics')}
    handlers = {'/nmpc/predicted_path': viz.pred_cb, '/nmpc/reference_path': viz.ref_cb,
                '/nmpc/global_path': viz.global_cb, '/nmpc/diagnostics': viz.diag_cb}
    seen_phases = {}
    next_odom = next_status = 0.0
    started = False
    n_cmd = 0
    odom_topic = '/fmu/out/vehicle_odometry'
    while sim.t < 45.0:
        sim.step(0.004)
        CLOCK.t = sim.t
        if sim.t >= next_odom:
            next_odom += 0.01
            od = sim.odometry()
            if od is not None:
                msg = cn.VehicleOdometry()
                msg.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
                msg.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
                msg.position, msg.velocity = list(od[1]), list(od[2])
                msg.q = list(euler_to_quat_wxyz(*od[3]))
                node.subscriptions[odom_topic](msg)
                viz.odom_cb(msg)
        if sim.t >= next_status:
            next_status += 0.1
            st = cn.VehicleStatus()
            st.arming_state = cn.VehicleStatus.ARMING_STATE_ARMED if sim.armed else 1
            st.nav_state = cn.VehicleStatus.NAVIGATION_STATE_OFFBOARD if sim.offboard else 4
            node.subscriptions['/fmu/out/vehicle_status_v1'](st)
        if not started and sim.t > 0.5:
            assert node.services['/nmpc/start'](None, types.SimpleNamespace(success=False, message='')).success
            started = True
        for timer in node.timers:
            if sim.t >= timer.next:
                timer.next += timer.period
                timer.cb()
        node.solve_once()
        # forward the node's publications to the visualizer (what DDS would do)
        for topic, cb in handlers.items():
            msgs = node.publishers[topic].msgs
            while forwarded[topic] < len(msgs):
                cb(msgs[forwarded[topic]])
                forwarded[topic] += 1
        mode = node.publishers['/fmu/in/offboard_control_mode'].msgs
        if mode:
            m = mode[-1]
            if m.position:
                t = node.publishers['/fmu/in/trajectory_setpoint'].msgs[-1]
                sim.set_setpoint(types.SimpleNamespace(kind='trajectory', position=np.asarray(t.position),
                                                       velocity=np.asarray(t.velocity),
                                                       acceleration=np.asarray(t.acceleration), yaw=t.yaw))
            else:
                a = node.publishers['/fmu/in/vehicle_attitude_setpoint_v1'].msgs[-1]
                sim.set_setpoint(types.SimpleNamespace(kind='attitude', q_d=np.asarray(a.q_d),
                                                       thrust_normalized=-a.thrust_body[2]))
        cmds = node.publishers['/fmu/in/vehicle_command'].msgs
        if len(cmds) > n_cmd:
            if any(c.command == cn.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and c.param1 == 1.0
                   for c in cmds[n_cmd:]):
                sim.request_offboard_and_arm()
            n_cmd = len(cmds)
        phase = node.rt.phase
        if phase not in seen_phases and viz.diag.get('phase') == phase:
            seen_phases[phase] = (_markers(viz), dict(viz.diag))
        if phase == 'GOAL' and 'GOAL' in seen_phases and sim.t - node.rt.t_goal > 1.0:
            break

    assert {'TAKEOFF', 'NMPC', 'GOAL'} <= set(seen_phases), seen_phases.keys()

    # --- during navigation every overlay/monitor field is populated ------------
    marks, diag = seen_phases['NMPC']
    for ns in ('global_path', 'reference_path', 'prediction_path', 'takeoff_stage', 'takeoff_corridor',
               'obstacle_physical', 'obstacle_safety', 'arena_bounds', 'vehicle', 'status_text'):
        assert ns in marks, ns
    head = marks['status_text'].text.splitlines()[0]
    assert head.startswith('NMPC/running') and 'n/a' not in head
    assert np.allclose(_enu(marks['takeoff_stage']), ned_to_enu(node.rt.takeoff_target), atol=2e-3)
    text, _ = nmpc_monitor.format_report(diag)
    assert '[NMPC] result=running' in text

    marks, diag = seen_phases['GOAL']
    assert marks['status_text'].text.startswith('GOAL/success')
    final_text, warn = nmpc_monitor.format_report(dict(viz.diag))
    assert 'result=success' in final_text and not warn
    assert int(viz.diag['solve_accepted']) > 20 and float(viz.diag['nmpc_control_percent']) > 99.0
    assert float(viz.diag['navigation_elapsed_s']) > 3.0
    assert 'n/a' not in final_text.split('\n')[4]      # counts line fully populated
    node.destroy_node()


def test_visualizer_uses_current_scenario_takeoff_and_controller_hold_override(tmp_path):
    scn = generate_scenario(EXP, 42)
    data = scn.to_dict()
    path = _write_json(tmp_path, data, 'takeoff.json')
    fake_ros.Node.OVERRIDES = {'scenario_json': str(path),
                               'nmpc_config_yaml': str(NMPC_YAML)}
    viz = nmpc_visualizer.NmpcVisualizer()
    expected = np.asarray(data['takeoff_ned'], float) - np.asarray(data['controller_origin_ned'], float)
    assert np.allclose(viz.takeoff_planned, expected)
    # the controller-reported target overrides the plan (e.g. lowered under an obstacle)
    viz.diag = {'takeoff_target_ned': '[0.000, 0.000, -0.900]'}
    assert np.allclose(viz.takeoff_point(), [0.0, 0.0, -0.9])
    viz.diag = {'hold_point_ned': '[1.000, 2.000, -1.500]'}
    assert 'px4_hold_point' in _markers(viz)


def test_visualizer_hot_reloads_scenario_file(tmp_path):
    """A live `nmpc_mission next` rewrite must update RViz geometry without relaunching."""
    base = generate_scenario(EXP, 42).to_dict()
    path = _write_json(tmp_path, base, 'live.json')
    fake_ros.Node.OVERRIDES = {'scenario_json': str(path), 'nmpc_config_yaml': str(NMPC_YAML)}
    viz = nmpc_visualizer.NmpcVisualizer()
    old_goal = viz.goal.copy()

    # Keep the persistent PX4 origin but move the mission-leg goal in arena NED.
    updated = dict(base)
    updated['controller_origin_ned'] = list(base['start_ned'])
    updated['goal_ned'] = [base['goal_ned'][0] - 0.25, base['goal_ned'][1], base['goal_ned'][2]]
    path.write_text(json.dumps(updated, indent=2) + '\n')
    # Ensure mtime changes even on coarse filesystems.
    import os, time
    time.sleep(0.002)
    os.utime(path, None)

    # Old planned paths must not survive a geometry hot reload.
    viz.predicted = object()
    viz.reference = object()
    viz.global_path = object()
    viz.prediction_received_ns = 123
    viz._maybe_reload_scenario()
    assert viz.predicted is None and viz.reference is None and viz.global_path is None
    assert viz.prediction_received_ns is None
    assert not np.allclose(viz.goal, old_goal)
    expected = np.asarray(updated['goal_ned'], float) - np.asarray(updated['controller_origin_ned'], float)
    assert np.allclose(viz.goal, expected)


def test_mission_goal_ack_requires_real_new_leg_and_continuation_preserves_settings():
    goal = np.array([1.0, 2.0, -3.0])
    # A staged environment already advertises the future goal, but is still
    # holding the old terminal point. That must NOT count as /nmpc/goal ACK.
    staged = {'goal_ned': '[1.000, 2.000, -3.000]', 'phase': 'GOAL', 'leg_index': '2'}
    assert not mission_cli._goal_acknowledged(staged, goal, 'GOAL', 2)
    committed = {'goal_ned': '[1.000, 2.000, -3.000]', 'phase': 'NMPC', 'leg_index': '3'}
    assert mission_cli._goal_acknowledged(committed, goal, 'GOAL', 2)
    rejected = {'goal_ned': '[0.000, 0.000, -1.000]', 'phase': 'GOAL', 'leg_index': '2'}
    assert not mission_cli._goal_acknowledged(rejected, goal, 'GOAL', 2)

    current = {'experiment_config': {
        'seed': 42, 'obstacle_count': 17, 'arena_north_m': 19.0,
        'takeoff_altitude_m': 1.2, 'start_ned_override': [1, 2, 3],
        'goal_ned_override': [4, 5, 6], 'controller_origin_ned': [0, 0, 0],
    }}
    cfg = mission_cli._continuation_config(current)
    assert cfg['obstacle_count'] == 17 and cfg['arena_north_m'] == 19.0
    for k in ('seed', 'takeoff_altitude_m', 'start_ned_override',
              'goal_ned_override', 'controller_origin_ned'):
        assert k not in cfg


def test_visualizer_starts_without_runtime_scenario_and_renders_config_geometry(tmp_path):
    """RViz marker production must not depend on running `nmpc_experiment new` first."""
    missing = tmp_path / 'does_not_exist.json'
    fake_ros.Node.OVERRIDES = {
        'scenario_json': str(missing),
        'fallback_scenario_json': '',
        'nmpc_config_yaml': str(NMPC_YAML),
    }
    viz = nmpc_visualizer.NmpcVisualizer()
    marks = _markers(viz)
    assert 'goal' in marks
    assert 'arena_bounds' in marks
    assert 'obstacle_physical' in marks
    assert np.allclose(viz.start, np.zeros(3))


def test_packaged_rviz_config_targets_marker_array_and_map_frame():
    """The saved RViz profile must subscribe to the marker stream the visualizer publishes."""
    rviz_path = SRC / 'px4_nmpc_bringup' / 'config' / 'nmpc.rviz'
    cfg = yaml.safe_load(rviz_path.read_text())
    manager = cfg['Visualization Manager']
    assert manager['Global Options']['Fixed Frame'] == 'map'
    displays = {d['Name']: d for d in manager['Displays']}
    markers = displays['NMPC Visualization']
    assert markers['Enabled'] is True
    assert markers['Class'] == 'rviz_default_plugins/MarkerArray'
    assert markers['Topic']['Value'] == '/nmpc/visualization'
    assert markers['Topic']['Reliability Policy'] == 'Reliable'
    assert markers['Topic']['Durability Policy'] == 'Volatile'


def _diag_array(level, values):
    arr = fake_ros.sys.modules['diagnostic_msgs.msg'].DiagnosticArray()
    st = fake_ros.sys.modules['diagnostic_msgs.msg'].DiagnosticStatus()
    st.level = level
    st.message = 'test'
    kv_cls = fake_ros.sys.modules['diagnostic_msgs.msg'].KeyValue
    for k, v in values.items():
        kv = kv_cls()
        kv.key, kv.value = k, v
        st.values.append(kv)
    arr.status = [st]
    return arr


def test_monitor_accepts_rclpy_byte_levels():
    """rclpy delivers DiagnosticStatus.level as bytes (b'\\x02'); v2.0.1 crashed on int(level)."""
    fake_ros.Node.OVERRIDES = {}
    mon = nmpc_monitor.NmpcMonitor()
    for raw, expected in ((b'\x00', 0), (b'\x01', 1), (b'\x02', 2)):
        mon.diag_cb(_diag_array(raw, dict(SAMPLE_DIAG)))
        assert mon.latest['_level'] == expected
    text, warn = nmpc_monitor.format_report(mon.latest)
    assert warn                                     # ERROR level is always reported as a warning
    mon.print_status()                              # must not raise
    # PX4 failsafe details are shown
    d = dict(SAMPLE_DIAG, px4_mode='AUTO_RTL', armed='True', px4_failsafe='True',
             px4_failsafe_flags='offboard_control_signal_lost')
    text, warn = nmpc_monitor.format_report(d)
    assert 'FAILSAFE' in text and 'offboard_control_signal_lost' in text and warn


def test_visualizer_colours_overlay_by_byte_level(tmp_path):
    path = _write_json(tmp_path, generate_scenario(EXP, 42).to_dict())
    fake_ros.Node.OVERRIDES = {'scenario_json': str(path), 'nmpc_config_yaml': str(NMPC_YAML)}
    viz = nmpc_visualizer.NmpcVisualizer()
    msg = cn.VehicleOdometry()
    msg.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
    msg.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
    msg.position, msg.velocity, msg.q = [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    viz.odom_cb(msg)
    viz.diag_cb(_diag_array(b'\x02', dict(SAMPLE_DIAG)))
    assert viz.diag['_level'] == '2'
    text = _markers(viz)['status_text']
    assert (text.color.r, text.color.g) == (1.0, 0.35)       # red = ERROR


def test_visualizer_can_use_ros_parameter_geometry(tmp_path):
    """scenario_json:=none -> nmpc.yaml (+ scenario_yaml) geometry, no packaged fallback."""
    extra = tmp_path / 'extra.yaml'
    extra.write_text(yaml.safe_dump({'px4_nmpc_controller': {'ros__parameters': {
        'goal_ned': [1.0, 2.0, -2.5], 'obstacle_centers_flat': [3.0, 0.0, -2.0], 'obstacle_radii': [0.5]}}}))
    fake_ros.Node.OVERRIDES = {'scenario_json': 'none', 'nmpc_config_yaml': str(NMPC_YAML),
                               'scenario_yaml': str(extra),
                               'fallback_scenario_json': str(CONFIG / 'scenario_example.json')}
    viz = nmpc_visualizer.NmpcVisualizer()
    assert np.allclose(viz.goal, [1.0, 2.0, -2.5])
    assert len(viz.obstacles) == 1


def test_experiment_load_keeps_the_fixed_reference_line(tmp_path, monkeypatch):
    """`nmpc_experiment load` must pass reference_start_ned (not a Scenario field) on to the controller: without it
    the slalom's fixed reference line starts at the vehicle on the ground (PX4 SITL flew into the floor)."""
    from px4_nmpc_tools import experiment_manager as em
    bundle = SRC / 'px4_nmpc_config' / 'config' / 'experiments' / 'infeasible_slalom'
    monkeypatch.setattr(sys, 'argv', ['nmpc_experiment', '--runtime-dir', str(tmp_path), 'load',
                                      str(bundle / 'scenario_10.json'), '--experiment', str(bundle / 'experiment.yaml')])
    em.main()
    params = yaml.safe_load((tmp_path / 'scenario.yaml').read_text())['px4_nmpc_controller']['ros__parameters']
    assert params['fixed_line_start_ned'] == pytest.approx([0.0, 0.0, -2.35])
    assert params['takeoff_height_m'] == pytest.approx(2.35)
    assert json.loads((tmp_path / 'scenario.json').read_text())['reference_start_ned'] == [-14.0, 0.0, -2.5]

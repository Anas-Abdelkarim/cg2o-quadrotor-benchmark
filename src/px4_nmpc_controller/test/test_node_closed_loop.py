"""Closed-loop test of the real ROS node code without ROS 2.

``fake_ros`` stubs rclpy and generates px4_msgs classes from the bundled
.msg files.  The node's timers/subscriptions are driven by the PX4-like
simulator; the published PX4 messages (field names, signs, NaNs, offboard
flags, versioned topic names) are decoded exactly as PX4 would use them.
"""
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
sys.path.insert(0, str(HERE))

import fake_ros  # noqa: E402

CLOCK = fake_ros.install(SRC / 'px4_msgs' / 'msg')

from px4_nmpc_controller import controller_node as cn  # noqa: E402
from px4_nmpc_core.math_utils import euler_to_quat_wxyz  # noqa: E402
from px4_nmpc_core.scenario import generate_scenario, scenario_parameters  # noqa: E402
from px4_nmpc_core.sim.px4_like import Px4LikeQuad, SimParams  # noqa: E402

CONFIG = SRC / 'px4_nmpc_config' / 'config'


def _params(interface, seed):
    p = yaml.safe_load((CONFIG / 'nmpc.yaml').read_text())['px4_nmpc_controller']['ros__parameters']
    exp = yaml.safe_load((CONFIG / 'experiment.yaml').read_text())['experiment']
    p.update(scenario_parameters(generate_scenario(exp, seed), float(exp['wall_margin_m'])))
    p['control_interface'] = interface
    if interface != 'trajectory':
        p['horizon_steps'] = 25
    return p


def _decode(node, px4):
    """Translate the last published offboard mode + setpoint as PX4 would."""
    mode_pub = node.publishers['/fmu/in/offboard_control_mode']
    if not mode_pub.msgs:
        return None
    mode = mode_pub.msgs[-1]
    if mode.position:
        m = node.publishers['/fmu/in/trajectory_setpoint'].msgs[-1]
        return types.SimpleNamespace(kind='trajectory', position=np.asarray(m.position, float),
                                     velocity=np.asarray(m.velocity, float),
                                     acceleration=np.asarray(m.acceleration, float), yaw=float(m.yaw),
                                     yawspeed=float(m.yawspeed))
    if mode.attitude:
        m = node.publishers['/fmu/in/vehicle_attitude_setpoint_v1'].msgs[-1]
        return types.SimpleNamespace(kind='attitude', q_d=np.asarray(m.q_d, float),
                                     thrust_normalized=-float(m.thrust_body[2]))
    if mode.body_rate:
        m = node.publishers['/fmu/in/vehicle_rates_setpoint'].msgs[-1]
        return types.SimpleNamespace(kind='body_rate', rates=np.array([m.roll, m.pitch, m.yaw], float),
                                     thrust_normalized=-float(m.thrust_body[2]))
    raise AssertionError('offboard mode without any setpoint type')


@pytest.mark.parametrize('interface', ['trajectory', 'attitude', 'body_rate', 'trajectory+yaw'])
def test_ros_node_flies_scenario_to_goal(interface):
    CLOCK.t = 0.0
    yaw_opt = interface.endswith('+yaw')
    interface = interface.split('+')[0]
    fake_ros.Node.OVERRIDES = _params(interface, 7)
    if yaw_opt:
        fake_ros.Node.OVERRIDES['yaw_mode'] = 'optimized'
    yawspeeds = []
    cn.NmpcNode.START_SOLVER_THREAD = False
    node = cn.NmpcNode()
    px4 = cn.VehicleOdometry
    status_cls = cn.VehicleStatus

    # PX4 v1.17 naming: VehicleStatus/VehicleAttitudeSetpoint have MESSAGE_VERSION=1 -> _v1
    assert '/fmu/out/vehicle_status_v1' in node.subscriptions
    assert '/fmu/out/vehicle_status' not in node.subscriptions
    assert '/fmu/in/vehicle_attitude_setpoint_v1' in node.publishers
    assert '/fmu/out/vehicle_odometry' in node.subscriptions

    sim = Px4LikeQuad(SimParams(wind_acc=(0.15, -0.1, 0.0)), rng=np.random.default_rng(1))
    dt = 0.004
    next_odom = next_status = 0.0
    started = False
    n_cmd_seen = 0
    min_clear = float('inf')
    obstacles = node.rt.obstacles_phys
    reached = None
    while sim.t < 45.0:
        sim.step(dt)
        CLOCK.t = sim.t
        if sim.t >= next_odom:
            next_odom += 0.01
            od = sim.odometry()
            if od is not None:
                msg = px4()
                msg.pose_frame = px4.POSE_FRAME_NED
                msg.velocity_frame = px4.VELOCITY_FRAME_NED
                msg.position = list(od[1])
                msg.velocity = list(od[2])
                msg.q = list(euler_to_quat_wxyz(*od[3]))
                node.subscriptions['/fmu/out/vehicle_odometry'](msg)
        if sim.t >= next_status:
            next_status += 0.1
            st = status_cls()
            st.arming_state = status_cls.ARMING_STATE_ARMED if sim.armed else 1
            st.nav_state = status_cls.NAVIGATION_STATE_OFFBOARD if sim.offboard else 4
            node.subscriptions['/fmu/out/vehicle_status_v1'](st)
        if not started and sim.t > 0.5:
            resp = node.services['/nmpc/start'](None, types.SimpleNamespace(success=False, message=''))
            assert resp.success, resp.message
            started = True
        for timer in node.timers:
            if sim.t >= timer.next:
                timer.next += timer.period
                timer.cb()
        node.solve_once()
        sp = _decode(node, px4)
        if sp is not None:
            sim.set_setpoint(sp)
            if sp.kind == 'trajectory' and np.isfinite(getattr(sp, 'yawspeed', np.nan)):
                yawspeeds.append(sp.yawspeed)
        cmds = node.publishers['/fmu/in/vehicle_command'].msgs
        if len(cmds) > n_cmd_seen:
            if any(c.command == cn.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and c.param1 == 1.0
                   for c in cmds[n_cmd_seen:]):
                sim.request_offboard_and_arm()
            n_cmd_seen = len(cmds)
        for c, r in obstacles:
            min_clear = min(min_clear, float(np.linalg.norm(sim.pos - c) - r - sim.p.phys_radius))
        if node.rt.phase == 'GOAL':
            reached = reached or sim.t
            # GOAL is entered inside goal_tolerance (0.35 m); give the PX4 position
            # hold 3 s (the same settling time as sim.episode.run_episode) to converge
            if sim.t - reached > 3.0:
                break
    assert reached is not None, f'goal not reached; phase={node.rt.phase} logs={node.logs[-5:]}'
    assert min_clear > 0.0
    assert np.linalg.norm(sim.pos - node.rt.goal) < 0.3
    if yaw_opt:
        # the planned yaw rate is forwarded to PX4 as TrajectorySetpoint.yawspeed
        assert yawspeeds and max(abs(v) for v in yawspeeds) > 0.05
        assert max(abs(v) for v in yawspeeds) < 1.2
    diag = node.publishers['/nmpc/diagnostics'].msgs[-1].status[0]
    keys = {kv.key for kv in diag.values}
    for k in ('phase', 'last_solve_ms', 'deadline_ratio', 'prediction_age_s', 'applied_source',
              'distance_to_goal_m', 'min_safety_clearance_m', 'warnings'):
        assert k in keys
    assert node.publishers['/nmpc/predicted_path'].msgs
    assert node.publishers['/nmpc/guide_path'].msgs
    node.destroy_node()


def test_live_goal_topic_resumes_from_goal():
    """The ROS goal callback must turn a completed trajectory mission into a new leg."""
    CLOCK.t = 12.0
    fake_ros.Node.OVERRIDES = _params('trajectory', 42)
    cn.NmpcNode.START_SOLVER_THREAD = False
    node = cn.NmpcNode()

    old_goal = node.rt.goal.copy()
    od = cn.VehicleOdometry()
    od.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
    od.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
    od.position = list(old_goal)
    od.velocity = [0.0, 0.0, 0.0]
    od.q = list(euler_to_quat_wxyz(0.0, 0.0, 0.0))
    node.subscriptions['/fmu/out/vehicle_odometry'](od)
    node.rt.phase = 'GOAL'
    node.rt.leg_index = 1
    node.rt.armed = True
    node.rt.offboard = True

    candidate = None
    for n in np.linspace(node.rt.arena_lb[0] + 0.5, node.rt.arena_ub[0] - 0.5, 5):
        for e in np.linspace(node.rt.arena_lb[1] + 0.5, node.rt.arena_ub[1] - 0.5, 5):
            q = np.array([n, e, -2.0])
            ok, _ = node.rt._validate_goal(q)
            if ok and np.linalg.norm(q - old_goal) > 1.5:
                candidate = q
                break
        if candidate is not None:
            break
    assert candidate is not None

    msg = cn.PoseStamped()
    msg.header = types.SimpleNamespace(frame_id='px4_local_ned')
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = candidate.tolist()
    node.subscriptions['/nmpc/goal'](msg)

    assert node.rt.phase == 'NMPC'
    assert node.rt.leg_index == 2
    assert np.allclose(node.rt.goal, candidate)
    assert np.allclose(node.rt.mission_start, old_goal)
    node.destroy_node()


def test_reload_scenario_stages_environment_while_goal_hold_is_preserved():
    """Controller reload must not move the aircraft before Gazebo scene commit."""
    CLOCK.t = 20.0
    fake_ros.Node.OVERRIDES = _params('trajectory', 42)
    cn.NmpcNode.START_SOLVER_THREAD = False
    node = cn.NmpcNode()

    old_goal = node.rt.goal.copy()
    od = cn.VehicleOdometry()
    od.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
    od.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
    od.position = list(old_goal)
    od.velocity = [0.0, 0.0, 0.0]
    od.q = list(euler_to_quat_wxyz(0.0, 0.0, 0.0))
    node.subscriptions['/fmu/out/vehicle_odometry'](od)
    node.rt.phase = 'GOAL'
    node.rt.leg_index = 1
    node.rt.armed = node.rt.offboard = True
    node.rt.goal_hold_point = old_goal.copy()

    candidate = None
    for n in np.linspace(node.rt.arena_lb[0] + 0.5, node.rt.arena_ub[0] - 0.5, 5):
        for e in np.linspace(node.rt.arena_lb[1] + 0.5, node.rt.arena_ub[1] - 0.5, 5):
            q = np.array([n, e, -2.0])
            ok, _ = node.rt._validate_goal(q)
            if ok and np.linalg.norm(q - old_goal) > 1.5:
                candidate = q
                break
        if candidate is not None:
            break
    assert candidate is not None

    node._load_scenario_geometry = lambda: (
        {}, {}, list(node.rt.obstacles_phys), node.rt.arena_lb.copy(), node.rt.arena_ub.copy(), {}, candidate)
    resp = node.reload_scenario_cb(None, types.SimpleNamespace(success=False, message=''))
    assert resp.success, resp.message
    assert node.rt.phase == 'GOAL'
    assert np.allclose(node.rt.goal, candidate)
    assert np.allclose(node.rt.goal_hold_point, old_goal)
    assert np.allclose(node.rt._setpoint(CLOCK.t).position, old_goal)
    node.destroy_node()


def test_stop_keeps_airborne_vehicle_in_landing_until_disarmed():
    """Stop must not make `next` think a descending vehicle is safely IDLE."""
    CLOCK.t = 30.0
    fake_ros.Node.OVERRIDES = _params('trajectory', 42)
    cn.NmpcNode.START_SOLVER_THREAD = False
    node = cn.NmpcNode()

    od = cn.VehicleOdometry()
    od.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
    od.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
    od.position = list(node.rt.goal)
    od.velocity = [0.0, 0.0, 0.0]
    od.q = list(euler_to_quat_wxyz(0.0, 0.0, 0.0))
    node.subscriptions['/fmu/out/vehicle_odometry'](od)
    node.rt.phase = 'GOAL'
    node.rt.armed = node.rt.offboard = True

    resp = node.stop_cb(None, types.SimpleNamespace(success=False, message=''))
    assert resp.success
    assert node.rt.phase == 'LANDING'
    assert any(c.command == cn.VehicleCommand.VEHICLE_CMD_NAV_LAND
               for c in node.publishers['/fmu/in/vehicle_command'].msgs)

    # While airborne/landing, reset is intentionally refused.
    reset = node.reset_cb(None, types.SimpleNamespace(success=False, message=''))
    assert not reset.success
    assert node.rt.phase == 'LANDING'

    st = cn.VehicleStatus()
    st.arming_state = 1
    st.nav_state = 4
    node.subscriptions['/fmu/out/vehicle_status_v1'](st)
    assert node.rt.phase == 'IDLE'
    node.destroy_node()



def test_stop_lands_armed_aborted_vehicle_instead_of_disarming():
    """An OFFBOARD abort can leave PX4 armed/airborne; stop must LAND, never disarm in air."""
    CLOCK.t = 35.0
    fake_ros.Node.OVERRIDES = _params('trajectory', 42)
    cn.NmpcNode.START_SOLVER_THREAD = False
    node = cn.NmpcNode()

    od = cn.VehicleOdometry()
    od.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
    od.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
    od.position = [0.0, 0.0, -2.0]
    od.velocity = [0.0, 0.0, 0.0]
    od.q = list(euler_to_quat_wxyz(0.0, 0.0, 0.0))
    node.subscriptions['/fmu/out/vehicle_odometry'](od)
    node.rt.phase = 'ABORTED'
    node.rt.armed = True
    node.rt.offboard = False

    before = len(node.publishers['/fmu/in/vehicle_command'].msgs)
    resp = node.stop_cb(None, types.SimpleNamespace(success=False, message=''))
    assert resp.success
    assert node.rt.phase == 'LANDING'
    cmds = node.publishers['/fmu/in/vehicle_command'].msgs[before:]
    assert any(c.command == cn.VehicleCommand.VEHICLE_CMD_NAV_LAND for c in cmds)
    assert not any(c.command == cn.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and c.param1 == 0.0
                   for c in cmds)
    node.destroy_node()

def test_commit_staged_goal_service_resumes_only_after_scene_stage():
    CLOCK.t = 40.0
    fake_ros.Node.OVERRIDES = _params('trajectory', 42)
    cn.NmpcNode.START_SOLVER_THREAD = False
    node = cn.NmpcNode()

    old_goal = node.rt.goal.copy()
    od = cn.VehicleOdometry()
    od.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
    od.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
    od.position = list(old_goal)
    od.velocity = [0.0, 0.0, 0.0]
    od.q = list(euler_to_quat_wxyz(0.0, 0.0, 0.0))
    node.subscriptions['/fmu/out/vehicle_odometry'](od)
    node.rt.phase = 'GOAL'
    node.rt.leg_index = 3
    node.rt.armed = node.rt.offboard = True
    node.rt.goal_hold_point = old_goal.copy()

    candidate = None
    for n in np.linspace(node.rt.arena_lb[0] + 0.5, node.rt.arena_ub[0] - 0.5, 5):
        for e in np.linspace(node.rt.arena_lb[1] + 0.5, node.rt.arena_ub[1] - 0.5, 5):
            q = np.array([n, e, -2.0])
            ok, _ = node.rt._validate_goal(q)
            if ok and np.linalg.norm(q - old_goal) > 1.5:
                candidate = q
                break
        if candidate is not None:
            break
    assert candidate is not None

    ok, msg = node.rt.set_environment(node.rt.obstacles_phys, node.rt.arena_lb, node.rt.arena_ub,
                                      goal=candidate, t=CLOCK.t, resume=False)
    assert ok, msg
    assert node.rt.phase == 'GOAL'
    assert np.allclose(node.rt._setpoint(CLOCK.t).position, old_goal)

    resp = node.commit_staged_goal_cb(None, types.SimpleNamespace(success=False, message=''))
    assert resp.success, resp.message
    assert node.rt.phase == 'NMPC'
    assert node.rt.leg_index == 4
    assert np.allclose(node.rt.goal, candidate)
    assert node.rt.goal_hold_point is None
    node.destroy_node()


# --------------------------------------------------------------------------
# v2.0.2 regressions: PX4 v1.17 topic names, failsafe handling, start guards
# --------------------------------------------------------------------------
def _odom(position, velocity=(0.0, 0.0, 0.0)):
    od = cn.VehicleOdometry()
    od.pose_frame = cn.VehicleOdometry.POSE_FRAME_NED
    od.velocity_frame = cn.VehicleOdometry.VELOCITY_FRAME_NED
    od.position = [float(v) for v in position]
    od.velocity = [float(v) for v in velocity]
    od.q = [1.0, 0.0, 0.0, 0.0]
    return od


def _status(armed, nav_state, failsafe=False):
    st = cn.VehicleStatus()
    st.arming_state = cn.VehicleStatus.ARMING_STATE_ARMED if armed else cn.VehicleStatus.ARMING_STATE_DISARMED
    st.nav_state = nav_state
    st.failsafe = failsafe
    return st


def _new_node(t=0.0, **overrides):
    CLOCK.t = t
    params = _params('trajectory', 42)
    params.update(overrides)
    fake_ros.Node.OVERRIDES = params
    cn.NmpcNode.START_SOLVER_THREAD = False
    return cn.NmpcNode()


def _resp():
    return types.SimpleNamespace(success=False, message='')


def test_topics_follow_px4_v117_naming_rule():
    node = _new_node()
    assert node.topics['status'] == '/fmu/out/vehicle_status_v1'
    assert node.topics['att'] == '/fmu/in/vehicle_attitude_setpoint_v1'
    assert node.topics['odom'] == '/fmu/out/vehicle_odometry'
    assert node.topics['traj'] == '/fmu/in/trajectory_setpoint'
    assert node.topics['rates'] == '/fmu/in/vehicle_rates_setpoint'
    assert node.topics['cmd'] == '/fmu/in/vehicle_command'
    assert node.topics['mode'] == '/fmu/in/offboard_control_mode'
    assert node.topics['ack'] == '/fmu/out/vehicle_command_ack'
    assert node.topics['failsafe'] == '/fmu/out/failsafe_flags'
    node.destroy_node()
    legacy = _new_node(px4_versioned_topics='never')
    assert legacy.topics['status'] == '/fmu/out/vehicle_status'
    legacy.destroy_node()
    # YAML booleans from older configs keep working
    old_true = _new_node(px4_versioned_topics=True)
    assert old_true.topics['status'] == '/fmu/out/vehicle_status_v1'
    old_true.destroy_node()


def test_start_is_refused_until_vehicle_status_arrives():
    node = _new_node()
    node.odom_cb(_odom([0.0, 0.0, 0.0]))
    resp = node.start_cb(None, _resp())
    assert not resp.success
    assert 'vehicle_status_v1' in resp.message
    assert node.rt.phase == 'IDLE'
    node.subscriptions['/fmu/out/vehicle_status_v1'](_status(False, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
    resp = node.start_cb(None, _resp())
    assert resp.success, resp.message
    assert node.rt.phase == 'ARMING'
    node.destroy_node()


def test_px4_failsafe_aborts_immediately_and_is_not_fought():
    node = _new_node(t=5.0)
    node.odom_cb(_odom([0.0, 0.0, -1.5]))
    node.rt.phase = 'NMPC'
    node.rt.armed = node.rt.offboard = True
    node.px4_nav_state = cn.VehicleStatus.NAVIGATION_STATE_OFFBOARD
    flags = cn.FailsafeFlags()
    flags.offboard_control_signal_lost = True
    node.subscriptions['/fmu/out/failsafe_flags'](flags)
    node.subscriptions['/fmu/out/vehicle_status_v1'](
        _status(True, cn.VehicleStatus.NAVIGATION_STATE_AUTO_RTL, failsafe=True))
    assert node.rt.phase == 'ABORTED'
    assert 'failsafe' in node.rt.abort_reason.lower() and 'AUTO_RTL' in node.rt.abort_reason
    assert any('offboard_control_signal_lost' in text for _lvl, text in node.logs)
    before = len(node.publishers['/fmu/in/vehicle_command'].msgs)
    n_mode = len(node.publishers['/fmu/in/offboard_control_mode'].msgs)
    for k in range(20):
        CLOCK.t = 5.0 + 0.05 * (k + 1)
        node.stream_tick()
    cmds = node.publishers['/fmu/in/vehicle_command'].msgs[before:]
    assert not any(c.command == cn.VehicleCommand.VEHICLE_CMD_DO_SET_MODE for c in cmds)
    assert len(node.publishers['/fmu/in/offboard_control_mode'].msgs) == n_mode   # PX4 keeps control
    # diagnostics carry the reason for the monitor / RViz
    node.publish_diagnostics()
    st = node.publishers['/nmpc/diagnostics'].msgs[-1].status[0]
    values = {kv.key: kv.value for kv in st.values}
    assert st.level == cn.DiagnosticStatus.ERROR
    assert values['px4_failsafe'] == 'True'
    assert 'offboard_control_signal_lost' in values['px4_failsafe_flags']
    node.destroy_node()


def test_pilot_mode_change_aborts_after_grace_without_rerequest():
    node = _new_node(t=10.0)
    node.odom_cb(_odom([0.0, 0.0, -1.5]))
    node.rt.phase = 'NMPC'
    node.rt.armed = node.rt.offboard = True
    status_cb = node.subscriptions['/fmu/out/vehicle_status_v1']
    status_cb(_status(True, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
    assert node.rt.phase == 'NMPC'            # grace period
    before = len(node.publishers['/fmu/in/vehicle_command'].msgs)
    for k in range(40):
        CLOCK.t = 10.0 + 0.05 * (k + 1)
        node.stream_tick()
        if k % 4 == 0:
            status_cb(_status(True, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
    assert node.rt.phase == 'ABORTED'
    assert 'POSCTL' in node.rt.abort_reason
    cmds = node.publishers['/fmu/in/vehicle_command'].msgs[before:]
    assert not any(c.command == cn.VehicleCommand.VEHICLE_CMD_DO_SET_MODE for c in cmds)
    node.destroy_node()


def test_arming_timeout_aborts_and_reports_px4_rejection():
    node = _new_node(t=0.0, arming_timeout_s=5.0)
    node.odom_cb(_odom([0.0, 0.0, 0.0]))
    status_cb = node.subscriptions['/fmu/out/vehicle_status_v1']
    status_cb(_status(False, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
    assert node.start_cb(None, _resp()).success
    ack = cn.VehicleCommandAck()
    ack.command = cn.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    ack.result = cn.VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED
    for k in range(200):
        CLOCK.t = 0.05 * (k + 1)
        node.odom_cb(_odom([0.0, 0.0, 0.0]))
        node.stream_tick()
        if k % 4 == 0:
            status_cb(_status(False, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
            node.subscriptions['/fmu/out/vehicle_command_ack'](ack)
        if node.rt.phase == 'ABORTED':
            break
    assert node.rt.phase == 'ABORTED'
    assert 'did not arm' in node.rt.abort_reason and 'ARM/DISARM=DENIED' in node.rt.abort_reason
    assert any('PX4 rejected ARM/DISARM: DENIED' in text for _lvl, text in node.logs)
    node.destroy_node()


def test_missing_status_after_arm_request_commands_land():
    """Legacy path (status lost after start): never go silent while PX4 may be in OFFBOARD."""
    node = _new_node(t=0.0, status_timeout_s=2.0)
    node.odom_cb(_odom([0.0, 0.0, 0.0]))
    node.subscriptions['/fmu/out/vehicle_status_v1'](_status(False, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
    assert node.start_cb(None, _resp()).success
    node.status_seen = False                  # simulate the status stream disappearing
    for k in range(100):
        CLOCK.t = 0.05 * (k + 1)
        node.odom_cb(_odom([0.0, 0.0, 0.0]))
        node.stream_tick()
    assert node.rt.phase == 'ABORTED'
    cmds = node.publishers['/fmu/in/vehicle_command'].msgs
    assert cmds[-1].command == cn.VehicleCommand.VEHICLE_CMD_NAV_LAND
    node.destroy_node()


def test_scenario_json_is_the_geometry_source(tmp_path):
    import json
    from px4_nmpc_core.scenario import controller_geometry
    exp = yaml.safe_load((CONFIG / 'experiment.yaml').read_text())['experiment']
    scenario = generate_scenario(exp, 11).to_dict()
    scenario['experiment_config'] = dict(exp, seed=11)
    path = tmp_path / 'scenario.json'
    path.write_text(json.dumps(scenario))
    node = _new_node(scenario_json=str(path))       # ROS params still hold seed 42
    geom = controller_geometry(scenario)
    assert np.allclose(node.rt.goal, geom['goal_local_ned'])
    assert len(node.rt.obstacles_phys) == len(scenario['obstacle_radii'])
    assert 'seed 11' in node.scenario_source
    node.destroy_node()
    # missing runtime file -> packaged example, clearly labelled
    node = _new_node(scenario_json=str(tmp_path / 'missing.json'),
                     fallback_scenario_json=str(CONFIG / 'scenario_example.json'))
    assert node.scenario_source.startswith('PACKAGED EXAMPLE')
    node.publish_diagnostics()
    values = {kv.key: kv.value for kv in node.publishers['/nmpc/diagnostics'].msgs[-1].status[0].values}
    assert values['scenario_source'].startswith('PACKAGED EXAMPLE')
    node.destroy_node()


def test_diagnostic_level_is_published_as_rclpy_bytes():
    node = _new_node()
    node.odom_cb(_odom([0.0, 0.0, 0.0]))
    node.subscriptions['/fmu/out/vehicle_status_v1'](_status(False, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
    node.publish_diagnostics()
    st = node.publishers['/nmpc/diagnostics'].msgs[-1].status[0]
    assert isinstance(st.level, bytes)
    assert st.level == cn.DiagnosticStatus.OK, st.message
    node.destroy_node()


def test_no_setpoints_are_injected_outside_offboard():
    """PX4 mixes DDS setpoints into its own uORB setpoints: never send them outside OFFBOARD.

    (PX4 v1.17 SITL: goal-hold setpoints streamed during AUTO_LAND made the
    landing vehicle climb again - 'Landing detected' followed by 'Takeoff detected'.)
    """
    node = _new_node(t=20.0)
    node.odom_cb(_odom([0.0, 0.0, -2.0]))
    traj = node.publishers['/fmu/in/trajectory_setpoint'].msgs
    mode = node.publishers['/fmu/in/offboard_control_mode'].msgs
    status_cb = node.subscriptions['/fmu/out/vehicle_status_v1']

    # GOAL hold in OFFBOARD: setpoints are streamed
    node.rt.phase = 'GOAL'
    node.rt.goal_hold_point = np.array([0.0, 0.0, -2.0])
    status_cb(_status(True, cn.VehicleStatus.NAVIGATION_STATE_OFFBOARD))
    node.stream_tick()
    assert len(traj) == 1 and len(mode) == 1

    # stop -> LAND: nothing at all is published while PX4 lands
    assert node.stop_cb(None, _resp()).success
    status_cb(_status(True, cn.VehicleStatus.NAVIGATION_STATE_AUTO_LAND))
    for k in range(10):
        CLOCK.t = 20.0 + 0.05 * (k + 1)
        node.stream_tick()
    assert len(traj) == 1 and len(mode) == 1

    # pilot switched to POSCTL during NMPC (grace period): heartbeat only, no setpoints
    node2 = _new_node(t=30.0)
    node2.odom_cb(_odom([0.0, 0.0, -2.0]))
    node2.rt.phase = 'NMPC'
    node2.rt.hold_point = np.array([0.0, 0.0, -2.0])
    node2.subscriptions['/fmu/out/vehicle_status_v1'](_status(True, cn.VehicleStatus.NAVIGATION_STATE_POSCTL))
    node2.stream_tick()
    assert len(node2.publishers['/fmu/in/offboard_control_mode'].msgs) == 1
    assert len(node2.publishers['/fmu/in/trajectory_setpoint'].msgs) == 0
    node.destroy_node()
    node2.destroy_node()


def test_reset_is_refused_while_px4_is_still_armed():
    """After an abort PX4 may still be flying its own RTL/Land: reset must not touch it."""
    node = _new_node(t=50.0)
    node.odom_cb(_odom([0.0, 0.0, -2.0]))
    node.rt.phase = 'ABORTED'
    node.subscriptions['/fmu/out/vehicle_status_v1'](
        _status(True, cn.VehicleStatus.NAVIGATION_STATE_AUTO_RTL, failsafe=True))
    before = len(node.publishers['/fmu/in/vehicle_command'].msgs)
    resp = node.reset_cb(None, _resp())
    assert not resp.success and 'still armed' in resp.message
    assert node.rt.phase == 'ABORTED'
    assert len(node.publishers['/fmu/in/vehicle_command'].msgs) == before     # no LAND/DISARM sent
    # landed + disarmed -> reset works and the vehicle is reported IDLE
    node.subscriptions['/fmu/out/vehicle_status_v1'](
        _status(False, cn.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER))
    node.odom_cb(_odom([0.0, 0.0, 0.0]))
    resp = node.reset_cb(None, _resp())
    assert resp.success, resp.message
    assert node.rt.phase == 'IDLE'
    node.destroy_node()

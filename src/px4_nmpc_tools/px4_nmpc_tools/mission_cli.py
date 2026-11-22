"""Interactive mission control with acknowledged goals and transactional scene changes.

Safety properties:
* A live goal is persisted/moved in Gazebo only after the controller reports
  that it accepted the goal.
* A new obstacle scene is staged in the controller while PX4 keeps holding the
  completed goal. Gazebo is then replaced. Only after Gazebo verifies the new
  entities does navigation resume.
* If Gazebo replacement fails, scenario files and the controller are rolled
  back to the previous environment and the old scene is re-applied best-effort.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import rclpy
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_srvs.srv import Trigger

from px4_nmpc_core.scenario import (
    arena_to_gazebo_enu, controller_geometry, controller_origin_ned_of,
    generate_scenario, scenario_parameters,
)
from .experiment_manager import (
    DEFAULT_WORLD, _load_experiment, _load_runtime_scenario, _create_model,
    _remove_model, _sphere_sdf, _scenario_paths, _write_scenario, apply_scene,
)


def _parse_vec(text):
    if text is None:
        return None
    try:
        a = np.asarray([float(x) for x in str(text).strip('[]').split(',')], dtype=float)
    except ValueError:
        return None
    return a if a.shape == (3,) and np.all(np.isfinite(a)) else None


def _same_vec(a, b, atol=5e-3):
    aa = _parse_vec(a) if isinstance(a, str) else np.asarray(a, float).reshape(3)
    bb = np.asarray(b, float).reshape(3)
    return aa is not None and np.allclose(aa, bb, atol=atol, rtol=0.0)


def _goal_acknowledged(diag, goal, before_phase, before_leg):
    if not diag or not _same_vec(diag.get('goal_ned'), goal):
        return False
    if before_phase in ('GOAL', 'NMPC'):
        try:
            leg = int(diag.get('leg_index', '0'))
        except (TypeError, ValueError):
            return False
        return diag.get('phase') == 'NMPC' and leg > int(before_leg or 0)
    return True


class MissionClient(Node):
    def __init__(self):
        super().__init__('nmpc_mission_cli')
        self.diag = None
        self.create_subscription(DiagnosticArray, '/nmpc/diagnostics', self._diag_cb, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/nmpc/goal', 10)

    def _diag_cb(self, msg):
        if not msg.status:
            return
        st = msg.status[0]
        d = {kv.key: kv.value for kv in st.values}
        d['_message'] = st.message
        self.diag = d

    def wait_diag(self, timeout=3.0):
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end and self.diag is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.diag is None:
            raise RuntimeError('No /nmpc/diagnostics received. Is nmpc.launch.py running?')
        return self.diag

    def trigger(self, name, timeout=5.0):
        client = self.create_client(Trigger, name)
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'Service {name} is not available')
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError(f'Timed out calling {name}')
        return future.result()

    def publish_goal(self, goal_ned):
        goal = np.asarray(goal_ned, float).reshape(3)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'px4_local_ned'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(v) for v in goal]
        msg.pose.orientation.w = 1.0
        end = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < end and self.goal_pub.get_subscription_count() < 1:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.goal_pub.get_subscription_count() < 1:
            raise RuntimeError('No subscriber on /nmpc/goal. Is the controller running?')
        self.goal_pub.publish(msg)

    def wait_goal_ack(self, goal, before_phase, before_leg, timeout=3.0):
        """Wait until diagnostics prove the controller accepted this publication.

        A GOAL/NMPC retarget must increment the leg and enter/remain NMPC. This
        prevents a staged environment (whose diagnostic goal already changed)
        from being mistaken for acknowledgement of the later /nmpc/goal event.
        """
        goal = np.asarray(goal, float).reshape(3)
        before_leg = int(before_leg or 0)
        end = time.monotonic() + timeout
        last = None
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            d = self.diag
            if not d:
                continue
            last = d
            if _goal_acknowledged(d, goal, before_phase, before_leg):
                return d
        seen = 'none' if last is None else f"phase={last.get('phase')} leg={last.get('leg_index')} goal={last.get('goal_ned')}"
        raise RuntimeError(
            f'Controller did not acknowledge goal {goal.tolist()} within {timeout:.1f}s ({seen}). '
            'The goal was not persisted; check controller logs for rejection reason.')


def _goal_from_args(args):
    if args.ned is not None:
        return np.asarray(args.ned, float)
    e, n, u = [float(v) for v in args.enu]
    return np.array([n, e, -u], dtype=float)


def _print_status(d):
    print(
        f"phase={d.get('phase','?')}  leg={d.get('leg_index','?')}  "
        f"completed={d.get('goals_completed','?')}  pos={d.get('position_ned','?')}  "
        f"goal={d.get('goal_ned','?')}  d_goal={d.get('distance_to_goal_m','?')} m  "
        f"solver={d.get('solver_status','?')}"
    )
    armed = 'armed' if str(d.get('armed', 'False')).lower() == 'true' else 'disarmed'
    line = f"px4={armed}/{d.get('px4_mode', '?')}"
    if str(d.get('px4_failsafe', 'False')).lower() == 'true':
        line += f"  FAILSAFE flags={d.get('px4_failsafe_flags', '?')}"
    if d.get('abort_reason'):
        line += f"  abort_reason={d['abort_reason']}"
    if d.get('warnings') not in (None, '', 'none'):
        line += f"  warnings={d['warnings']}"
    print(line)


def _persist_live_goal(goal_local, runtime_dir=None):
    """Persist an already-ACKNOWLEDGED live target for replay/visualization."""
    scenario = _load_runtime_scenario(runtime_dir)
    origin = controller_origin_ned_of(scenario)
    scenario['goal_ned'] = [float(v) for v in origin + np.asarray(goal_local, float)]
    geom = controller_geometry(scenario)
    scenario['controller_goal_local_ned'] = [float(v) for v in geom['goal_local_ned']]
    yaml_path, json_path = _scenario_paths(runtime_dir)
    wall = float(scenario.get('experiment_config', {}).get('wall_margin_m', 0.45))
    params = {'px4_nmpc_controller': {'ros__parameters': scenario_parameters(scenario, wall)}}
    yaml_path.write_text(yaml.safe_dump(params, sort_keys=False))
    json_path.write_text(json.dumps(scenario, indent=2) + '\n')
    try:
        _remove_model('nmpc_goal_marker', DEFAULT_WORLD)
        _create_model(_sphere_sdf(
            'nmpc_goal_marker', arena_to_gazebo_enu(scenario['goal_ned'], origin), 0.28,
            (0.10, 0.90, 0.25, 0.85), collision=False, transparency=0.15), DEFAULT_WORLD)
    except Exception as exc:
        print(f'Note: controller accepted the goal, but Gazebo goal marker was not moved: {exc}')


def _continuation_config(current, config_path=None):
    """Preserve the current experiment geometry/tuning unless explicitly overridden."""
    if config_path:
        cfg = copy.deepcopy(_load_experiment(config_path))
    else:
        cfg = copy.deepcopy(current.get('experiment_config') or _load_experiment(None))
    for key in ('seed', 'start_ned_override', 'goal_ned_override', 'controller_origin_ned'):
        cfg.pop(key, None)
    # This is a continuation while already airborne, not a second take-off.
    cfg.pop('takeoff_altitude_m', None)
    cfg.pop('takeoff_corridor_clearance_m', None)
    return cfg


def _snapshot_files(runtime_dir=None):
    yaml_path, json_path = _scenario_paths(runtime_dir)
    return yaml_path, json_path, (
        yaml_path.read_text() if yaml_path.exists() else None,
        json_path.read_text() if json_path.exists() else None,
    )


def _restore_files(yaml_path, json_path, snap):
    for p, text in ((yaml_path, snap[0]), (json_path, snap[1])):
        if text is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(text)


def _rollback_environment(node, old_scenario, yaml_path, json_path, snap):
    errors = []
    _restore_files(yaml_path, json_path, snap)
    try:
        result = node.trigger('/nmpc/reload_scenario')
        if not result.success:
            errors.append('controller rollback: ' + result.message)
    except Exception as exc:
        errors.append('controller rollback: ' + str(exc))
    try:
        apply_scene(old_scenario, DEFAULT_WORLD)
    except Exception as exc:
        errors.append('Gazebo rollback: ' + str(exc))
    return errors


def main(args=None):
    parser = argparse.ArgumentParser(description='Live mission control for PX4 NMPC.')
    parser.add_argument('--runtime-dir', default=None, help='scenario directory (default ~/.ros/px4_nmpc)')
    parser.add_argument('--config', default=None, help='optional experiment.yaml override for `next`')
    sub = parser.add_subparsers(dest='command', required=True)

    p_goal = sub.add_parser('goal', help='Send a new target without restarting PX4/Gazebo')
    group = p_goal.add_mutually_exclusive_group(required=True)
    group.add_argument('--ned', nargs=3, type=float, metavar=('N', 'E', 'D'), help='PX4-local NED target')
    group.add_argument('--enu', nargs=3, type=float, metavar=('E', 'N', 'U'), help='RViz/map ENU target')

    p_next = sub.add_parser('next', help='Transactionally replace obstacles and start a new leg at current pose')
    p_next.add_argument('--seed', type=int, default=None)
    p_next.add_argument('--goal-local-ned', nargs=3, type=float, metavar=('N', 'E', 'D'),
                        help='optional fixed new goal in PX4-local NED; otherwise sample one')
    p_next.add_argument('--no-start', action='store_true', help='if controller is IDLE, stage but do not call /nmpc/start')

    sub.add_parser('status', help='Show current mission status')
    sub.add_parser('start', help='Start from the current measured position')
    sub.add_parser('stop', help='Stop (LAND if airborne)')
    sub.add_parser('reset', help='Reset controller state after PX4 is landed/disarmed (physical pose unchanged)')

    ns = parser.parse_args(args=args)
    rclpy.init()
    node = MissionClient()
    try:
        if ns.command == 'status':
            _print_status(node.wait_diag())
            return
        if ns.command in ('start', 'stop', 'reset'):
            srv = {
                'start': '/nmpc/start',
                'stop': '/nmpc/stop',
                'reset': '/nmpc/reset_controller',
            }[ns.command]
            result = node.trigger(srv)
            if not result.success:
                raise RuntimeError(result.message)
            print(result.message)
            return
        if ns.command == 'goal':
            goal = _goal_from_args(ns)
            before = dict(node.wait_diag())
            node.diag = None
            node.publish_goal(goal)
            ack = node.wait_goal_ack(goal, before.get('phase', ''), before.get('leg_index', 0))
            try:
                _persist_live_goal(goal, ns.runtime_dir)
            except Exception as exc:
                print(f'Note: goal was accepted, but scenario bookkeeping was not updated: {exc}')
            print(f'New goal accepted: {goal.tolist()} PX4-local NED')
            _print_status(ack)
            return

        # NEXT: only replace physical geometry while stopped or at terminal hold.
        before = dict(node.wait_diag())
        phase = before.get('phase', '')
        if phase not in ('IDLE', 'GOAL'):
            raise RuntimeError(
                f'`next` is allowed only in IDLE or GOAL, current phase is {phase}. '
                'Use `nmpc_mission goal` to retarget during flight without changing obstacles.')
        if phase == 'IDLE' and str(before.get('armed', 'False')).lower() == 'true':
            raise RuntimeError(
                '`next` refused: controller is IDLE but PX4 still reports armed. Wait for touchdown/disarm first.')
        pos_local = _parse_vec(before.get('position_ned'))
        if pos_local is None:
            raise RuntimeError('Controller diagnostics do not contain a valid position_ned')

        current = _load_runtime_scenario(ns.runtime_dir)
        origin = controller_origin_ned_of(current)
        start_arena = origin + pos_local
        cfg = _continuation_config(current, ns.config)
        cfg['start_ned_override'] = start_arena.tolist()
        cfg['controller_origin_ned'] = origin.tolist()
        if ns.goal_local_ned is not None:
            cfg['goal_ned_override'] = (origin + np.asarray(ns.goal_local_ned, float)).tolist()
        seed = int(ns.seed) if ns.seed is not None else int(current.get('seed', 0)) + 1
        scenario = generate_scenario(cfg, seed)
        scenario_dict = scenario.to_dict()
        new_goal_local = np.asarray(controller_geometry(scenario_dict)['goal_local_ned'], float)

        yaml_path, json_path, snap = _snapshot_files(ns.runtime_dir)
        try:
            _write_scenario(scenario, yaml_path, json_path, cfg)
        except Exception:
            _restore_files(yaml_path, json_path, snap)
            raise

        # Phase 1: stage controller geometry while preserving the old terminal
        # hold point. No motion toward the new target is allowed yet.
        staged = node.trigger('/nmpc/reload_scenario')
        if not staged.success:
            _restore_files(yaml_path, json_path, snap)
            rollback_note = ''
            try:
                old_reload = node.trigger('/nmpc/reload_scenario')
                if not old_reload.success:
                    rollback_note = ' Old-controller reload also failed: ' + old_reload.message
            except Exception as rb_exc:
                rollback_note = ' Old-controller reload also failed: ' + str(rb_exc)
            raise RuntimeError(
                'Controller rejected new environment before Gazebo was changed: ' + staged.message + rollback_note)

        # Phase 2: replace and verify the physical Gazebo scene. Roll back both
        # file/controller state and the old scene if this fails halfway through.
        try:
            apply_scene(_load_runtime_scenario(ns.runtime_dir), DEFAULT_WORLD)
        except Exception as exc:
            rb = _rollback_environment(node, current, yaml_path, json_path, snap)
            extra = '' if not rb else ' Rollback warnings: ' + '; '.join(rb)
            raise RuntimeError(f'Gazebo scene update failed; old environment restored where possible: {exc}.{extra}')

        print(f'New live scenario committed: {json_path}')
        print(f'Leg start (current): {pos_local.tolist()} PX4-local NED')
        print(f'New goal: {new_goal_local.tolist()} PX4-local NED')

        if phase == 'GOAL':
            # Explicitly resume only now that the physical scene is verified.
            # Use a synchronous service for the staged goal so a lost one-shot
            # topic publication cannot leave the CLI and controller disagreeing.
            committed = node.trigger('/nmpc/commit_staged_goal')
            if not committed.success:
                rb = _rollback_environment(node, current, yaml_path, json_path, snap)
                extra = '' if not rb else ' Rollback warnings: ' + '; '.join(rb)
                raise RuntimeError('Could not start the staged goal; old environment restored where possible: '
                                   + committed.message + extra)
            # The synchronous service result is the authoritative commit ACK.
            # Diagnostics are useful confirmation, but a delayed diagnostic sample
            # must not turn a successful commit into a false CLI failure after the
            # Gazebo scene has already been replaced.
            try:
                ack = node.wait_goal_ack(new_goal_local, phase, before.get('leg_index', 0))
            except RuntimeError as exc:
                print(f'Gazebo verified; staged goal committed. Diagnostics confirmation was delayed: {exc}')
            else:
                print('Gazebo verified; new NMPC leg started.')
                _print_status(ack)
        elif not ns.no_start:
            started = node.trigger('/nmpc/start')
            if not started.success:
                raise RuntimeError(started.message)
            print(started.message)
        else:
            print('Environment staged in IDLE; run `nmpc_mission start` when ready.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

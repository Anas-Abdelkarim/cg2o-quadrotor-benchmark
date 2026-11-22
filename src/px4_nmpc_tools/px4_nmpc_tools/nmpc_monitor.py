"""Compact live experiment/solver monitor for /nmpc/diagnostics.

Prints a structured multi-line report of the controller diagnostics
(mission state, vehicle, solver, control source, PX4 mode/failsafe).
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node

from px4_nmpc_core.px4_interface import diagnostic_level


def _get(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in ('', None):
            return d[k]
    return default


def _f(d, *keys, default=float('nan')):
    try:
        return float(_get(d, *keys, default=default))
    except (TypeError, ValueError):
        return default


def _i(d, *keys, default=0):
    try:
        return int(float(_get(d, *keys, default=default)))
    except (TypeError, ValueError):
        return default


def _fmt(x, digits=2, suffix=''):
    if not math.isfinite(x):
        return 'n/a'
    return f'{x:.{digits}f}{suffix}'


def format_report(d):
    """Return (text, is_warning) for one diagnostics snapshot (pure, unit-tested)."""
    state = _get(d, 'mission_state', 'phase', default='n/a')
    result = _get(d, 'experiment_result', default='n/a')
    iface = _get(d, 'interface', default='')
    backend = _get(d, 'backend', default='')
    elapsed = _f(d, 'navigation_elapsed_s')
    dgoal = _f(d, 'distance_to_goal_m')
    speed = _f(d, 'speed_mps')
    roll, pitch, yaw = _f(d, 'roll_deg'), _f(d, 'pitch_deg'), _f(d, 'yaw_deg')
    yaw_opt = str(_get(d, 'yaw_mode', default='fixed')) == 'optimized'
    yaw_cmd, yaw_rate_cmd = _f(d, 'yaw_cmd_deg'), _f(d, 'yaw_rate_cmd_dps')
    clearance = _f(d, 'min_safety_clearance_m')
    clearance_seen = _f(d, 'min_clearance_seen_m')

    solve = _f(d, 'last_solve_ms')
    ema = _f(d, 'solve_ms_ema')
    ratio = _f(d, 'deadline_ratio')
    pred_age = _f(d, 'prediction_age_s')
    iterations = _i(d, 'last_solver_iterations', 'iterations', default=-1)
    solver = _get(d, 'solver_status', default='n/a')

    total = _i(d, 'solve_total', 'solves')
    accepted = _i(d, 'solve_accepted', 'usable_solves')
    failed = _i(d, 'solve_failed', 'failed_solves')
    rejected = _i(d, 'solve_rejected', 'rejected_unsafe')
    deadline_miss = _i(d, 'deadline_misses', 'deadline_rejected')
    holds = _i(d, 'recovery_entries', 'hold_events')

    nmpc_pct = _f(d, 'nmpc_control_percent')
    hold_pct = _f(d, 'recovery_control_percent')
    source = _get(d, 'applied_source', default='n/a')
    stage = _get(d, 'applied_stage', default='n/a')
    dist = _get(d, 'disturbance_estimate', default='')
    warnings = _get(d, 'warnings', default='none')
    abort_reason = _get(d, 'abort_reason', default='')

    px4_mode = _get(d, 'px4_mode', default='')
    px4_armed = _get(d, 'armed', default='')
    px4_failsafe = str(_get(d, 'px4_failsafe', default='False')).lower() == 'true'
    px4_flags = _get(d, 'px4_failsafe_flags', default='none')
    px4_ack = _get(d, 'px4_last_ack', default='none')
    scenario = _get(d, 'scenario_source', default='')

    tag = f' [{iface}/{backend}]' if iface else ''
    text = (
        f'\nNMPC MONITOR [{state}] result={result}{tag}  nav_t={_fmt(elapsed, 1, " s")}\n'
        f'  vehicle : d_goal={_fmt(dgoal)} m  speed={_fmt(speed)} m/s  '
        f'RPY=[{_fmt(roll, 1)}, {_fmt(pitch, 1)}, {_fmt(yaw, 1)}] deg  '
        + (f'yaw_cmd={_fmt(yaw_cmd, 1)} deg ({_fmt(yaw_rate_cmd, 1)} deg/s)  ' if yaw_opt else '') +
        f'clearance={_fmt(clearance)} m (min {_fmt(clearance_seen)} m)\n'
        f'  solver  : {solver}  solve={_fmt(solve, 1)} ms (ema {_fmt(ema, 1)})  ratio={_fmt(ratio, 2)}x  '
        f'iter={iterations}  pred_age={_fmt(pred_age, 2)} s\n'
        f'  counts  : accepted={accepted}/{total}  failed={failed}  rejected={rejected}  '
        f'late={deadline_miss}  hold_entries={holds}\n'
        f'  control : {source}[{stage}]  NMPC={_fmt(nmpc_pct, 1)}%  hold/recovery={_fmt(hold_pct, 1)}%'
        + (f'  d_hat={dist}' if dist and dist != '[0.00, 0.00, 0.00]' else '') + '\n'
        f'  status  : warnings={warnings}'
    )
    if abort_reason:
        text += f'  abort_reason={abort_reason}'
    if px4_mode:
        armed_text = 'armed' if str(px4_armed).lower() == 'true' else 'disarmed'
        text += f'\n  px4     : {armed_text}, mode={px4_mode}'
        if px4_failsafe:
            text += f', FAILSAFE flags={px4_flags}'
        if px4_ack not in ('', 'none'):
            text += f', last_ack: {px4_ack}'
    if scenario:
        text += f'\n  scenario: {scenario}'
    is_warning = (warnings not in ('none', 'n/a') or result == 'aborted' or px4_failsafe
                  or diagnostic_level(d.get('_level', 0)) > 0)
    return text, is_warning


class NmpcMonitor(Node):
    """Compact live experiment/solver monitor for /nmpc/diagnostics."""

    def __init__(self):
        super().__init__('nmpc_monitor')
        self.declare_parameter('print_hz', 1.0)
        self.latest = None
        self.create_subscription(DiagnosticArray, '/nmpc/diagnostics', self.diag_cb, 10)
        hz = max(0.2, float(self.get_parameter('print_hz').value))
        self.create_timer(1.0 / hz, self.print_status)
        self.get_logger().info('NMPC monitor listening on /nmpc/diagnostics')

    def diag_cb(self, msg):
        if not msg.status:
            return
        s = msg.status[0]
        d = {kv.key: kv.value for kv in s.values}
        d['_message'] = s.message
        d['_level'] = diagnostic_level(s.level)
        self.latest = d

    def print_status(self):
        if self.latest is None:
            self.get_logger().info('Waiting for /nmpc/diagnostics ...', throttle_duration_sec=5.0)
            return
        try:
            text, warn = format_report(self.latest)
        except Exception as exc:  # noqa: BLE001 - a monitor must never take the launch down
            self.get_logger().error(f'Could not format diagnostics: {exc}', throttle_duration_sec=5.0)
            return
        if warn:
            self.get_logger().warning(text)
        else:
            self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = NmpcMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

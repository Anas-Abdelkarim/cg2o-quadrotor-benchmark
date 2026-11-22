"""Fly the infeasible 3-D slalom in PX4 SITL + Gazebo with the NMPC (cg2o ISPD) and record RViz and Gazebo.

    source scripts/setup_env.sh            # with ROS (sources install/setup.bash)
    python3 -m px4_nmpc_tools.sitl_video                                      # ISPD, warm start from the final iterate
    python3 -m px4_nmpc_tools.sitl_video --tuned results/tuning/<stamp>_ispd_warm_final
    python3 -m px4_nmpc_tools.sitl_video --variant ipopt_warm_primal_dual     # the same flight with IPOPT
    python3 -m px4_nmpc_tools.sitl_video --headless       # no windows, no video: flight check with logs/plots

Everything runs from this repository: Micro XRCE-DDS agent, PX4 SITL (gz_x500, built by
scripts/build_externals/px4.sh), the slalom scene in Gazebo, nmpc.launch.py (controller, visualizer, RViz) with the
slalom overrides of the experiment bundle. RViz (chase view) and the Gazebo GUI (chase camera) are placed side by side
and kept on top; one ffmpeg x11grab capture records both, so the two views are synchronised. PX4 odometry and
/nmpc/diagnostics are logged. No ground station is needed: NAV_DLL_ACT is 0 during the flight and restored after it.

Output: results/sitl/<yymmdd_hhmmss>_slalom_<solver>/
    slalom_sitl.mp4          RViz | Gazebo side by side with a title bar (solver, scenario, horizon)
    screen.mp4               the raw, synchronised screen capture of both windows
    flight.csv, solves.csv   PX4 odometry (NED) and NMPC diagnostics over time
    flight.png/.pdf          top view with obstacle / collision boundary / safety region + 3-D clearance
    summary.json             outcome, time to goal, minimum clearance, solve-time statistics
    logs/                    agent, PX4, launch, mission
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from px4_nmpc_core.paths import repo_root

import numpy as np
import yaml



ROOT = repo_root()
CFG = ROOT / 'src' / 'px4_nmpc_config' / 'config'
BUNDLE = CFG / 'experiments' / 'infeasible_slalom'
PX4_PARAM_STATE = ROOT / 'build' / 'sitl_px4_params.json'      # PX4 parameters changed for the flight (original values)


class Procs:
    """Background processes, each in its own process group, stopped in reverse order."""

    def __init__(self, logdir):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.items = []

    def start(self, name, cmd, cwd=None, env=None, stdin=None):
        log = open(self.logdir / f'{name}.log', 'w')
        log.write(' '.join(map(str, cmd)) + '\n')
        log.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=stdin, start_new_session=True)
        self.items.append((name, p, log))
        print(f'[sitl] started {name} (pid {p.pid})', flush=True)
        return p

    def stop(self, name=None, sig=signal.SIGINT, timeout=15.0):
        for n, p, log in reversed(list(self.items)):
            if name is not None and n != name:
                continue
            if p.poll() is None:
                try:
                    os.killpg(p.pid, sig)
                    p.wait(timeout=timeout)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            log.close()
            self.items.remove((n, p, log))


def run(cmd, timeout=30.0, env=None):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def nmpc_status(env):
    try:
        rc, out = run(['ros2', 'service', 'call', '/nmpc/status', 'std_srvs/srv/Trigger', '{}'], timeout=15, env=env)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"message='([^']*)'", out)
    if not m or 'success=True' not in out:
        return None
    msg = m.group(1)
    fields = dict(re.findall(r'(\w+)=([^,]+)', msg))
    return {'raw': msg, **fields}


GZ_GUI_CONFIG = """<?xml version="1.0"?>
<dialog name="quick_start" show_again="false"/>
<window>
  <width>{w}</width>
  <height>{h}</height>
  <position_x>{x}</position_x>
  <position_y>{y}</position_y>
  <menus><drawer visible="false"/><plugins visible="false"/></menus>
  <dialog_on_exit>false</dialog_on_exit>
</window>
<plugin filename="MinimalScene" name="3D View">
  <gz-gui>
    <title>3D View</title>
    <property type="bool" key="showTitleBar">false</property>
    <property type="string" key="state">docked</property>
  </gz-gui>
  <engine>ogre2</engine>
  <scene>scene</scene>
  <ambient_light>0.5 0.5 0.5</ambient_light>
  <background_color>0.85 0.88 0.92</background_color>
  <camera_pose>-6 -4 5 0 0.45 0.6</camera_pose>
</plugin>
""" + "".join("""<plugin filename="@P@" name="@P@">
  <gz-gui>
    <property key="resizable" type="bool">false</property>
    <property key="width" type="double">5</property>
    <property key="height" type="double">5</property>
    <property key="state" type="string">floating</property>
    <property key="showTitleBar" type="bool">false</property>
  </gz-gui>@EXTRA@
</plugin>
""".replace('@P@', p).replace('@EXTRA@', e) for p, e in (
    ('GzSceneManager', ''), ('InteractiveViewControl', ''), ('CameraTracking', ''), ('MarkerManager', '')))


def gz_environment(px4_build, env):
    """Environment of PX4's gz_env.sh (model/world/plugin paths) on top of `env`."""
    script = px4_build / 'rootfs' / 'gz_env.sh'
    if not script.exists():
        return dict(env)
    r = subprocess.run(['bash', '-c', f'. "{script}" && env -0'], capture_output=True, env=env, timeout=20)
    out = dict(env)
    for item in r.stdout.split(b'\0'):
        if b'=' in item:
            k, v = item.split(b'=', 1)
            out[k.decode()] = v.decode(errors='replace')
    return out


def place_window(title_re, x, y, w, h, timeout=40.0):
    """Move/resize the top-level window whose title matches `title_re` and keep it above the others (EWMH, via
    python-xlib). Returns the absolute client geometry (x, y, w, h) or None."""
    try:
        from Xlib import X, display, protocol
    except ImportError:
        print('[sitl] python-xlib not installed: windows are not placed (pip install python-xlib)', flush=True)
        return None
    d = display.Display()
    root = d.screen().root
    atom = d.intern_atom
    t0 = time.time()
    while time.time() - t0 < timeout:
        clients = root.get_full_property(atom('_NET_CLIENT_LIST'), X.AnyPropertyType)
        for wid in (clients.value if clients else []):
            win = d.create_resource_object('window', wid)
            try:
                name = win.get_full_property(atom('_NET_WM_NAME'), atom('UTF8_STRING'))
                title = name.value.decode('utf-8', 'replace') if name else (win.get_wm_name() or '')
            except Exception:  # noqa: BLE001  (window vanished)
                continue
            if not re.search(title_re, title):
                continue

            def msg(kind, data):
                ev = protocol.event.ClientMessage(window=win, client_type=atom(kind), data=(32, data))
                root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
            msg('_NET_WM_STATE', [0, atom('_NET_WM_STATE_MAXIMIZED_VERT'), atom('_NET_WM_STATE_MAXIMIZED_HORZ'), 1, 0])
            msg('_NET_MOVERESIZE_WINDOW', [(1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | 1, x, y, w, h])
            msg('_NET_WM_STATE', [1, atom('_NET_WM_STATE_ABOVE'), 0, 1, 0])
            msg('_NET_ACTIVE_WINDOW', [1, X.CurrentTime, 0, 0, 0])
            d.sync()
            time.sleep(1.5)
            g = win.get_geometry()
            pos = root.translate_coords(win, 0, 0)
            print(f'[sitl] window "{title}" at {pos.x},{pos.y} {g.width}x{g.height}', flush=True)
            return (pos.x, pos.y, g.width, g.height)
        time.sleep(1.0)
    return None


def find_window(pattern, timeout=40.0):
    """Absolute geometry (x, y, w, h) of the first X window whose title matches `pattern`."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rc, out = run(['xwininfo', '-root', '-tree'], timeout=10)
        for line in out.splitlines():
            if re.search(pattern, line, re.I):
                wid = line.split()[0]
                rc2, info = run(['xwininfo', '-id', wid], timeout=10)
                vals = {k: int(v) for k, v in re.findall(r'(Absolute upper-left [XY]|Width|Height):\s+(-?\d+)', info)}
                if vals.get('Width', 0) > 200 and vals.get('Height', 0) > 200:
                    return (vals['Absolute upper-left X'], vals['Absolute upper-left Y'], vals['Width'], vals['Height'])
        time.sleep(1.0)
    return None


class Logger:
    """rclpy subscriber: PX4 odometry and NMPC diagnostics -> CSV rows (runs in a thread)."""

    def __init__(self):
        import rclpy
        from diagnostic_msgs.msg import DiagnosticArray
        from px4_msgs.msg import VehicleOdometry
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from px4_nmpc_core.px4_interface import px4_topic
        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node('sitl_video_logger')
        self.odom, self.diag = [], []
        self.t0 = time.time()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.node.create_subscription(VehicleOdometry, px4_topic('/fmu', 'out', 'vehicle_odometry', VehicleOdometry),
                                      self._odom, qos)
        self.node.create_subscription(DiagnosticArray, '/nmpc/diagnostics', self._diag, 10)
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        from rclpy.executors import ExternalShutdownException
        while not self.stop_evt.is_set():
            try:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
            except ExternalShutdownException:
                break

    def _odom(self, m):
        self.odom.append([time.time() - self.t0, *[float(v) for v in m.position], *[float(v) for v in m.velocity]])

    def _diag(self, msg):
        for st in msg.status:
            kv = {k.key: k.value for k in st.values}
            if 'phase' in kv:
                kv['t'] = time.time() - self.t0
                self.diag.append(kv)

    def close(self):
        self.stop_evt.set()
        self.thread.join(timeout=2)
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def variant_from_study(name, tuned):
    spec = yaml.safe_load((BUNDLE / 'study.yaml').read_text())
    variants = spec['variants']
    v = copy.deepcopy(variants[name])
    if tuned:
        folder = Path(tuned)
        if (folder / 'best_config.json').exists():
            values = json.loads((folder / 'best_config.json').read_text())
            for k, val in values.items():
                if val is None:
                    continue
                cur = v
                keys = k.split('.')
                for kk in keys[:-1]:
                    cur = cur.setdefault(kk, {})
                cur[keys[-1]] = val
        else:
            # The direct-rate tuner validates its winner on every PX4
            # navigation problem and stores the native settings in report.json.
            report = json.loads((folder / 'report.json').read_text())
            winner = report.get('best')
            if not winner or not winner.get('eligible') or not winner.get('settings'):
                raise ValueError('tuning report has no full-trace eligible winner')
            v['cg2o'] = deep_merge(v.get('cg2o', {}), winner['settings'])
        v['label'] = v.get('label', name) + ', tuned'
    return v


def pin_controller(cpus):
    """Pin every running nmpc_controller process (all threads) to `cpus`; returns {pid: affinity}."""
    pinned = {}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(a.endswith(b'/nmpc_controller') for a in argv):
            subprocess.run(['taskset', '-a', '-cp', str(cpus), d.name], stdout=subprocess.DEVNULL, check=False)
            pinned[d.name] = sorted(os.sched_getaffinity(int(d.name)))
    print(f'[sitl] controller pinned to CPUs {cpus}: {pinned}', flush=True)
    if not pinned:
        print('[sitl] WARNING: no nmpc_controller process found to pin', flush=True)
    return pinned


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='ispd_warm_final', help='variant of study.yaml (default ISPD warm final)')
    ap.add_argument('--challenging-vs-max', type=float, default=2.5,
                    help='challenging MPC: guide progress speed cap [m/s]')
    ap.add_argument('--challenging-start', default=None,
                    help='challenging MPC: take-off offset "dN,dE" [m] from the course start (randomized repeats)')
    ap.add_argument('--challenging-yaw-deg', type=float, default=0.0,
                    help='challenging MPC: initial heading [deg, NED], 0 = along the course (North)')
    ap.add_argument('--challenging-cg2o', default=None,
                    help='challenging MPC: tuned cg2o settings (selected_parameters.json of a tuning study)')
    ap.add_argument('--challenging-cg2o-cold', default=None,
                    help='challenging MPC: cold-start tuned cg2o values, used for the cold solves of a warm flight')
    ap.add_argument('--prediction-model', choices=('linear','nonlinear','challenging'), default='linear',
                    help='Change prediction formulation; retain guidance, tracking and mission')
    ap.add_argument('--control-interface', choices=('trajectory','body_rate'), default='trajectory',
                    help='PX4 offboard command interface; use body_rate for direct nonlinear commands')
    ap.add_argument('--nonlinear-profile', choices=('ported','matched','matched_fixed_yaw'), default='matched',
                    help='Native formulation defaults, or bounds/cost units matched to the old acceleration/jerk model')
    ap.add_argument('--nonlinear-tracking-margin',type=float,default=0.,
                    help='additional optimizer-only obstacle reserve [m] for direct-rate tracking error')
    ap.add_argument('--nonlinear-backtracking-min',type=float,default=.03,
                    help='minimum ISPD inequality line-search step for the nonlinear formulation')
    ap.add_argument('--nonlinear-recovery-buffer',type=float,default=.02,
                    help='clearance retained when the extra tracking reserve adapts near an obstacle')
    ap.add_argument('--nonlinear-rate-slew-weight',type=float,default=None,
                    help='optional body-rate change weight; keep default unless testing a control ablation')
    ap.add_argument('--fixed-nonlinear-margin',action='store_true',
                    help='keep the full extra obstacle reserve near obstacles; use the NLP slack for recovery')
    ap.add_argument('--cold-every-solve',action='store_true',
                    help='discard all primal/dual warm-start history at every MPC update')
    ap.add_argument('--tuned', default=None, help='tuning folder whose validated winner is applied to the variant')
    ap.add_argument('--max-iterations', type=int, default=40,
                    help='solver iteration budget per NMPC solve (default: the flight budget of cg2o.yaml; a larger '
                         'budget allows solves of a second or more, during which PX4 follows a stale plan)')
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds from mission start to abort')
    ap.add_argument('--speed', type=float, default=None, help='reference speed [m/s] (default: the bundle, 2.7 m/s)')
    ap.add_argument('--guidance-mode', choices=('fixed_line','planner','direct'), default=None,
                    help='reference route for a separate control ablation; the paper stress case uses fixed_line')
    ap.add_argument('--hover-throttle',type=float,default=None,
                    help='normalized PX4 throttle corresponding to hover; useful for measured direct-rate calibration')
    ap.add_argument('--accel', type=float, default=None,
                    help='reference acceleration and braking limit [m/s^2] (default: the bundle, 1.8 / 2.0)')
    ap.add_argument('--no-acceleration-feedforward', action='store_true',
                    help='send position + velocity only (nmpc.yaml default). By default the planned acceleration is '
                         'sent as feed-forward, as the PX4-like simulator of the study applies it; without it PX4 lags '
                         'the slalom manoeuvres by more than the safety margin')
    ap.add_argument('--reseed-pos', type=float, default=0.15,
                    help='the NMPC plans from the measured state once PX4 is this far [m] from the plan (nmpc.yaml: '
                         '0.5). Must be below the safety margin (0.35 m), otherwise the tracking error of real PX4 '
                         'can use up the whole margin unnoticed')
    ap.add_argument('--reseed-vel', type=float, default=0.4, help='same for the velocity error [m/s] (nmpc.yaml: 0.8)')
    ap.add_argument('--study-settings', action='store_true',
                    help='solve with the solver settings of the paired slalom study (study_paper.yaml): iteration '
                         'budget 250 and wall-time limit 2.5 s instead of 40 / 0.25 s. The OCP is the same in all '
                         'cases. Re-seeding keeps --reseed-pos/--reseed-vel (0.15 m / 0.4 m/s); the study simulator '
                         'uses 0.5 m / 0.8 m/s, which with real PX4 lets the tracking error use up the safety margin '
                         '(add --reseed-pos 0.5 --reseed-vel 0.8 to reproduce that)')
    ap.add_argument('--controller-cpus', default=None,
                    help='pin the NMPC controller process (all threads, solver included) to these CPUs, e.g. 2,3 = '
                         'one isolated P-core (isolcpus=0-3); Gazebo, PX4, RViz and the recorder stay on the others. '
                         'Use it whenever the solve times are reported')
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--no-gazebo-gui', action='store_true', help='headless Gazebo (RViz video only)')
    ap.add_argument('--headless', action='store_true',
                    help='no windows and no recording: the same PX4 SITL flight with logs, plots and summary.json')
    ap.add_argument('--output', default=None)
    args = ap.parse_args(argv)
    if args.headless:
        args.no_gazebo_gui = True
    wall_time_s = 0.25
    if args.study_settings:
        study = yaml.safe_load((BUNDLE / 'study_paper.yaml').read_text())['study']
        bundle_rp = yaml.safe_load((BUNDLE / 'nmpc.yaml').read_text())['px4_nmpc_controller']['ros__parameters']
        args.max_iterations = int(study.get('budget', 250))
        wall_time_s = float(bundle_rp.get('solver_max_wall_time_s', 2.5))

    from px4_nmpc_compare.common import unique_run_dir
    from px4_nmpc_core.scenario import load_scenario_json, controller_origin_ned_of

    variant = variant_from_study(args.variant, args.tuned)
    backend = variant['backend']
    tag = ('ispd' if backend == 'cg2o' else 'ipopt') + ('_tuned' if args.tuned else '')
    if args.prediction_model == 'nonlinear':tag += '_nonlinear'
    if args.prediction_model == 'challenging':
        tag += '_challenging'
        if args.control_interface != 'body_rate':
            ap.error('--prediction-model challenging sends thrust and body rates: use --control-interface body_rate')
    out = unique_run_dir(ROOT, f'slalom_{tag}', args.output, group='sitl')
    procs = Procs(out / 'logs')
    env = dict(os.environ)
    # System RViz cannot load the GTK modules inherited from Snap VS Code.
    for key in ('GTK_PATH','GTK_EXE_PREFIX','GTK_MODULES','GIO_MODULE_DIR','GDK_PIXBUF_MODULE_FILE','GDK_PIXBUF_MODULEDIR','GTK_IM_MODULE_FILE'):
        env.pop(key,None)
    if args.prediction_model == 'nonlinear':
        env['PX4_NMPC_NONLINEAR_LOG'] = str(out/'nonlinear_solves.jsonl')
        env['PX4_NMPC_NONLINEAR_TRACE'] = str(out/'nonlinear_trace.jsonl')
    if args.prediction_model == 'challenging':
        env['PX4_NMPC_CHALLENGING_TRACE'] = str(out/'challenging_trace.jsonl')
        if args.challenging_cg2o:
            shutil.copy(args.challenging_cg2o, out / 'challenging_cg2o_tuned.json')
            env['PX4_NMPC_CHALLENGING_CG2O'] = str(out / 'challenging_cg2o_tuned.json')
        if args.challenging_cg2o_cold:
            shutil.copy(args.challenging_cg2o_cold, out / 'challenging_cg2o_cold_tuned.json')
            env['PX4_NMPC_CHALLENGING_CG2O_COLD'] = str(out / 'challenging_cg2o_cold_tuned.json')
    env.setdefault('TMPDIR', str(ROOT / 'build' / 'tmp'))

    # ---- solver settings for the ROS node (dicts are not ROS parameters: YAML files via environment)
    if backend == 'cg2o':
        base = yaml.safe_load((CFG / 'cg2o.yaml').read_text())
        cg = deep_merge(base, variant.get('cg2o') or {})
        if args.prediction_model == 'nonlinear':
            cg.setdefault('ispd', {})['ineq_backtracking_step_min'] = args.nonlinear_backtracking_min
        cg['max_iterations'] = int(args.max_iterations)
        (out / 'cg2o_used.yaml').write_text(yaml.safe_dump(cg, sort_keys=False))
        env['PX4_NMPC_CG2O_CONFIG'] = str(out / 'cg2o_used.yaml')
    else:
        base = yaml.safe_load((CFG / 'ipopt_cpp.yaml').read_text())
        ip = deep_merge(base.get('ipopt_cpp', base), variant.get('ipopt_cpp') or {})
        (out / 'ipopt_cpp_used.yaml').write_text(yaml.safe_dump({'ipopt_cpp': ip}, sort_keys=False))
        env['PX4_NMPC_IPOPT_CPP_CONFIG'] = str(out / 'ipopt_cpp_used.yaml')
    extra = yaml.safe_load((BUNDLE / 'nmpc.yaml').read_text())
    rp = extra['px4_nmpc_controller']['ros__parameters']
    rp['prediction_model'] = args.prediction_model
    rp['control_interface'] = args.control_interface
    rp['nonlinear_profile'] = args.nonlinear_profile
    rp['nonlinear_tracking_margin'] = float(args.nonlinear_tracking_margin)
    rp['nonlinear_backtracking_min'] = float(args.nonlinear_backtracking_min)
    rp['nonlinear_recovery_buffer'] = float(args.nonlinear_recovery_buffer)
    if args.nonlinear_rate_slew_weight is not None:
        if not math.isfinite(args.nonlinear_rate_slew_weight) or args.nonlinear_rate_slew_weight < 0.:
            ap.error('--nonlinear-rate-slew-weight must be finite and nonnegative')
        rp['nonlinear_rate_slew_weight'] = float(args.nonlinear_rate_slew_weight)
    rp['nonlinear_margin_adaptive'] = not args.fixed_nonlinear_margin
    rp['force_cold_start'] = bool(args.cold_every_solve)
    scenario_source = BUNDLE / 'scenario_10.json'
    if args.prediction_model == 'challenging':
        from px4_nmpc_compare import challenging as challenging_course
        scenario_source = out / 'challenging_scenario.json'
        start_offset = [float(v) for v in args.challenging_start.split(',')] if args.challenging_start else None
        course_scenario = challenging_course.write_scenario(challenging_course.load_course(), scenario_source,
                                                            start_offset)
        rp['challenging_vs_max'] = float(args.challenging_vs_max)
        rp['challenging_origin_ned'] = [float(v) for v in controller_origin_ned_of(course_scenario)]
        rp['horizon_steps'] = 40
        rp['control_dt'] = 0.10
    if args.guidance_mode is not None:
        rp['guidance_mode'] = args.guidance_mode
        rp['planner_enabled'] = args.guidance_mode == 'planner'
    rp['solver_max_iter'] = int(args.max_iterations)
    rp['solver_max_wall_time_s'] = wall_time_s
    if args.speed is not None:
        rp['desired_speed'] = float(args.speed)
    if args.hover_throttle is not None:
        rp['hover_throttle'] = float(args.hover_throttle)
    if args.accel is not None:
        rp['reference_accel_max'] = rp['reference_decel_max'] = float(args.accel)
    rp['trajectory_acceleration_feedforward'] = not args.no_acceleration_feedforward
    args.speed = float(rp['desired_speed'])
    rp['reseed_position_error_m'] = float(args.reseed_pos)
    rp['reseed_velocity_error_mps'] = float(args.reseed_vel)
    (out / 'nmpc_slalom.yaml').write_text(yaml.safe_dump(extra, sort_keys=False))

    # ---- runtime scenario (the fixed slalom geometry)
    runtime = out / 'runtime'
    rc, txt = run(['python3', '-m', 'px4_nmpc_tools.experiment_manager', '--runtime-dir', runtime, 'load',
                   scenario_source, '--experiment', BUNDLE / 'experiment.yaml'], env=env)
    (out / 'logs' / 'scenario.log').write_text(txt)
    if rc != 0:
        raise SystemExit(f'loading the slalom scenario failed:\n{txt}')
    scenario = load_scenario_json(runtime / 'scenario.json')
    start_alt = -float(controller_origin_ned_of(scenario)[2])

    # ---- RViz: no side panels, chase view on the vehicle (map = ENU: x East, y North; the slalom runs north)
    rv = yaml.safe_load((ROOT / 'src/px4_nmpc_bringup/config/nmpc.rviz').read_text())
    rv['Panels'] = []
    view = rv['Visualization Manager']['Views']['Current']
    view.update({'Target Frame': 'nmpc_vehicle', 'Distance': 9.0, 'Pitch': 0.42, 'Yaw': -0.55,     # behind, slightly aside
                 'Focal Point': {'X': 0.0, 'Y': 0.0, 'Z': 0.0}})
    rv['Window Geometry'].update({'Width': 960, 'Height': 720, 'X': 0, 'Y': 0, 'Hide Left Dock': True,
                                  'Hide Right Dock': True})
    rviz_cfg = out / 'slalom.rviz'
    rviz_cfg.write_text(yaml.safe_dump(rv, sort_keys=False))

    px4_build = ROOT / 'build' / 'externals' / 'PX4-Autopilot' / 'px4_sitl_default'
    if not (px4_build / 'bin' / 'px4').exists():
        raise SystemExit('PX4 SITL not built: PX4_SETUP=OFF ./scripts/build_externals/px4.sh')
    logger = None
    nav_dll_act = None
    penv = dict(env)
    summary = {'variant': variant, 'backend': backend, 'prediction_model': args.prediction_model,
               'challenging_vs_max': float(args.challenging_vs_max) if args.prediction_model == 'challenging' else None,
               'challenging_start_offset': args.challenging_start, 'challenging_yaw_deg': args.challenging_yaw_deg,
               'challenging_cg2o': args.challenging_cg2o,
               'challenging_cg2o_cold': args.challenging_cg2o_cold,
               'nonlinear_profile': args.nonlinear_profile, 'tuned': args.tuned, 'speed_mps': args.speed,
               'nonlinear_tracking_margin_m': args.nonlinear_tracking_margin,
               'nonlinear_backtracking_min': args.nonlinear_backtracking_min,
               'nonlinear_recovery_buffer_m': args.nonlinear_recovery_buffer,
               'nonlinear_rate_slew_weight': float(rp.get('nonlinear_rate_slew_weight',-1.)),
               'nonlinear_margin_adaptive': not args.fixed_nonlinear_margin,
               'cold_every_solve': bool(args.cold_every_solve),
               'guidance_mode': rp['guidance_mode'],
               'accel_mps2': rp['reference_accel_max'], 'decel_mps2': rp['reference_decel_max'],
               'acceleration_feedforward': rp['trajectory_acceleration_feedforward'],
               'study_settings': bool(args.study_settings), 'solver_max_wall_time_s': wall_time_s,
               'max_iterations': args.max_iterations, 'reseed_position_error_m': args.reseed_pos,
               'reseed_velocity_error_mps': args.reseed_vel,
               'hover_throttle': float(rp.get('hover_throttle',.729))}
    rec_rviz = None
    try:
        procs.start('xrce_agent', ['MicroXRCEAgent', 'udp4', '-p', '8888'], env=env)
        # PX4 starts the Gazebo server only (HEADLESS); the GUI is started below with our own window and camera
        # Challenging course: spawn facing North (Gazebo ENU yaw pi/2 = NED yaw 0), the course
        # direction. With the default spawn the NMPC took over facing East (NED yaw ~97 deg) and had
        # to yaw 90 deg while accelerating North sideways (47 deg commanded, 67 deg true tilt).
        # Gazebo ENU yaw = pi/2 - NED yaw.
        spawn_yaw = (math.pi / 2 - math.radians(args.challenging_yaw_deg)
                     if args.prediction_model == 'challenging' else 0.0)
        penv = dict(env, PX4_SIM_MODEL='gz_x500', GZ_IP='127.0.0.1',
                    PX4_GZ_MODEL_POSE=f'0,0,{start_alt:.3f},0,0,{spawn_yaw:.6f}',
                    HEADLESS='1')
        procs.start('px4', [px4_build / 'bin' / 'px4', '-d'], cwd=px4_build / 'rootfs', env=penv)
        t0 = time.time()
        while time.time() - t0 < 90:
            rc, topics = run(['gz', 'topic', '-l'], timeout=10, env=env)
            if '/world/default/clock' in topics and 'x500_0' in topics:
                break
            time.sleep(2.0)
        else:
            raise SystemExit('Gazebo/PX4 did not come up (see logs/px4.log)')
        time.sleep(5.0)
        rc, txt = run(['python3', '-m', 'px4_nmpc_tools.experiment_manager', '--runtime-dir', runtime, 'apply'],
                      timeout=90, env=env)
        (out / 'logs' / 'apply_scene.log').write_text(txt)
        # PX4 refuses to arm without a ground station when NAV_DLL_ACT > 0 (data-link-loss action). No QGC runs
        # here: 0 for this flight, the previous value is restored and saved at the end. The original value is kept in
        # build/sitl_px4_params.json until the restore succeeded (survives a crash of this script).
        rc, txt = run([px4_build / 'bin' / 'px4-param', 'show', 'NAV_DLL_ACT'], timeout=15, env=penv)
        (out / 'logs' / 'px4_param.log').write_text(txt)
        m = re.search(r'NAV_DLL_ACT\s*\[[^]]*\]\s*:\s*(-?\d+)', txt)
        state = json.loads(PX4_PARAM_STATE.read_text()) if PX4_PARAM_STATE.exists() else {}
        if m and 'NAV_DLL_ACT' not in state:
            state['NAV_DLL_ACT'] = int(m.group(1))
            PX4_PARAM_STATE.write_text(json.dumps(state))
        nav_dll_act = state.get('NAV_DLL_ACT')
        run([px4_build / 'bin' / 'px4-param', 'set', 'NAV_DLL_ACT', '0'], timeout=15, env=penv)
        print(f'[sitl] NAV_DLL_ACT {nav_dll_act} -> 0 for this flight (restored afterwards)', flush=True)
        if not args.no_gazebo_gui:
            (out / 'gazebo_gui.config').write_text(GZ_GUI_CONFIG.format(x=960, y=0, w=960, h=720))
            procs.start('gz_gui', ['gz', 'sim', '-g', '--gui-config', out / 'gazebo_gui.config'],
                        env=dict(gz_environment(px4_build, env), GZ_IP='127.0.0.1'))
        procs.start('launch', ['ros2', 'launch', 'px4_nmpc_bringup', 'nmpc.launch.py',
                               f'solver_backend:={backend}', f'scenario_json:={runtime / "scenario.json"}',
                               f'scenario_file:={runtime / "scenario.yaml"}', f'extra_params_file:={out / "nmpc_slalom.yaml"}',
                               f'rviz_config:={rviz_cfg}', 'launch_monitor:=false', 'rviz_respawn:=false',
                               f"launch_rviz:={'false' if args.headless else 'true'}"], env=env)
        logger = Logger()
        t0 = time.time()
        status = None
        while time.time() - t0 < 90:
            status = nmpc_status(env)
            if status is not None:
                break
            time.sleep(2.0)
        if status is None:
            raise SystemExit('controller did not report status (see logs/launch.log)')
        print(f"[sitl] controller ready: {status['raw']}", flush=True)
        if status.get('backend', '').strip() not in (backend, 'cg2o', 'ipopt_cpp') or \
                (backend == 'cg2o' and 'cg2o' not in status.get('backend', '')):
            print(f"[sitl] WARNING: controller backend is {status.get('backend')}", flush=True)
        summary['controller_backend'] = status.get('backend')
        if args.controller_cpus:
            summary['controller_cpus'] = pin_controller(args.controller_cpus)

        # ---- one screen capture of both windows (RViz left, Gazebo right): the two views stay synchronised
        geom = ggeom = None
        if not args.headless:
            geom = place_window(r'RViz\*?$', 0, 0, 960, 720) or find_window(r'"[^"]*RViz\*?"')
        if not args.no_gazebo_gui:
            ggeom = place_window(r'^Gazebo Sim', 960, 0, 960, 720)
            # chase camera: legacy follow (always available) + smoothed follow / look-at (Harmonic /gui/track)
            run(['gz', 'service', '-s', '/gui/follow', '--reqtype', 'gz.msgs.StringMsg', '--reptype',
                 'gz.msgs.Boolean', '--timeout', '5000', '--req', 'data: "x500_0"'], env=env)
            run(['gz', 'service', '-s', '/gui/follow/offset', '--reqtype', 'gz.msgs.Vector3d', '--reptype',
                 'gz.msgs.Boolean', '--timeout', '5000', '--req', 'x: -3.2, y: -0.7, z: 2.0'], env=env)
            run(['gz', 'topic', '-t', '/gui/track', '-m', 'gz.msgs.CameraTrack', '-p',
                 'track_mode: FOLLOW_LOOK_AT, follow_target: {name: "x500_0"}, track_target: {name: "x500_0"}, '
                 'follow_offset: {x: -3.2, y: -0.7, z: 2.0}, follow_pgain: 0.08, track_pgain: 0.3'], env=env)
        # windows settle (Gazebo resizes itself after loading): place again and read the final geometry
        time.sleep(2.0)
        if not args.headless:
            geom = place_window(r'RViz\*?$', 0, 0, 960, 720) or geom
        if not args.no_gazebo_gui:
            ggeom = place_window(r'^Gazebo Sim', 960, 0, 960, 720) or ggeom
        rects = [g for g in (geom, ggeom) if g is not None]
        if rects:
            # each window's own rectangle only (nothing else of the desktop), both inputs in one ffmpeg process
            cmd = ['ffmpeg', '-y', '-loglevel', 'error']
            for g in rects:
                cmd += ['-f', 'x11grab', '-draw_mouse', '0', '-framerate', args.fps,
                        '-video_size', f'{g[2] - g[2] % 2}x{g[3] - g[3] % 2}', '-i',
                        f"{env.get('DISPLAY', ':0')}+{g[0]},{g[1]}"]
            if len(rects) == 2:
                cmd += ['-filter_complex', '[0:v]scale=960:720,setsar=1[a];[1:v]scale=960:720,setsar=1[b];'
                                           '[a][b]hstack=inputs=2[v]', '-map', '[v]']
            cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out / 'screen.mp4']
            rec_rviz = procs.start('rec_screen', cmd, env=env, stdin=subprocess.PIPE)
        elif not args.headless:
            print('[sitl] no RViz / Gazebo window found: nothing is recorded', flush=True)
        summary['recording'] = {'rviz': geom, 'gazebo': ggeom}

        # ---- mission
        time.sleep(2.0)
        rc, txt = run(['ros2', 'run', 'px4_nmpc_tools', 'nmpc_mission', 'start'], timeout=60, env=env)
        (out / 'logs' / 'mission_start.log').write_text(txt)
        t_start = time.time()
        phases = []
        outcome = 'timeout'
        goal_since = None
        while time.time() - t_start < args.timeout:
            st = nmpc_status(env)
            if st is not None:
                ph = st.get('phase', '?')
                phases.append((round(time.time() - t_start, 2), st['raw']))
                if ph == 'GOAL':
                    goal_since = goal_since or time.time()
                    if time.time() - goal_since > 3.0:
                        outcome = 'goal'
                        break
                if ph == 'ABORTED':
                    outcome = 'aborted'
                    break
            time.sleep(0.5)
        summary['outcome'] = outcome
        summary['mission_wall_s'] = round(time.time() - t_start, 2)
        (out / 'logs' / 'status.log').write_text('\n'.join(f'{t}: {s}' for t, s in phases))
        print(f'[sitl] mission outcome: {outcome}', flush=True)
        time.sleep(1.0)
    finally:
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if rec_rviz is not None:
            procs.stop('rec_screen', sig=signal.SIGINT)
        try:
            run(['ros2', 'run', 'px4_nmpc_tools', 'nmpc_mission', 'stop'], timeout=30, env=env)
            time.sleep(8.0)
        except Exception:  # noqa: BLE001
            pass
        if nav_dll_act is not None:
            try:
                bin_param = px4_build / 'bin' / 'px4-param'
                run([bin_param, 'set', 'NAV_DLL_ACT', str(nav_dll_act)], timeout=15, env=penv)
                rc_s, _ = run([bin_param, 'save'], timeout=15, env=penv)
                rc_c, txt_c = run([bin_param, 'show', 'NAV_DLL_ACT'], timeout=15, env=penv)
                if rc_s == 0 and re.search(rf'NAV_DLL_ACT\s*\[[^]]*\]\s*:\s*{nav_dll_act}\b', txt_c):
                    PX4_PARAM_STATE.unlink(missing_ok=True)
                    print(f'[sitl] NAV_DLL_ACT restored to {nav_dll_act} and saved', flush=True)
                else:
                    print(f'[sitl] WARNING: NAV_DLL_ACT={nav_dll_act} not confirmed; the next run restores it '
                          f'({PX4_PARAM_STATE})', flush=True)
            except Exception:  # noqa: BLE001
                print(f'[sitl] WARNING: could not restore NAV_DLL_ACT={nav_dll_act}; the next run restores it '
                      f'({PX4_PARAM_STATE})', flush=True)
        if logger is not None:
            logger.close()
            write_logs(out, logger)
        procs.stop()
        # the Gazebo server started by PX4 (only this repository's PX4 world, not other Gazebo sessions)
        subprocess.run(['pkill', '-f', f'gz sim.*{ROOT}/externals/PX4-Autopilot'], check=False)
        signal.signal(signal.SIGINT, previous_sigint)

    analyse(out, runtime, summary)
    compose(out, variant, args)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    video = out / 'slalom_sitl.mp4'
    print(json.dumps({'folder': str(out), 'video': str(video) if video.exists() else None,
                      'outcome': summary.get('outcome')}))
    return 0


def write_logs(out, logger):
    with open(out / 'flight.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'pN', 'pE', 'pD', 'vN', 'vE', 'vD'])
        w.writerows(logger.odom)
    keys = ['t', 'phase', 'last_solve_ms', 'last_solver_iterations', 'solver_status', 'usable_solves', 'solves',
            'distance_to_goal_m', 'speed_mps', 'min_clearance_seen_m', 'applied_source', 'backend']
    with open(out / 'solves.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for d in logger.diag:
            w.writerow([d.get(k, '') for k in keys])


def analyse(out, runtime, summary):
    """Flight figure (top view + clearance) and numbers of the SITL flight."""
    from px4_nmpc_compare import study_figures as sf
    rows = list(csv.reader(open(out / 'flight.csv')))[1:] if (out / 'flight.csv').exists() else []
    if not rows:
        return
    a = np.asarray(rows, float)
    params = yaml.safe_load((runtime / 'scenario.yaml').read_text())['px4_nmpc_controller']['ros__parameters']
    slalom = yaml.safe_load((BUNDLE / 'nmpc.yaml').read_text())['px4_nmpc_controller']['ros__parameters']
    params.update({k: slalom[k] for k in ('uav_radius', 'safety_margin')})
    pos = a[:, 1:4]
    diag = list(csv.DictReader(open(out / 'solves.csv')))
    nav = [d for d in diag if d['phase'] in ('NMPC', 'GOAL')]
    t_nav = [float(d['t']) for d in diag if d['phase'] == 'NMPC']
    t_goal = [float(d['t']) for d in diag if d['phase'] == 'GOAL']
    if t_nav:
        sel = (a[:, 0] >= t_nav[0]) & (a[:, 0] <= (t_goal[0] if t_goal else a[-1, 0]))
        pos_nav = pos[sel] if sel.any() else pos
    else:
        pos_nav = pos
    clear = sf.clearance_along(params, pos_nav)
    ms = np.asarray([float(d['last_solve_ms']) for d in nav if d['last_solve_ms'] not in ('', 'nan')])
    it = np.asarray([float(d['last_solver_iterations']) for d in nav if d.get('last_solver_iterations') not in ('', 'nan', None)])
    summary.update({
        'time_to_goal_s': (t_goal[0] - t_nav[0]) if (t_nav and t_goal) else None,
        'min_physical_clearance_m': float(clear.min()),
        'safety_margin_m': float(params['safety_margin']),
        'diagnostic_solve_ms_median': float(np.median(ms)) if ms.size else None,
        'diagnostic_solve_ms_p95': float(np.percentile(ms, 95)) if ms.size else None,
        'diagnostic_solve_ms_max': float(ms.max()) if ms.size else None,
        'diagnostic_iterations_median': float(np.median(it)) if it.size else None,
        'final_usable_solves': nav[-1]['usable_solves'] if nav else None, 'final_solves': nav[-1]['solves'] if nav else None,
    })
    plt = sf._plt()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.2), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0]})
    handles = sf.draw_obstacles_top(axes[0], params, pos_nav)
    axes[0].plot(pos_nav[:, 0], pos_nav[:, 1], color=sf.PALETTE['orange'], lw=1.6, label='PX4 SITL flight')
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[0].set_ylabel('East [m]')
    axes[0].set_title('PX4 SITL + Gazebo: slalom flown with the NMPC (top view)', loc='left')
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + handles, fontsize=7.5, loc='upper left',
                   bbox_to_anchor=(1.0, 1.05))
    axes[1].plot(pos_nav[:, 0], clear, color=sf.PALETTE['orange'], lw=1.4)
    axes[1].axhline(0.0, color='#e34948', lw=1.0)
    axes[1].axhline(float(params['safety_margin']), color='#b8860b', lw=1.0, ls='--')
    axes[1].set_ylabel('clearance [m]')
    axes[1].set_xlabel('North [m]')
    axes[1].set_ylim(-0.1, 2.0)
    fig.tight_layout()
    fig.savefig(out / 'flight.png', dpi=170, bbox_inches='tight')
    fig.savefig(out / 'flight.pdf', bbox_inches='tight')
    plt.close(fig)


def compose(out, variant, args):
    """The synchronised screen capture (RViz | Gazebo) with a title bar."""
    screen = out / 'screen.mp4'
    if not screen.exists() or screen.stat().st_size < 10000:
        return
    cg = dict(variant.get('cg2o') or {})
    ws = {0: 'primal shift', 1: 'final iterate', 2: 'middle iterate'}.get(
        int(dict(cg.get('warm_start') or {}).get('strategy', 1)), '')
    solver = (f"cg2o ISPD (warm start: {ws})" if variant['backend'] == 'cg2o'
              else f"IPOPT ({dict(variant.get('ipopt_cpp') or {}).get('warm_start', 'primal')} warm start)")
    route = ('waypoint-guided route' if args.guidance_mode == 'planner'
             else 'infeasible 3-D slalom')
    lines = [f"PX4 SITL + Gazebo   |   {args.prediction_model} prediction   |   NMPC solver: {solver}" + (', tuned' if args.tuned else ''),
             f'{route}: 10 obstacles, reference {args.speed:g} m/s, horizon 4 s (N = 40)   |   '
             'left: RViz (prediction, safety shells, solver status)   right: Gazebo']
    if args.prediction_model == 'challenging':
        lines = [f"PX4 SITL + Gazebo   |   thrust + body-rate NMPC to PX4   |   solver: {solver}"
                 + (', cold every solve' if args.cold_every_solve else ''),
                 f'infeasible guide line, 4 spheres + 4 posts + 2 bars, progress up to {args.challenging_vs_max:g} m/s, '
                 'horizon 4 s (N = 40)   |   left: RViz   right: Gazebo']
    esc = lambda t: t.replace('\\', '').replace(':', r'\:').replace("'", '').replace('%', r'\%')
    filt = (f"[0:v]scale=1920:-2,pad=iw:ih+64:0:64:color=white,"
            f"drawtext=text='{esc(lines[0])}':x=14:y=10:fontsize=20:fontcolor=black,"
            f"drawtext=text='{esc(lines[1])}':x=14:y=38:fontsize=16:fontcolor=0x444444[o]")
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(screen), '-filter_complex', filt, '-map', '[o]',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', str(out / 'slalom_sitl.mp4')]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        (out / 'logs' / 'compose.log').write_text(r.stderr)


if __name__ == '__main__':
    raise SystemExit(main())

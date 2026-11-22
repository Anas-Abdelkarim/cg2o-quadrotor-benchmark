"""Challenging nonlinear MPC: course, problem data, hashing and the paired
offline closed loop (cg2o drives, IPOPT solves the identical problem in shadow).

Every problem passed to a backend is built here, hashed (SHA-256 over all
numerical data) and logged, so cg2o and IPOPT results can be paired exactly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
COURSE = ROOT / 'src/px4_nmpc_config/config/experiments/challenging_course/course.yaml'
CG2O_YAML = ROOT / 'src/px4_nmpc_config/config/cg2o.yaml'
CG2O_COURSE = COURSE.parent / 'cg2o_overrides.yaml'
PARAM_FIELDS = ('N', 'dt', 'mass', 'gravity', 'drag', 'tau_rate', 'tmin', 'tmax', 'rate_xy', 'rate_z', 'vmax',
                'cos_tilt', 'vs_max', 's_goal', 'vs_cold', 'e_amp', 'e_freq', 'e_phase', 'alt0',
                'alt_amp', 'alt_freq', 'alt_phase', 'body', 'body_radius', 'margin', 'lower', 'upper',
                'w_lag', 'w_lag_terminal', 'w_contour', 'w_contour_terminal', 'w_heading',
                'w_attitude', 'w_thrust', 'w_rate', 'w_progress', 'v_bar', 'w_goal', 'goal_delta', 'w_slew', 'w_slack',
                'obstacles')


def extension():
    directory = str(ROOT / 'build/challenging_mpc')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import _challenging
    return _challenging


def load_course(path=COURSE):
    path = Path(path)
    if not path.is_absolute():          # relative paths (shipped data) refer to the repository root
        path = ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def make_params(course, overrides=None):
    """Params of the native model from the course file (plus flat overrides)."""
    ext = extension()
    c, v, m = course['course'], course['vehicle'], course['mpc']
    o = dict(overrides or {})
    p = ext.Params()
    p.N = int(o.get('N', m['N']))
    p.dt = float(o.get('dt', m['dt']))
    p.mass = float(v['mass'])
    p.gravity = 9.81
    p.drag = np.asarray(o.get('drag', [0., 0., 0.]), float)
    p.tau_rate = np.asarray(o.get('tau_rate', m['tau_rate']), float)
    hover = p.mass * p.gravity
    p.tmin = float(m['thrust_min_over_hover']) * hover
    p.tmax = float(m['thrust_max_margin']) * float(v['thrust_max_over_hover']) * hover
    p.rate_xy = float(m['rate_xy'])
    p.rate_z = float(m['rate_z'])
    p.vs_max = float(o.get('vs_max', m['vs_max']))
    p.vmax = p.vs_max + float(m['vmax_margin'])
    p.cos_tilt = math.cos(math.radians(float(o.get('tilt_max_deg', m['tilt_max_deg']))))
    p.vs_cold = p.vs_max
    p.v_bar = p.vs_max
    p.s_goal = float(c['goal_s'])
    for k, val in c['guide'].items():
        setattr(p, k, float(val))
    p.body = np.asarray(v['body'], float)
    p.body_radius = np.asarray(v['body_radius'], float)
    p.margin = float(o.get('margin', m['margin']))
    p.lower = np.asarray(c['arena']['lower'], float)
    p.upper = np.asarray(c['arena']['upper'], float)
    w = m['weights']
    p.w_lag, p.w_lag_terminal = float(w['lag']), float(w['lag_terminal'])
    p.w_contour, p.w_contour_terminal = float(w['contour']), float(w['contour_terminal'])
    p.w_heading, p.w_attitude = float(w['heading']), float(w['attitude'])
    p.w_thrust, p.w_rate, p.w_progress = float(w['thrust']), float(w['rate']), float(w['progress'])
    p.w_goal = float(w['goal'])
    p.goal_delta = float(w['goal_delta'])
    p.w_slew = np.asarray(w['slew'], float)
    p.w_slack = float(w['slack'])
    p.obstacles = np.asarray(c['obstacles'], float)
    return p


def make_problem(p, x0, uprev):
    ext = extension()
    b = ext.Problem()
    b.p = p
    b.x0 = np.asarray(x0, float)
    b.uprev = np.asarray(uprev, float)
    b.build()
    return b


def problem_sha256(b):
    """Hash of every number that defines the NLP (parameters, x0, previous input)."""
    h = hashlib.sha256()
    p = b.p
    for name in PARAM_FIELDS:
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(getattr(p, name), float)).tobytes())
    h.update(np.ascontiguousarray(b.x0, float).tobytes())
    h.update(np.ascontiguousarray(b.uprev, float).tobytes())
    return h.hexdigest()


def array_sha256(z):
    return hashlib.sha256(np.ascontiguousarray(z, float).tobytes()).hexdigest()


def measured_state(p, pos, vel, rpy, omega=(0., 0., 0.)):
    """MPC state from a measurement [p, v, rpy, body rates]; the guide progress
    is the closest point of the guide line."""
    ext = extension()
    s = ext.project(np.asarray(pos, float), p)
    return np.r_[pos, vel, rpy, omega, s]


def shift_primal(b, z_prev, shift=1):
    """Shifted previous solution, the same rule as cg2o's shift: stage k takes
    stage k + shift, the last stages are rolled out with the dynamics, x0 is the
    new measurement (dynamics defect at the first stage)."""
    ext = extension()
    N = b.p.N
    X, U, S = b.states(z_prev), b.inputs(z_prev), b.slacks(z_prev)
    n = max(1, int(shift))
    idx = np.minimum(np.arange(N) + n, N - 1)
    U, S = U[:, idx], S[idx]
    Xn = X[:, np.minimum(np.arange(N + 1) + n, N)].copy()
    Xn[:, 0] = b.x0
    for k in range(max(0, N - n), N):
        Xn[:, k + 1] = ext.step(Xn[:, k], U[:, k], b.p, b.p.dt)
    return b.pack(Xn, U, S)


SLALOM_SCENARIO = ROOT / 'src/px4_nmpc_config/config/experiments/infeasible_slalom/scenario_10.json'


def write_scenario(course, path, start_offset=None):
    """Runtime scenario JSON of the course: same hall/start/goal schema as the slalom,
    spheres in obstacle_centers_ned/obstacle_radii, every obstacle in typed_obstacles.
    ``start_offset`` (dN, dE) [m] moves the take-off point (and with it the PX4
    local origin) for randomized repeat flights; obstacles stay in the hall frame."""
    data = json.loads(SLALOM_SCENARIO.read_text())
    if start_offset is not None:
        dn, de = (float(v) for v in start_offset)
        for key in ('start_ned', 'controller_origin_ned', 'reference_start_ned'):
            if key in data:
                data[key] = [data[key][0] + dn, data[key][1] + de, data[key][2]]
    obstacles = [list(map(float, o)) for o in course['course']['obstacles']]
    spheres = [o for o in obstacles if int(o[0]) == 0]
    data['obstacle_centers_ned'] = [o[1:4] for o in spheres]
    data['obstacle_radii'] = [o[4] for o in spheres]
    data['typed_obstacles'] = obstacles
    data['challenging_course'] = True
    data['guide'] = {k: float(v) for k, v in course['course']['guide'].items()}
    goal_s = float(course['course']['goal_s'])
    g = course['course']['guide']
    alt = g['alt0'] + g['alt_amp'] * math.sin(g['alt_freq'] * goal_s + g['alt_phase'])
    data['goal_ned'] = [goal_s, 0.0, -alt]
    for key in [k for k in data if k.startswith('controller_') and k not in ('controller_frame', 'controller_origin_ned')]:
        data.pop(key)
    data.get('experiment_config', {})['obstacle_count'] = len(spheres)
    Path(path).write_text(json.dumps(data, indent=2) + '\n')
    return data


def apply_dotted(settings, values, prefix='cg2o.'):
    """Merge tuned values {'cg2o.section.key': v} (selected_parameters.json) into nested settings."""
    for key, v in values.items():
        if key.startswith(prefix):
            section, name = key[len(prefix):].split('.', 1)
            settings.setdefault(section, {})[name] = v
    return settings


def cg2o_settings(overrides=CG2O_COURSE):
    """Repository cg2o.yaml with the course's documented overrides merged in."""
    with open(CG2O_YAML) as f:
        s = yaml.safe_load(f)
    if overrides and Path(overrides).exists():
        with open(overrides) as f:
            for section, values in (yaml.safe_load(f) or {}).items():
                if isinstance(values, dict):
                    s.setdefault(section, {}).update(values)
                else:
                    s[section] = values
    return s


# ------------------------------------------------------------------ closed loop
def hover_start(p, course):
    s0 = float(course['course']['start_s'])
    ext = extension()
    pos = ext.path(s0, p)
    return np.r_[pos, np.zeros(9)]   # [p, v, rpy, body rates]


def run_closed_loop(args):
    ext = extension()
    course = load_course(args.course)
    overrides = dict(vs_max=args.vs_max)
    if args.N:
        overrides['N'] = args.N
    p = make_params(course, overrides)
    plant = make_params(course, dict(overrides, drag=[args.plant_drag] * 3))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cg = ext.CGSession()
    cg.configure(cg2o_settings())
    ip = ext.IPOPTSession()
    ip_options = dict(hessian_approximation=args.ipopt_hessian)
    if args.ipopt_options:
        ip_options.update(json.loads(args.ipopt_options))
    ip.set_options(ip_options)
    rows = []
    x = hover_start(p, course)
    uprev = np.array([p.hover(), 0., 0., 0., 0.])
    z_ip = None
    trajectory = []
    warm_ok = False
    t = 0.
    log = open(out / 'problems.jsonl', 'w')
    for step in range(int(args.t_max / p.dt)):
        x0 = measured_state(p, x[:3], x[3:6], x[6:9], x[9:12])
        b = make_problem(p, x0, uprev)
        cold = b.cold_start()
        h = problem_sha256(b)
        initial = b.evaluate(cold)
        warm = args.mode == 'warm' and warm_ok
        r_cg = cg.solve(b, cold, args.budget, args.tol, warm=warm, shift=1)
        z_ip_start = shift_primal(b, z_ip) if (args.mode == 'warm' and z_ip is not None) else cold
        r_ip = ip.solve(b, z_ip_start, args.budget, args.tol, warm=args.mode == 'warm' and z_ip is not None,
                        shift=1, dual=True) if args.shadow else None
        usable = bool(np.isfinite(r_cg['z']).all() and r_cg['final_max_eq_residual'] <= 10 * args.tol
                      and r_cg['final_max_ineq_violation'] <= 10 * args.tol)
        row = dict(step=step, t=t, problem_sha256=h, cold_sha256=array_sha256(cold),
                   s0=float(x0[-1]), speed=float(np.linalg.norm(x[3:6])),
                   tilt_deg=math.degrees(math.acos(max(-1., min(1., math.cos(x[6]) * math.cos(x[7]))))),
                   cold_obstacle_rows_inside_reserve=initial['obstacle_rows_inside_reserve'],
                   cold_max_reserve_penetration_m=initial['max_reserve_penetration_m'],
                   cold_max_eq_residual=initial['final_max_eq_residual'])
        for name, r in (('cg2o', r_cg), ('ipopt', r_ip)):
            if r is None:
                continue
            for key in ('iterations', 'solve_time_ms', 'callback_ms', 'converged', 'status', 'final_cost',
                        'final_max_eq_residual', 'final_max_ineq_violation', 'min_body_clearance_m',
                        'max_slack', 'warm', 'warm_duals', 'warm_start_used', 'restoration_iterations',
                        'regularized_iterations'):
                row[f'{name}_{key}'] = r[key] if not isinstance(r[key], (np.floating,)) else float(r[key])
        row['cg2o_usable'] = usable
        if r_ip is not None:
            row['ipopt_usable'] = bool(r_ip['final_max_eq_residual'] <= 10 * args.tol
                                       and r_ip['final_max_ineq_violation'] <= 10 * args.tol)
            z_ip = r_ip['z'] if row['ipopt_usable'] else None
        rows.append(row)
        log.write(json.dumps(dict(problem_sha256=h, x0=x0.tolist(), uprev=uprev.tolist())) + '\n')
        if usable:
            U = b.inputs(r_cg['z'])
            u = U[:, 0]
            warm_ok = True
        else:
            # Hold the previous command for one period and restart cold.
            u = uprev
            warm_ok = False
            cg.reset()
        # Plant: the same rigid-body model (optionally with drag), 10 substeps.
        xs = np.r_[x, x0[-1]]
        for _ in range(10):
            xs = ext.step(xs, u, plant, p.dt / 10)
        x = xs[:12]
        uprev = u
        t += p.dt
        trajectory.append(dict(t=t, x=x.tolist(), u=u.tolist()))
        if x0[-1] >= p.s_goal - 0.5 and np.linalg.norm(x[3:6]) < 0.5:
            break
        if not np.isfinite(x).all() or x[2] > 0.5:
            break
        if args.verbose:
            print(f"{step:4d} s={x0[-1]:6.2f} v={row['speed']:.2f} tilt={row['tilt_deg']:5.1f} "
                  f"cg2o {row['cg2o_iterations']:3d}it {row['cg2o_solve_time_ms']:7.1f}ms "
                  f"clr={row['cg2o_min_body_clearance_m']:.3f}"
                  + (f" | ipopt {row['ipopt_iterations']:3d}it {row['ipopt_solve_time_ms']:7.1f}ms "
                     f"restor={row['ipopt_restoration_iterations']}" if r_ip else ''), flush=True)
    log.close()
    with open(out / 'paired.json', 'w') as f:
        json.dump(rows, f)
    with open(out / 'trajectory.json', 'w') as f:
        json.dump(trajectory, f)
    summary = summarize(rows, p, trajectory)
    summary.update(mode=args.mode, vs_max=args.vs_max, plant_drag=args.plant_drag, budget=args.budget,
                   tol=args.tol, ipopt_options=ip_options, N=p.N)
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def true_clearance(p, trajectory):
    """Minimum true clearance of the five body spheres along the plant trajectory."""
    ext = extension()
    best = 1e9
    for row in trajectory:
        x = np.asarray(row['x'])
        R = ext.rotation(x[6], x[7], x[8])
        for i in range(5):
            w = x[:3] + R @ np.asarray(p.body)[i]
            for o in np.asarray(p.obstacles):
                d = w - o[1:4]
                if o[0] == 1:
                    d[2] = 0.
                elif o[0] == 2:
                    d[1] = 0.
                best = min(best, float(np.linalg.norm(d) - o[4] - p.body_radius[i]))
    return best


def summarize(rows, p, trajectory):
    def stats(key, mask=None):
        v = np.array([r[key] for r in rows if key in r and (mask is None or mask(r))], float)
        if v.size == 0:
            return None
        return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                    max=float(v.max()))
    out = dict(problems=len(rows),
               cold_infeasible_problems=int(sum(r['cold_obstacle_rows_inside_reserve'] > 0 for r in rows)),
               cold_max_penetration_m=float(max(r['cold_max_reserve_penetration_m'] for r in rows)),
               hashes_logged=len({r['problem_sha256'] for r in rows}),
               goal_reached=bool(rows and rows[-1]["s0"] >= p.s_goal - 0.5),
               flight_time_s=float(trajectory[-1]['t']) if trajectory else 0.,
               peak_tilt_deg=float(max(r['tilt_deg'] for r in rows)),
               peak_speed=float(max(r['speed'] for r in rows)),
               true_min_clearance_m=true_clearance(p, trajectory))
    for name in ('cg2o', 'ipopt'):
        if f'{name}_iterations' not in rows[0]:
            continue
        out[name] = dict(usable=int(sum(r[f'{name}_usable'] for r in rows)),
                         converged=int(sum(bool(r[f'{name}_converged']) for r in rows)),
                         iterations=stats(f'{name}_iterations'),
                         solve_time_ms=stats(f'{name}_solve_time_ms'),
                         callback_ms=stats(f'{name}_callback_ms'),
                         restoration_iterations=int(sum(r[f'{name}_restoration_iterations'] for r in rows)),
                         regularized_iterations=int(sum(r[f'{name}_regularized_iterations'] for r in rows)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out')
    ap.add_argument('--course', default=str(COURSE))
    ap.add_argument('--mode', choices=('warm', 'cold'), default='warm')
    ap.add_argument('--vs-max', type=float, default=2.5)
    ap.add_argument('--N', type=int, default=0)
    ap.add_argument('--budget', type=int, default=250)
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('--t-max', type=float, default=40.)
    ap.add_argument('--plant-drag', type=float, default=0.0, help='model mismatch: body drag [1/s] in the plant')
    ap.add_argument('--no-shadow', dest='shadow', action='store_false')
    ap.add_argument('--ipopt-hessian', default='exact', choices=('exact', 'limited-memory'))
    ap.add_argument('--ipopt-options', default='', help='JSON dict of extra IPOPT options')
    ap.add_argument('--verbose', action='store_true')
    run_closed_loop(ap.parse_args(argv))


if __name__ == '__main__':
    main()

"""Paired replay of a PX4/Gazebo flight of the challenging MPC.

Reads ``challenging_trace.jsonl`` of a flight, rebuilds every problem from the
logged data, verifies its SHA-256 against the flight, and solves it with cg2o
and with IPOPT, one solver after the other, on the calling process' CPUs (run
under ``taskset -c 0-3``). Each solver keeps its own warm-start history; warm
or cold follows the flight's mode. The flight's own cg2o times were measured
next to ROS and PX4 and are kept only as a record.

    taskset -c 0-3 python -m px4_nmpc_compare.challenging_replay RUN OUT [--ipopt-hessian exact]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from px4_nmpc_compare import challenging as C


def load_trace(run):
    rows = [json.loads(line) for line in (Path(run) / 'challenging_trace.jsonl').read_text().splitlines() if line]
    meta = [r for r in rows if r.get('kind') == 'metadata']
    problems = [r for r in rows if r.get('kind') == 'problem']
    if len(meta) != 1:
        raise SystemExit(f'expected one metadata row, found {len(meta)}')
    return meta[0], problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--ipopt-hessian', default='exact', choices=('exact', 'gauss-newton', 'limited-memory'))
    ap.add_argument('--ipopt-options', default='', help='JSON dict of extra IPOPT options')
    ap.add_argument('--cg2o-params', default='', help='tuned cg2o values (selected_parameters.json) '
                    'merged over the flight\'s cg2o settings')
    ap.add_argument('--ipopt-params', default='', help='tuned IPOPT values (selected_parameters.json); '
                    'sets the options, the warm-start kind and the HSL library')
    ap.add_argument('--cg2o-params-cold', default='', help='cold-start tuned cg2o values merged over the warm '
                    'settings for cold solves (default: the flight\'s cg2o_cold_settings, if any)')
    ap.add_argument('--ipopt-params-cold', default='', help='cold-start tuned IPOPT values (selected_parameters.json) '
                    'for cold solves; warm solves keep --ipopt-params')
    ap.add_argument('--start-index', type=int, default=-1,
                    help='first problem; default: the first navigation (NMPC phase) problem if the trace logs the '
                         'phase, else 0')
    ap.add_argument('--count', type=int, default=0, help='number of problems (0 = all)')
    ap.add_argument('--budget', type=int, default=0, help='iteration budget (0 = the flight budget)')
    args = ap.parse_args(argv)
    meta, problems = load_trace(args.run)
    ext = C.extension()
    course = C.load_course(meta['course'])
    p = C.make_params(course, meta['overrides'])
    budget, tol = int(args.budget or meta['budget']), float(meta['tol'])
    cg_settings = meta['cg2o_settings']
    if args.cg2o_params:
        cg_settings = C.apply_dotted(cg_settings, json.loads(Path(args.cg2o_params).read_text()))
    cg_cold_settings = meta.get('cg2o_cold_settings')
    if args.cg2o_params_cold:
        cg_cold_settings = C.apply_dotted(copy.deepcopy(cg_settings),
                                          json.loads(Path(args.cg2o_params_cold).read_text()))
    cg = ext.CGSession()
    cg.configure(cg_settings)
    cg_active = 'warm'
    ip = ext.IPOPTSession()
    options, dual = dict(hessian_approximation=args.ipopt_hessian), True
    if args.ipopt_params:
        from px4_nmpc_compare.challenging_tune import ipopt_options
        options, dual = ipopt_options(json.loads(Path(args.ipopt_params).read_text()))
    if args.ipopt_options:
        options.update(json.loads(args.ipopt_options))
    ip.set_options(options)
    options_cold = None
    if args.ipopt_params_cold:
        from px4_nmpc_compare.challenging_tune import ipopt_options
        options_cold, _ = ipopt_options(json.loads(Path(args.ipopt_params_cold).read_text()))
        if args.ipopt_options:
            options_cold.update(json.loads(args.ipopt_options))
    ip_active = 'warm'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    z_ip = None
    cg_ok = False
    mismatches = 0
    start = args.start_index
    if start < 0:
        nav = [i for i, d in enumerate(problems) if d.get('phase') == 'NMPC']
        start = nav[0] if nav else 0
    stop = start + args.count if args.count else None
    for d in problems[start:stop]:
        b = C.make_problem(p, np.asarray(d['x0_hall']), np.asarray(d['uprev']))
        h = C.problem_sha256(b)
        cold = b.cold_start()
        if h != d['problem_sha256'] or C.array_sha256(cold) != d['cold_sha256']:
            mismatches += 1
        warm_flight = bool(d['warm'])
        shift = max(0, d['shift'])
        cg_warm = warm_flight and cg_ok
        want = 'warm' if (cg_warm or cg_cold_settings is None) else 'cold'
        if want != cg_active:
            cg.configure(cg_settings if want == 'warm' else cg_cold_settings)
            cg_active = want
        r_cg = cg.solve(b, cold, budget, tol, warm=cg_warm, shift=shift)
        ip_warm = warm_flight and z_ip is not None
        want = 'warm' if (ip_warm or options_cold is None) else 'cold'
        if want != ip_active:
            ip.set_options(options if want == 'warm' else options_cold)
            ip_active = want
        z0 = C.shift_primal(b, z_ip, shift) if ip_warm else cold
        r_ip = ip.solve(b, z0, budget, tol, warm=ip_warm, shift=shift, dual=dual)
        row = dict(problem_index=d['problem_index'], problem_sha256=h, flight_hash_match=h == d['problem_sha256'],
                   warm=warm_flight, phase=d.get('phase'), cg2o_warm=cg_warm, ipopt_warm=ip_warm, flight_cg2o_ms=d['solve_ms'], flight_cg2o_iterations=d['iterations'])
        initial = b.evaluate(cold)
        row['cold_obstacle_rows_inside_reserve'] = initial['obstacle_rows_inside_reserve']
        row['cold_max_reserve_penetration_m'] = initial['max_reserve_penetration_m']
        for name, r in (('cg2o', r_cg), ('ipopt', r_ip)):
            usable = bool(np.isfinite(r['z']).all() and r['final_max_eq_residual'] <= 10 * tol
                          and r['final_max_ineq_violation'] <= 10 * tol)
            row[f'{name}_usable'] = usable
            for key in ('iterations', 'solve_time_ms', 'callback_ms', 'hessian_ms', 'converged', 'status', 'final_cost',
                        'final_max_eq_residual', 'final_max_ineq_violation', 'min_body_clearance_m',
                        'restoration_iterations', 'regularized_iterations'):
                v = r[key]
                row[f'{name}_{key}'] = float(v) if isinstance(v, (float, np.floating)) else v
        cg_ok = row['cg2o_usable']
        if not cg_ok:
            cg.reset()
        z_ip = r_ip['z'] if row['ipopt_usable'] else None
        rows.append(row)
    (out / 'paired.json').write_text(json.dumps(rows))
    summary = dict(run=str(args.run), start_index=start, problems=len(rows), hash_mismatches=mismatches, budget=budget,
                   cg2o_settings=cg_settings, cg2o_cold_settings=cg_cold_settings, ipopt_options=options,
                   ipopt_cold_options=options_cold, ipopt_primal_dual_warm_start=dual)
    for name in ('cg2o', 'ipopt'):
        t = np.array([r[f'{name}_solve_time_ms'] for r in rows])
        # Conservative timing: the exact Hessian is used but its evaluation time
        # is excluded (Hessian assumed available at no cost). Other callbacks
        # (f, gradient, g, Jacobian) use the same code as cg2o's linearization
        # and stay included.
        t_free = t - np.array([r[f'{name}_hessian_ms'] for r in rows])
        it = np.array([r[f'{name}_iterations'] for r in rows])
        summary[name] = dict(usable=int(sum(r[f'{name}_usable'] for r in rows)),
                             mean_ms=float(t.mean()), median_ms=float(np.median(t)),
                             p95_ms=float(np.percentile(t, 95)), max_ms=float(t.max()),
                             hessian_free_median_ms=float(np.median(t_free)),
                             hessian_free_p95_ms=float(np.percentile(t_free, 95)),
                             hessian_free_mean_ms=float(t_free.mean()),
                             mean_iterations=float(it.mean()), iteration_limit_hits=int(sum(it >= budget)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

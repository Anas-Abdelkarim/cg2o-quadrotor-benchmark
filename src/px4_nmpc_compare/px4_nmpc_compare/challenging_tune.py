"""Equal-budget Bayesian tuning of cg2o and IPOPT on stored PX4 problem sequences.

Used by ``python -m tuning run`` (tools/solver_tuning) through
``evaluate_cg2o`` / ``evaluate_ipopt``. One evaluation replays a whole flight's
navigation problems in order (warm starts need the sequence), with the given
solver settings, the flight's warm/cold mode and shifts, and the 80-iteration
flight budget. Both solvers are scored with the same metrics and gates:

* ``solve_ms_mean`` / ``solve_ms_p95``: solver call time. For IPOPT the exact
  Hessian evaluation is excluded (Hessian assumed free); every other callback
  shares cg2o's derivative code and is included.
* ``unusable_fraction``, ``hit_max_fraction``, ``eq_residual_max``,
  ``ineq_violation_max`` (over usable solutions).
* ``cost_excess_rel_median`` / ``_p95``: relative cost above the per-problem
  reference (best usable cost of either solver in the baseline replay at a
  250-iteration budget).

Run under ``taskset -c 0-3`` with ``jobs: 1`` (timing).
"""
from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from px4_nmpc_compare import challenging as C

ROOT = Path(__file__).resolve().parents[3]
HSL = ROOT / 'install/externals/hsl/lib/x86_64-linux-gnu/libcoinhsl.so'
BUDGET = 80
# Dataset whose recorded tuning flights are replayed: the shipped data by default, or a new run folder
# (results/<run>, written by benchmark/flight/tuning_flights.sh). Set with the environment variable BENCH_DATA.
DATA = os.environ.get('BENCH_DATA', 'data')
TRACES = {
    # case: (flight run, first navigation problem if the trace does not log the phase, reference replay)
    'warm_headless': (f'{DATA}/sitl/challenging_lag_warm_v25_headless', 51,
                      f'{DATA}/challenging_mpc/replay_lag_warm_v25_ma57_b250'),
    'warm_video': (f'{DATA}/sitl/challenging_lag_warm_v25_video', 54,
                   f'{DATA}/challenging_mpc/replay_lag_warm_video_ma57_b250'),
    'cold_headless': (f'{DATA}/sitl/challenging_lag_cold_v25_headless', 63,
                      f'{DATA}/challenging_mpc/replay_lag_cold_v25_ma57_b250'),
    # validation of cold-start settings: the video flight's problems, every one solved cold
    'video_as_cold': (f'{DATA}/sitl/challenging_lag_warm_v25_video', 54,
                      f'{DATA}/challenging_mpc/replay_lag_warm_video_ma57_b250'),
}
FORCE_COLD = {'video_as_cold'}


@lru_cache(maxsize=None)
def sequence(case):
    run, start, replay = TRACES[case]
    rows = [json.loads(l) for l in (ROOT / run / 'challenging_trace.jsonl').read_text().splitlines() if l]
    meta = rows[0]
    problems = [r for r in rows[1:] if r.get('kind') == 'problem']
    if any('phase' in r for r in problems):   # newer traces log the phase: start at the first navigation problem
        start = next(i for i, r in enumerate(problems) if r.get('phase') == 'NMPC')
    problems = problems[start:]
    p = C.make_params(C.load_course(meta['course']), meta['overrides'])
    built = []
    for d in problems:
        b = C.make_problem(p, np.asarray(d['x0_hall']), np.asarray(d['uprev']))
        if C.problem_sha256(b) != d['problem_sha256']:
            raise RuntimeError(f'{case}: problem {d["problem_index"]} does not match its flight hash')
        built.append((d, b, b.cold_start()))
    paired = json.loads((ROOT / replay / 'paired.json').read_text())
    reference = {}
    for r in paired:
        costs = [r[f'{s}_final_cost'] for s in ('cg2o', 'ipopt') if r[f'{s}_usable']]
        if costs:
            reference[r['problem_index']] = min(costs)
    return built, reference, float(meta['tol'])


def _score(case, fidelity, solve_one):
    built, reference, tol = sequence(case)
    if fidelity:
        built = built[:int(fidelity)]
    times, iters, eq, ineq, excess = [], [], [], [], []
    unusable = hits = 0
    for d, b, cold in built:
        r, ms = solve_one(d, b, cold)
        usable = bool(np.isfinite(r['z']).all() and r['final_max_eq_residual'] <= 10 * tol
                      and r['final_max_ineq_violation'] <= 10 * tol)
        times.append(ms)
        iters.append(r['iterations'])
        hits += r['iterations'] >= BUDGET
        if not usable:
            unusable += 1
            continue
        eq.append(r['final_max_eq_residual'])
        ineq.append(r['final_max_ineq_violation'])
        ref = reference.get(d['problem_index'])
        if ref is not None:
            excess.append((r['final_cost'] - ref) / max(1., abs(ref)))
    n = len(built)
    times = np.asarray(times)
    return dict(solve_ms_mean=float(times.mean()), solve_ms_p95=float(np.percentile(times, 95)),
                iterations_mean=float(np.mean(iters)), unusable_fraction=unusable / n,
                hit_max_fraction=hits / n, eq_residual_max=float(max(eq, default=0.)),
                ineq_violation_max=float(max(ineq, default=0.)),
                cost_excess_rel_median=float(np.median(excess)) if excess else 1.,
                cost_excess_rel_p95=float(np.percentile(excess, 95)) if excess else 1.)


def evaluate_cg2o(values, context):
    ext = C.extension()
    settings = C.apply_dotted(C.cg2o_settings(), values)
    session = ext.CGSession()
    session.configure(copy.deepcopy(settings))
    state = dict(ok=False)
    tol = float(sequence(context.case)[2])

    def solve_one(d, b, cold):
        warm = bool(d['warm']) and state['ok'] and context.case not in FORCE_COLD
        r = session.solve(b, cold, BUDGET, tol, warm=warm, shift=max(0, d['shift']))
        # same usable test as the scoring and the IPOPT evaluator
        state['ok'] = bool(np.isfinite(r['z']).all() and r['final_max_eq_residual'] <= 10 * tol
                           and r['final_max_ineq_violation'] <= 10 * tol)
        if not state['ok']:
            session.reset()
        return r, float(r['solve_time_ms'])
    return _score(context.case, context.fidelity, solve_one)


def ipopt_options(values):
    """Tuned values {'ipopt.<option>': v} -> (IPOPT options, primal-dual warm start?)."""
    options = {k[len('ipopt.'):]: v for k, v in values.items() if k.startswith('ipopt.')}
    dual = options.pop('warm_start', 'primal_dual') == 'primal_dual'
    options['hessian_approximation'] = options.pop('hessian', 'exact')
    if options.get('linear_solver', 'mumps').startswith('ma'):
        options['hsllib'] = str(HSL)
    return options, dual


def evaluate_ipopt(values, context):
    ext = C.extension()
    options, dual = ipopt_options(values)
    session = ext.IPOPTSession()
    session.set_options(options)
    state = dict(z=None)
    tol = float(sequence(context.case)[2])

    def solve_one(d, b, cold):
        warm = bool(d['warm']) and state['z'] is not None and context.case not in FORCE_COLD
        start = C.shift_primal(b, state['z'], max(0, d['shift'])) if warm else cold
        r = session.solve(b, start, BUDGET, tol, warm=warm, shift=max(0, d['shift']), dual=dual)
        ok = bool(np.isfinite(r['z']).all() and r['final_max_eq_residual'] <= 10 * tol
                  and r['final_max_ineq_violation'] <= 10 * tol)
        state['z'] = r['z'] if ok else None
        return r, float(r['solve_time_ms'] - r['hessian_ms'])
    return _score(context.case, context.fidelity, solve_one)

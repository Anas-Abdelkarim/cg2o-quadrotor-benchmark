#!/usr/bin/env python3
"""Tables IX-XII and Figs. 10-11 of the paper (Sec. VII) from one experiment folder.

    python3 experiments/paper_results.py [FOLDER] [--dataset DIR] [--tuning DIR] [--compare]

FOLDER   an experiment folder written by run_experiment.py (results/experiments/<stamp>_<preset>).
         Default: the newest complete folder that is not a --quick or --no-hsl run.
--dataset where the flights live (challenging_campaign/). Default: the dataset recorded in FOLDER/run.json.
--tuning  where the four tuning studies live (tuning/). Default: the dataset's tuning/ if it holds all four
          studies, else data/tuning (the paper's studies).
--compare compare with the paper's numbers (experiments/paper_reference.yaml): values that do not depend on
          timing must be the same; times are listed. Writes comparison_with_paper.txt; exit code 1 on a difference.

Writes into FOLDER:
    paper_table_ix.{txt,json}    selected solver settings            (Table IX)
    paper_table_x.{txt,json}     closed-loop flights                 (Table X)
    paper_table_xi.{txt,json}    paired comparison                   (Table XI)
    paper_table_xii.{txt,json}   sensitivity of cg2o                 (Table XII)
    paper_fig10_course.pdf       the 20 flights, top and side view   (Fig. 10)
    paper_fig11_solve_times.pdf  solve-time distributions            (Fig. 11)
    analysis/                    full outputs of benchmark/analysis (campaign_summary.json, sensitivity.json)
The analysis code in benchmark/analysis is used unchanged; this script only lays out its numbers as in the paper.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'results/experiments'
ANALYSIS = ROOT / 'benchmark/analysis'
SHIPPED_TUNING = ROOT / 'data/tuning'
IPOPT_COLD_DEFAULT = ROOT / 'src/px4_nmpc_config/config/experiments/challenging_course/ipopt_cold_default.json'
PERIOD_MS = 100.0
STUDIES = ('cg2o_warm', 'cg2o_cold', 'ipopt_warm', 'ipopt_cold')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def newest_complete():
    """Newest experiment folder whose run.json says complete, not quick and with IPOPT on HSL (as in the paper)."""
    best = None
    for p in sorted(EXPERIMENTS.glob('*/run.json')):
        info = json.loads(p.read_text())
        if info.get('status') == 'complete' and not info.get('quick') and info.get('ipopt_hsl', True):
            best = p.parent
    if best is None:
        sys.exit(f'no complete experiment folder in {EXPERIMENTS}; run experiments/run_experiment.py first '
                 f'or give a folder')
    return best


def study(tuning, name):
    """Newest study folder of a kind (e.g. <tuning>/20261110_133924_challenging_cg2o_cold)."""
    found = sorted(glob.glob(str(tuning / f'*_challenging_{name}')))
    return Path(found[-1]) if found else None


def tuning_dir(dataset, override):
    if override:
        return Path(override)
    own = dataset / 'tuning'
    if all(study(own, s) for s in STUDIES):
        return own
    return SHIPPED_TUNING


def rel(path):
    """A path relative to the repository root when it is inside it (outputs carry no machine-specific paths)."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def write(folder, name, text, data):
    (folder / f'{name}.txt').write_text(text)
    (folder / f'{name}.json').write_text(json.dumps(data, indent=1))
    print(text)


# --------------------------------------------------------------------------------------------- Table IX
SYMBOLS = {   # selected parameter -> notation of the paper
    'cg2o.warm_start.strategy': 'warm start (1 = final iterate)',
    'cg2o.ispd.kappa_final': 'kappa_final', 'cg2o.ispd.tau': 'tau',
    'cg2o.ispd.ineq_backtracking_step_min': 'zeta_ineq_min', 'cg2o.ispd.aux_backtracking_step_min': 'zeta_aux_min',
    'cg2o.ispd.init_kappa_strategy': 'kappa_0 rule (0 = constant)', 'cg2o.ispd.kappa_initial': 'kappa_0',
    'cg2o.ispd.update_kappa_strategy': 'barrier update (2 = gap only)', 'cg2o.ispd.kappa_update_factor': 'nu',
    'cg2o.ispd.step_size_strategy': 'step rule (1 = R2)',
    'cg2o.ispd.keep_aux_positive_strategy': 'positivity (0 = scaling)', 'cg2o.ispd.aux_scaling_factor': 'beta',
    'cg2o.ispd.slack_variable_initial': 's_pos', 'cg2o.ispd.lagrange_multiplier_initial': 'lambda_pos',
    'ipopt.warm_start': 'warm start', 'ipopt.linear_solver': 'linear solver', 'ipopt.hessian': 'Hessian',
    'ipopt.mu_strategy': 'mu strategy', 'ipopt.mu_oracle': 'mu oracle', 'ipopt.mu_init': 'mu_0',
    'ipopt.bound_push': 'bound_push', 'ipopt.bound_frac': 'bound_frac', 'ipopt.barrier_tol_factor': 'barrier_tol_factor',
    'ipopt.nlp_scaling_method': 'nlp_scaling_method',
}


def fmt_value(v):
    if isinstance(v, float):
        return f'{v:.3g}'
    return str(v)


def table_ix(folder, tuning):
    data, lines = {}, ['TABLE IX  Selected solver settings of the quadrotor benchmark', f'tuning studies: {rel(tuning)}', '']
    for name in STUDIES:
        s = study(tuning, name)
        sel = s / 'selected_parameters.json' if s else None
        if sel and sel.exists():
            params, source = json.loads(sel.read_text()), rel(sel)
        elif name == 'ipopt_cold':
            # no IPOPT cold setting passed the quality limits: the default options with MA57 are used
            params, source = json.loads(IPOPT_COLD_DEFAULT.read_text()), f'{rel(IPOPT_COLD_DEFAULT)} (no setting selected)'
        else:
            params, source = {}, 'missing'
        data[name] = dict(source=source, parameters=params)
        lines.append(f'{name.replace("_", ", ")}   ({source})')
        for k, v in params.items():
            lines.append(f'    {SYMBOLS.get(k, k):34s} {fmt_value(v)}')
        lines.append('')
    write(folder, 'paper_table_ix', '\n'.join(lines), data)


# --------------------------------------------------------------------------------------------- Table X
def table_x(folder, campaign, summary):
    data, lines = {}, ['TABLE X  Closed-loop flights with cg2o in PX4 SITL', '',
                       f'{"Start":6} {"Goal":>6} {"Time to goal (s)":>17} {"Max. tilt (deg)":>16} '
                       f'{"Feasible solves":>16} {"Solve time (ms) med / P95":>26}']
    for mode in ('warm', 'cold'):
        if mode not in summary:
            continue
        fl = summary[mode]['flights']
        t = [f['time_to_goal_s'] for f in fl if f['time_to_goal_s'] is not None]
        tilt = [f['max_tilt_deg'] for f in fl if f['max_tilt_deg'] is not None]
        ms = [r['solve_ms'] for d in sorted(campaign.glob(f'{mode}_[0-9][0-9]'))
              for r in map(json.loads, (d / 'challenging_trace.jsonl').read_text().splitlines())
              if r.get('kind') == 'problem' and r.get('phase') == 'NMPC']
        n = sum(f['nav_solves'] for f in fl)
        feas = n - sum(f['nav_unusable'] for f in fl)
        d = dict(flights=len(fl), goal=sum(f['outcome'] == 'goal' for f in fl), time_min_s=min(t), time_max_s=max(t),
                 tilt_median_deg=float(np.median(tilt)), tilt_max_deg=max(tilt), feasible=feas, problems=n,
                 solve_ms_median=float(np.median(ms)), solve_ms_p95=float(np.percentile(ms, 95)),
                 within_period_pct=float(100 * np.mean(np.asarray(ms) <= PERIOD_MS)),
                 min_clearance_m=min(f['min_clearance_m'] for f in fl), hold_samples=sum(f['hold_samples'] for f in fl))
        data[mode] = d
        lines.append(f'{mode.capitalize():6} {d["goal"]:>3}/{d["flights"]:<2} {d["time_min_s"]:>8.1f}-{d["time_max_s"]:<8.1f}'
                     f'{d["tilt_median_deg"]:>9.0f} / {d["tilt_max_deg"]:<5.0f} {d["feasible"]:>8}/{d["problems"]:<7}'
                     f'{d["solve_ms_median"]:>13.1f} / {d["solve_ms_p95"]:.1f}')
    lines += ['', 'Max. tilt: median / largest over the flights. Solve times measured in flight (navigation phase).']
    for mode, d in data.items():
        lines.append(f'{mode}: {d["within_period_pct"]:.1f}% of the in-flight solves within 100 ms; smallest airframe '
                     f'clearance {d["min_clearance_m"]:.3f} m; position-hold samples {d["hold_samples"]}')
    write(folder, 'paper_table_x', '\n'.join(lines), data)


# --------------------------------------------------------------------------------------------- Table XI
def table_xi(folder, summary, ipopt_hsl=True):
    data, lines = {}, ['TABLE XI  Paired comparison on the MPC problems of the 20 flights (iteration limit 80)', '',
                       f'{"Start":6} {"Solver":7} {"Iter. avg":>9} {"Avg":>7} {"Med.":>7} {"P95":>7} {"Max":>7} '
                       f'{"<= Ts (%)":>9} {"Feasible":>9}']
    total = 0
    infeasible = 0
    for mode in ('cold', 'warm'):
        p = summary.get(mode, {}).get('paper')
        if not p:
            continue
        data[mode] = p
        total += p['problems']
        infeasible += p['cold_start_infeasible']
        for i, s in enumerate(('ipopt', 'cg2o')):
            r = p[s]
            lines.append(f'{mode.capitalize() if i == 0 else "":6} {"IPOPT" if s == "ipopt" else "cg2o":7} '
                         f'{r["mean_iterations"]:9.1f} {r["mean_ms"]:7.1f} {r["median_ms"]:7.1f} {r["p95_ms"]:7.1f} '
                         f'{r["max_ms"]:7.1f} {r["within_period_pct"]:9.1f} {r["usable_pct"]:8.1f}%')
    lines += ['', ' '.join(f'{data[m]["problems"]} {m}' for m in data) + ' problems. <= Ts: solves within the control '
              'period of 100 ms. IPOPT time without the evaluation of its exact Hessian.',
              'IPOPT linear solver: HSL MA57 (cold) and MA27 (warm), as in the paper.' if ipopt_hsl else
              'IPOPT linear solver: MUMPS (run with --no-hsl; the paper used HSL MA57/MA27).', '']
    for mode, p in data.items():
        lines.append(f'{mode}: cg2o {p["ipopt"]["mean_ms"] / p["cg2o"]["mean_ms"]:.1f}x faster than IPOPT (mean); '
                     f'median cost of cg2o relative to IPOPT {p["cg2o_cost_excess_median_pct"]:+.1f}%; '
                     f'hash mismatches {p["hash_mismatches"]}')
    if total:
        rows = [p for p in data.values()]
        lines.append(f'cold starts infeasible w.r.t. the obstacles: {infeasible}/{total} ({100 * infeasible / total:.0f}%); '
                     f'violated obstacle rows (mean over infeasible) '
                     f'{np.average([p["cold_start_rows_inside_mean"] for p in rows], weights=[p["cold_start_infeasible"] for p in rows]):.0f}; '
                     f'largest penetration {max(p["cold_start_penetration_max_m"] for p in rows):.2f} m')
        data['cold_start_infeasible_total'] = dict(infeasible=infeasible, problems=total)
    write(folder, 'paper_table_xi', '\n'.join(lines), data)


# --------------------------------------------------------------------------------------------- Table XII
def table_xii(folder, cold_study, sens_json):
    sens = load_module('sensitivity')
    trials, t0, _ = sens.load(cold_study)
    lin = [t for t in trials if t['linear_full']]
    groups = []
    for code, name in ((0, 'lower-bounded (29)'), (1, 'bounded by kappa_0'), (2, 'gap only'), (3, 'fixed factor (nested)')):
        groups.append(sens.group(lin, 'Barrier update', name, lambda t, c=code: t['update'] == c))
    for code, name in ((0, 'scaling (35)'), (1, 'correction (36)')):
        groups.append(sens.group(lin, 'Positivity', name, lambda t, c=code: t['positivity'] == c))
    for key, label, edges in (('zeta_aux', 'zeta_aux_min', (0.5, 0.8)), ('kappa_final', 'kappa_final', (1000, 3000)),
                              ('beta', 'beta', (0.05, 0.3))):
        lo = -np.inf
        for hi in list(edges) + [np.inf]:
            name = f'< {hi:g}' if lo == -np.inf else (f'>= {lo:g}' if hi == np.inf else f'[{lo:g}, {hi:g})')
            groups.append(sens.group(lin, label, name, lambda t, a=lo, b=hi, k=key: a <= t[k] < b))
            lo = hi
    summary = json.loads(Path(sens_json).read_text())
    nonlin = [t for t in trials if not t['linear_full']]
    lines = [f'TABLE XII  Sensitivity of cg2o to its options: {len(lin)} trials of the cold-start search', '',
             f'{"Option":15} {"Setting":24} {"Trials":>6} {"Rejected":>9} {"Med. time":>9}']
    last = None
    for g in groups:
        rej = '-' if g['rejected'] is None else f'{100 * g["rejected"]:.0f}%'
        med = '-' if g['median_rel_time'] is None else f'{g["median_rel_time"]:.2f}'
        lines.append(f'{g["option"] if g["option"] != last else "":15} {g["setting"]:24} {g["trials"]:6d} {rej:>9} {med:>9}')
        last = g['option']
    report = (Path(cold_study) / 'report.txt').read_text().splitlines()
    score = next((float(l.split(':')[1]) for l in report if l.startswith('score:')), None)
    lines += ['', 'Med. time: median over the accepted trials of the mean solve time, relative to the untuned setting.',
              f'{len(trials)} search trials; {len(nonlin)} used a nonlinear or damped auxiliary update, '
              f'{sum(not t["accepted"] for t in nonlin)} of them rejected (not in the table).',
              f'median accepted trial: {1 / summary["accepted_median_rel"]:.3f}x faster than the untuned setting '
              f'({t0:.1f} ms)']
    if score is not None:
        lines.append(f'selected (validated) setting: {100 * (1 - score):.1f}% faster than the untuned setting')
    write(folder, 'paper_table_xii', '\n'.join(lines),
          dict(study=rel(cold_study), trials=len(lin), groups=groups, nonlinear_trials=len(nonlin),
               nonlinear_rejected=sum(not t['accepted'] for t in nonlin),
               accepted_median_speedup=1 / summary['accepted_median_rel'], selected_score=score))


# --------------------------------------------------------------------------------------------- compare
def compare(folder):
    """Compare the tables of a folder with experiments/paper_reference.yaml; returns the number of mismatches."""
    import yaml
    ref = yaml.safe_load((ROOT / 'experiments/paper_reference.yaml').read_text())
    lines, bad = [], 0

    build_dependent = set(ref['table_xi'].get('build_dependent', []))

    def check(label, got, want, exact):
        nonlocal bad
        nd = len(str(want).split('.')[1]) if isinstance(want, float) else 0
        same = round(float(got), nd) == round(float(want), nd)
        if exact and label in build_dependent:
            tag = 'same' if same else 'differs (IPOPT build)'
        else:
            if exact and not same:
                bad += 1
            tag = ('same' if same else 'DIFFERENT') if exact else ('same' if same else 'differs (timing)')
        lines.append(f'  {label:44s} paper {want!s:>8}   here {round(float(got), nd)!s:>8}   {tag}')

    x = json.loads((folder / 'paper_table_x.json').read_text())
    lines.append('Table X (flights)')
    for mode, want in ref['table_x'].items():
        if mode == 'timing':
            continue
        for k, v in want.items():
            check(f'{mode} {k}', x[mode][k], v, True)
    for mode, want in ref['table_x']['timing'].items():
        for k, v in want.items():
            check(f'{mode} {k}', x[mode][k], v, False)
    xi = json.loads((folder / 'paper_table_xi.json').read_text())
    lines.append('Table XI (paired comparison)')
    for mode in ('cold', 'warm'):
        if mode not in xi:
            continue
        check(f'{mode} problems', xi[mode]['problems'], ref['table_xi']['exact'][mode]['problems'], True)
        for s in ('ipopt', 'cg2o'):
            for k, v in ref['table_xi']['exact'][mode][s].items():
                check(f'{mode} {s} {k}', xi[mode][s][k], v, True)
            for k, v in ref['table_xi']['timing'][mode][s].items():
                check(f'{mode} {s} {k}', xi[mode][s][k], v, False)
    if 'cold_start_infeasible_total' in xi:
        check('cold starts infeasible', xi['cold_start_infeasible_total']['infeasible'],
              ref['table_xi']['exact']['cold_start_infeasible'], True)
    xii = json.loads((folder / 'paper_table_xii.json').read_text())
    lines.append('Table XII (sensitivity)')
    for g in xii['groups']:
        want = ref['table_xii'].get(g['option'], {}).get(g['setting'])
        if want is None:
            continue
        check(f"{g['option']} {g['setting']} trials", g['trials'], want[0], True)
        check(f"{g['option']} {g['setting']} rejected %", 100 * g['rejected'], want[1], True)
        check(f"{g['option']} {g['setting']} med. time", g['median_rel_time'], want[2], True)
    lines.append(f'{bad} difference(s) in values that do not depend on timing or on the IPOPT build')
    text = '\n'.join(lines)
    (folder / 'comparison_with_paper.txt').write_text(text + '\n')
    print(text)
    return bad


# --------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('folder', nargs='?', help='experiment folder (default: newest complete, not --quick)')
    ap.add_argument('--dataset', help='folder holding challenging_campaign/ (default: from FOLDER/run.json)')
    ap.add_argument('--tuning', help='folder holding the four tuning studies')
    ap.add_argument('--compare', action='store_true',
                    help='also compare with experiments/paper_reference.yaml (writes comparison_with_paper.txt)')
    a = ap.parse_args(argv)
    folder = Path(a.folder).resolve() if a.folder else newest_complete()
    info = json.loads((folder / 'run.json').read_text()) if (folder / 'run.json').exists() else {}
    dataset = Path(a.dataset or info.get('dataset') or folder)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    campaign = dataset / 'challenging_campaign'
    if not campaign.is_dir():
        sys.exit(f'no challenging_campaign/ in {dataset}')
    tuning = tuning_dir(dataset, a.tuning or (ROOT / info['tuning'] if info.get('tuning') else None))
    print(f'experiment folder: {folder}\nflights: {campaign}\ntuning: {tuning}\n')
    out = folder / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    subprocess.run([py, str(ANALYSIS / 'campaign_summary.py'), str(campaign), str(out / 'campaign_summary.json')],
                   check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((out / 'campaign_summary.json').read_text())
    cold_study = study(tuning, 'cg2o_cold')
    subprocess.run([py, str(ANALYSIS / 'sensitivity.py'), str(cold_study), str(out / 'sensitivity.json')],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([py, str(ANALYSIS / 'paper_figures.py'), str(out / 'figures'), str(campaign)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copy(out / 'figures/challenging_course.pdf', folder / 'paper_fig10_course.pdf')
    shutil.copy(out / 'figures/challenging_cdf.pdf', folder / 'paper_fig11_solve_times.pdf')
    table_ix(folder, tuning)
    print()
    table_x(folder, campaign, summary)
    print()
    table_xi(folder, summary, info.get('ipopt_hsl', True))
    print()
    table_xii(folder, cold_study, out / 'sensitivity.json')
    print(f'\nwritten to {folder}: paper_table_{{ix,x,xi,xii}}.{{txt,json}}, paper_fig10_course.pdf, '
          f'paper_fig11_solve_times.pdf, analysis/')
    if a.compare:
        print()
        return 1 if compare(folder) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())

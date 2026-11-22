"""Sensitivity table of cg2o from a cold-start Bayesian tuning study (same grouping as the paper's slalom table).

Search trials only (the baseline = course default setting is the time reference). A trial is rejected if a quality
gate failed (status REJECTED) or the evaluation failed. Med. time = median over the accepted trials of a group of
mean solve time / mean solve time of the default setting.

    python benchmark/analysis/sensitivity.py <dataset>/tuning/<study> [out.json]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def load(study):
    lines = [l for l in Path(study, 'trials.csv').read_text().splitlines() if not l.startswith('#')]
    rows = list(csv.DictReader(lines))
    case = next(k for k in rows[0] if k.endswith('_solve_ms_mean')).rsplit('_solve_ms_mean', 1)[0]
    base = next(r for r in rows if r['phase'] == 'baseline')
    t0 = float(base[f'{case}_solve_ms_mean'])
    trials = []
    for r in rows:
        if r['phase'] != 'search':
            continue
        t = r[f'{case}_solve_ms_mean']
        f = lambda k: float(r[k]) if r.get(k) not in (None, '') else np.nan
        trials.append(dict(
            trial=int(r['trial']), accepted=r['status'] == 'ok', reasons=r.get('reasons', ''),
            rel=float(t) / t0 if t not in ('', None) else np.nan,
            linear_full=f('ineq_prediction_strategy') == 0 and f('ineq_prediction_step_strategy') == 0,
            update=int(f('update_kappa_strategy')), positivity=int(f('keep_aux_positive_strategy')),
            zeta_aux=f('aux_backtracking_step_min'), zeta_ineq=f('ineq_backtracking_step_min'),
            kappa_final=f('kappa_final'), beta=f('aux_scaling_factor'), nu=f('kappa_update_factor'),
            tau=f('tau'), kappa0=f('kappa_initial'), init_kappa=int(f('init_kappa_strategy')),
            step=int(f('step_size_strategy')), unusable=f(f'{case}_unusable_fraction'),
            hits=f(f'{case}_hit_max_fraction')))
    return trials, t0, case


def group(trials, label, setting, pred):
    g = [t for t in trials if pred(t)]
    acc = [t['rel'] for t in g if t['accepted']]
    return dict(option=label, setting=setting, trials=len(g),
                rejected=(1 - len(acc) / len(g)) if g else None,
                median_rel_time=float(np.median(acc)) if acc else None)


def main(study, out=None):
    trials, t0, case = load(study)
    lin = [t for t in trials if t['linear_full']]
    rows = [group(trials, 'Prediction', 'linearized, full step (rec.)', lambda t: t['linear_full']),
            group(trials, 'Prediction', 'nonlinear or damped', lambda t: not t['linear_full'])]
    for code, name in ((0, 'lower-bounded (rec.)'), (1, 'bounded by kappa0'), (2, 'duality gap only'),
                       (3, 'fixed factor (nested)')):
        rows.append(group(lin, 'Barrier update', name, lambda t, c=code: t['update'] == c))
    for code, name in ((0, 'scaling (rec.)'), (1, 'correction')):
        rows.append(group(lin, 'Positivity', name, lambda t, c=code: t['positivity'] == c))
    bins = dict(zeta_aux=[0.5, 0.8], kappa_final=[1000, 3000, 8000], beta=[0.05, 0.3], zeta_ineq=[0.03, 0.3])
    for key, edges in bins.items():
        lo = -np.inf
        for hi in edges + [np.inf]:
            name = f'[{lo:g}, {hi:g})'
            rows.append(group(lin, key, name, lambda t, a=lo, b=hi, k=key: a <= t[k] < b))
            lo = hi
    acc = [t for t in lin if t['accepted']]
    corr = {}
    for key in ('zeta_aux', 'zeta_ineq', 'kappa_final', 'beta', 'nu', 'tau', 'kappa0'):
        pairs = [(t[key], t['rel']) for t in acc if np.isfinite(t[key]) and np.isfinite(t['rel'])
                 and not (key == 'tau' and t['update'] != 0) and not (key == 'kappa0' and t['init_kappa'] != 0)]
        if len(pairs) >= 5:
            rho = spearmanr(*zip(*pairs))[0]
            corr[key] = dict(n=len(pairs), rho=float(rho))
    ranked = sorted((t for t in trials if np.isfinite(t['rel'])), key=lambda t: t['rel'])
    summary = dict(study=str(study), case=case, default_solve_ms_mean=t0, search_trials=len(trials),
                   accepted=sum(t['accepted'] for t in trials),
                   accepted_median_rel=float(np.median([t['rel'] for t in trials if t['accepted']])),
                   fastest_five=[dict(trial=t['trial'], rel=t['rel'], accepted=t['accepted'], reasons=t['reasons'])
                                 for t in ranked[:5]],
                   groups=rows, spearman_accepted_linear=corr)
    print(f"{case}: default {t0:.1f} ms; {len(trials)} search trials, {summary['accepted']} accepted, "
          f"median accepted {summary['accepted_median_rel']:.2f}x default")
    print(f'{"option":15} {"setting":30} {"trials":>6} {"rej":>6} {"med":>6}')
    for r in rows:
        rej = '-' if r['rejected'] is None else f"{100 * r['rejected']:.0f}%"
        med = '-' if r['median_rel_time'] is None else f"{r['median_rel_time']:.2f}"
        print(f"{r['option']:15} {r['setting']:30} {r['trials']:6d} {rej:>6} {med:>6}")
    print('spearman (accepted, linear):', {k: (v['n'], round(v['rho'], 2)) for k, v in corr.items()})
    print('fastest five:', [(t['trial'], round(t['rel'], 2), t['accepted']) for t in ranked[:5]])
    if out is None:   # never write into the shipped data
        shipped = Path(study).resolve().is_relative_to(Path(__file__).resolve().parents[2] / 'data')
        out = Path('results/analysis') / f'{Path(study).name}_sensitivity.json' if shipped else Path(study, 'sensitivity.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=1))
    print(f'written: {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])

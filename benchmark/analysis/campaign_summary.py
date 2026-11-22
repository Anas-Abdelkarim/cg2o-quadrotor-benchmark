"""Summary of the repeat PX4/Gazebo campaign of the challenging MPC.

For every flight folder results/challenging_campaign/{warm,cold}_NN: the truth evaluation
(challenging_evaluation.json), the in-flight solves of the navigation phase (challenging_trace.jsonl),
the PX4 position holds (solves.csv) and, if present, the paired replay (replay/summary.json and
replay/paired.json). Writes campaign_summary.json and prints one table per mode.

    python benchmark/analysis/campaign_summary.py CAMPAIGN_DIR [OUT_JSON]

CAMPAIGN_DIR: <dataset>/challenging_campaign (data/ or a run folder). OUT_JSON: default
CAMPAIGN_DIR/campaign_summary.json; give another path for the shipped data, which is never written.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PERIOD_MS = 100.0


def flight(folder):
    f = Path(folder)
    out = dict(name=f.name)
    ev = json.loads((f / 'challenging_evaluation.json').read_text()) if (f / 'challenging_evaluation.json').exists() else {}
    out.update(outcome=ev.get('outcome'), time_to_goal_s=ev.get('time_to_goal_s'),
               min_clearance_m=ev.get('min_true_body_clearance_m'), max_tilt_deg=ev.get('max_true_tilt_deg'),
               jerk_rms=ev.get('filtered_jerk_rms_mps3'))
    rows = [json.loads(l) for l in (f / 'challenging_trace.jsonl').read_text().splitlines() if l]
    nav = [r for r in rows if r.get('kind') == 'problem' and r.get('phase') == 'NMPC']
    out.update(nav_solves=len(nav), nav_unusable=sum(not r['usable'] for r in nav),
               nav_limit_hits=sum(r['iterations'] >= 80 for r in nav),
               nav_over_period=sum(r['solve_ms'] > PERIOD_MS for r in nav),
               nav_solve_ms_median=float(np.median([r['solve_ms'] for r in nav])) if nav else None,
               nav_solve_ms_max=float(max(r['solve_ms'] for r in nav)) if nav else None)
    with open(f / 'solves.csv') as fh:
        s = list(csv.DictReader(fh))
    out['hold_samples'] = sum(r['phase'] == 'NMPC' and r['applied_source'] == 'hold' for r in s)
    rep = f / 'replay'
    if (rep / 'paired.json').exists():
        paired = json.loads((rep / 'paired.json').read_text())
        for name in ('cg2o', 'ipopt'):
            t = np.array([r[f'{name}_solve_time_ms'] - r[f'{name}_hessian_ms'] for r in paired])
            it = np.array([r[f'{name}_iterations'] for r in paired])
            out[f'replay_{name}'] = dict(
                problems=len(paired), unusable=sum(not r[f'{name}_usable'] for r in paired),
                limit_hits=int((it >= 80).sum()), over_period=int((t > PERIOD_MS).sum()),
                mean_ms=float(t.mean()), median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                max_ms=float(t.max()), mean_iterations=float(it.mean()))
        out['replay_hash_mismatches'] = sum(not r['flight_hash_match'] for r in paired)
        out['replay_cold_infeasible'] = sum(r['cold_obstacle_rows_inside_reserve'] > 0 for r in paired)
    return out


def pool(flights, name):
    """Replay statistics over all problems of all flights of one mode."""
    t, it, unusable, n = [], [], 0, 0
    for fl in flights:
        p = Path(fl) / 'replay/paired.json'
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            t.append(r[f'{name}_solve_time_ms'] - r[f'{name}_hessian_ms'])
            it.append(r[f'{name}_iterations'])
            unusable += not r[f'{name}_usable']
            n += 1
    if not n:
        return None
    t, it = np.array(t), np.array(it)
    return dict(problems=n, unusable=unusable, limit_hits=int((it >= 80).sum()), over_period=int((t > PERIOD_MS).sum()),
                mean_ms=float(t.mean()), median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                max_ms=float(t.max()), mean_iterations=float(it.mean()))


def paper_table(flights):
    """Navigation-phase statistics of the paired replay, as reported in the paper."""
    rows = []
    for fl in flights:
        p = Path(fl) / 'replay/paired.json'
        if p.exists():
            rows += [r for r in json.loads(p.read_text()) if r.get('phase') == 'NMPC']
    if not rows:
        return None
    out = dict(problems=len(rows))
    for name in ('cg2o', 'ipopt'):
        t = np.array([r[f'{name}_solve_time_ms'] - r[f'{name}_hessian_ms'] for r in rows])
        it = np.array([r[f'{name}_iterations'] for r in rows])
        usable = np.array([bool(r[f'{name}_usable']) for r in rows])
        out[name] = dict(mean_iterations=float(it.mean()), mean_ms=float(t.mean()), median_ms=float(np.median(t)),
                         p95_ms=float(np.percentile(t, 95)), max_ms=float(t.max()),
                         within_period_pct=float(100 * (t <= PERIOD_MS).mean()),
                         usable_pct=float(100 * usable.mean()), unusable=int((~usable).sum()),
                         limit_hits=int((it >= 80).sum()))
    inside = np.array([r['cold_obstacle_rows_inside_reserve'] for r in rows])
    out['cold_start_infeasible'] = int((inside > 0).sum())
    out['cold_start_rows_inside_mean'] = float(inside[inside > 0].mean()) if (inside > 0).any() else 0.0
    out['cold_start_penetration_max_m'] = float(max(r['cold_max_reserve_penetration_m'] for r in rows))
    both = [r for r in rows if r['cg2o_usable'] and r['ipopt_usable']]
    rel = [(r['cg2o_final_cost'] - r['ipopt_final_cost']) / max(1.0, abs(r['ipopt_final_cost'])) for r in both]
    out['cg2o_cost_excess_median_pct'] = float(100 * np.median(rel)) if rel else None
    out['hash_mismatches'] = sum(not r['flight_hash_match'] for r in rows)
    return out


def main(root='data/challenging_campaign', out=None):
    root = Path(root)
    result = {}
    for mode in ('warm', 'cold'):
        folders = sorted(p for p in root.glob(f'{mode}_[0-9][0-9]') if (p / 'challenging_trace.jsonl').exists())
        if not folders:
            continue
        rows = [flight(p) for p in folders]
        result[mode] = dict(flights=rows, pooled_cg2o=pool(folders, 'cg2o'), pooled_ipopt=pool(folders, 'ipopt'),
                            paper=paper_table(folders))
        print(f'\n== {mode}: {len(rows)} flights')
        print(f'{"flight":8} {"outcome":8} {"goal s":>7} {"clear m":>8} {"tilt":>6} {"nav":>4} {"unus":>4} '
              f'{"hits":>4} {"holds":>5} | replay cg2o/ipopt: unusable, mean ms')
        for r in rows:
            rep = ''
            if 'replay_cg2o' in r:
                a, b = r['replay_cg2o'], r['replay_ipopt']
                rep = f"{a['unusable']}/{b['unusable']}, {a['mean_ms']:.1f}/{b['mean_ms']:.1f}"
            fmt = lambda v, f: (f % v) if isinstance(v, (int, float)) else '-'
            print(f"{r['name']:8} {str(r['outcome']):8} {fmt(r['time_to_goal_s'], '%7.1f')} "
                  f"{fmt(r['min_clearance_m'], '%8.3f')} {fmt(r['max_tilt_deg'], '%6.1f')} {r['nav_solves']:4d} "
                  f"{r['nav_unusable']:4d} {r['nav_limit_hits']:4d} {r['hold_samples']:5d} | {rep}")
        for name in ('cg2o', 'ipopt'):
            if result[mode][f'pooled_{name}']:
                print(f'pooled replay {name}:', {k: round(v, 2) for k, v in result[mode][f'pooled_{name}'].items()})
        if result[mode]['paper']:
            print('paper table (navigation phase):', json.dumps(result[mode]['paper'], indent=1))
    if out is None:   # never write into the shipped data
        shipped = root.resolve().is_relative_to(Path(__file__).resolve().parents[2] / 'data')
        out = Path('results/analysis/campaign_summary.json') if shipped else root / 'campaign_summary.json'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])

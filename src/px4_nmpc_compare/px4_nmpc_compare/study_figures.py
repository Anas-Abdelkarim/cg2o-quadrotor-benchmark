"""Figures, tables and REPORT.md of a paired slalom study (see slalom_study.py)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

# reference categorical palette (fixed order, light surface); a variant keeps its colour in every figure
PALETTE = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
           'magenta': '#e87ba4', 'green': '#008300', 'violet': '#6250d6', 'red': '#e34948'}
VARIANT_COLOR = {
    # IPOPT (native C++, default options): blue family
    'ipopt': 'blue', 'ipopt_warm_primal': 'blue', 'ipopt_warm_primal_dual': 'aqua',
    # ISPD (cg2o): warm colours; a tuned variant is dashed in the colour of its base variant
    'ispd': 'orange', 'ispd_warm_final': 'orange', 'ispd_warm_primal': 'yellow', 'ispd_warm_middle': 'magenta',
}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def color_of(name, fallback_index=0):
    if name.endswith('_tuned'):          # a tuned variant keeps its base variant's colour (dashed line)
        name = name[:-len('_tuned')]
    if name in VARIANT_COLOR:
        return PALETTE[VARIANT_COLOR[name]]
    return list(PALETTE.values())[fallback_index % len(PALETTE)]


def style_of(name):
    return '--' if name.endswith('_tuned') else '-'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
        'axes.spines.right': False, 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
        'legend.frameon': False, 'lines.linewidth': 1.6, 'axes.titlecolor': INK,
    })
    return plt


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        q = {}
        for k, v in r.items():
            if v in ('True', 'False'):
                q[k] = v == 'True'
            else:
                try:
                    q[k] = float(v)
                except (TypeError, ValueError):
                    q[k] = v
        out.append(q)
    return out


def col(rows, key):
    return np.asarray([r.get(key, np.nan) if isinstance(r.get(key, np.nan), float) else np.nan for r in rows], float)


def summary_of(path):
    p = Path(str(path) + '.json')
    return json.loads(p.read_text()) if p.exists() else {}


def _save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')


def label_of(variants, name):
    return variants.get(name, {}).get('label', name)


# ------------------------------------------------------------------ figures
def fig_per_step(plt, out, variants, names, folder, title, fname):
    data = [(n, read_csv(folder / f'{n}.csv')) for n in names]
    data = [(n, r) for n, r in data if r]
    if not data:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    for i, (n, rows) in enumerate(data):
        t = col(rows, 't0')
        kw = dict(color=color_of(n, i), ls=style_of(n), label=label_of(variants, n), lw=1.4)
        axes[0].plot(t, col(rows, 'iterations'), **kw)
        axes[1].plot(t, col(rows, 'solve_ms'), **kw)
    axes[0].set_ylabel('iterations per OCP')
    axes[1].set_ylabel('solve time per OCP [ms]')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('time of the OCP in the slalom flight [s]')
    axes[0].set_title(title, loc='left')
    axes[0].legend(ncol=2, fontsize=8, loc='upper right')
    fig.tight_layout()
    _save(fig, out, fname)
    plt.close(fig)


def fig_bars(plt, out, variants, groups, fname, title):
    """groups: [(group title, [(name, summary)])]; panels: mean solve time (p95 tick), mean iterations."""
    groups = [(g, [(n, s) for n, s in items if s]) for g, items in groups]
    groups = [(g, items) for g, items in groups if items]
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 2, figsize=(10.5, 1.0 + 0.42 * max(len(i) for _, i in groups) * len(groups)),
                             squeeze=False)
    for gi, (gtitle, items) in enumerate(groups):
        names = [n for n, _ in items]
        y = np.arange(len(items))[::-1]
        for ax, key, xlabel in ((axes[gi, 0], 'solve_ms', 'solve time per OCP [ms]: mean, p95 tick'),
                                (axes[gi, 1], 'iterations', 'iterations per OCP: mean, p95 tick')):
            mean = [s.get(f'{key}_mean', np.nan) for _, s in items]
            p95 = [s.get(f'{key}_p95', np.nan) for _, s in items]
            colors = [color_of(n, i) for i, n in enumerate(names)]
            ax.barh(y, mean, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
            ax.scatter(p95, y, marker='|', s=160, color=INK, zorder=3, linewidths=1.6)
            for yy, m, q in zip(y, mean, p95):
                if np.isfinite(m):
                    ax.text(np.nanmax([m, q]), yy, f'  {m:.1f}', va='center', ha='left', fontsize=8, color=INK)
            ax.set_yticks(y)
            ax.set_yticklabels([label_of(variants, n) for n in names] if ax is axes[gi, 0] else [''] * len(names))
            ax.set_xlabel(xlabel)
            ax.grid(axis='y', visible=False)
            ax.set_xlim(0, max([v for v in p95 + mean if np.isfinite(v)] + [1]) * 1.25)
        axes[gi, 0].set_title(gtitle, loc='left')
    fig.suptitle(title, x=0.01, ha='left', fontweight='bold', color=INK)
    fig.tight_layout()
    _save(fig, out, fname)
    plt.close(fig)


def fig_cdf(plt, out, variants, cold, warm, base):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True)
    for ax, names, folder, title in ((axes[0], cold, base / 'cold', 'cold (infeasible reference rollout)'),
                                     (axes[1], warm, base / 'warm', 'warm (own previous solution, shifted)')):
        for i, n in enumerate(names):
            rows = read_csv(folder / f'{n}.csv')
            v = np.sort(col(rows, 'solve_ms'))
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            ax.step(v, np.arange(1, v.size + 1) / v.size, where='post', color=color_of(n, i), ls=style_of(n),
                    label=label_of(variants, n))
        ax.set_xscale('log')
        ax.axvline(100.0, color=INK2, lw=1, ls=':')
        ax.text(100.0, 1.02, '100 ms control period', color=INK2, fontsize=8, ha='center', va='bottom')
        ax.set_xlabel('solve time per OCP [ms]')
        ax.set_title(title, loc='left', pad=14)
        ax.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, -0.16), ncol=2)
    axes[0].set_ylabel('fraction of OCPs solved within')
    fig.tight_layout()
    _save(fig, out, 'solve_time_cdf')
    plt.close(fig)


def fig_first(plt, out, variants, names, base):
    items = []
    for n in names:
        rows = read_csv(base / 'first' / f'{n}.csv')
        if rows:
            r = rows[0]
            items.append((n, r))
    if not items:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.9 + 0.42 * len(items)))
    y = np.arange(len(items))[::-1]
    names_ = [n for n, _ in items]
    colors = [color_of(n, i) for i, n in enumerate(names_)]
    ms = [r.get('solve_ms', np.nan) for _, r in items]
    it = [r.get('iterations', np.nan) for _, r in items]
    ok = [bool(r.get('usable')) and bool(r.get('success')) for _, r in items]
    for ax, vals, xlabel, fmt in ((axes[0], ms, 'solve time [ms] (median of repeats)', '{:.1f}'),
                                  (axes[1], it, 'iterations', '{:.0f}')):
        ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        for yy, v, good in zip(y, vals, ok):
            if np.isfinite(v):
                ax.text(v, yy, ' ' + fmt.format(v) + ('' if good else '  not converged'), va='center', fontsize=8,
                        color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([label_of(variants, n) for n in names_] if ax is axes[0] else [''] * len(names_))
        ax.set_xlabel(xlabel)
        ax.grid(axis='y', visible=False)
        ax.set_xlim(0, max([v for v in vals if np.isfinite(v)] + [1]) * 1.35)
    axes[0].set_title('First slalom OCP, cold: the reference rollout penetrates all 10 safety shells', loc='left')
    fig.tight_layout()
    _save(fig, out, 'first_ocp')
    plt.close(fig)


def fig_cost(plt, out, variants, cold, warm, base):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.9 + 0.36 * max(len(cold), len(warm))), sharex=True)
    for ax, names, folder, title in ((axes[0], cold, base / 'cold', 'cold'), (axes[1], warm, base / 'warm', 'warm')):
        data, labels, colors = [], [], []
        for i, n in enumerate(names):
            v = col(read_csv(folder / f'{n}.csv'), 'cost_excess_rel')
            v = v[np.isfinite(v)]
            if v.size:
                data.append(np.clip(v, -1, 1))
                labels.append(label_of(variants, n))
                colors.append(color_of(n, i))
        if not data:
            continue
        bp = ax.boxplot(data[::-1], vert=False, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, markeredgecolor=INK2),
                        medianprops=dict(color=INK))
        for patch, c in zip(bp['boxes'], colors[::-1]):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
        ax.set_yticks(np.arange(1, len(labels) + 1))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xscale('symlog', linthresh=1e-6)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(f'{title}: (cost - reference) / max(1, |reference|)', loc='left')
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    _save(fig, out, 'cost_vs_reference')
    plt.close(fig)


def obstacle_geometry(params):
    centers = np.asarray(params['obstacle_centers_flat'], float).reshape(-1, 3)
    radii = np.asarray(params['obstacle_radii'], float)
    uav, margin = float(params['uav_radius']), float(params['safety_margin'])
    return centers, radii, radii + uav, radii + uav + margin, uav, margin


def passing_side(params, pos):
    """For every obstacle: how the flight `pos` passes it at its closest approach: 'above', 'below' or 'side'."""
    centers = obstacle_geometry(params)[0]
    out = []
    for c in centers:
        i = int(np.argmin(np.linalg.norm(pos - c, axis=1)))
        dh = float(np.linalg.norm(pos[i, 0:2] - c[0:2]))
        dz = float(c[2] - pos[i, 2])             # NED: positive = the vehicle is above the obstacle centre
        out.append(('above' if dz > 0 else 'below') if abs(dz) > dh else 'side')
    return out


def clearance_along(params, pos):
    """3-D clearance between the vehicle body and the nearest obstacle surface (0 = contact)."""
    centers, radii, _, _, uav, _ = obstacle_geometry(params)
    d = np.linalg.norm(pos[:, None, :] - centers[None, :, :], axis=2) - radii[None, :] - uav
    return d.min(axis=1)


def draw_obstacles_top(ax, params, pos=None, label=True, annotate=True):
    """Top view of every obstacle sphere: obstacle (filled), collision boundary (obstacle + UAV radius) and safety
    region (+ safety margin, the NMPC constraint). With `pos`, obstacles that the flight avoids vertically are marked
    'above'/'below': their circles overlap the path in the top view although the 3-D clearance is positive."""
    from matplotlib.patches import Circle
    centers, r, r_col, r_saf, _, _ = obstacle_geometry(params)
    sides = passing_side(params, pos) if (pos is not None and annotate) else ['side'] * len(r)
    for k, (c, a, b, s, side) in enumerate(zip(centers, r, r_col, r_saf, sides)):
        ax.add_patch(Circle(c[0:2], s, facecolor='#f3e7c9', edgecolor='#b8860b', lw=0.9, ls='--', zorder=1))
        ax.add_patch(Circle(c[0:2], b, facecolor='none', edgecolor=INK, lw=0.9, zorder=2))
        ax.add_patch(Circle(c[0:2], a, facecolor='#9c9a94', edgecolor=INK, lw=0.6, zorder=3))
        if side != 'side':
            up = c[1] > 0.1 or (abs(c[1]) <= 0.1 and k % 2 == 0)
            ax.text(c[0], c[1] + (s + 0.1) * (1 if up else -1), 'over' if side == 'above' else 'under', ha='center',
                    va='bottom' if up else 'top', fontsize=7, color=INK2, zorder=5)
    if label:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        return [Patch(facecolor='#9c9a94', edgecolor=INK, label='obstacle'),
                Line2D([], [], color=INK, lw=0.9, label='collision boundary (+ UAV radius)'),
                Patch(facecolor='#f3e7c9', edgecolor='#b8860b', ls='--', label='safety region (+ margin = NMPC constraint)')]
    return []


def altitude_of(params, ned):
    """Height above the arena floor of NED positions in the controller frame (floor = arena origin)."""
    floor_d = float(np.asarray(params.get('arena_origin_local_ned', [0.0, 0.0, 0.0]), float)[2])
    return floor_d - np.asarray(ned, float)[..., 2]


def draw_obstacles_side(ax, params):
    """Side view (North, altitude) of every obstacle sphere through its centre: obstacle, collision boundary and
    safety region, as in draw_obstacles_top."""
    from matplotlib.patches import Circle
    centers, r, r_col, r_saf, _, _ = obstacle_geometry(params)
    alt = altitude_of(params, centers)
    for c, h, a, b, s in zip(centers, alt, r, r_col, r_saf):
        ax.add_patch(Circle((c[0], h), s, facecolor='#f3e7c9', edgecolor='#b8860b', lw=0.9, ls='--', zorder=1))
        ax.add_patch(Circle((c[0], h), b, facecolor='none', edgecolor=INK, lw=0.9, zorder=2))
        ax.add_patch(Circle((c[0], h), a, facecolor='#9c9a94', edgecolor=INK, lw=0.6, zorder=3))


def fig_flights(plt, out, variants, flights, base, params):
    data = []
    for i, n in enumerate(flights):
        f = base / 'flights' / n / 'flight.npz'
        if f.exists():
            d = np.load(f)
            s = json.loads((base / 'flights' / n / 'flight_summary.json').read_text())
            data.append((n, d, s))
    if not data:
        return
    _, _, _, _, uav, margin = obstacle_geometry(params)
    ref = next((d for n, d, _ in data if n.startswith('ispd') or 'cg2o' in n), data[0][1])
    centers, _, _, r_saf, _, _ = obstacle_geometry(params)
    allp = np.concatenate([d['pos'] for _, d, _ in data])
    xr = (float(allp[:, 0].min()) - 0.6, float(allp[:, 0].max()) + 0.6)
    yr = (min(float(allp[:, 1].min()), float((centers[:, 1] - r_saf).min())) - 0.7,
          max(float(allp[:, 1].max()), float((centers[:, 1] + r_saf).max())) + 0.7)
    top_h = 0.78 * 11.5 * (yr[1] - yr[0]) / (xr[1] - xr[0])
    fig, axes = plt.subplots(2, 1, figsize=(11.5, top_h + 3.6), sharex=True,
                             gridspec_kw={'height_ratios': [top_h, 2.2]})
    ax = axes[0]
    handles = draw_obstacles_top(ax, params, ref['pos'])
    for i, (n, d, s) in enumerate(data):
        ax.plot(d['pos'][:, 0], d['pos'][:, 1], color=color_of(n, i), ls=style_of(n), lw=1.5, zorder=4,
                label=f"{label_of(variants, n)}: {s.get('outcome')}, {s.get('time_to_goal', float('nan')):.1f} s")
    ax.axhline(0.0, color=INK2, lw=0.8, ls=':', zorder=0)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('East [m]')
    ax.set_title('Closed-loop slalom flights, top view (dotted: fixed straight reference; over/under: obstacle '
                 'avoided vertically)', loc='left')
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles, fontsize=7.5, loc='upper left',
              bbox_to_anchor=(1.0, 1.05))
    ax = axes[1]
    for i, (n, d, s) in enumerate(data):
        ax.plot(d['pos'][:, 0], clearance_along(params, d['pos']), color=color_of(n, i), ls=style_of(n), lw=1.3)
    ax.axhspan(-0.5, 0.0, color='#e34948', alpha=0.12, lw=0)
    ax.axhline(0.0, color='#e34948', lw=1.0)
    ax.axhline(margin, color='#b8860b', lw=1.0, ls='--')
    ax.text(ax.get_xlim()[0], 0.0, ' collision', color='#e34948', va='bottom', fontsize=8)
    ax.text(ax.get_xlim()[0], margin, ' safety margin', color='#b8860b', va='bottom', fontsize=8)
    ax.set_ylim(-0.1, max(2.0, margin + 0.5))
    ax.set_ylabel('clearance [m]')
    ax.set_xlabel('North [m]')
    ax.set_title('3-D distance between the vehicle body and the nearest obstacle surface', loc='left')
    fig.tight_layout()
    _save(fig, out, 'flights')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    for i, (n, d, s) in enumerate(data):
        kw = dict(color=color_of(n, i), ls=style_of(n), label=label_of(variants, n), lw=1.3)
        axes[0].plot(d['solve_t'], d['solve_iterations'], **kw)
        axes[1].plot(d['solve_t'], d['solve_ms'], **kw)
    axes[0].set_ylabel('iterations')
    axes[1].set_ylabel('solve time [ms]')
    axes[1].set_yscale('log')
    axes[1].axhline(100.0, color=INK2, lw=1, ls=':')
    axes[1].set_xlabel('simulation time [s]')
    axes[0].set_title('Solver effort during each closed-loop flight', loc='left')
    axes[0].legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    _save(fig, out, 'flight_solver_effort')
    plt.close(fig)


# ------------------------------------------------------------------ report
def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '-'
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f'{v:.1e}'
        return f'{v:.{nd}f}'
    return str(v)


def table(rows, headers):
    out = ['| ' + ' | '.join(h for h, _ in headers) + ' |', '|' + '---|' * len(headers)]
    for r in rows:
        out.append('| ' + ' | '.join(_fmt(f(r)) if callable(f) else _fmt(r.get(f)) for _, f in headers) + ' |')
    return '\n'.join(out)


def write_study_report(base, variants, cold, warm, flights, st):
    from .common import write_csv
    base = Path(base)
    out = base / 'figures'
    out.mkdir(exist_ok=True)
    params = json.loads((base / 'params.json').read_text())
    summary_rows = []
    for protocol, names, folder in (('cold', cold, 'cold'), ('warm', warm, 'warm'), ('first_ocp', cold, 'first')):
        for n in names:
            s = summary_of(base / folder / f'{n}.csv')
            if s:
                summary_rows.append(dict(protocol=protocol, variant=n, label=label_of(variants, n), **s))
    flight_rows = []
    for n in flights:
        f = base / 'flights' / n / 'flight_summary.json'
        if f.exists():
            s = json.loads(f.read_text())
            flight_rows.append(dict(variant=n, label=label_of(variants, n), **{k: v for k, v in s.items()
                                                                              if not isinstance(v, (list, dict))}))
    write_csv(summary_rows, base / 'summary.csv')
    write_csv(flight_rows, base / 'flights.csv')

    plt = _plt()
    fig_first(plt, out, variants, cold, base)
    fig_per_step(plt, out, variants, cold, base / 'cold',
                 'Every slalom OCP solved COLD from the infeasible reference rollout', 'cold_per_ocp')
    fig_per_step(plt, out, variants, warm, base / 'warm',
                 'Every slalom OCP solved WARM from the variant\'s own previous solution', 'warm_per_ocp')
    by = {(r['protocol'], r['variant']): r for r in summary_rows}
    fig_bars(plt, out, variants,
             [('cold', [(n, by.get(('cold', n))) for n in cold]), ('warm', [(n, by.get(('warm', n))) for n in warm])],
             'summary_bars', 'Paired slalom OCPs: solve time and iterations (same OCPs for every solver)')
    fig_cdf(plt, out, variants, cold, warm, base)
    fig_cost(plt, out, variants, cold, warm, base)
    fig_flights(plt, out, variants, flights, base, params)

    meta = json.loads((base / 'trace' / 'trace_meta.json').read_text())
    ds = meta.get('driver_summary', {})
    hdr = [('variant', 'label'), ('OCPs', 'steps'), ('usable', 'usable_rate'), ('converged', 'success_rate'),
           ('at limit', 'hit_max_fraction'), ('iter mean', 'iterations_mean'), ('iter p95', 'iterations_p95'),
           ('ms mean', 'solve_ms_mean'), ('ms median', 'solve_ms_median'), ('ms p95', 'solve_ms_p95'),
           ('ms max', 'solve_ms_max'), ('ms/iter', 'ms_per_iteration'),
           ('cost excess median', 'cost_excess_rel_median'), ('other local opt.', 'other_optimum_fraction'),
           ('eq res max', 'eq_residual_max'),
           ('ineq viol max', 'ineq_violation_max')]
    fhdr = [('variant', 'label'), ('outcome', 'outcome'), ('time to goal [s]', 'time_to_goal'),
            ('min phys. clearance [m]', 'min_physical_clearance'), ('solves', 'solves'),
            ('usable', 'usable_rate'), ('ms median', 'solve_ms_median'), ('ms p95', 'solve_ms_p95'),
            ('ms max', 'solve_ms_max'), ('iter median', 'iterations_median'), ('iter max', 'iterations_max'),
            ('hold events', 'hold_events')]
    sel = lambda p: [r for r in summary_rows if r['protocol'] == p]
    lines = [
        '# Paired ISPD (cg2o) vs IPOPT study: infeasible 3-D slalom', '',
        f"Driver (flies the recorded OCP sequence): **{label_of(variants, st['driver'])}**, outcome "
        f"`{ds.get('outcome')}`, {meta.get('requests')} OCPs, horizon N = {meta.get('N')}, "
        f"K = {meta.get('K')} obstacle slots, dt = {meta.get('dt')} s.",
        f"Common iteration budget for every solver: **{st.get('budget')}**. Reference cost of every OCP: "
        f"{label_of(variants, st['reference'])}. All solutions are scored by the same function "
        '(objective and constraint violation of the CasADi formulation); `cost excess` = (cost - reference) / '
        'max(1, |reference|); `other local opt.` = fraction of OCPs whose cost is > 1 % above the reference: the slalom '
        'OCP is nonconvex and such solutions are other feasible local optima (e.g. passing an obstacle on its other '
        'side), not inaccurate ones. The remaining small positive excess of ISPD is its barrier gap (it stops at '
        'kappa_final, i.e. s*lambda = 1/kappa_final per inequality row).', '',
        'Times are optimizer-only (`solve_ms`), one process per solver pinned to one core, single-threaded BLAS. '
        'They are machine dependent: compare solvers within this folder. Iteration counts are not comparable '
        'between IPOPT and ISPD one-to-one (different work per iteration); time is.', '',
        '## First OCP, cold (Level 1: the reference rollout penetrates all safety shells)', '',
        table(sel('first_ocp'), hdr), '', '![first OCP](figures/first_ocp.png)', '',
        '## All OCPs of the flight, cold', '', table(sel('cold'), hdr), '',
        '![cold](figures/cold_per_ocp.png)', '',
        '## All OCPs of the flight, warm (receding-horizon warm starts)', '', table(sel('warm'), hdr), '',
        '![warm](figures/warm_per_ocp.png)', '', '![summary](figures/summary_bars.png)', '',
        '![cdf](figures/solve_time_cdf.png)', '', '![cost](figures/cost_vs_reference.png)', '',
        '## Closed-loop flights (each solver flies the slalom itself)', '', table(flight_rows, fhdr), '',
        '![flights](figures/flights.png)', '', '![effort](figures/flight_solver_effort.png)', '',
        '## Files', '',
        '`cold/*.csv`, `warm/*.csv`, `first/*.csv`: one row per OCP (iterations, solve_ms, solver and common cost, '
        'residuals, warm-start information); `*.csv.json`: their summaries. `summary.csv`, `flights.csv`: the tables '
        'above. `trace/`: the recorded OCP sequence. `study_used.yaml`, `run_context.yaml`, `config_used/`: exact '
        'settings. `logs/`: one log per process.',
    ]
    (base / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    try:
        from .paper_figures import write_paper_material
        write_paper_material(base, variants, cold, warm, flights, st)
    except Exception as exc:  # noqa: BLE001 - the study report must not be lost because of a paper figure
        print(f'[report] paper figures failed: {exc!r}', flush=True)
    return summary_rows, flight_rows

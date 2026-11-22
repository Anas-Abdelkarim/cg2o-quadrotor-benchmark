"""Paper figures of the challenging MPC campaign (results/challenging_campaign).

    python benchmark/analysis/paper_figures.py OUT_DIR [CAMPAIGN_DIR]   (default data/challenging_campaign)

challenging_course.pdf: top and side view of the course with the 20 PX4/Gazebo flights (ground truth).
challenging_cdf.pdf:    solve-time distributions of the paired replay, navigation phase, (a) cold, (b) warm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[2]
CAMP = ROOT / 'data/challenging_campaign'
BLUE, ORANGE, GRAY, INK = '#2a78d6', '#eb6834', '#9a9a96', '#3c3c3a'
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'legend.frameon': False, 'pdf.fonttype': 42})


def flights(mode):
    return sorted(CAMP.glob(f'{mode}_[0-9][0-9]'))


def course_figure(out):
    sc = json.loads((flights('warm')[0] / 'runtime/scenario.json').read_text())
    obs = np.asarray(sc['typed_obstacles'], float)
    fig, (top, side) = plt.subplots(2, 1, figsize=(3.5, 2.3), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1, 1], hspace=0.1))
    for kind, a, b, c, r in obs:
        if kind == 0:      # sphere (N, E, D)
            top.add_patch(Circle((a, b), r, color=GRAY, lw=0))
            side.add_patch(Circle((a, -c), r, color=GRAY, lw=0))
        elif kind == 1:    # vertical post (N, E)
            top.add_patch(Circle((a, b), r, color=INK, lw=0))
            side.add_patch(Rectangle((a - r, 0), 2 * r, 6, color=INK, alpha=0.35, lw=0))
        else:              # horizontal bar along East through (N, D)
            top.add_patch(Rectangle((a - r, -5), 2 * r, 10, color=INK, alpha=0.35, lw=0))
            side.plot(a, -c, 'o', ms=3, color=INK)   # bar cross-section (drawn enlarged)
    s = np.linspace(-14, 14, 200)
    for ax, y in ((top, 0 * s), (side, 2.5 - np.sin(np.pi * s / 14))):
        ax.plot(s, y, color=INK, lw=0.6, ls=(0, (3, 2)), label='guide line')
    for mode, color in (('warm', BLUE), ('cold', ORANGE)):
        for i, f in enumerate(flights(mode)):
            T = np.load(f / 'truth.npz')
            p = T['position'][:, T['navigation']]
            kw = dict(color=color, lw=0.6, alpha=0.75, label=f'{mode} start' if i == 0 else None)
            top.plot(p[0], p[1], **kw)
            side.plot(p[0], -p[2], **kw)
    top.set(ylim=(-1.9, 2.1), ylabel='East (m)')
    side.set(ylim=(0.5, 4.3), ylabel='Altitude (m)', xlabel='North (m)', xlim=(-14.8, 14.8))
    for ax in (top, side):
        ax.spines[['top', 'right']].set_visible(False)
    top.legend(loc='upper left', ncol=3, fontsize=6.5, bbox_to_anchor=(0, 1.32), handlelength=1.6)
    top.text(0.99, 0.04, '(a) top view', transform=top.transAxes, va='bottom', ha='right', fontsize=6.5, color=INK)
    side.text(0.99, 0.97, '(b) side view', transform=side.transAxes, va='top', ha='right', fontsize=6.5, color=INK)
    fig.savefig(out / 'challenging_course.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def times(mode, solver):
    t = []
    for f in flights(mode):
        for r in json.loads((f / 'replay/paired.json').read_text()):
            if r.get('phase') == 'NMPC':
                t.append(r[f'{solver}_solve_time_ms'] - r[f'{solver}_hessian_ms'])
    return np.sort(np.asarray(t))


def cdf_figure(out):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, mode, tag in zip(axes, ('cold', 'warm'), ('(a) cold start', '(b) warm start')):
        for solver, color, ls, name in (('cg2o', BLUE, '-', 'cg2o'), ('ipopt', ORANGE, (0, (4, 1.5)), 'IPOPT')):
            t = times(mode, solver)
            ax.step(t, np.arange(1, len(t) + 1) / len(t), where='post', color=color, ls=ls, lw=1.0, label=name)
        ax.axvline(100, color=INK, lw=0.5, ls=':')
        ax.set(xscale='log', xlim=(5, 700), xlabel='Solve time (ms)')
        ax.set_title(tag, fontsize=7.5, color=INK)
        ax.grid(True, which='major', lw=0.3, color='#dcdcd8')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Fraction of problems')
    axes[0].legend(loc='upper left', fontsize=6.5, handlelength=2)
    fig.savefig(out / 'challenging_cdf.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    if len(sys.argv) > 2:
        CAMP = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    course_figure(out)
    cdf_figure(out)

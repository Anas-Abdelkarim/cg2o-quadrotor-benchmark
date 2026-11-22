"""Optional plots use only actual results and the final selected trial.

search_progress.png      screening history: accepted trials, rejected trials, best feasible score so far
parameters.png           score against value for the numeric parameters most correlated with the score (accepted search trials)
selected_vs_baseline.png objective metric per case: validation baseline against the selected configuration (only when a
                         configuration was selected by validation)
"""

from __future__ import annotations
import json
from pathlib import Path
from .storage import Store
from .tables import case_metrics


def _case_key(case):
    return json.dumps(case, sort_keys=True, separators=(",", ":"))


def plot_study(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    if not (folder / "trials.sqlite3").is_file():
        raise ValueError("study database missing")
    db = Store(folder)
    try:
        rows = db.results()
    finally:
        db.close()
    report = (
        json.loads((folder / "report.json").read_text())
        if (folder / "report.json").exists()
        else None
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    out = []
    fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
    search = [e for e in rows if e.phase == "search"]
    best = None
    xs, ys = [], []
    for i, e in enumerate(search, 1):
        if e.accepted:
            ax.plot(i, e.score, "o", color="#0072B2", ms=3)
            best = e.score if best is None else min(best, e.score)
        else:
            ax.plot(
                i, 0.97, "x", color="#D55E00", ms=3, transform=ax.get_xaxis_transform()
            )
        if best is not None:
            xs.append(i)
            ys.append(best)
    ax.step(
        xs, ys, where="post", color="black", lw=1, label="Best feasible screening score"
    )
    ax.set(
        xlabel="Search attempt (completed)",
        ylabel="Score (lower is better)",
        title="Screening history; crosses indicate rejected trials",
    )
    ax.grid(alpha=0.2)
    if xs:
        ax.legend(frameon=False)
    if report:
        fig.suptitle(
            f"Final selection: {report['status']} (trial {report['selected_trial']})",
            fontsize=10,
        )
    path = folder / "search_progress.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    out.append(path)

    accepted = [e for e in search if e.accepted]
    effects = []
    try:
        import numpy as np

        for key in [p["key"] for p in manifest["space"]]:
            pts = [(e.values[key], e.score) for e in accepted if key in e.values]
            if len(pts) < 8 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v, _ in pts
            ):
                continue
            v = np.array([p[0] for p in pts], float)
            sc = np.array([p[1] for p in pts], float)
            if np.ptp(v) == 0:
                continue
            rho = np.corrcoef(v.argsort().argsort(), sc.argsort().argsort())[0, 1]
            effects.append((abs(float(rho)), key, v, sc))
    except ImportError:
        effects = []
    effects = sorted(effects, key=lambda t: -t[0])[:6]
    if effects:
        fig, axes = plt.subplots(
            2, 3, figsize=(7.5, 4.4), layout="constrained", squeeze=False
        )
        for ax, (_, key, v, sc) in zip(axes.ravel(), effects):
            ax.plot(v, sc, "o", ms=2.5, color="#0072B2", alpha=0.6)
            if v.min() > 0 and v.max() / v.min() > 50:
                ax.set_xscale("log")
            ax.set_title(key.rsplit(".", 1)[-1], fontsize=8)
            ax.grid(alpha=0.2)
        for ax in axes.ravel()[len(effects) :]:
            ax.set_visible(False)
        fig.supylabel("score")
        path = folder / "parameters.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        out.append(path)

    if report and report.get("selected_trial") is not None:
        base = next((e for e in rows if e.phase == "validation_baseline"), None)
        sel = next((e for e in rows if e.trial_id == report["selected_trial"]), None)
        if base and sel:
            metric = manifest["objective"]["metric"]
            cases = report["validation_cases"]
            constraints = []
            fig, ax = plt.subplots(
                figsize=(max(3.5, 1.2 * len(cases) + 2), 3), layout="constrained"
            )
            width = 0.38
            for j, (label, e, color) in enumerate(
                (("validation baseline", base, "#999999"), ("selected", sel, "#009E73"))
            ):
                vals = [
                    case_metrics(e, _case_key(c), metric, constraints) or 0
                    for c in cases
                ]
                xs = [i + (j - 0.5) * width for i in range(len(cases))]
                ax.bar(xs, vals, width, color=color, label=label)
                for x, v in zip(xs, vals):
                    ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(cases)), [str(c) for c in cases])
            ax.set_ylabel(metric)
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.2, axis="y")
            path = folder / "selected_vs_baseline.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            out.append(path)
    return out

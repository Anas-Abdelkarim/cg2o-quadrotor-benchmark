"""Human-readable live views of a study (the SQLite journal stays the source of truth).

trials.csv   one row appended and flushed after EVERY completed evaluation, so it can be opened (editor, Excel, pandas) while
             the study runs. Lines starting with '#' are comments: the command that started the study, and the meaning of the
             columns (read with `pandas.read_csv(path, comment="#")`). Columns:
               trial, time, phase, status (ok / REJECTED), score, fidelity,
               c<case>_<metric>   per case: the objective metric (mean over repeats) and every constrained metric (worst
                                  repeat), e.g. c30_iterations, c30_hit_fraction
               <parameter>        one column per tuned parameter (empty = the option is absent)
               reasons            why a trial was rejected
status       `tuning status <folder>`: command, counts, baseline, best so far with the changed parameters, latest evaluations.
"""

from __future__ import annotations
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import threading

from .model import Evaluation


def _short_names(keys):
    short = [k.rsplit(".", 1)[-1] for k in keys]
    return {k: (s if short.count(s) == 1 else k) for k, s in zip(keys, short)}


def metric_names(objective, constraints):
    """The metrics shown per case: the objective metric first, then every constrained metric (no duplicates)."""
    names = [objective.metric]
    for c in constraints:
        if c.metric not in names:
            names.append(c.metric)
    return names


def case_metrics(evaluation, case_key, name, constraints):
    """Mean over repeats for the objective metric, worst repeat for a constrained metric."""
    values = [
        r["metrics"][name]
        for r in evaluation.runs
        if r["case_key"] == case_key and name in r["metrics"]
    ]
    if not values:
        return None
    limit = [c for c in constraints if c.metric == name]
    if limit:
        return max(values) if limit[0].op == "<=" else min(values)
    return sum(values) / len(values)


class LiveTable:
    def __init__(self, path, space, objective, constraints, cases, header):
        self.path = Path(path)
        self.keys = list(space.keys)
        self.names = _short_names(self.keys)
        self.objective, self.constraints = objective, tuple(constraints)
        self.cases = list(cases)
        self.metrics = metric_names(objective, constraints)
        self.header = header
        self.lock = threading.Lock()
        self._start()

    def columns(self):
        cols = ["trial", "time", "phase", "status", "score", "fidelity"]
        for case in self.cases:
            cols += [f"c{case}_{m}" for m in self.metrics]
        return cols + [self.names[k] for k in self.keys] + ["reasons"]

    def _start(self):
        new = not self.path.exists() or self.path.stat().st_size == 0
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock, open(self.path, "a", newline="") as f:
            f.write(
                f"# command: {self.header or 'python API'} | {'started' if new else 'resumed'}: {stamp}\n"
            )
            if new:
                f.write(
                    "# score: lower is better; status REJECTED = a quality constraint failed (see reasons); phase: baseline | "
                    "search | validation_baseline | validation; c<case>_<metric>: objective metric = mean over repeats, "
                    "constrained metric = worst repeat\n"
                )
                csv.writer(f).writerow(self.columns())

    def row(self, e: Evaluation):
        row = [
            e.trial_id,
            datetime.now().strftime("%H:%M:%S"),
            e.phase,
            "ok" if e.accepted else "REJECTED",
            f"{e.score:.6g}" if e.score is not None and math.isfinite(e.score) else "",
            e.fidelity,
        ]
        for case in self.cases:
            key = json.dumps(case, sort_keys=True, separators=(",", ":"))
            for m in self.metrics:
                v = case_metrics(e, key, m, self.constraints)
                row.append("" if v is None else f"{v:.6g}")
        row += [e.values.get(k, "") for k in self.keys]
        row.append("; ".join(e.reasons))
        return row

    def append(self, e: Evaluation):
        with self.lock, open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(self.row(e))
            f.flush()

    def rewrite(self, evaluations):
        """Complete table from the journal (used at exit; keeps the comment lines of every run)."""
        with self.lock:
            comments = [
                ln for ln in self.path.read_text().splitlines() if ln.startswith("#")
            ]
            with open(self.path, "w", newline="") as f:
                for ln in comments:
                    f.write(ln + "\n")
                w = csv.writer(f)
                w.writerow(self.columns())
                for e in evaluations:
                    w.writerow(self.row(e))


def status_text(folder, last=10):
    from .storage import Store

    folder = Path(folder)
    if not (folder / "trials.sqlite3").is_file():
        raise ValueError("study database missing")
    manifest = json.loads((folder / "manifest.json").read_text())
    objective = manifest["objective"]["metric"]
    cases = manifest["settings"]["cases"]
    db = Store(folder)
    try:
        records = db.records()
        results = db.results()
    finally:
        db.close()
    out = []
    head = (
        next(
            (
                ln
                for ln in (folder / "trials.csv").read_text().splitlines()
                if ln.startswith("# command")
            ),
            "",
        )
        if (folder / "trials.csv").exists()
        else ""
    )
    if head:
        out.append(head)
    running = sum(r["state"] == "running" for r in records)
    abandoned = sum(r["state"] == "abandoned" for r in records)
    ok = [e for e in results if e.accepted]
    out.append(
        f"{len(results)} completed evaluations ({len(ok)} accepted, {len(results) - len(ok)} rejected)"
        + (f", {running} running now" if running else "")
        + (f", {abandoned} abandoned" if abandoned else "")
    )

    def per_case(e):
        txt = []
        for case in cases:
            key = json.dumps(case, sort_keys=True, separators=(",", ":"))
            vals = [
                r["metrics"][objective]
                for r in e.runs
                if r["case_key"] == key and objective in r["metrics"]
            ]
            txt.append(f"{case}: {sum(vals) / len(vals):.4g}" if vals else f"{case}: -")
        return "  ".join(txt)

    base = next((e for e in results if e.phase == "baseline"), None)
    if base:
        out.append(
            f"baseline: score {base.score if base.score is not None else float('nan'):.6g}  {objective} per case  {per_case(base)}"
            + ("" if base.accepted else "  (fails the quality gates)")
        )
    search = [e for e in results if e.phase == "search" and e.accepted]
    if search:
        best = min(search, key=lambda e: e.score)
        out.append(
            f"best so far:  trial {best.trial_id}  score {best.score:.6g}  {objective} per case  {per_case(best)}"
        )
        start = manifest.get("baseline", {})
        for k, v in best.values.items():
            if start.get(k) != v:
                out.append(f"    {k} = {v}   (baseline: {start.get(k, 'absent')})")
        for k in start:
            if k not in best.values:
                out.append(f"    {k} removed   (baseline: {start[k]})")
    if (folder / "report.json").exists():
        rep = json.loads((folder / "report.json").read_text())
        out.append(f"selection: {rep['status']}  (trial {rep['selected_trial']})")
    if last:
        out.append(f"\nlatest {min(last, len(results))} evaluations:")
        for e in results[-last:]:
            score = (
                f"{e.score:.4g}" if e.score is not None and e.accepted else "REJECTED"
            )
            line = f"  #{e.trial_id:<5} {e.phase:<20} {score:<9} {per_case(e)}"
            if not e.accepted:
                line += "   " + "; ".join(e.reasons)[:70]
            out.append(line)
    return "\n".join(out)

"""One configuration -> durable request, raw runs, quality checks and scalar score."""

from __future__ import annotations
import copy
import math
from .model import RunContext, RunResult, Evaluation
from .util import atomic_json, canonical, digest


class Evaluator:
    def __init__(
        self, space, experiment, objective, constraints, directory, settings, store
    ):
        self.space, self.experiment, self.objective = space, experiment, objective
        self.constraints, self.directory, self.settings, self.store = (
            constraints,
            directory,
            settings,
            store,
        )

    def evaluate(
        self,
        values,
        phase,
        fidelity,
        cases,
        repeats,
        baseline_runs=(),
        is_baseline=False,
    ):
        values = copy.deepcopy(values)
        domain_error = ""
        if not is_baseline:
            try:
                values = self.space.normalize(values)
            except ValueError as exc:
                domain_error = str(exc)
        # Common random numbers: all candidates and baseline get the same seed
        # for a given case/repeat; validation has a separate seed namespace.
        namespace = "validation" if phase.startswith("validation") else "screen"
        key = digest([values, fidelity, cases, repeats, namespace, is_baseline])
        request = {
            "values": values,
            "fidelity": fidelity,
            "cases": cases,
            "repeats": repeats,
            "cache_key": key,
            "baseline": is_baseline,
        }
        trial_id = self.store.reserve(phase, request)
        if domain_error:
            evaluation = Evaluation(
                trial_id, phase, values, fidelity, (), None, (domain_error,)
            )
            self.store.complete(evaluation)
            return evaluation
        cached = (
            self.store.cached(key) if self.settings.cache and not is_baseline else None
        )
        if cached:
            result = Evaluation(
                trial_id,
                phase,
                values,
                fidelity,
                tuple(copy.deepcopy(cached.runs)),
                cached.score,
                tuple(cached.reasons),
                cached.trial_id,
            )
            self.store.complete(result)
            return result
        runs, reasons = [], []
        directory = self.directory / "artifacts" / f"trial-{trial_id:06d}"
        atomic_json(directory / "request.json", request)
        for ci, case in enumerate(cases):
            for rep in range(repeats):
                seed = int(
                    digest([self.settings.seed, namespace, case, rep])[:8], 16
                ) % (2**31)
                run_dir = directory / f"case-{ci:03d}" / f"repeat-{rep:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                ctx = RunContext(
                    copy.deepcopy(case), fidelity, seed, run_dir, is_baseline
                )
                try:
                    value = self.experiment.run(copy.deepcopy(values), ctx)
                    result = (
                        value
                        if isinstance(value, RunResult)
                        else RunResult(dict(value))
                    )
                    error = result.validate()
                except Exception as exc:
                    result = RunResult(error=f"{type(exc).__name__}: {exc}")
                    error = result.error
                metrics = {
                    k: float(v)
                    for k, v in result.metrics.items()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                }
                row = {
                    "case": case,
                    "case_key": canonical(case),
                    "repeat": rep,
                    "seed": seed,
                    "metrics": metrics,
                    "error": error,
                    "directory": str(run_dir.relative_to(self.directory)),
                }
                runs.append(row)
                checks = (
                    [error]
                    if error
                    else [c.violation(metrics) for c in self.constraints]
                )
                reasons.extend(f"case={case} repeat={rep}: {x}" for x in checks if x)
                atomic_json(run_dir / "metrics.json", row)
        score = None
        # Measure objective even if a gate fails: useful diagnostics/normalization,
        # but accepted remains False and invalid baselines can never be selected.
        try:
            if any(r["error"] for r in runs):
                raise ValueError("one or more experiment runs failed")
            score = self.objective.score(runs, baseline_runs or runs)
        except (KeyError, ValueError, ZeroDivisionError, OverflowError) as exc:
            reasons.append(f"objective: {exc}")
        evaluation = Evaluation(
            trial_id, phase, values, fidelity, tuple(runs), score, tuple(reasons)
        )
        self.store.complete(evaluation)
        return evaluation

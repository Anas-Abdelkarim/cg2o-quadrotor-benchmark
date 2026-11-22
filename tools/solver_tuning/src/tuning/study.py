"""Experiment-independent orchestration, quality checks and validated selection."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
from .model import Objective
from .evaluator import Evaluator
from .storage import Store, exclusive_study
from .tables import LiveTable
from .strategies import make_strategy
from .util import atomic_json, canonical, digest, file_hash


@dataclass(frozen=True)
class Settings:
    cases: tuple = (0,)
    validation_cases: tuple = ()
    screening_fidelity: int = 200
    validation_fidelity: int = 0
    repeats: int = 1
    validation_repeats: int = 2
    validate_top: int = 5
    jobs: int = 1
    seed: int = 1
    cache: bool = True
    strategy_options: dict | None = None

    def __post_init__(self):
        for key in (
            "screening_fidelity",
            "validation_fidelity",
            "repeats",
            "validation_repeats",
            "validate_top",
            "jobs",
            "seed",
        ):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if not self.cases or len({canonical(c) for c in self.cases}) != len(self.cases):
            raise ValueError("nonempty unique screening cases required")
        if min(self.screening_fidelity, self.validation_fidelity) < 0:
            raise ValueError("fidelity must be nonnegative (0=full)")
        if min(self.repeats, self.validation_repeats, self.validate_top, self.jobs) < 1:
            raise ValueError("repeats, validate_top and jobs must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


class Study:
    def __init__(
        self,
        space,
        experiment,
        objective: Objective,
        constraints=(),
        *,
        directory,
        settings=None,
        baseline=None,
        method="tpe",
        strategy_factory=None,
        strategy_revision="",
    ):
        self.space, self.experiment, self.objective = space, experiment, objective
        self.constraints = tuple(constraints)
        self.directory = Path(directory).resolve()
        self.settings = settings or Settings()
        self.method = method
        self.strategy_factory = strategy_factory
        self.strategy_revision = strategy_revision
        self.command = ""  # text of the command line, shown in the first line of trials.csv (set by the CLI)
        self.table = None
        self.start = copy.deepcopy(
            experiment.baseline(space) if baseline is None else baseline
        )
        if set(self.start) - set(space.keys):
            raise ValueError("baseline contains unknown parameter keys")
        # No clipping or default insertion here: baseline is exactly as supplied.
        canonical(self.start)
        self.validation_cases = tuple(
            json.loads(c)
            for c in dict.fromkeys(
                canonical(c)
                for c in (*self.settings.cases, *self.settings.validation_cases)
            )
        )
        if experiment.isolated_timing:
            if self.settings.jobs != 1:
                raise ValueError(
                    "timing experiments require jobs=1 (no overlapping simulations)"
                )
            if self.settings.cache:
                raise ValueError(
                    "timing experiments require cache=False for fresh measurements"
                )
        self.store = None

    def _manifest(self):
        versions = {}
        for p in ("PyYAML", "optuna", "numpy"):
            try:
                versions[p] = importlib.metadata.version(p)
            except importlib.metadata.PackageNotFoundError:
                pass
        source = Path(__file__).parent
        code = digest(
            {
                str(p.relative_to(source)): file_hash(p)
                for p in sorted(source.rglob("*.py"))
            }
        )
        custom = None
        if self.strategy_factory is not None:
            import inspect

            fn = self.strategy_factory
            source_file = inspect.getsourcefile(fn)
            if source_file is None and not self.strategy_revision:
                raise ValueError(
                    "custom strategy needs inspectable source or an explicit revision"
                )
            custom = {
                "name": fn.__module__ + ":" + fn.__qualname__,
                "source": file_hash(source_file) if source_file else None,
                "revision": self.strategy_revision,
            }
        return {
            "custom_strategy": custom,
            "schema": 2,
            "package_code": code,
            "python": platform.python_version(),
            "dependencies": versions,
            "space": self.space.to_dict(),
            "relations": [asdict(r) for r in self.space.relations],
            "experiment": self.experiment.fingerprint(),
            "baseline": self.start,
            "objective": asdict(self.objective),
            "constraints": [asdict(c) for c in self.constraints],
            "settings": asdict(self.settings),
            "method": self.method,
        }

    def _open(self, resume):
        manifest = self._manifest()
        path = self.directory / "manifest.json"
        if path.exists():
            if not resume:
                raise ValueError("study exists; explicitly use resume=True / --resume")
            if canonical(json.loads(path.read_text())) != canonical(manifest):
                raise ValueError(
                    "resume incompatible: inputs, code, dependencies, space, scoring, cases or settings changed; create a new study"
                )
        else:
            if resume:
                raise ValueError("cannot resume: manifest missing")
            if (self.directory / "trials.sqlite3").exists():
                raise ValueError("database exists without manifest")
            atomic_json(path, manifest)
        self.store = Store(self.directory)
        self.store.recover()
        # live view: one CSV row per completed evaluation, flushed at once
        self.table = LiveTable(
            self.directory / "trials.csv",
            self.space,
            self.objective,
            self.constraints,
            self.validation_cases,
            self.command,
        )
        self.store.on_complete = self.table.append
        self.evaluator = Evaluator(
            self.space,
            self.experiment,
            self.objective,
            self.constraints,
            self.directory,
            self.settings,
            self.store,
        )

    def _evaluate(self, *args, **kwargs):
        return self.evaluator.evaluate(*args, **kwargs)

    def _phase_baseline(self, phase, fidelity, cases, repeats):
        previous = self.store.results(phase)
        # Reuse the original complete measurement on resume, not a new denominator.
        return (
            previous[0]
            if previous
            else self._evaluate(
                self.start, phase, fidelity, cases, repeats, is_baseline=True
            )
        )

    def run(self, trials=100, *, resume=False, validate=True, max_seconds=None):
        if trials < 0:
            raise ValueError("trials must be nonnegative")
        if max_seconds is not None and (
            not math.isfinite(max_seconds) or max_seconds <= 0
        ):
            raise ValueError("max_seconds must be positive and finite")
        self.directory.mkdir(parents=True, exist_ok=True)
        with exclusive_study(self.directory):
            self._open(resume)
            try:
                s = self.settings
                self._write_report(None, "running; selection pending")
                strategy = (
                    self.strategy_factory(
                        self.space, s.seed, **(s.strategy_options or {})
                    )
                    if self.strategy_factory
                    else make_strategy(
                        self.method, self.space, s.seed, s.strategy_options
                    )
                )
                baseline = self._phase_baseline(
                    "baseline", s.screening_fidelity, s.cases, s.repeats
                )
                # Existing complete searches are used as training data, not mutated.
                history = self.store.results("search")
                issued = len(self.store.records("search"))
                t0 = time.monotonic()
                try:
                    while issued < trials and (
                        max_seconds is None or time.monotonic() - t0 < max_seconds
                    ):
                        batch = min(s.jobs, trials - issued)
                        if strategy.batch_limit:
                            batch = min(batch, strategy.batch_limit)
                        proposals = [
                            strategy.ask(issued + i, history, self.start)
                            for i in range(batch)
                        ]
                        with ThreadPoolExecutor(max_workers=s.jobs) as pool:
                            futures = [
                                pool.submit(
                                    self._evaluate,
                                    p.values,
                                    "search",
                                    s.screening_fidelity,
                                    s.cases,
                                    s.repeats,
                                    baseline.runs,
                                )
                                for p in proposals
                            ]
                            # Controller owns search state; worker exceptions propagate.
                            for p, future in zip(proposals, futures):
                                result = future.result()
                                strategy.tell(p, result)
                                history.append(result)
                                print(
                                    f"trial {result.trial_id}: "
                                    + (
                                        f"score={result.score:.6g}"
                                        if result.accepted
                                        else "REJECTED "
                                        + "; ".join(result.reasons)[:180]
                                    ),
                                    flush=True,
                                )
                        issued += batch
                except KeyboardInterrupt:
                    atomic_json(
                        self.directory / "interrupted.json",
                        {
                            "message": "search interrupted; completed trials retained; resume to continue"
                        },
                    )
                    self._write_report(None, "interrupted; no new winner exported")
                    return None
                return (
                    self._validate(baseline, history)
                    if validate
                    else self._screen_report(history, baseline)
                )
            finally:
                self._export_tables()
                self.store.close()
                self.store = None

    def _screen_report(self, history, baseline):
        accepted = [e for e in [baseline, *history] if e.accepted]
        best = min(accepted, key=lambda e: e.score) if accepted else None
        self._write_report(best, "screening only; not validated", export=False)
        return best

    def _validate(self, baseline, history):
        s = self.settings
        base = self._phase_baseline(
            "validation_baseline",
            s.validation_fidelity,
            self.validation_cases,
            s.validation_repeats,
        )
        distinct = {}
        for e in sorted((e for e in history if e.accepted), key=lambda e: e.score):
            if e.values != self.start:
                distinct.setdefault(canonical(e.values), e)
        finalists = list(distinct.values())[: s.validate_top]
        contenders = [base]
        # Finalists are independent measurements: validate them concurrently (at most `jobs` simulations at a time).
        with ThreadPoolExecutor(max_workers=s.jobs) as pool:
            futures = [
                pool.submit(
                    self._evaluate,
                    e.values,
                    "validation",
                    s.validation_fidelity,
                    self.validation_cases,
                    s.validation_repeats,
                    base.runs,
                )
                for e in finalists
            ]
            contenders.extend(f.result() for f in futures)
        good = [e for e in contenders if e.accepted]
        winner = min(good, key=lambda e: e.score) if good else None
        self._write_report(
            winner,
            "validated" if winner else "no configuration passed validation",
            export=True,
        )
        return winner

    def _write_report(self, winner, status, export=False):
        # Clear stale exports before reporting, so an older selection cannot look current.
        for p in (
            "best_config.json",
            "selected_parameters.json",
            "selected_config.yaml",
        ):
            (self.directory / p).unlink(missing_ok=True)
        value = {
            "status": status,
            "selected_trial": winner.trial_id if winner else None,
            "objective": asdict(self.objective),
            "winner": winner.to_dict() if winner else None,
            "validation_cases": self.validation_cases,
            "limitation": "Best among evaluated finalists under these criteria; not a global optimum or independent scenario validation.",
        }
        atomic_json(self.directory / "report.json", value)
        if winner and export:
            self.experiment.export(copy.deepcopy(winner.values), self.directory)
            atomic_json(self.directory / "best_config.json", winner.values)
        lines = [
            status,
            f"method: {self.method}",
            f"objective: {self.objective.metric} ({self.objective.mode})",
            f"validation cases: {self.validation_cases}",
        ]
        if winner:
            lines += [
                f"selected trial: {winner.trial_id}",
                f"score: {winner.score:.8g}",
                json.dumps(winner.values, indent=2),
            ]
        lines += [value["limitation"]]
        (self.directory / "report.txt").write_text("\n".join(lines) + "\n")

    def _export_tables(self):
        if not self.store:
            return
        evaluations = self.store.results()
        with open(self.directory / "log.jsonl", "w") as f:
            for e in evaluations:
                f.write(canonical(e.to_dict()) + "\n")
        if self.table is not None:
            self.table.rewrite(evaluations)

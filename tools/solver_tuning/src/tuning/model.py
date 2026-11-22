from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
import statistics
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunContext:
    case: Any
    fidelity: int
    seed: int
    directory: Path
    baseline: bool = False


@dataclass(frozen=True)
class RunResult:
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""

    def validate(self):
        if self.error:
            return self.error
        if not self.metrics:
            return "empty metrics"
        for key, v in self.metrics.items():
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            ):
                return f"{key}: nonfinite or nonnumeric metric"
        return ""


@dataclass(frozen=True)
class Constraint:
    metric: str
    limit: float
    op: str = "<="

    def __post_init__(self):
        if self.op not in ("<=", ">=") or not math.isfinite(self.limit):
            raise ValueError("constraint requires <= or >= and finite limit")

    def violation(self, metrics):
        v = metrics.get(self.metric)
        if v is None or not math.isfinite(v):
            return f"{self.metric}: unavailable"
        good = v <= self.limit if self.op == "<=" else v >= self.limit
        return "" if good else f"{self.metric}={v:g} violates {self.op}{self.limit:g}"


@dataclass(frozen=True)
class Objective:
    metric: str
    mode: str = "raw"
    direction: str = "minimize"
    penalties: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ("raw", "relative") or self.direction not in (
            "minimize",
            "maximize",
        ):
            raise ValueError("invalid objective mode/direction")
        if any(not math.isfinite(v) or v < 0 for v in self.penalties.values()):
            raise ValueError("penalty weights must be finite and nonnegative")

    def score(self, runs, baseline_runs):
        parts = []
        for case in sorted({r["case_key"] for r in runs}):
            group = [r["metrics"] for r in runs if r["case_key"] == case]
            val = statistics.mean(m[self.metric] for m in group)
            if self.mode == "relative":
                base = [
                    r["metrics"][self.metric]
                    for r in baseline_runs
                    if r["case_key"] == case and self.metric in r["metrics"]
                ]
                if not base or statistics.mean(base) <= 0:
                    raise ValueError(
                        f"{case}: relative objective needs a positive measured baseline"
                    )
                val /= statistics.mean(base)
            sign = 1 if self.direction == "minimize" else -1
            parts.append(
                sign * val
                + sum(
                    w * statistics.mean(m[k] for m in group)
                    for k, w in self.penalties.items()
                )
            )
        score = statistics.mean(parts)
        if not math.isfinite(score):
            raise ValueError("nonfinite objective")
        return score


@dataclass(frozen=True)
class Evaluation:
    trial_id: int
    phase: str
    values: dict
    fidelity: int
    runs: tuple[dict, ...]
    score: float | None
    reasons: tuple[str, ...]
    cached_from: int | None = None

    @property
    def accepted(self):
        return self.score is not None and not self.reasons and math.isfinite(self.score)

    def to_dict(self):
        return {**asdict(self), "accepted": self.accepted}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k != "accepted"})

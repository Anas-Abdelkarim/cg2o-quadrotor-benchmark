from __future__ import annotations
import json
import math
import random
import threading
import time
import pytest
from tuning import (
    Experiment,
    Study,
    Space,
    Float,
    Log,
    Int,
    Cat,
    OneMinusLog,
    Objective,
    Constraint,
    Settings,
    Evaluation,
    Relation,
)
from tuning.storage import Store, exclusive_study
from tuning.strategies import DifferentialEvolution, make_strategy


class Quadratic(Experiment):
    def __init__(self, bad=False, revision="1", fail_case=None):
        self.bad, self.revision, self.fail_case = bad, revision, fail_case
        self.contexts = []

    def fingerprint(self):
        return {
            "type": "quadratic-test",
            "bad": self.bad,
            "revision": self.revision,
            "fail_case": self.fail_case,
        }

    def run(self, v, c):
        self.contexts.append(c)
        return {
            "loss": 1 + (v["x"] - 1) ** 2,
            "violation": 1.0 if self.bad or c.case == self.fail_case else 0.0,
        }


def study(tmp_path, experiment=None, settings=None, method="random", baseline=None):
    return Study(
        Space([Float("x", -2, 2, default=0)]),
        experiment or Quadratic(),
        Objective("loss", mode="relative"),
        [Constraint("violation", 0)],
        directory=tmp_path,
        settings=settings or Settings(cases=("train",), validation_repeats=1, jobs=1),
        method=method,
        baseline=baseline,
    )


@pytest.mark.parametrize(
    "param",
    [
        Float("p", -3, 8),
        Log("p", 1e-8, 1e4),
        Int("p", -2, 10),
        Cat("p", ["yes", "no"]),
        OneMinusLog("p", 1e-4, 0.1),
    ],
)
def test_sampling_bounds_and_grids(param):
    rng = random.Random(14)
    for _ in range(200):
        param.validate(param.sample(rng))
    for value in param.candidates():
        param.validate(value)


def test_conditions_and_unset():
    space = Space(
        [
            Cat("mode", ["a", "b"], default="a"),
            Log("extra", 0.01, 1, optional=True, active_if={"mode": ["b"]}),
        ]
    )
    assert space.normalize({"mode": "a", "extra": 0.5}) == {"mode": "a"}
    assert space.normalize({"mode": "b", "extra": None}) == {"mode": "b"}
    assert space.normalize({"mode": "b", "extra": 0.1}) == {"mode": "b", "extra": 0.1}


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: Log("x", 0, 1),
        lambda: Float("x", 1, 1),
        lambda: Space([Float("x", 0, 1), Float("x", 0, 1)]),
        lambda: Space([Float("x", 0, 1, active_if={"later": ["x"]})]),
    ],
)
def test_invalid_domains_rejected(constructor):
    with pytest.raises(ValueError):
        constructor()


def test_no_clipped_baseline(tmp_path):
    s = study(tmp_path, baseline={"x": 50})
    s.run(0)
    baseline = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert baseline["values"] == {"x": 50}


def test_invalid_baseline_never_exported(tmp_path):
    winner = study(tmp_path, Quadratic(bad=True)).run(5)
    assert winner is None
    assert not (tmp_path / "best_config.json").exists()
    assert (
        json.loads((tmp_path / "report.json").read_text())["status"]
        == "no configuration passed validation"
    )
    assert all(
        not row["accepted"]
        for row in map(json.loads, (tmp_path / "log.jsonl").read_text().splitlines())
    )


def test_validation_adds_cases(tmp_path):
    q = Quadratic(fail_case="holdout")
    s = study(
        tmp_path,
        q,
        Settings(cases=("train",), validation_cases=("holdout",), validation_repeats=1),
    )
    assert s.run(4) is None
    validation = [c for c in q.contexts if c.fidelity == 0]
    assert {c.case for c in validation} == {"train", "holdout"}


def test_repeat_gate_is_not_averaged(tmp_path):
    class Once(Quadratic):
        def run(self, v, c):
            return {"loss": 1.0, "violation": 1.0 if c.seed % 2 else 0.0}

    # A strict raw-run gate, with an unconditional explicit bad repeat for certainty.
    class Worse(Quadratic):
        def run(self, v, c):
            return {
                "loss": 1.0,
                "violation": 1.0 if "repeat-001" in str(c.directory) else 0.0,
            }

    s = study(
        tmp_path, Worse(), Settings(cases=("a",), repeats=2, validation_repeats=2)
    )
    assert s.run(2) is None


def test_resume_fixed_baseline_and_budget(tmp_path):
    s = study(tmp_path)
    s.run(3, validate=False)
    old = (tmp_path / "log.jsonl").read_text().splitlines()[0]
    study(tmp_path).run(6, resume=True)
    db = Store(tmp_path)
    try:
        assert len(db.records("baseline")) == 1
        assert len(db.records("search")) == 6
    finally:
        db.close()
    assert (tmp_path / "log.jsonl").read_text().splitlines()[0] == old


def test_resume_refuses_changed_inputs(tmp_path):
    study(tmp_path).run(1)
    with pytest.raises(ValueError, match="resume incompatible"):
        study(tmp_path, Quadratic(revision="2")).run(2, resume=True)


def test_failed_run_not_cached_and_nan_rejected(tmp_path):
    class Nonfinite(Quadratic):
        def run(self, v, c):
            return {"loss": float("nan"), "violation": 0.0}

    assert study(tmp_path, Nonfinite()).run(1) is None


def test_cache_returns_new_object(tmp_path):
    db = Store(tmp_path)
    i = db.reserve("search", {"cache_key": "key"})
    db.complete(
        Evaluation(
            i, "search", {"x": 1}, 1, ({"metrics": {"loss": 1}, "error": ""},), 1, ()
        )
    )
    one = db.cached("key")
    one.values["x"] = 100
    two = db.cached("key")
    assert two.values == {"x": 1}
    db.close()


def test_crash_recovery_marks_abandoned(tmp_path):
    db = Store(tmp_path)
    db.reserve("search", {"values": {"x": 1}})
    db.recover()
    assert db.records()[0]["state"] == "abandoned"
    db.close()


def test_one_global_concurrency_budget(tmp_path):
    class Count(Quadratic):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def run(self, v, c):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.005)
            with self.lock:
                self.active -= 1
            return super().run(v, c)

    q = Count()
    study(tmp_path, q, Settings(cases=(1, 2, 3), jobs=2, validation_repeats=1)).run(8)
    assert q.peak <= 2


def test_timing_requires_isolation(tmp_path):
    q = Quadratic()
    q.isolated_timing = True
    with pytest.raises(ValueError, match="jobs=1"):
        study(tmp_path, q, Settings(jobs=2))


def test_single_controller_lock(tmp_path):
    with exclusive_study(tmp_path):
        with pytest.raises(RuntimeError):
            with exclusive_study(tmp_path):
                pass


@pytest.mark.parametrize("method", ["random", "coordinate", "de"])
def test_strategies_end_to_end(tmp_path, method):
    result = study(tmp_path, method=method).run(30)
    assert result.accepted
    assert result.score <= 1


def test_de_rejects_categorical():
    with pytest.raises(ValueError, match="numeric"):
        DifferentialEvolution(Space([Cat("x", ["a", "b"])]), 1)


def test_tpe_rejection_has_no_finite_magic_penalty(tmp_path):
    pytest.importorskip("optuna")
    strategy = make_strategy(
        "tpe", Space([Float("x", -2, 2, default=0)]), 10, {"startup": 3}
    )
    history = []
    for i in range(8):
        p = strategy.ask(i, history, {"x": 0})
        ev = Evaluation(
            i,
            "search",
            p.values,
            1,
            (),
            100.0 if i % 2 else None,
            () if i % 2 else ("bad",),
        )
        strategy.tell(p, ev)
        history.append(ev)
    records = strategy.study.trials
    assert any(t.value == 100 for t in records)
    assert all(
        math.isinf(t.value) for t in records if t.user_attrs["constraints"] == [1.0]
    )


def test_tpe_resume_can_continue(tmp_path):
    pytest.importorskip("optuna")
    study(tmp_path, method="tpe").run(4, validate=False)
    assert study(tmp_path, method="tpe").run(8, resume=True).accepted


def test_resume_random_matches_uninterrupted(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    study(one).run(8)
    study(two).run(3, validate=False)
    study(two).run(8, resume=True)
    a, b = Store(one), Store(two)
    try:
        assert [e.values for e in a.results("search")] == [
            e.values for e in b.results("search")
        ]
    finally:
        a.close()
        b.close()


def test_tpe_resume_matches_uninterrupted_after_startup(tmp_path):
    pytest.importorskip("optuna")
    opts = Settings(
        cases=("train",), validation_repeats=1, strategy_options={"startup": 3}
    )
    one = tmp_path / "one"
    two = tmp_path / "two"
    study(one, method="tpe", settings=opts).run(12, validate=False)
    study(two, method="tpe", settings=opts).run(6, validate=False)
    study(two, method="tpe", settings=opts).run(12, resume=True, validate=False)
    a, b = Store(one), Store(two)
    try:
        assert [e.values for e in a.results("search")] == [
            e.values for e in b.results("search")
        ]
    finally:
        a.close()
        b.close()


def test_parameter_relation_rejects_before_simulation(tmp_path):
    s = Study(
        Space(
            [Int("lo", 1, 5, default=1), Int("hi", 1, 5, default=5)],
            relations=[Relation("lo", "hi")],
        ),
        Quadratic(),
        Objective("loss"),
        directory=tmp_path,
        method="random",
    )
    s.directory.mkdir(exist_ok=True)
    s._open(False)
    try:
        ev = s._evaluate({"lo": 5, "hi": 1}, "search", 1, (0,), 1)
        assert not ev.accepted
        assert not ev.runs
        assert "relation" in ev.reasons[0]
    finally:
        s.store.close()

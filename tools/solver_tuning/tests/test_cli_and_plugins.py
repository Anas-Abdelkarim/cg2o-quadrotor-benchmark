import json
from pathlib import Path
import yaml
import pytest
from tuning import Study, Space, Float, Objective, Settings
from tuning.cli import main
from tuning.experiment import Experiment
from tuning.strategies import Proposal


class Demo(Experiment):
    def fingerprint(self):
        return {"example": 1}

    def run(self, values, context):
        return {"loss": 1 + values["x"] ** 2}


class Custom:
    batch_limit = 1

    def __init__(self, space, seed, **options):
        self.space = space

    def ask(self, index, history, start):
        return Proposal({"x": 0.0})

    def tell(self, proposal, result):
        pass


def test_custom_strategy_and_cache(tmp_path):
    study = Study(
        Space([Float("x", -1, 1, default=0.9)]),
        Demo(),
        Objective("loss"),
        directory=tmp_path,
        method="custom",
        strategy_factory=Custom,
        settings=Settings(validation_repeats=1),
    )
    result = study.run(3)
    assert result.accepted
    assert result.values == {"x": 0.0}
    rows = [json.loads(s) for s in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(r["cached_from"] for r in rows)
    assert len({r["trial_id"] for r in rows}) == len(rows)


def test_cli_function_run_status_and_bad_resume(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    cfg = yaml.safe_load((examples / "toy.yaml").read_text())
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    assert main(["run", str(config), "--out", str(out), "--trials", "4", "--no-stamp"]) == 0
    assert main(["status", str(out)]) == 0
    cfg["settings"]["seed"] = 999
    config.write_text(yaml.safe_dump(cfg))
    assert (
        main(["run", str(config), "--out", str(out), "--trials", "6", "--resume"]) == 2
    )


def test_cli_new_run_stamps_out_folder(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    config = tmp_path / "study.yaml"
    config.write_text((examples / "toy.yaml").read_text())
    assert main(["run", str(config), "--out", str(tmp_path / "foo"), "--trials", "2", "--screen-only"]) == 0
    (stamped,) = [p for p in tmp_path.iterdir() if p.name.endswith("_foo")]
    assert not (tmp_path / "foo").exists()
    assert len(stamped.name) == len("20261027_153000_foo")
    assert main(["run", str(config), "--out", str(stamped), "--trials", "3", "--resume", "--screen-only"]) == 0
    assert [p for p in tmp_path.iterdir() if p.name.endswith("_foo")] == [stamped]


def test_worker_error_propagates_without_false_export(tmp_path):
    class Broken(Custom):
        def tell(self, proposal, result):
            raise RuntimeError("broken strategy")

    s = Study(
        Space([Float("x", -1, 1, default=0.9)]),
        Demo(),
        Objective("loss"),
        directory=tmp_path,
        strategy_factory=Broken,
        method="broken",
    )
    with pytest.raises(RuntimeError, match="broken strategy"):
        s.run(2)
    assert not (tmp_path / "best_config.json").exists()
    assert (tmp_path / "trials.sqlite3").exists()

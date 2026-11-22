from __future__ import annotations
import subprocess
import sys
import pytest
from tuning import RunContext
from tuning.adapters.command import CommandExperiment
from tuning.adapters.process import execute


def test_json_program_adapter(tmp_path):
    script = tmp_path / "program.py"
    script.write_text(
        'import json,sys\nr=json.load(open(sys.argv[1]))\njson.dump({"metrics":{"loss":r["parameters"]["x"]**2}},open(sys.argv[2],"w"))\n'
    )
    adapter = CommandExperiment(
        [sys.executable, str(script), "{request}", "{result}"], inputs=[script]
    )
    directory = tmp_path / "trial"
    directory.mkdir()
    ctx = RunContext("test", 0, 1, directory)
    result = adapter.run({"x": 3}, ctx)
    assert result.metrics == {"loss": 9}
    assert (directory / "stdout.log").exists()


def test_external_timeout(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        execute([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, 0.05)
    assert (tmp_path / "stderr.log").exists()

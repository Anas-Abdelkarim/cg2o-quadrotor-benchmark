"""Any external program: read request JSON, write result JSON with metrics."""

from __future__ import annotations
from pathlib import Path
import json
import shutil
from ..experiment import Experiment
from ..model import RunResult
from ..util import atomic_json, file_hash
from .process import execute


class CommandExperiment(Experiment):
    def __init__(
        self, command, *, baseline=None, timeout=300, inputs=(), revision="", env=None
    ):
        if not command:
            raise ValueError("command must not be empty")
        self.command = list(command)
        binary = shutil.which(str(self.command[0]))
        if not binary:
            raise ValueError(f"command not found: {self.command[0]}")
        self.command[0] = str(Path(binary).resolve())
        self.start, self.timeout, self.revision = baseline, timeout, revision
        self.inputs = [Path(p).resolve() for p in inputs]
        self.env = env or {}

    def fingerprint(self):
        return {
            "adapter": "command",
            "command": self.command,
            "timeout": self.timeout,
            "revision": self.revision,
            "inputs": {
                str(p): file_hash(p) for p in [Path(self.command[0]), *self.inputs]
            },
            "env": self.env,
        }

    def baseline(self, space):
        return dict(self.start) if self.start is not None else super().baseline(space)

    def run(self, values, context):
        req = context.directory / "input.json"
        result = context.directory / "output.json"
        atomic_json(
            req,
            {
                "parameters": values,
                "case": context.case,
                "fidelity": context.fidelity,
                "seed": context.seed,
                "baseline": context.baseline,
            },
        )
        result.unlink(missing_ok=True)
        cmd = [
            part.replace("{request}", str(req)).replace("{result}", str(result))
            for part in self.command
        ]
        atomic_json(context.directory / "command.json", cmd)
        execute(cmd, context.directory, self.timeout, self.env)
        data = json.loads(result.read_text())
        return RunResult(data.get("metrics", {}), data.get("error", ""))

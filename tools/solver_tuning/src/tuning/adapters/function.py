"""Adapt an importable Python function(values, context) returning metrics."""

from __future__ import annotations
import importlib
import inspect
from pathlib import Path
from ..experiment import Experiment
from ..model import RunResult
from ..util import file_hash


class PythonFunction(Experiment):
    def __init__(self, entrypoint, baseline=None, revision="", inputs=()):
        self.entrypoint, self.start, self.revision = entrypoint, baseline, revision
        self.inputs = [Path(p).resolve() for p in inputs]
        module, name = entrypoint.split(":", 1)
        self.function = getattr(importlib.import_module(module), name)

    def fingerprint(self):
        source = inspect.getsourcefile(self.function)
        if not source and not self.revision:
            raise ValueError(
                "function without a source file needs an explicit revision"
            )
        return {
            "adapter": "python",
            "entrypoint": self.entrypoint,
            "revision": self.revision,
            "source": file_hash(source) if source else None,
            "inputs": {str(p): file_hash(p) for p in self.inputs},
        }

    def baseline(self, space):
        return dict(self.start) if self.start is not None else super().baseline(space)

    def run(self, values, context):
        value = self.function(values, context)
        return value if isinstance(value, RunResult) else RunResult(dict(value))

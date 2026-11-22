from __future__ import annotations
from abc import ABC, abstractmethod
from .model import RunContext, RunResult


class Experiment(ABC):
    """Implement a thread-safe run(). Each call receives a unique artifact directory.

    fingerprint() must describe every external input influencing a run. Resume
    refuses a different fingerprint. Export must not modify the input experiment.
    """

    isolated_timing = False

    @abstractmethod
    def fingerprint(self) -> dict: ...

    @abstractmethod
    def run(self, values: dict, context: RunContext) -> RunResult: ...

    def baseline(self, space):
        return space.defaults()

    def export(self, values, directory):
        from .util import atomic_json

        atomic_json(directory / "selected_parameters.json", values)

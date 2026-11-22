"""Shared subprocess boundary: no shell interpolation, logs and timeout cleanup."""

from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess


def execute(command, directory, timeout, env=None):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (
        open(directory / "stdout.log", "w") as stdout,
        open(directory / "stderr.log", "w") as stderr,
    ):
        proc = subprocess.Popen(
            list(map(str, command)),
            cwd=directory,
            stdout=stdout,
            stderr=stderr,
            env={**os.environ, **(env or {})},
            start_new_session=os.name == "posix",
        )
        try:
            code = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.wait()
            raise
    if code != 0:
        raise RuntimeError(
            f"external command exited {code}; see stdout.log / stderr.log"
        )

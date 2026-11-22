from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import re
import shlex
import signal
import sys
import yaml
from . import Space, Relation, Constraint, Objective, Settings, Study


def load_study(path, directory):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    space = Space.from_dict(
        data["parameters"], [Relation(**r) for r in data.get("relations", [])]
    )
    adapter = dict(data["experiment"])
    kind = adapter.pop("type")

    def resolve(value):
        return str((path.parent / value).resolve())

    if kind == "python":
        from .adapters.function import PythonFunction

        adapter["inputs"] = [resolve(p) for p in adapter.get("inputs", [])]
        experiment = PythonFunction(**adapter)
    elif kind == "command":
        from .adapters.command import CommandExperiment

        adapter["inputs"] = [resolve(p) for p in adapter.get("inputs", [])]
        experiment = CommandExperiment(**adapter)
    else:
        raise ValueError("experiment type must be python or command")
    settings = Settings(**data.get("settings", {}))
    return Study(
        space,
        experiment,
        Objective(**data["objective"]),
        [Constraint(**c) for c in data.get("constraints", [])],
        directory=directory,
        settings=settings,
        baseline=data.get("baseline"),
        method=data.get("method", "tpe"),
    )


def _stamped(out):
    """<parent>/<YYYYmmdd_HHMMSS>_<name> for a new study; a name that already starts with a stamp is kept."""
    out = Path(out)
    if re.match(r"\d{8}_\d{6}_", out.name):
        return out
    return out.with_name(f"{datetime.now():%Y%m%d_%H%M%S}_{out.name}")


def _stop(signum, frame):
    # Also for background jobs (nohup / &), which would otherwise ignore SIGINT: the study stops scheduling new work, keeps
    # every completed evaluation and writes an "interrupted" report. `run --finish` then validates what was found.
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description="General constrained parameter tuning")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume a reproducible study")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="study folder; a new run prefixes its name with a time stamp (results/tuning/foo -> "
        "results/tuning/20261027_153000_foo); --resume/--finish take the stamped folder as it is",
    )
    run.add_argument(
        "--no-stamp", action="store_true", help="use --out exactly as given for a new run"
    )
    run.add_argument(
        "--trials",
        type=int,
        default=100,
        help="total search-attempt budget including previous attempts",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--finish",
        action="store_true",
        help="no further search: validate the best finalists found so far and select (implies --resume)",
    )
    run.add_argument("--screen-only", action="store_true")
    run.add_argument(
        "--max-seconds",
        type=float,
        help="search scheduling budget; baseline and validation are additional",
    )
    status = sub.add_parser(
        "status", help="live summary of a running or finished study (reads the journal)"
    )
    status.add_argument("folder", type=Path)
    status.add_argument(
        "--last", type=int, default=10, help="number of latest evaluations to show"
    )
    plot = sub.add_parser("plot")
    plot.add_argument("folder", type=Path)
    a = parser.parse_args(argv)
    try:
        if a.command == "run":
            signal.signal(signal.SIGINT, _stop)
            signal.signal(signal.SIGTERM, _stop)
            if not (a.resume or a.finish or a.no_stamp):
                a.out = _stamped(a.out)
                print(f"study folder: {a.out}", file=sys.stderr, flush=True)
            s = load_study(a.config, a.out)
            s.command = shlex.join(["python", "-m", "tuning", *(argv or sys.argv[1:])])
            s.run(
                0 if a.finish else a.trials,
                resume=a.resume or a.finish,
                validate=not a.screen_only,
                max_seconds=a.max_seconds,
            )
            print(a.out.resolve() / "report.txt")
        elif a.command == "status":
            from .tables import status_text

            print(status_text(a.folder, a.last))
        else:
            from .reporting import plot_study

            for path in plot_study(a.folder):
                print(path)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Search method factory; implementations depend only on the generic domain/model."""

from .base import Proposal, Strategy
from .random_search import RandomSearch
from .coordinate import CoordinateSearch
from .evolutionary import DifferentialEvolution

__all__ = [
    "Proposal",
    "Strategy",
    "RandomSearch",
    "CoordinateSearch",
    "DifferentialEvolution",
    "make_strategy",
]


def make_strategy(method, space, seed, options=None):
    classes = {
        "random": RandomSearch,
        "coordinate": CoordinateSearch,
        "de": DifferentialEvolution,
    }
    allowed = {
        "random": set(),
        "coordinate": {"grid_points"},
        "de": {"population", "mutation", "crossover"},
        "tpe": {"startup"},
    }
    if method not in allowed:
        raise ValueError(f"unknown strategy {method}")
    extra = set(options or {}) - allowed[method]
    if extra:
        raise ValueError(f"unknown strategy options: {sorted(extra)}")
    if method == "tpe":
        from .tpe import TPE

        return TPE(space, seed, **(options or {}))
    if method not in classes:
        raise ValueError(f"unknown strategy {method}")
    return classes[method](space, seed, **(options or {}))

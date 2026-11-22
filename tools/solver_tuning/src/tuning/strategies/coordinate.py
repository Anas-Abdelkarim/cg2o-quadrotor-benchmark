from __future__ import annotations
import random
from .base import Proposal
from .random_search import RandomSearch
from ..util import canonical


class CoordinateSearch(RandomSearch):
    """Sequential greedy coordinate proposals; resume derives progress from history."""

    batch_limit = 1

    def __init__(self, space, seed, grid_points=5, **kwargs):
        super().__init__(space, seed)
        if grid_points < 2:
            raise ValueError("grid_points must be >=2")
        self.grid_points = grid_points

    def ask(self, index, history, start):
        valid = [e for e in history if e.accepted]
        best = min(valid, key=lambda e: e.score).values if valid else start
        # Normalize only proposals; never change the measured baseline.
        try:
            best = self.space.normalize(best)
        except ValueError:
            best = self.space.sample(random.Random(self.seed))
        seen = {canonical(e.values) for e in history}
        for p in self.space:
            if not p.active(best):
                continue
            for value in p.candidates(self.grid_points):
                v = dict(best)
                if value is None:
                    v.pop(p.key, None)
                else:
                    v[p.key] = value
                try:
                    v = self.space.normalize(v)
                except ValueError:
                    continue
                if canonical(v) not in seen and v != best:
                    return Proposal(v)
        # Restart after all neighbors have been exhausted, rather than falsely
        # claiming that a coordinate optimum is a global optimum.
        return super().ask(index, history, start)

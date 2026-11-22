from __future__ import annotations
import random
from .base import Proposal
from .random_search import RandomSearch


class DifferentialEvolution(RandomSearch):
    """DE/rand/1/bin for numeric spaces, with integer decoding and log coordinates.

    Not a mixed categorical evolutionary optimizer: categorical/conditional/optional
    domains are deliberately rejected. Failed individuals cannot win selection.
    """

    batch_limit = None

    def __init__(
        self, space, seed, population=12, mutation=0.8, crossover=0.7, **kwargs
    ):
        super().__init__(space, seed)
        if population < 4 or not 0 < mutation <= 2 or not 0 <= crossover <= 1:
            raise ValueError(
                "DE requires population>=4, mutation in (0,2], crossover in [0,1]"
            )
        if any(p.kind == "cat" or p.optional or p.active_if for p in space):
            raise ValueError(
                "DE supports unconditional numeric parameters only; use TPE or random for mixed spaces"
            )
        self.population, self.mutation, self.crossover = population, mutation, crossover

    def _encode(self, values):
        import math

        out = []
        for p in self.space:
            x = values[p.key]
            if p.kind in ("log", "one_minus_log"):
                x = 1 - x if p.kind == "one_minus_log" else x
                out.append((math.log(x) - math.log(p.low)) / math.log(p.high / p.low))
            else:
                out.append((x - p.low) / (p.high - p.low))
        return out

    def _decode(self, xs):
        import math

        values = {}
        for p, x in zip(self.space, xs):
            x = min(1.0, max(0.0, x))
            if p.kind in ("log", "one_minus_log"):
                v = math.exp(math.log(p.low) + x * math.log(p.high / p.low))
                v = min(p.high, max(p.low, v))
                values[p.key] = 1 - v if p.kind == "one_minus_log" else v
            else:
                v = p.low + x * (p.high - p.low)
                values[p.key] = round(v) if p.kind == "int" else v
        return values

    def ask(self, index, history, start):
        pool = [None] * self.population
        # Each proposal competes with one population slot. History order is issue order.
        for i, e in enumerate(history):
            j = i % self.population
            incumbent = pool[j]
            if incumbent is None or (
                e.accepted and (not incumbent.accepted or e.score < incumbent.score)
            ):
                pool[j] = e
        if any(e is None for e in pool):
            return super().ask(index, history, start)
        rng = random.Random(self.seed + index * 104729)
        target = index % self.population
        ids = rng.sample([i for i in range(self.population) if i != target], 3)
        aa, bb, cc = [self._encode(pool[i].values) for i in ids]
        old = self._encode(pool[target].values)
        forced = rng.randrange(len(old))
        return Proposal(
            self._decode(
                [
                    aa[j] + self.mutation * (bb[j] - cc[j])
                    if j == forced or rng.random() < self.crossover
                    else old[j]
                    for j in range(len(old))
                ]
            )
        )

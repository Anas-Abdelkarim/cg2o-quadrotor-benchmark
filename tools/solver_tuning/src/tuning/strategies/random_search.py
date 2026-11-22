from __future__ import annotations
import random
from .base import Proposal


class RandomSearch:
    batch_limit = None

    def __init__(self, space, seed, **kwargs):
        self.space, self.seed = space, seed

    def ask(self, index, history, start):
        return Proposal(self.space.sample(random.Random(self.seed + index * 104729)))

    def tell(self, proposal, result):
        pass

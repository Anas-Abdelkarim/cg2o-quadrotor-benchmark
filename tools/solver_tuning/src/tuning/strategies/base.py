from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Proposal:
    values: dict
    metadata: dict = field(default_factory=dict)


class Strategy(Protocol):
    batch_limit: int | None

    def ask(self, index: int, history: list, start: dict) -> Proposal: ...
    def tell(self, proposal: Proposal, result) -> None: ...

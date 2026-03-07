"""Difficult conditions benchmarking framework.

Collects per-condition accuracy when run against real or synthetic data,
useful for evaluating model robustness to foil glare, blur, rotation, etc.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConditionResult:
    """Result for a single difficult condition."""

    condition: str
    total_images: int
    correct_top_1: int
    correct_top_5: int
    top_1_accuracy: float
    top_5_accuracy: float
    mean_latency_ms: float


@dataclass
class DifficultConditionsReport:
    """Aggregate report across all difficult conditions."""

    conditions: list[ConditionResult] = field(default_factory=list)
    overall_top_1_accuracy: float = 0.0
    overall_top_5_accuracy: float = 0.0
    total_images: int = 0
    worst_condition: str = ""

    def add_condition(self, result: ConditionResult) -> None:
        self.conditions.append(result)
        self._recompute()

    def _recompute(self) -> None:
        if not self.conditions:
            return
        total = sum(c.total_images for c in self.conditions)
        correct_1 = sum(c.correct_top_1 for c in self.conditions)
        correct_5 = sum(c.correct_top_5 for c in self.conditions)
        self.total_images = total
        self.overall_top_1_accuracy = correct_1 / total if total > 0 else 0.0
        self.overall_top_5_accuracy = correct_5 / total if total > 0 else 0.0
        self.worst_condition = min(self.conditions, key=lambda c: c.top_1_accuracy).condition

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

"""In-process latency metrics.

A bounded ring buffer of recent samples per stage, with percentiles computed on
read. Deliberately not Prometheus: this keeps the service dependency-light, and
percentiles over the last N requests are what you actually look at when tuning
retrieval. Swap in a real exporter if you need cross-process aggregation.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

STAGES = ("plan", "embed", "search", "rerank", "generate", "total")


class LatencyRegistry:
    def __init__(self, window: int = 1000):
        self._window = window
        self._samples: dict[str, deque[float]] = {s: deque(maxlen=window) for s in STAGES}
        self._counts: dict[str, int] = {"queries": 0, "ingests": 0, "errors": 0}
        self._lock = threading.Lock()

    def record_query(self, timings: dict[str, float]) -> None:
        with self._lock:
            self._counts["queries"] += 1
            for stage in STAGES:
                value = timings.get(f"{stage}_ms")
                if value is not None:
                    self._samples[stage].append(float(value))

    def increment(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[counter] = self._counts.get(counter, 0) + amount

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            samples = {stage: list(values) for stage, values in self._samples.items()}

        stages: dict[str, Any] = {}
        for stage, values in samples.items():
            if values:
                stages[stage] = summarize(values)
        return {"counts": counts, "window": self._window, "latency_ms": stages}

    def reset(self) -> None:
        with self._lock:
            for values in self._samples.values():
                values.clear()
            self._counts = {"queries": 0, "ingests": 0, "errors": 0}


def summarize(values: list[float]) -> dict[str, float]:
    """Count, mean, and p50/p95/p99 for a sample list."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": round(percentile(ordered, 50), 2),
        "p95": round(percentile(ordered, 95), 2),
        "p99": round(percentile(ordered, 99), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
    }


def percentile(ordered: list[float], pct: float) -> float:
    """Linear-interpolated percentile over a pre-sorted list."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


registry = LatencyRegistry()

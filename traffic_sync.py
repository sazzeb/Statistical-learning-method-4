"""Reusable traffic-signal synchronization helpers.

This module is designed to be imported from the notebook.
It keeps the math exact (LCM) and also supports feature building for K-Means.

Assumption:
- "turn green" happens at the start of each cycle (t % cycle == 0).
"""

from __future__ import annotations

from math import gcd, pi
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def lcm_pair(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if a <= 0 or b <= 0:
        raise ValueError("cycle lengths must be positive integers")
    return a // gcd(a, b) * b


def lcm_all(cycle_lengths: Sequence[int]) -> int:
    if not cycle_lengths:
        raise ValueError("cycle_lengths must be non-empty")
    running = int(cycle_lengths[0])
    for v in cycle_lengths[1:]:
        running = lcm_pair(running, int(v))
    return running


def next_triple_sync_minutes(cycle_lengths_sec: Sequence[int]) -> float:
    """Return minutes until the next time all signals align (t>0)."""
    return lcm_all([int(x) for x in cycle_lengths_sec]) / 60.0


def dual_sync_times_minutes(cycle_lengths_sec: Sequence[int], duration_hours: int) -> List[float]:
    """Times (minutes) when exactly two signals align, within duration.

    Exactly two means: among the three cycles, exactly two divide t.
    We exclude t=0.
    """

    if len(cycle_lengths_sec) != 3:
        raise ValueError("expected exactly three cycle lengths")
    if int(duration_hours) <= 0:
        raise ValueError("duration_hours must be positive")

    horizon = int(duration_hours) * 3600
    cycles = np.array([int(x) for x in cycle_lengths_sec], dtype=np.int64)

    ticks = np.arange(0, horizon + 1, dtype=np.int64)
    divisibility = (ticks[:, None] % cycles[None, :]) == 0  # (n, 3) boolean
    how_many = divisibility.sum(axis=1)

    dual_mask = (how_many == 2) & (ticks > 0)
    dual_ticks = ticks[dual_mask]

    return (dual_ticks.astype(np.float64) / 60.0).tolist()


def sync_count_truth(ticks_sec: np.ndarray, cycle_lengths_sec: Sequence[int]) -> np.ndarray:
    """For each tick, count how many signals start a new cycle."""
    cycles = np.array([int(x) for x in cycle_lengths_sec], dtype=np.int64)
    ticks_sec = np.asarray(ticks_sec, dtype=np.int64)
    return ((ticks_sec[:, None] % cycles[None, :]) == 0).sum(axis=1)


def phase_features(ticks_sec: np.ndarray, cycle_lengths_sec: Sequence[int]) -> np.ndarray:
    """Cyclic phase features: [sin(2π t/c), cos(2π t/c)] for each cycle."""
    ticks = np.asarray(ticks_sec, dtype=np.float64)
    cols = []
    for c in cycle_lengths_sec:
        c = float(c)
        theta = (2.0 * pi * ticks) / c
        cols.append(np.cos(theta))
        cols.append(np.sin(theta))
    return np.column_stack(cols)

"""Reusable traffic-signal synchronization helpers.

Generated/updated from the notebook so you can re-import cleanly.
"""

from __future__ import annotations

from functools import reduce
from math import gcd, pi
from typing import List, Sequence

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
    return reduce(lcm_pair, [int(x) for x in cycle_lengths])


def next_triple_sync_minutes(cycle_lengths_sec: Sequence[int]) -> float:
    """Minutes until the next triple alignment (t>0)."""
    return lcm_all(cycle_lengths_sec) / 60.0


def dual_sync_times_minutes(cycle_lengths_sec: Sequence[int], duration_hours: int) -> List[float]:
    """Times (minutes) when exactly two signals align within the window."""
    if len(cycle_lengths_sec) != 3:
        raise ValueError("expected exactly three cycle lengths")
    horizon = int(duration_hours) * 3600
    cycles = np.array([int(x) for x in cycle_lengths_sec], dtype=np.int64)
    ticks = np.arange(0, horizon + 1, dtype=np.int64)
    is_boundary = (ticks[:, None] % cycles[None, :]) == 0
    count = is_boundary.sum(axis=1)
    dual_ticks = ticks[(count == 2) & (ticks > 0)]
    return (dual_ticks.astype(np.float64) / 60.0).tolist()


def sync_count_truth(ticks_sec: np.ndarray, cycle_lengths_sec: Sequence[int]) -> np.ndarray:
    cycles = np.array([int(x) for x in cycle_lengths_sec], dtype=np.int64)
    ticks_sec = np.asarray(ticks_sec, dtype=np.int64)
    return ((ticks_sec[:, None] % cycles[None, :]) == 0).sum(axis=1)


def phase_features(ticks_sec: np.ndarray, cycle_lengths_sec: Sequence[int]) -> np.ndarray:
    ticks = np.asarray(ticks_sec, dtype=np.float64)
    cols = []
    for c in cycle_lengths_sec:
        c = float(c)
        theta = (2.0 * pi * ticks) / c
        cols.append(np.cos(theta))
        cols.append(np.sin(theta))
    return np.column_stack(cols)

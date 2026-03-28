"""Traffic-light synchronization utilities.

Assumptions (kept simple and consistent with the prompt):
- Each light starts a new cycle at t=0 seconds and turns GREEN at that instant.
- "Turn green simultaneously" means "cycle boundary alignment" (t is a multiple of the cycle length).

The deterministic answer uses LCM.
The K-Means portion is intentionally educational: we generate many event times and
cluster them using cyclic (sin/cos) phase features, then evaluate cluster-to-class
accuracy against the known event-type labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, pi
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def lcm(a: int, b: int) -> int:
    if a <= 0 or b <= 0:
        raise ValueError("cycle lengths must be positive integers")
    return a // gcd(a, b) * b


def lcm_many(values: Sequence[int]) -> int:
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    running = int(values[0])
    for v in values[1:]:
        running = lcm(running, int(v))
    return running


def next_triple_sync_minutes(cycles_sec: Sequence[int]) -> float:
    """Minutes from t=0 to the next triple alignment (strictly after 0)."""
    t_sync = lcm_many([int(x) for x in cycles_sec])
    return t_sync / 60.0


def dual_sync_times_minutes(
    cycles_sec: Sequence[int],
    duration_hours: int,
) -> List[float]:
    """All times (in minutes) within duration where exactly two cycles align.

    Notes:
    - We list alignments at t>0 only.
    - "Exactly two" means: divisible by LCM of some pair, and NOT divisible by
      the LCM of all three.
    """

    if len(cycles_sec) != 3:
        raise ValueError("this helper expects exactly 3 cycles")
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")

    a, b, c = [int(x) for x in cycles_sec]
    horizon_sec = int(duration_hours) * 3600

    l_ab = lcm(a, b)
    l_ac = lcm(a, c)
    l_bc = lcm(b, c)
    l_abc = lcm_many([a, b, c])

    candidate_times: set[int] = set()

    for step in (l_ab, l_ac, l_bc):
        # start at step (exclude t=0)
        for t in range(step, horizon_sec + 1, step):
            candidate_times.add(t)

    dual_only = [t for t in candidate_times if (t % l_abc) != 0]
    dual_only.sort()
    return [t / 60.0 for t in dual_only]


def _alignment_mask(t_sec: int, cycles_sec: Sequence[int]) -> Tuple[bool, ...]:
    return tuple((t_sec % int(c)) == 0 for c in cycles_sec)


def event_type_label(t_sec: int, cycles_sec: Sequence[int]) -> str:
    """Ground-truth label for an event time.

    Returns one of: 'triple', 'dual', 'single'
    """

    mask = _alignment_mask(t_sec, cycles_sec)
    aligned_count = int(sum(mask))

    if aligned_count == 3:
        return "triple"
    if aligned_count == 2:
        return "dual"
    if aligned_count == 1:
        return "single"
    raise ValueError("event_type_label expects t_sec to be an event boundary")


def generate_event_times(
    cycles_sec: Sequence[int],
    duration_hours: int,
) -> np.ndarray:
    """Union of all cycle-boundary times within the given horizon.

    Returns a sorted int array of seconds.
    """

    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")

    horizon_sec = int(duration_hours) * 3600
    unique_times: set[int] = set()

    for c in cycles_sec:
        step = int(c)
        if step <= 0:
            raise ValueError("cycle lengths must be positive integers")
        for t in range(0, horizon_sec + 1, step):
            unique_times.add(t)

    return np.array(sorted(unique_times), dtype=np.int64)


def make_phase_features(
    times_sec: np.ndarray,
    cycles_sec: Sequence[int],
) -> np.ndarray:
    """Cyclic (sin/cos) features of phase for each cycle.

    For each cycle length c we compute:
      sin(2π t / c), cos(2π t / c)

    Result shape: (n_samples, 2 * len(cycles_sec))
    """

    times_sec = np.asarray(times_sec, dtype=np.float64)
    feats: list[np.ndarray] = []

    for c in cycles_sec:
        c = float(c)
        theta = (2.0 * pi * times_sec) / c
        feats.append(np.sin(theta))
        feats.append(np.cos(theta))

    return np.column_stack(feats)


@dataclass(frozen=True)
class KMeansReport:
    k: int
    n_samples: int
    accuracy: float
    adjusted_rand_index: float


def kmeans_cluster_and_score(
    times_sec: np.ndarray,
    cycles_sec: Sequence[int],
    k: int = 3,
    random_state: int = 42,
) -> KMeansReport:
    """Cluster event times and compute label-mapped accuracy.

    This uses *ground truth* labels only for evaluation.

    - k defaults to 3 because our event labels are: single/dual/triple.
    - We map each cluster id to the majority true label inside that cluster.
    """

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for kmeans_cluster_and_score; "
            "install with `conda install scikit-learn` or `pip install scikit-learn`"
        ) from exc

    times_sec = np.asarray(times_sec, dtype=np.int64)
    if times_sec.ndim != 1:
        raise ValueError("times_sec must be a 1D array")

    # keep only event boundaries (union of multiples) and remove t=0 to avoid
    # the always-triple starting point dominating small datasets
    event_times = times_sec[times_sec > 0]

    x_mat = make_phase_features(event_times, cycles_sec)

    true_labels = np.array([event_type_label(int(t), cycles_sec) for t in event_times])
    label_vocab = {name: idx for idx, name in enumerate(["single", "dual", "triple"])}
    y_true = np.array([label_vocab[s] for s in true_labels], dtype=np.int64)

    model = KMeans(n_clusters=int(k), n_init="auto", random_state=int(random_state))
    cluster_ids = model.fit_predict(x_mat)

    # Majority-vote mapping from cluster -> label
    cluster_to_label: dict[int, int] = {}
    for cid in range(int(k)):
        members = y_true[cluster_ids == cid]
        if members.size == 0:
            # unused cluster: map to most common label overall
            values, counts = np.unique(y_true, return_counts=True)
            cluster_to_label[cid] = int(values[np.argmax(counts)])
        else:
            values, counts = np.unique(members, return_counts=True)
            cluster_to_label[cid] = int(values[np.argmax(counts)])

    y_pred = np.array([cluster_to_label[int(cid)] for cid in cluster_ids], dtype=np.int64)

    acc = float((y_pred == y_true).mean())
    ari = float(adjusted_rand_score(y_true, cluster_ids))

    return KMeansReport(k=int(k), n_samples=int(event_times.size), accuracy=acc, adjusted_rand_index=ari)

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

Shape = Tuple[int, int, int]
BucketKey = int
DistributionKey = Tuple[int, int]

DEFAULT_M_SPLIT = 16
DEFAULT_N_SPLIT = 4096
DEFAULT_K_SPLIT = 6144

M_BY_BIN = {
    0: [1, 2],
    1: [4, 8],
    2: [16, 32],
    3: [64],
}
K_BY_BIN = {
    0: [1024, 1536, 2048],
    1: [3072, 4096, 6144, 8192],
}
M_CHOICES = [1, 2, 4, 8, 16, 32, 64]
N_CHOICES = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
K_CHOICES = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
ALL_BUCKET_KEYS: List[BucketKey] = list(range(8))
ALL_DISTRIBUTION_KEYS: List[DistributionKey] = [(m_bin, k_bin) for m_bin in range(4) for k_bin in range(2)]


def bucket_key(
    M: int,
    N: int,
    K: int,
    m_split: int = DEFAULT_M_SPLIT,
    n_split: int = DEFAULT_N_SPLIT,
    k_split: int = DEFAULT_K_SPLIT,
) -> BucketKey:
    """Bucket heuristic with 3 binary splits on M/N/K.

    bit2: M <= m_split
    bit1: N <= n_split
    bit0: K <= k_split
    """
    b2 = 1 if M <= m_split else 0
    b1 = 1 if N <= n_split else 0
    b0 = 1 if K <= k_split else 0
    return (b2 << 2) | (b1 << 1) | b0


def bucket_key_to_str(key: BucketKey) -> str:
    return f"bucket_g{key}"


def _build_bucket_candidates(m_split: int, n_split: int, k_split: int) -> Dict[BucketKey, List[Shape]]:
    out: Dict[BucketKey, List[Shape]] = defaultdict(list)
    for m in M_CHOICES:
        for n in N_CHOICES:
            for k in K_CHOICES:
                out[bucket_key(m, n, k, m_split, n_split, k_split)].append((m, n, k))
    return dict(out)


def _sample_shape_from_distribution_key(rng: random.Random, key: DistributionKey) -> Shape:
    m_bin, k_bin = key
    m = rng.choice(M_BY_BIN[m_bin])
    n = rng.choice(N_CHOICES)
    k = rng.choice(K_BY_BIN[k_bin])
    return (m, n, k)


def _sample_shape_for_bucket_key(
    rng: random.Random,
    key: BucketKey,
    bucket_candidates: Dict[BucketKey, List[Shape]],
) -> Shape:
    return rng.choice(bucket_candidates[key])


def _build_positions_by_key(
    shapes: List[Shape],
    m_split: int,
    n_split: int,
    k_split: int,
) -> Dict[BucketKey, List[int]]:
    positions_by_key: Dict[BucketKey, List[int]] = defaultdict(list)
    for idx, (m, n, k) in enumerate(shapes):
        key = bucket_key(m, n, k, m_split, n_split, k_split)
        positions_by_key[key].append(idx)
    return positions_by_key


def _fill_missing_bucket_keys(
    rng: random.Random,
    out: List[Shape],
    positions_by_key: Dict[BucketKey, List[int]],
    bucket_candidates: Dict[BucketKey, List[Shape]],
) -> None:
    required_keys = sorted(bucket_candidates.keys())
    missing = [key for key in required_keys if key not in positions_by_key]
    for miss_key in missing:
        donor_key = None
        donor_count = -1
        for key, positions in positions_by_key.items():
            if len(positions) > 1 and len(positions) > donor_count:
                donor_key = key
                donor_count = len(positions)
        if donor_key is None:
            raise RuntimeError("BUG: 无法在不破坏已有覆盖的前提下补齐 bucket key 覆盖")
        replace_pos = positions_by_key[donor_key].pop()
        out[replace_pos] = _sample_shape_for_bucket_key(rng, miss_key, bucket_candidates)
        positions_by_key[miss_key].append(replace_pos)


def _ensure_bucket_key_coverage(
    rng: random.Random,
    shapes: List[Shape],
    bucket_candidates: Dict[BucketKey, List[Shape]],
    m_split: int,
    n_split: int,
    k_split: int,
) -> List[Shape]:
    out = list(shapes)
    positions_by_key = _build_positions_by_key(out, m_split, n_split, k_split)
    if all(key in positions_by_key for key in bucket_candidates.keys()):
        return out
    _fill_missing_bucket_keys(rng, out, positions_by_key, bucket_candidates)
    return out


def _sample_base_shapes(rng: random.Random, tune_size: int) -> List[Shape]:
    base = tune_size // len(ALL_DISTRIBUTION_KEYS)
    rem = tune_size % len(ALL_DISTRIBUTION_KEYS)
    out: List[Shape] = []
    for i, key in enumerate(ALL_DISTRIBUTION_KEYS):
        cnt = base + (1 if i < rem else 0)
        for _ in range(cnt):
            out.append(_sample_shape_from_distribution_key(rng, key))
    return out


def build_tune_set(
    rng: random.Random,
    tune_size: int,
    m_split: int,
    n_split: int,
    k_split: int,
) -> tuple[Shape, ...]:
    bucket_candidates = _build_bucket_candidates(m_split, n_split, k_split)
    reachable_keys = sorted(bucket_candidates.keys())
    if tune_size < len(reachable_keys):
        raise RuntimeError(
            f"tune_size={tune_size} 太小，必须 >= {len(reachable_keys)} 以覆盖当前可达 key: {reachable_keys}"
        )
    out = _sample_base_shapes(rng, tune_size)
    out = _ensure_bucket_key_coverage(rng, out, bucket_candidates, m_split, n_split, k_split)
    rng.shuffle(out)
    return tuple(out)


def build_eval_set(rng: random.Random, eval_size: int) -> tuple[Shape, ...]:
    out: List[Shape] = []
    for _ in range(eval_size):
        key = rng.choice(ALL_DISTRIBUTION_KEYS)
        out.append(_sample_shape_from_distribution_key(rng, key))
    return tuple(out)

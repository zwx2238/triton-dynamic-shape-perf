from __future__ import annotations

from typing import Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.bucket_policy import BucketTunePolicy

Shape = Tuple[int, int, int]
BucketKey = int
DEFAULT_M_SPLIT = 16
DEFAULT_N_SPLIT = 4096
DEFAULT_K_SPLIT = 6144


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


class BucketAutotunePolicy(BucketTunePolicy[Shape, MatmulConfig, BucketKey]):
    def __init__(
        self,
        candidates: Sequence[MatmulConfig],
        m_split: int = DEFAULT_M_SPLIT,
        n_split: int = DEFAULT_N_SPLIT,
        k_split: int = DEFAULT_K_SPLIT,
    ) -> None:
        self.m_split = int(m_split)
        self.n_split = int(n_split)
        self.k_split = int(k_split)
        super().__init__(
            candidates=candidates,
            key_fn=lambda shape: bucket_key(shape[0], shape[1], shape[2], self.m_split, self.n_split, self.k_split),
            key_to_str=bucket_key_to_str,
        )

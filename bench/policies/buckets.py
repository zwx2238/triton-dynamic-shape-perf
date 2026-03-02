from __future__ import annotations

from itertools import product
from typing import Iterable, Tuple

Shape = Tuple[int, int, int]
BucketKey = Tuple[int, int, int]

_REPRESENTATIVE_MAP = {
    0: 32,
    1: 128,
    2: 512,
    3: 2048,
}


def bin4(x: int) -> int:
    if x <= 32:
        return 0
    if x <= 128:
        return 1
    if x <= 512:
        return 2
    return 3


def bucket_key(M: int, N: int, K: int) -> BucketKey:
    return (bin4(M), bin4(N), bin4(K))


def bucket_key_of_shape(shape: Shape) -> BucketKey:
    return bucket_key(*shape)


def bucket_key_to_str(key: BucketKey) -> str:
    return f"{key[0]}_{key[1]}_{key[2]}"


def all_bucket_keys() -> Iterable[BucketKey]:
    return product(range(4), range(4), range(4))


def representative_shape(key: BucketKey) -> Shape:
    return (
        _REPRESENTATIVE_MAP[key[0]],
        _REPRESENTATIVE_MAP[key[1]],
        _REPRESENTATIVE_MAP[key[2]],
    )

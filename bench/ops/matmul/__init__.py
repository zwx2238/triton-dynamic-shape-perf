from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

from .configs import BASE_CONFIGS, CONFIG_MAP, MatmulConfig
from .prototype import (
    derive_candidate_pool_from_typical_shapes,
    pick_typical_shapes,
    write_prototype_report_csv,
)
from .records import make_bucket_record, make_torch_record
from .shapes import (
    ALL_BUCKET_KEYS,
    ALL_DISTRIBUTION_KEYS,
    BucketKey,
    DistributionKey,
    Shape,
    bucket_key,
    bucket_key_to_str,
    build_eval_set,
    build_tune_set,
)
from .torch_baseline import TorchMatmulEvaluator
from .triton_kernel import TritonMatmulEvaluator


class MatmulOperator:
    name = "matmul"

    def get_candidates(self) -> List[MatmulConfig]:
        return list(BASE_CONFIGS)

    def get_candidate(self, config_id: str) -> MatmulConfig:
        return CONFIG_MAP[config_id]

    def make_tune_set(
        self, rng: random.Random, tune_size: int, splits: Tuple[int, ...],
    ) -> tuple[Shape, ...]:
        m_split, n_split, k_split = splits
        return build_tune_set(rng, tune_size, m_split, n_split, k_split)

    def make_eval_set(self, rng: random.Random, eval_size: int) -> tuple[Shape, ...]:
        return build_eval_set(rng, eval_size)

    def eval_key(self, shape: Shape, splits: Tuple[int, ...]) -> BucketKey:
        m_split, n_split, k_split = splits
        return bucket_key(shape[0], shape[1], shape[2], m_split, n_split, k_split)

    def eval_key_str(self, key: BucketKey) -> str:
        return bucket_key_to_str(key)

    def all_bucket_keys(self) -> Tuple[BucketKey, ...]:
        return tuple(ALL_BUCKET_KEYS)

    def shape_to_str(self, shape: Shape) -> str:
        return f"{shape[0]}x{shape[1]}x{shape[2]}"

    def pick_typical_shapes(self, shapes: Sequence[Shape], count: int) -> List[Shape]:
        return pick_typical_shapes(shapes, count)

    def derive_candidate_pool_from_typical_shapes(
        self,
        typical_shapes: Sequence[Shape],
        candidates: Sequence[MatmulConfig],
        eval_batch: Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]],
    ) -> tuple[List[MatmulConfig], List[Dict[str, object]]]:
        return derive_candidate_pool_from_typical_shapes(typical_shapes, candidates, eval_batch)

    def write_prototype_report_csv(self, path: str, rows: Sequence[Dict[str, object]]) -> None:
        write_prototype_report_csv(path, rows)

    def make_bucket_record(self, *args, **kwargs) -> Dict[str, object]:
        return make_bucket_record(*args, **kwargs)

    def make_torch_record(self, *args, **kwargs) -> Dict[str, object]:
        return make_torch_record(*args, **kwargs)

    def build_eval_entry(self, row: Dict[str, object], cfg: MatmulConfig) -> tuple:
        return ((int(row["M"]), int(row["N"]), int(row["K"])), cfg)

    def create_triton_evaluator(self, dtype: str, device: str, warmup: int, repeat: int) -> TritonMatmulEvaluator:
        return TritonMatmulEvaluator(dtype=dtype, device=device, warmup=warmup, repeat=repeat)

    def create_torch_evaluator(self, dtype: str, device: str, warmup: int, repeat: int) -> TorchMatmulEvaluator:
        return TorchMatmulEvaluator(dtype=dtype, device=device, warmup=warmup, repeat=repeat)

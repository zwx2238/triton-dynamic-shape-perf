from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MatmulConfig:
    config_id: str
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int


_BASE_BLOCKS: List[tuple[int, int, int]] = [
    (16, 128, 128),
    (32, 128, 128),
    (64, 128, 128),
    (64, 256, 64),
    (64, 128, 256),
    (32, 32, 1024),
    (64, 64, 512),
    (64, 64, 256),
]

BASE_CONFIGS: List[MatmulConfig] = [
    MatmulConfig(f"c{idx}", bm, bn, bk)
    for idx, (bm, bn, bk) in enumerate(_BASE_BLOCKS)
]


CONFIG_MAP: Dict[str, MatmulConfig] = {cfg.config_id: cfg for cfg in BASE_CONFIGS}


def get_default_config() -> MatmulConfig:
    for cfg in BASE_CONFIGS:
        if (cfg.BLOCK_M, cfg.BLOCK_N, cfg.BLOCK_K) == (64, 128, 128):
            return cfg
    raise RuntimeError("缺少默认配置 (64,128,128)")


def ids_to_configs(config_ids: List[str]) -> List[MatmulConfig]:
    return [CONFIG_MAP[cid] for cid in config_ids if cid in CONFIG_MAP]

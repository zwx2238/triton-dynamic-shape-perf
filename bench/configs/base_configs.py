from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MatmulConfig:
    config_id: str
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    num_warps: int
    num_stages: int
    GROUP_M: int


BASE_CONFIGS: List[MatmulConfig] = [
    MatmulConfig("c00", 32, 32, 32, 2, 2, 8),
    MatmulConfig("c01", 64, 32, 32, 2, 2, 8),
    MatmulConfig("c02", 32, 64, 32, 2, 2, 8),
    MatmulConfig("c03", 64, 64, 32, 4, 2, 8),
    MatmulConfig("c04", 64, 64, 64, 4, 3, 8),
    MatmulConfig("c05", 128, 64, 32, 4, 3, 8),
    MatmulConfig("c06", 64, 128, 32, 4, 3, 8),
    MatmulConfig("c07", 128, 128, 32, 8, 3, 8),
    MatmulConfig("c08", 128, 128, 64, 8, 4, 8),
    MatmulConfig("c09", 128, 64, 64, 8, 3, 8),
    MatmulConfig("c10", 64, 128, 64, 8, 3, 8),
    MatmulConfig("c11", 32, 128, 32, 4, 2, 8),
]


CONFIG_MAP: Dict[str, MatmulConfig] = {cfg.config_id: cfg for cfg in BASE_CONFIGS}


def get_default_config() -> MatmulConfig:
    return CONFIG_MAP["c03"]


def ids_to_configs(config_ids: List[str]) -> List[MatmulConfig]:
    return [CONFIG_MAP[cid] for cid in config_ids if cid in CONFIG_MAP]

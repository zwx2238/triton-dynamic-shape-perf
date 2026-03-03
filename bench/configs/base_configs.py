from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MatmulConfig:
    config_id: str
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int


BASE_CONFIGS: List[MatmulConfig] = [
    MatmulConfig("c00", 32, 32, 32),
    MatmulConfig("c01", 64, 32, 32),
    MatmulConfig("c02", 32, 64, 32),
    MatmulConfig("c03", 64, 64, 32),
    MatmulConfig("c04", 64, 64, 64),
    MatmulConfig("c05", 128, 64, 32),
    MatmulConfig("c06", 64, 128, 32),
    MatmulConfig("c07", 128, 128, 32),
    MatmulConfig("c08", 128, 128, 64),
    MatmulConfig("c09", 128, 64, 64),
    MatmulConfig("c10", 64, 128, 64),
    MatmulConfig("c11", 32, 128, 32),
    MatmulConfig("c12", 128, 128, 128),
    MatmulConfig("c13", 128, 128, 96),
    MatmulConfig("c14", 128, 128, 64),
    MatmulConfig("c15", 128, 96, 128),
    MatmulConfig("c16", 96, 128, 128),
    MatmulConfig("c17", 128, 64, 128),
    MatmulConfig("c18", 64, 128, 128),
    MatmulConfig("c19", 256, 128, 64),
    MatmulConfig("c20", 128, 256, 64),
    MatmulConfig("c21", 256, 64, 64),
    MatmulConfig("c22", 64, 256, 64),
    MatmulConfig("c23", 128, 128, 128),
    # 小 M 场景补充（例如 decode 阶段 M=1/2/4/8/16）
    MatmulConfig("c24", 16, 128, 128),
    MatmulConfig("c25", 16, 64, 128),
    MatmulConfig("c26", 16, 128, 64),
    MatmulConfig("c27", 16, 256, 64),
    MatmulConfig("c28", 16, 64, 64),
    MatmulConfig("c29", 16, 256, 128),
]


CONFIG_MAP: Dict[str, MatmulConfig] = {cfg.config_id: cfg for cfg in BASE_CONFIGS}


def get_default_config() -> MatmulConfig:
    return CONFIG_MAP["c03"]


def ids_to_configs(config_ids: List[str]) -> List[MatmulConfig]:
    return [CONFIG_MAP[cid] for cid in config_ids if cid in CONFIG_MAP]

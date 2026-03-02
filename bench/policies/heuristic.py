from __future__ import annotations

from typing import Dict, Tuple

from bench.configs.base_configs import MatmulConfig, get_default_config
from bench.policies.common import SelectionResult

Shape = Tuple[int, int, int]


def heuristic_pick_id(M: int, N: int, K: int) -> str:
    if M <= 32 and N >= 64:
        return "c11"

    if M <= 64 and N <= 64 and K <= 64:
        return "c03"

    if M >= 128 and N >= 128 and K >= 512:
        return "c08"

    if M >= 128 and N < 128:
        return "c05"

    if N >= 128 and M < 128:
        return "c06"

    if K >= 256:
        if M >= N:
            return "c09"
        return "c10"

    return "c03"


class HeuristicPolicy:
    method = "C"

    def __init__(self, config_map: Dict[str, MatmulConfig]) -> None:
        self.config_map = config_map

    def select(self, shape: Shape) -> SelectionResult:
        M, N, K = shape
        cid = heuristic_pick_id(M, N, K)
        cfg = self.config_map.get(cid, get_default_config())
        return SelectionResult(
            config=cfg,
            cache_key=f"{M}x{N}x{K}",
            tune_time_ms=0.0,
            premeasure=None,
            notes="",
        )

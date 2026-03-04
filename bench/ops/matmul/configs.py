from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MatmulConfig:
    config_id: str
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int


_BM_FIXED = 16
_TILE_MULTIPLE = 16
_L0C_CAP_BYTES = 128 * 1024
_L1_CAP_BYTES = 512 * 1024
_L1_DOUBLE_BUFFER = 2
_NUM_BASE_CONFIGS = 8
_DIVERSITY_TOP_MIN_LEVELS = 16


@dataclass(frozen=True)
class _TileCandidate:
    bn: int
    bk: int
    min_tile: int
    util_l0c: float
    util_l1: float
    util_geo: float


def _is_feasible_tile(bn: int, bk: int) -> bool:
    l0c_bytes = _BM_FIXED * bn * 4
    l1_bytes = (_BM_FIXED * bk + bk * bn) * 4 * _L1_DOUBLE_BUFFER
    return l0c_bytes <= _L0C_CAP_BYTES and l1_bytes <= _L1_CAP_BYTES


def _iter_tile_candidates() -> List[_TileCandidate]:
    out: List[_TileCandidate] = []
    for bn in range(_TILE_MULTIPLE, 8192 + _TILE_MULTIPLE, _TILE_MULTIPLE):
        l0c_bytes = _BM_FIXED * bn * 4
        if l0c_bytes > _L0C_CAP_BYTES:
            break
        for bk in range(_TILE_MULTIPLE, 8192 + _TILE_MULTIPLE, _TILE_MULTIPLE):
            l1_bytes = (_BM_FIXED * bk + bk * bn) * 4 * _L1_DOUBLE_BUFFER
            if l1_bytes > _L1_CAP_BYTES:
                break
            # Mirror tiles are equivalent for this candidate seed pool.
            if bn < bk:
                continue
            util_l0c = l0c_bytes / _L0C_CAP_BYTES
            util_l1 = l1_bytes / _L1_CAP_BYTES
            out.append(
                _TileCandidate(
                    bn=bn,
                    bk=bk,
                    min_tile=min(bn, bk),
                    util_l0c=util_l0c,
                    util_l1=util_l1,
                    util_geo=math.sqrt(util_l0c * util_l1),
                )
            )
    return out


def _pick_even_indices(total: int, count: int) -> List[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    idxs = []
    used = set()
    for i in range(count):
        idx = int(round(i * (total - 1) / (count - 1)))
        while idx in used and idx + 1 < total:
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        if idx in used:
            continue
        used.add(idx)
        idxs.append(idx)
    return sorted(idxs)


def _build_base_configs() -> List[MatmulConfig]:
    candidates = _iter_tile_candidates()
    if not candidates:
        raise RuntimeError("no feasible BM/BN/BK tiles found under 512KB constraints")

    by_min: Dict[int, List[_TileCandidate]] = {}
    for cand in candidates:
        by_min.setdefault(cand.min_tile, []).append(cand)

    min_levels = sorted(by_min.keys(), reverse=True)
    focus_levels = min_levels[: min(len(min_levels), _DIVERSITY_TOP_MIN_LEVELS)]
    level_indices = _pick_even_indices(len(focus_levels), _NUM_BASE_CONFIGS)

    selected_tiles: List[_TileCandidate] = []
    for idx in level_indices:
        level = focus_levels[idx]
        # Primary objective: larger min(BN, BK); tie-break by balanced buffer utilization.
        best = max(
            by_min[level],
            key=lambda x: (x.util_geo, x.util_l1, x.util_l0c, x.bn, x.bk),
        )
        selected_tiles.append(best)

    out: List[MatmulConfig] = []
    for i, tile in enumerate(selected_tiles):
        out.append(MatmulConfig(f"c{i:02d}", _BM_FIXED, tile.bn, tile.bk))
    return out


BASE_CONFIGS: List[MatmulConfig] = _build_base_configs()


CONFIG_MAP: Dict[str, MatmulConfig] = {cfg.config_id: cfg for cfg in BASE_CONFIGS}


def get_default_config() -> MatmulConfig:
    return BASE_CONFIGS[min(3, len(BASE_CONFIGS) - 1)]


def ids_to_configs(config_ids: List[str]) -> List[MatmulConfig]:
    return [CONFIG_MAP[cid] for cid in config_ids if cid in CONFIG_MAP]

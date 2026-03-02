from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_searched_configs(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, list):
            result[k] = [x for x in v if isinstance(x, str)]
    return result


def save_searched_configs(path: str, mapping: Dict[str, List[str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)

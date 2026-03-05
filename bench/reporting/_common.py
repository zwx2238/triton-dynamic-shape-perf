from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Tuple


def fmt(value: float) -> str:
    return f"{value:.6f}"


def geometric_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(statistics.fmean(math.log(v) for v in vals))


def shape_sort_key(shape_id: str) -> Tuple[int, str]:
    if not shape_id:
        return (10**9, "")
    tail = shape_id.split("_")[-1]
    if tail.isdigit():
        return (int(tail), shape_id)
    return (10**9, shape_id)


def print_section(title: str, rows: List[Dict[str, object]], headers: List[str]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    widths = _compute_column_widths(rows, headers)
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def _compute_column_widths(rows: List[Dict[str, object]], headers: List[str]) -> Dict[str, int]:
    widths: Dict[str, int] = {}
    for h in headers:
        widths[h] = len(h)
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    return widths

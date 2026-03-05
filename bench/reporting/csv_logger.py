from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _get_fieldnames(records: List[Dict[str, object]]) -> List[str]:
    seen: dict[str, None] = {}
    for rec in records:
        for k in rec:
            if k not in seen:
                seen[k] = None
    return list(seen)


def _resolve_fieldnames(
    path: Path,
    records: List[Dict[str, object]],
) -> List[str]:
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
            if existing is not None:
                return existing
    return _get_fieldnames(records)


def _write_records(
    path: Path,
    records: List[Dict[str, object]],
    fieldnames: List[str],
    write_header: bool,
) -> int:
    count = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})
            count += 1
    return count


def append_records(
    path: str,
    records: Iterable[Dict[str, object]],
    fieldnames: Optional[List[str]] = None,
) -> int:
    records = list(records)
    if not records:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _resolve_fieldnames(p, records)
    write_header = not p.exists() or p.stat().st_size == 0
    return _write_records(p, records, fieldnames, write_header)


def reset_csv(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()

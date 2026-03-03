"""Op registry — framework entry point for operator plugins."""

from __future__ import annotations

from typing import Any, Dict


def _load_builtins() -> Dict[str, Any]:
    from .matmul import MatmulOperator
    return {"matmul": MatmulOperator()}


_OPERATORS: Dict[str, Any] = _load_builtins()


def register_operator(name: str, operator: Any) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("operator 名称不能为空")
    _OPERATORS[key] = operator


def get_operator(name: str) -> Any:
    if not name:
        raise ValueError("operator 名称不能为空")
    key = name.strip().lower()
    if key not in _OPERATORS:
        raise ValueError(f"不支持 operator={name}，当前仅支持: {', '.join(sorted(_OPERATORS.keys()))}")
    return _OPERATORS[key]


def list_operators() -> list[str]:
    return sorted(_OPERATORS.keys())

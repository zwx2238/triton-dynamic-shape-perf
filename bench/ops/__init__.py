"""Op registry — framework entry point for operator plugins."""

from __future__ import annotations

from typing import Any, Callable, Dict


def _create_matmul_operator() -> Any:
    from .matmul import MatmulOperator
    return MatmulOperator()


_BUILTIN_FACTORIES: Dict[str, Callable[[], Any]] = {
    "matmul": _create_matmul_operator,
}
_OPERATORS: Dict[str, Any] = {}


def _ensure_operator_loaded(key: str) -> None:
    if key in _OPERATORS:
        return
    factory = _BUILTIN_FACTORIES.get(key)
    if factory is None:
        return
    _OPERATORS[key] = factory()


def register_operator(name: str, operator: Any) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("operator 名称不能为空")
    _BUILTIN_FACTORIES.pop(key, None)
    _OPERATORS[key] = operator


def get_operator(name: str) -> Any:
    if not name:
        raise ValueError("operator 名称不能为空")
    key = name.strip().lower()
    _ensure_operator_loaded(key)
    if key not in _OPERATORS:
        supported = sorted(set(_OPERATORS.keys()) | set(_BUILTIN_FACTORIES.keys()))
        raise ValueError(f"不支持 operator={name}，当前仅支持: {', '.join(supported)}")
    return _OPERATORS[key]


def list_operators() -> list[str]:
    return sorted(set(_OPERATORS.keys()) | set(_BUILTIN_FACTORIES.keys()))

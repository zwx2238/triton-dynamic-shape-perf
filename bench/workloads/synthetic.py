from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

Shape = Tuple[int, int, int]
WorkloadSplits = Dict[str, List[Shape]]


def _sample_unique(
    rng: random.Random,
    count: int,
    sampler: Callable[[], Shape],
    max_attempts: int = 200000,
) -> List[Shape]:
    seen = set()
    out: List[Shape] = []
    attempts = 0
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        shape = sampler()
        if shape in seen:
            continue
        seen.add(shape)
        out.append(shape)
    if len(out) < count:
        raise RuntimeError(f"无法生成足够唯一 shape: 需要 {count}, 实际 {len(out)}")
    return out


def _pick_probe(eval_shapes: List[Shape], n: int = 8) -> List[Shape]:
    if len(eval_shapes) <= n:
        return eval_shapes[:]
    idxs = [int(i * len(eval_shapes) / n) for i in range(n)]
    selected: List[Shape] = []
    used = set()
    for idx in idxs:
        idx = min(idx, len(eval_shapes) - 1)
        if idx in used:
            continue
        used.add(idx)
        selected.append(eval_shapes[idx])
    while len(selected) < n:
        selected.append(eval_shapes[len(selected)])
    return selected[:n]


def _uniform_shapes(rng: random.Random, count: int) -> List[Shape]:
    choices = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

    def sampler() -> Shape:
        return (rng.choice(choices), rng.choice(choices), rng.choice(choices))

    return _sample_unique(rng, count, sampler)


def _llm_style_shapes(rng: random.Random, count: int) -> List[Shape]:
    m_choices = [1, 2, 4, 8, 16, 32, 64]
    nk_choices = [1024, 2048, 4096, 8192]

    def sampler() -> Shape:
        return (rng.choice(m_choices), rng.choice(nk_choices), rng.choice(nk_choices))

    return _sample_unique(rng, count, sampler)


def _training_style_shapes(rng: random.Random, count: int) -> List[Shape]:
    choices = [512, 1024, 1536, 2048, 3072, 4096]

    def sampler() -> Shape:
        while True:
            m = rng.choice(choices)
            n = rng.choice(choices)
            ratio = m / n
            if 0.5 <= ratio <= 2:
                k = rng.choice(choices)
                return (m, n, k)

    return _sample_unique(rng, count, sampler)


def _adversarial_eval_shapes() -> List[Shape]:
    shapes: List[Shape] = []

    # 36: M 非常小，N/K 很大
    for m in [1, 2, 4, 8, 16, 32]:
        for n in [1024, 4096, 8192]:
            for k in [1024, 4096]:
                shapes.append((m, n, k))

    # 24: N 非常小，M/K 偏大
    for n in [1, 2, 4, 8]:
        for m in [1024, 2048, 4096]:
            for k in [512, 2048]:
                shapes.append((m, n, k))

    # 4: K 极深
    shapes.extend(
        [
            (64, 64, 8192),
            (128, 128, 8192),
            (2048, 128, 8192),
            (128, 2048, 8192),
        ]
    )

    deduped: List[Shape] = []
    seen = set()
    for s in shapes:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    if len(deduped) != 64:
        raise RuntimeError(f"adversarial 形状数量错误: 期望 64, 实际 {len(deduped)}")
    return deduped


def _build_single_workload(
    seed: int,
    name: str,
    generator: Callable[[random.Random, int], List[Shape]],
) -> WorkloadSplits:
    rng_tune = random.Random(seed)
    rng_eval = random.Random(seed + 1)
    tune = generator(rng_tune, 32)
    eval_set = generator(rng_eval, 64)
    probe = _pick_probe(eval_set, 8)
    return {"tune": tune, "eval": eval_set, "probe": probe}


def build_synthetic_workloads(seed: int = 20260302) -> Dict[str, WorkloadSplits]:
    workloads: Dict[str, WorkloadSplits] = {}
    workloads["uniform"] = _build_single_workload(seed + 11, "uniform", _uniform_shapes)
    workloads["llm_style"] = _build_single_workload(seed + 23, "llm_style", _llm_style_shapes)
    workloads["training_style"] = _build_single_workload(seed + 37, "training_style", _training_style_shapes)

    adversarial_eval = _adversarial_eval_shapes()
    adversarial_tune = adversarial_eval[:32]
    adversarial_probe = _pick_probe(adversarial_eval, 8)
    workloads["adversarial"] = {
        "tune": adversarial_tune,
        "eval": adversarial_eval,
        "probe": adversarial_probe,
    }

    return workloads

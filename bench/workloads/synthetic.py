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
        # 当唯一组合空间不足时，补齐剩余样本（允许重复），保证总样本规模可控。
        missing = count - len(out)
        for _ in range(missing):
            out.append(sampler())
    return out


def _sample_with_replacement(
    rng: random.Random,
    count: int,
    sampler: Callable[[], Shape],
) -> List[Shape]:
    return [sampler() for _ in range(count)]


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
    # 扩展 nk 取值，确保可以支持更大的 eval 集（例如 256）
    nk_choices = [1024, 1536, 2048, 3072, 4096, 6144, 8192]

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
    generator: Callable[[random.Random, int], List[Shape]],
    tune_count: int,
    eval_count: int,
    probe_count: int,
) -> WorkloadSplits:
    rng_tune = random.Random(seed)
    rng_eval = random.Random(seed + 1)
    tune = generator(rng_tune, tune_count)
    eval_set = generator(rng_eval, eval_count)
    probe = _pick_probe(eval_set, probe_count)
    return {"tune": tune, "eval": eval_set, "probe": probe}


def _expand_to_count(base: List[Shape], count: int) -> List[Shape]:
    if count <= len(base):
        return base[:count]
    out: List[Shape] = []
    i = 0
    while len(out) < count:
        out.append(base[i % len(base)])
        i += 1
    return out


def build_synthetic_workloads(
    seed: int = 20260302,
    tune_count: int = 32,
    eval_count: int = 64,
    probe_count: int = 8,
) -> Dict[str, WorkloadSplits]:
    workloads: Dict[str, WorkloadSplits] = {}
    workloads["uniform"] = _build_single_workload(
        seed + 11, _uniform_shapes, tune_count=tune_count, eval_count=eval_count, probe_count=probe_count
    )
    workloads["llm_style"] = _build_single_workload(
        seed + 23, _llm_style_shapes, tune_count=tune_count, eval_count=eval_count, probe_count=probe_count
    )
    workloads["training_style"] = _build_single_workload(
        seed + 37, _training_style_shapes, tune_count=tune_count, eval_count=eval_count, probe_count=probe_count
    )

    adversarial_eval = _adversarial_eval_shapes()
    adversarial_tune = _expand_to_count(adversarial_eval[:32], tune_count)
    adversarial_eval = _expand_to_count(adversarial_eval, eval_count)
    adversarial_probe = _pick_probe(adversarial_eval, probe_count)
    workloads["adversarial"] = {
        "tune": adversarial_tune,
        "eval": adversarial_eval,
        "probe": adversarial_probe,
    }

    return workloads

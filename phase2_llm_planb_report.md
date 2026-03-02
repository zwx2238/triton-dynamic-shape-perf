# Phase 2 - llm_style 专项：Plan B 优化结果

## 1. 目标

仅针对 `llm_style` workload，提升 Plan B（bucket autotune）性能。

## 2. 关键改动

在 `B2` 中加入 llm 专项通道（默认开启）：

1. 形状识别
- 条件：`M <= 64 and N >= 1024 and K >= 1024`

2. llm 专项 key
- 对 llm 形状按 `shape(M,N,K)` 精细缓存（避免粗桶首样本误导）。

3. llm 专项候选池
- 默认候选：`c11,c00,c04,c02,c10,c01`
- 通过 `--b2-llm-candidates` 可改。

4. 非 llm 形状
- 仍走原 B2 的 bucket + 两阶段评分逻辑。

## 3. 实验设置

- 方法：`A / U / B / B2`
- workload：仅 `llm_style`
- 每方法预算：300s
- warmup=10, repeat=50
- 每方法独立进程 + 独立 `TRITON_CACHE_DIR`
- 数据文件：`results/phase2_llm_group4.csv`

## 4. 结果

| Method | median runtime (us) | mean TFLOPS | total tune time (ms) |
| --- | ---: | ---: | ---: |
| A | 76.208 | 3.251 | 25379.30 |
| U | 176.016 | 1.781 | 3077.02 |
| B | 171.224 | 1.814 | 4515.19 |
| B2 | 67.312 | 3.298 | 14238.66 |

## 5. 结论

1. `B2` 在 llm_style 上已明显超过 `B` 与 `U`。
- vs B：runtime `+60.69%`（171.224 -> 67.312 us）
- vs U：runtime `+61.76%`

2. `B2` 也超过了 `A`（本轮实测）。
- runtime `+11.67%`（76.208 -> 67.312 us）
- mean TFLOPS 更高（3.298 vs 3.251）
- tune_time 低于 A（14.24s vs 25.38s）

3. 结论：在 llm_style 目标下，当前最优方案是 `B2`（启用 llm 专项通道）。

## 6. 产物

- 数据：`results/phase2_llm_group4.csv`
- 图：`results/phase2_llm_group4.png`

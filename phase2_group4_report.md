# Phase 2 四组对比报告（A / U / B / B2）

## 1. 对比范围

本轮只保留 4 组：

- A: full autotune
- U: fixed static baseline
- B: bucket autotune (v1)
- B2: improved bucket autotune (v2)

## 2. 实验设置

- GPU: RTX 4060
- dtype: bf16
- workload: uniform / llm_style / training_style / adversarial
- 每方法预算: 300s
- warmup=10, repeat=50
- 每方法独立进程 + 独立 `TRITON_CACHE_DIR`
- 输出: `results/phase2_group4.csv`

## 3. 结果总览（全部 workload 汇总）

| Method | median runtime (us) | mean TFLOPS | total tune time (ms) | total compile time (ms) |
| --- | ---: | ---: | ---: | ---: |
| A | 83.968 | 10.637 | 84614.31 | 184.85 |
| U | 132.800 | 9.448 | 3055.21 | 0.00 |
| B | 136.704 | 9.632 | 8991.63 | 186.99 |
| B2 | 111.104 | 9.801 | 26282.66 | 120.75 |

相对 U 的整体 runtime 改善：

- A: `+36.77%`
- B2: `+16.34%`
- B: `-2.94%`（比 U 更慢）

## 4. 分 workload 对比（median us / mean TFLOPS）

### adversarial

- A: `64.960 / 1.721`
- U: `109.568 / 0.873`
- B: `92.160 / 0.931`
- B2: `81.624 / 1.429`

### llm_style

- A: `79.848 / 3.179`
- U: `174.592 / 1.881`
- B: `230.912 / 1.256`
- B2: `161.712 / 2.007`

### training_style

- A: `316.624 / 27.303`
- U: `317.576 / 26.065`
- B: `320.784 / 26.554`
- B2: `308.024 / 25.887`

### uniform

- A: `42.928 / 10.344`
- U: `63.944 / 8.975`
- B: `53.216 / 9.789`
- B2: `44.544 / 9.881`

## 5. 结论

1. 四组里性能上限仍是 A。
2. B2 明显优于 B，且整体显著优于 U（尤其在 llm_style 与 adversarial）。
3. B 在本轮分布下整体不如 U，说明 v1 bucket key 与首次建桶策略不稳。
4. 若只在这 4 组中选“非 A 的实用方案”，B2 是当前最优。

## 6. 产物

- 数据: `results/phase2_group4.csv`
- 图: `results/phase2_group4.png`

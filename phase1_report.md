# Triton 动态 Shape 性能实验报告（Phase 1）

## 1. 实验目标

本实验目标是对比 6 种配置选择策略在动态 shape 场景下的效果：

- A: full autotune
- B: bucket autotune
- C: heuristic rule
- D: offline table
- F: script search
- U: unoptimized fixed-static baseline（新增）

重点评估三类指标：

- compile time
- runtime cost
- runtime perf

其中 `U` 方法用于验证一个核心问题：
“在某个静态 shape 上表现好的固定分块，是否能泛化到动态 shape 分布”。

---

## 2. 实验环境

- GPU: NVIDIA GeForce RTX 4060 (desktop)
- dtype: bf16
- kernel: Triton matmul
- workload: synthetic（uniform / llm_style / training_style / adversarial）
- 随机种子: 20260302
- 预算: 每方法 300 秒
- 测量: warmup=10, repeat=50, 记录 p50 延迟与 p99

数据规模：

- 每个 workload: tune 32 + eval 64 = 96 shapes
- 4 个 workload 合计每方法 384 shapes
- 6 个方法总计 2304 条记录（CSV 数据行）

---

## 3. 方法定义（含新增 U）

### 3.1 U（新增）

U 的实现方式：

1. 在静态参考 shape `2048x2048x2048` 上，对 12 个 config 做一次 autotune。
2. 选出的最优 config 固定用于所有动态 shape。
3. 本次实测固定 config 为 `c07`。

这等价于“静态优化结果直接套到动态分布”的基线。

---

## 4. 实验执行

为避免不同方法共享进程缓存导致 compile 指标串扰，本次采用“每个方法独立进程 + 独立 TRITON_CACHE_DIR”运行。

Runner 输出的单方法墙钟时间：

| 方法 | 墙钟时间（s） |
| --- | ---: |
| C | 25.33 |
| D | 29.74 |
| B | 31.62 |
| A | 91.35 |
| F | 47.44 |
| U | 26.98 |

---

## 5. 总体结果

总体统计（全部 workload 汇总）：

| 方法 | 中位延迟 us（越低越好） | 平均 TFLOPS（越高越好） | compile_time 总和 ms | tune_time 总和 ms |
| --- | ---: | ---: | ---: | ---: |
| A | 82.960 | 10.595 | 182.06 | 84482.59 |
| B | 127.904 | 9.700 | 185.26 | 9012.53 |
| C | 133.272 | 8.907 | 1826.35 | 0.00 |
| D | 140.728 | 9.760 | 0.00 | 0.00 |
| F | 125.424 | 9.977 | 0.00 | 2773.00 |
| U | 161.280 | 9.071 | 0.00 | 3051.53 |

相对 `U` 的整体延迟收益：

- A: `+48.56%`
- F: `+22.23%`
- B: `+20.69%`
- C: `+17.37%`
- D: `+12.74%`

相对 `C` 的整体延迟收益：

- A: `+37.75%`
- F: `+5.89%`
- B: `+4.03%`
- D: `-5.59%`（慢于 C）
- U: `-21.02%`（显著慢于 C）

---

## 6. 分 workload 结果

按 workload 的中位延迟（us）与平均吞吐（TFLOPS）：

### 6.1 adversarial

| 方法 | median us | mean TFLOPS |
| --- | ---: | ---: |
| A | 64.512 | 1.755 |
| B | 76.696 | 1.355 |
| C | 99.328 | 1.032 |
| D | 98.720 | 0.985 |
| F | 104.912 | 1.089 |
| U | 182.272 | 0.709 |

### 6.2 llm_style

| 方法 | median us | mean TFLOPS |
| --- | ---: | ---: |
| A | 76.800 | 3.255 |
| B | 257.024 | 1.310 |
| C | 169.696 | 2.058 |
| D | 182.160 | 1.820 |
| F | 173.312 | 1.942 |
| U | 215.512 | 1.565 |

### 6.3 training_style

| 方法 | median us | mean TFLOPS |
| --- | ---: | ---: |
| A | 319.464 | 27.168 |
| B | 318.464 | 26.698 |
| C | 335.840 | 24.315 |
| D | 314.320 | 26.779 |
| F | 312.600 | 27.166 |
| U | 321.944 | 25.752 |

### 6.4 uniform

| 方法 | median us | mean TFLOPS |
| --- | ---: | ---: |
| A | 43.800 | 10.202 |
| B | 56.376 | 9.439 |
| C | 65.400 | 8.223 |
| D | 55.936 | 9.457 |
| F | 48.576 | 9.709 |
| U | 65.080 | 8.258 |

---

## 7. 关键结论

1. `A` 是性能上限。
在 4 个 workload 中，`A` 在 3 个（adversarial / llm_style / uniform）上 runtime/perf 都最好；整体延迟最优。

2. `F` 是较好的折中方案。
`F` 在 `training_style` 上拿到最佳延迟（312.600 us），且总体性能接近 A，但在线调参成本远低于 A。

3. `U` 明确证明“静态最优不等于动态最优”。
`U` 在整体上最慢（161.280 us），在 adversarial / llm_style 尤其明显落后，验证了固定分块对动态 shape 分布的泛化不足。

4. `B` 和 `D` 各有取舍。
`B` runtime 比 `D` 更好，但有在线 autotune 成本；`D` 无在线 tune，工程可控性更高，但这版规则下性能不如 B/F。

5. `C` 适合快速 baseline，不适合性能目标。
无调参成本、流程最简单，但运行性能明显落后于 A/F。

---

## 8. 工程口径说明（避免误读）

1. `compile_time_ms` 为“该记录对应 config 的首编译时间”。
因此 `D/F` 的在线记录里 compile 可能接近 0，因为编译开销可能发生在离线建表或 probe 阶段。

2. `tune_time_ms` 是按记录行写入。
例如 `U` 的一次性选 config 开销写在首条记录，用于保留方法内调参成本。

3. 本次已通过“方法分进程 + 独立缓存”降低方法间缓存串扰。

---

## 9. 结论建议

若目标是“纯性能优先”：

- 选 `A`（full autotune）。

若目标是“性能-成本平衡”：

- 优先考虑 `F`，其次 `B`。

若目标是“部署简单、零在线 tune”：

- 可用 `D`，但要接受一定性能损失。

`U` 建议仅保留为对照基线，不建议用于生产动态 shape 场景。

---

## 10. 产物路径

- 原始结果 CSV: `results/phase1_raw.csv`
- 离线表: `results/offline_table.csv`
- 搜索结果: `results/searched_configs.json`
- 可视化图: `results/phase1_summary.png`

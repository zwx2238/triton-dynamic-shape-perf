# Phase 1 实施文档 v1（可直接执行）

## 1. 已确认约束

- Kernel: `Triton matmul`
- GPU: `RTX 4060 (desktop)`
- dtype: `bf16`
- workload: 先用合成数据
- 比较方式: 指标独立比较，不做加权总分
- 关键指标: `compile time`, `runtime cost`, `runtime perf`
- 调参预算: 每个方法 `5 分钟`（300 秒）
- 需要我先给初版: `config 空间`、`bucket`、`heuristic rule`
- offline table: 必须实际离线跑一次生成
- 结果格式: `CSV`

---

## 2. Phase 1 范围

仅实现并比较：

- A: `full autotune`
- B: `bucket autotune`
- C: `heuristic rule`
- D: `offline table`
- F: `script search`

不包含 LLM（G/H/I 放到 Phase 2/3）。

---

## 3. 初版 Matmul Config 空间（v1）

先固定一组相对保守、在 4060 上可跑概率高的候选（12 个）：

| config_id | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages | GROUP_M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c00 | 32  | 32  | 32 | 2 | 2 | 8 |
| c01 | 64  | 32  | 32 | 2 | 2 | 8 |
| c02 | 32  | 64  | 32 | 2 | 2 | 8 |
| c03 | 64  | 64  | 32 | 4 | 2 | 8 |
| c04 | 64  | 64  | 64 | 4 | 3 | 8 |
| c05 | 128 | 64  | 32 | 4 | 3 | 8 |
| c06 | 64  | 128 | 32 | 4 | 3 | 8 |
| c07 | 128 | 128 | 32 | 8 | 3 | 8 |
| c08 | 128 | 128 | 64 | 8 | 4 | 8 |
| c09 | 128 | 64  | 64 | 8 | 3 | 8 |
| c10 | 64  | 128 | 64 | 8 | 3 | 8 |
| c11 | 32  | 128 | 32 | 4 | 2 | 8 |

说明：

- 这是 baseline 空间，不追求“专家最优”，目标是先稳定可比。
- 若某 config 在 4060 上编译失败，直接从池中剔除并记录 `invalid_config=1`。

---

## 4. 初版 Bucket 方案（v1）

### 4.1 分桶函数

```python
def bin4(x: int) -> int:
    if x <= 32:
        return 0
    if x <= 128:
        return 1
    if x <= 512:
        return 2
    return 3

def bucket_key(M: int, N: int, K: int) -> tuple[int, int, int]:
    return (bin4(M), bin4(N), bin4(K))
```

- 总桶数最多 `4 x 4 x 4 = 64`。
- v1 不加长宽比维度，先控制复杂度与 key 数量。

### 4.2 代表 shape（给 D 用）

离线建表时，每个 bucket 取一个代表 shape：

- `bin=0 -> 32`
- `bin=1 -> 128`
- `bin=2 -> 512`
- `bin=3 -> 2048`

例如 `(1,2,3)` 对应代表 shape `(128, 512, 2048)`。

---

## 5. 初版 Heuristic Rule（v1）

只使用上面的 12 个 config，保证可解释与可复现：

```python
def heuristic_pick(M: int, N: int, K: int) -> str:
    # 小 batch / skinny M
    if M <= 32 and N >= 64:
        return "c11"

    # 小矩阵
    if M <= 64 and N <= 64 and K <= 64:
        return "c03"

    # 大方阵且 K 深
    if M >= 128 and N >= 128 and K >= 512:
        return "c08"

    # 偏高矩阵
    if M >= 128 and N < 128:
        return "c05"

    # 偏宽矩阵
    if N >= 128 and M < 128:
        return "c06"

    # K 深但不是大方阵
    if K >= 256:
        if M >= N:
            return "c09"
        return "c10"

    # 默认
    return "c03"
```

---

## 6. 合成 Workload（v1）

每类 workload 分 `tune_set` 和 `eval_set`：

- `tune_set`: 32 shapes
- `eval_set`: 64 shapes

四类 workload：

1. `uniform`
   - `M,N,K` 从集合 `{64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096}` 独立均匀采样
2. `llm_style`
   - `M` 从 `{1,2,4,8,16,32,64}`
   - `N,K` 从 `{1024,2048,4096,8192}` 采样
3. `training_style`
   - `M,N,K` 从 `{512,1024,1536,2048,3072,4096}` 采样，并约束 `0.5 <= M/N <= 2`
4. `adversarial`
   - 手工固定 64 个极端 shape（如 `M=1`、`K` 超大、极端 skinny/fat）

随机种子固定：`seed=20260302`。

---

## 7. 五种方法的可执行定义

### A full autotune

- key: `(M,N,K)`
- 在首次出现某 shape 时，遍历全部候选 config（12 个）
- 记录最佳 config 并缓存
- 调参预算上限：300 秒

### B bucket autotune

- key: `bucket_key(M,N,K)`
- 首次出现某 bucket 时，使用当前 shape 对 12 configs autotune
- 记录该 bucket 最优 config
- 调参预算上限：300 秒

### C heuristic rule

- 直接走 `heuristic_pick(M,N,K)`，无 autotune
- 仅统计编译与运行指标

### D offline table

- 离线阶段：对已定义 bucket 的代表 shape 进行 autotune，生成 `offline_table.csv`
- 在线阶段：按 bucket 查表取 config
- 离线阶段预算上限：300 秒

### F script search

纯脚本策略，不用 LLM：

1. 在每类 workload 的 `probe_set(8 shapes)` 上评估 12 configs
2. 按几何平均速度保留 top-6 configs，生成 `searched_configs.json`
3. 后续按 B 的 bucket autotune 流程运行，但候选池改为 top-6
4. 总预算上限：300 秒（含 probe + bucket tune）

---

## 8. 计时与统计口径（严格）

每条 shape 的记录包含：

- `compile_time_ms`
  - 该 `(kernel_signature, config)` 首次触发 JIT 到可执行结束的 wall time
- `runtime_cost_us`
  - 稳态单次延迟（warmup 后测量）
- `runtime_perf_tflops`
  - `2*M*N*K / latency_seconds / 1e12`

建议测量参数（v1）：

- warmup: `10` 次
- repeat: `50` 次
- 统计 `p50` 作为 `runtime_cost_us`
- 同时写入 `p99_us`（后续分析稳定性时有用）

---

## 9. CSV 输出格式（统一）

文件：`results/phase1_raw.csv`

列：

```text
timestamp,method,workload,split,shape_id,M,N,K,dtype,gpu,
config_id,BLOCK_M,BLOCK_N,BLOCK_K,num_warps,num_stages,GROUP_M,
compile_time_ms,tune_time_ms,runtime_cost_us,p99_us,runtime_perf_tflops,
bucket_m,bucket_n,bucket_k,cache_key,invalid_config,notes
```

说明：

- `method in {A,B,C,D,F}`
- `split in {tune,eval}`
- `tune_time_ms` 在非 autotune 方法可置 `0`
- `cache_key` 对 A 是 shape key，对 B/D/F 是 bucket key

---

## 10. 目录与脚本建议

```text
.
├── draft.md
├── phase1_implementation.md
├── bench/
│   ├── kernels/
│   │   └── triton_matmul.py
│   ├── configs/
│   │   ├── base_configs.py
│   │   └── searched_configs.py
│   ├── policies/
│   │   ├── full_autotune.py
│   │   ├── bucket_autotune.py
│   │   ├── heuristic.py
│   │   ├── offline_table.py
│   │   └── script_search.py
│   ├── workloads/
│   │   └── synthetic.py
│   ├── runner.py
│   └── csv_logger.py
└── results/
    └── phase1_raw.csv
```

---

## 11. 一键执行顺序（建议）

1. 先跑 C（验证 kernel 和日志链路）
2. 再跑 D（确认 offline table 生成与加载）
3. 再跑 B（验证在线 bucket autotune）
4. 再跑 A（full autotune，通常最慢）
5. 最后跑 F（script search）

这样可以最早暴露工程问题，并减少返工。

---

## 12. 验收标准（Phase 1 完成条件）

- 五种方法都能在同一环境跑通并输出同一 CSV schema
- 每种方法都遵守 `300 秒` 预算上限
- 每种 workload 都有可比较的 `compile/runtime cost/runtime perf`
- D 的 `offline_table.csv` 由实际离线运行产生，不是手写


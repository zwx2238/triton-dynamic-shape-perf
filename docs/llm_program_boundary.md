# triton-dynamic-shape-perf: LLM 与程序职责边界

## 1. 目的
这份文档定义本项目里两类职责：
- `程序自动执行`：一次 pipeline 运行中必须稳定、可复现地完成的步骤。
- `LLM 可辅助`：围绕实验设计、结果解读、故障排查、代码演进的人类决策与工程支持。

核心原则：
- 需要“逐 shape 执行、依赖真实硬件测量、可重复”的任务，交给程序。
- 需要“跨实验归纳、策略取舍、代码改造建议”的任务，交给 LLM。

## 2. 程序自动执行范围（单次 run 内）

入口是 `python run_bucket_tune_pipeline.py`。以下步骤由代码自动串行完成：

| Stage | 主要代码入口 | 程序自动做什么 | 主要产物 |
|---|---|---|---|
| `setup_benchmark` | `run_bucket_tune_pipeline.py` + `bench/bucket_tune/runtime.py` | 构建 tune/eval 集合；校验 eval key 被 tune key 覆盖；初始化 triton/torch evaluator；准备输出文件 | `status.txt`、`stage_times.csv`、结果 CSV 初始化 |
| `prototype` | `bench/stages/prototype_stage.py` | 在典型 shape 上自动评估候选 config，收缩候选池 | `prototype_best.csv` |
| `benchmark_bucket` | `bench/stages/bucket_stage.py` + `bench/policies/bucket_policy.py` | 对 tune 集做 bucket 调优与缓存命中；对 eval 集仅复用缓存 config（不再调优）；写入 BUCKET 结果 | `bucket_torch.csv`（BUCKET 行） |
| `benchmark_torch` | `bench/stages/torch_stage.py` | 对 eval 集跑 torch baseline 并记录 | `bucket_torch.csv`（TORCH 行） |
| `case_compare` | `bench/reporting/compare_case_runtime.py` | 按 shape 聚合 TORCH/BUCKET 运行时间，对齐并计算 ratio/delta | `bucket_torch_case_compare_eval.csv` |
| `summary` | `bench/reporting/summarize_results.py` | 自动汇总 overall/tune/by-bucket 指标并打印表格 | `*_summary_overall.csv`、`*_summary_tune.csv`、`*_summary_by_bucket.csv` |

程序在 run 内还会自动做硬约束校验并失败退出，例如：
- `prototype_count > 0`
- `tune_size >= 8`（matmul 当前分桶需覆盖 8 个 key）
- `eval key ⊆ tune key`
- bucket policy 在 `eval` 阶段不允许再次调参

## 3. LLM 可做范围（run 外和跨 run）

### 3.1 实验设计与参数建议
- 根据目标给出 `tune-size / eval-size / prototype-count / bucket-splits / warmup / repeat` 的组合建议。
- 设计对照实验矩阵（例如只改一个变量，控制其它参数不变）。
- 识别明显无效或高风险参数（如不满足 key 覆盖约束）。

### 3.2 结果解读与归因
- 读取 `summary` 与 `case_compare` 结果，解释为何 BUCKET 快/慢。
- 识别异常分桶或异常 shape，给出下一轮实验建议。
- 形成面向团队的结论文档与行动项。

### 3.3 故障定位与修复
- 基于报错栈定位到具体模块和数据流。
- 直接修改代码并回归验证（例如策略缓存、指标口径、CSV 字段一致性）。
- 补充保护性校验与可观测性日志。

### 3.4 代码演进
- 新增 operator、policy、report 指标。
- 重构阶段边界，减少耦合。
- 增加自动化测试和回归脚本。

## 4. 明确边界：哪些不应让 LLM“替代程序”

以下任务应始终由程序执行，LLM 不能用“主观判断”替代：
- 针对每个 shape 的真实 runtime/compile/tune 测量。
- 候选 config 的逐项打分与最优选择。
- bucket 命中与缓存复用逻辑。
- CSV 明细写入、聚合计算、排序和比值计算。
- 运行时约束校验与失败退出。

换句话说：LLM 可以“决定下一次 run 怎么做”，但不应“在本次 run 内代替测量和打分引擎”。

## 5. 推荐协作流程

1. 人或 LLM 先定义实验目标和参数。
2. 程序执行一次完整 pipeline，产出标准化 CSV 与日志。
3. LLM 基于产物做解读、发现问题并提出下一轮改动。
4. 若需改代码，LLM提交补丁后由程序再次跑回归验证。
5. 重复以上闭环，直到指标达到目标。

## 6. 最小交接清单

每次交流建议至少给出：
- 运行命令与关键参数。
- `run_dir` 路径。
- 是否跑完到 `DONE`。
- 三类输出文件：`bucket_torch.csv`、`case_compare.csv`、`summary*.csv`。
- 如失败，提供完整报错栈与失败 stage。


# Phase 2 - Plan B（Bucket Autotune）优化报告

## 1. 目标

在不改 kernel/config space（仍是 12 configs）的前提下，专注优化 `B(bucket autotune)`，并给出可量化提升。

## 2. 改动内容

新增 `B2` 策略（文件：`bench/policies/bucket_autotune_v2.py`），核心改动：

1. 更细粒度分桶
- 从 `4x4x4`（仅 M/N/K 三维）升级到 `5x5x5 + ratio(3)`。
- 新 key: `(bin5(M), bin5(N), bin5(K), ratio3(M,N))`。

2. 两阶段打分
- Stage-1：在当前 shape 上评估全部候选，得到 primary 排名。
- Stage-2：仅对前 `top_k`（默认 4）做 anchor 校正（代表 shape），减少调参开销和误导。

3. 极端形状保护
- 对 `min(M,N)<=8` 或 `max(M,N)/min(M,N)>=8` 的极端 shape 关闭 anchor（`anchor_skipped`），避免被代表 shape 拉偏。

4. 参数化
- `--b2-anchor-alpha`（默认 0.1）
- `--b2-anchor-top-k`（默认 4）
- `--b2-disable-anchor`

## 3. 实验设置

- 时间：2026-03-02
- GPU: RTX 4060
- dtype: bf16
- workload: uniform / llm_style / training_style / adversarial
- 每方法预算：300s
- warmup=10, repeat=50
- 运行方式：独立进程 + 独立 TRITON_CACHE_DIR

## 4. 扫描结果（总览）

来源文件：
- `results/B_sweep_baseline.csv`
- `results/B2_sweep_a01_k4.csv`
- `results/B2_sweep_no_anchor.csv`
- 其余参数组合见 `results/B2_sweep_*.csv`

按整体 median runtime(us) 排名：

1. `B2_sweep_a01_k4`: `103.936 us`, `9.870 TFLOPS`, `tune_ms=25829.27`
2. `B2_sweep_no_anchor`: `105.656 us`, `10.100 TFLOPS`, `tune_ms=18536.11`
3. `B_sweep_baseline`: `138.752 us`, `9.405 TFLOPS`, `tune_ms=9114.56`

## 5. 相对基线 B 的提升幅度

以 `B_sweep_baseline` 为基线：

### 5.1 最快延迟方案（B2 a01_k4）

- 整体 median runtime: `138.752 -> 103.936 us`（`+25.09%`）
- 整体 mean TFLOPS: `9.405 -> 9.870`（`+4.95%`）
- tune_time: `9.11s -> 25.83s`（约 `2.83x`）

分 workload runtime 改善：
- adversarial: `+25.62%`
- llm_style: `+33.29%`
- training_style: `+3.57%`
- uniform: `+16.89%`

### 5.2 吞吐优先方案（B2 no_anchor）

- 整体 median runtime: `+23.85%`
- 整体 mean TFLOPS: `+7.40%`（最高）
- tune_time: `18.54s`（低于 a01_k4）

## 6. 结论

1. Plan B 可以被显著优化，已经做到：
- 延迟方向：相对 B 基线约 `25%` 改善。
- 吞吐方向：相对 B 基线约 `7.4%` 改善。

2. 目前最佳配置（延迟优先）是：
- `B2 + alpha=0.1 + top_k=4`（已设为默认）。

3. 如果上线更重视吞吐/调参开销平衡：
- 可考虑 `--b2-disable-anchor`。

## 7. 复现命令

延迟优先（当前默认）：

```bash
python -m bench.runner --methods B2 \
  --workloads uniform llm_style training_style adversarial \
  --budget-seconds 300 --dtype bf16 --warmup 10 --repeat 50
```

吞吐优先：

```bash
python -m bench.runner --methods B2 \
  --workloads uniform llm_style training_style adversarial \
  --budget-seconds 300 --dtype bf16 --warmup 10 --repeat 50 \
  --b2-disable-anchor
```

---

# 研究初始文档 v0

## 题目（工作名）

**Dynamic-Shape Kernel Tuning in Triton without Training:
Policy Synthesis vs Autotune vs Heuristic**

或中文：

> 在无训练条件下的 Triton 动态 shape kernel tuning：
> autotune / heuristic / policy synthesis 的比较研究

---

# 1. 研究背景

Triton kernel 的性能依赖 meta-parameters，例如：

```
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
group_size
vector_width
layout
```

在动态 shape 场景中：

```
shape → 最优 config 不同
```

常见解决方法：

* autotune
* bucket + autotune
* heuristic rule
* offline table
* cost model
* learned model

但在实际工程约束下：

```
不能训练新模型
不能做RL
不能长期收集数据优化预测器
不能runtime调用LLM
必须确定性
必须低延迟
```

因此需要研究：

> 在无学习能力条件下，如何设计 tuning policy，使动态 shape 下性能稳定且接近最优。

---

# 2. 问题定义（第一性原理）

给定：

```
kernel K
hardware H
shape s ∈ S
config c ∈ C
```

性能函数：

```
T = f(K, H, s, c)
```

目标：

```
选择 policy P
使得

c = P(s)

并最小化

E_s [ f(K,H,s,P(s)) ]
```

同时满足：

```
compile cost 低
tuning cost 低
runtime latency 低
deterministic
cache可控
```

注意：

```
P 不是训练得到
P 必须是程序 / 规则 / 表
```

---

# 3. 关键约束

本研究严格限制：

### 3.1 不允许训练

```
no RL
no ML cost model
no regression
no offline training
```

### 3.2 不允许 runtime LLM

```
no LLM per shape
no agent loop online
no inference in hot path
```

### 3.3 允许

```
offline分析
规则生成
程序生成
自动代码生成
离线测量
脚本搜索
```

### 3.4 LLM允许用途

仅限：

```
生成策略代码
生成config集合
生成bucket规则
生成kernel模板
生成搜索脚本
```

不允许：

```
runtime决策
预测性能
选择config在线
```

---

# 4. 真实困难来源分析

kernel tuning 的困难不在于：

```
config选择本身
```

而在于：

```
config空间设计
bucket设计
policy设计
kernel结构设计
```

原因：

* config空间依赖 kernel
* kernel依赖 arch
* arch依赖 dtype
* shape分布未知
* tuning cost高

因此：

> tuning 是系统设计问题，不是单次优化问题

---

# 5. 方法空间分类（最终研究分类）

仅保留在约束下合理的方法。

| ID | 方法                         | 是否autotune | 是否bucket | 是否学习 | 是否LLM | runtime调用LLM |
| -- | -------------------------- | ---------- | -------- | ---- | ----- | ------------ |
| A  | full autotune              | ✔          | ✘        | ✘    | ✘     | ✘            |
| B  | bucket autotune            | ✔          | ✔        | ✘    | ✘     | ✘            |
| C  | heuristic rule             | ✘          | ✔        | ✘    | ✘     | ✘            |
| D  | offline table              | ✘          | ✔        | ✘    | ✘     | ✘            |
| E  | analytic policy            | ✘          | ✔        | ✘    | ✘     | ✘            |
| F  | script search              | ✔          | ✔        | ✘    | ✘     | ✘            |
| G  | LLM policy synthesis       | ✘          | ✔        | ✘    | ✔     | ✘            |
| H  | LLM config space synthesis | ✔          | ✔        | ✘    | ✔     | ✘            |
| I  | LLM kernel synthesis       | varies     | varies   | ✘    | ✔     | ✘            |

不研究：

```
LLM runtime agent
RL autotune
learned cost model
```

---

# 6. 每类方法定义

## A full autotune

```
key = M,N,K
configs = full
```

优点：

* 最优

缺点：

* tune多
* 不稳定
* runtime慢

baseline。

---

## B bucket autotune

```
key = bucket(M,N,K)
```

优点：

* tune少

缺点：

* 顺序依赖
* 次优

工程常用。

---

## C heuristic rule

```
config = f(M,N,K)
```

优点：

* 稳定

缺点：

* 难设计

工业常用。

---

## D offline table

```
bucket → best config
```

优点：

* 稳定
* 快

缺点：

* bucket依赖

TensorRT / CUTLASS 风格。

---

## E analytic policy

基于公式：

```
tile based on occupancy
```

优点：

* 无tune

缺点：

* 难写

研究方向。

---

## F script search

脚本自动搜索 config空间。

优点：

* 不用LLM

缺点：

* config空间要人设计

重要对照组。

---

## G LLM policy synthesis

LLM生成：

```
pick_config()
bucket()
```

特点：

* offline生成
* runtime不用LLM

研究重点。

---

## H LLM config space synthesis

LLM生成：

```
configs = [...]
```

然后：

```
autotune
```

研究：

config空间设计。

---

## I LLM kernel synthesis

LLM生成：

```
更robust kernel
```

目标：

减少 tuning。

高级方向。

---

# 7. 实验设计

## 7.1 kernel

选择：

```
Triton matmul
Triton attention
Triton fused kernel
```

至少一个。

---

## 7.2 hardware

固定：

```
1 GPU
1 dtype
```

避免变量过多。

---

## 7.3 shape workload

### uniform

```
M,N,K ∈ [64,4096]
```

### LLM style

```
M small
K large
```

### training style

```
all large
```

### adversarial

```
extreme shapes
```

---

## 7.4 metrics

必须记录：

```
compile time
tune time
runtime throughput
p99 latency
variance
cache size
key count
config count
```

否则无法比较。

---

# 8. LLM使用方式（严格限制）

LLM只允许：

```
生成代码
生成规则
生成config
生成bucket
生成policy
```

不允许：

```
预测性能
在线选择
在线推理
```

LLM输出必须：

```
可执行代码
可复现
可验证
```

---

# 9. 研究目标

本研究目标不是：

```
证明LLM更快
```

而是：

```
分析 tuning policy 设计空间
比较不同策略
评估 LLM 在 policy synthesis 中的作用
```

核心问题：

```
LLM能否减少人工设计成本
同时保持性能
```

不是：

```
LLM能否超过autotune
```

---

# 10. 后续阶段计划

### Phase 1

实现：

A B C D F

### Phase 2

实现：

G H

### Phase 3

实现：

I

### Phase 4

分析

```
性能
稳定性
复杂度
开发成本
```

---

# 11. 下一步建议

下一步应该做：

```
确定 kernel
确定 config集合
确定 bucket方案
确定 workload
写 baseline
```

然后再引入 LLM。

不要反过来。

---

如果你愿意，下一步我可以写：

> Phase 1 实施文档（具体到代码层面）

包括：

* Triton kernel
* configs
* bucket函数
* benchmark脚本
* 结果记录格式

这个才是真正能开始做实验的版本。


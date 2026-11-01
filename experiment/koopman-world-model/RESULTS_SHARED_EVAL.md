# 共享固定窗口评估（plan9 P0）

在线训练日志里的 `eval_nse_*` 每个 run 从各自的 replay buffer 采样，窗口和分布不同，
不能直接比较。这里把所有 checkpoint 放到同一个固定 `WindowBatch` 上计算开环 NSE：

```bash
.venv/bin/python scripts/eval_wm_checkpoints.py \
  --checkpoints /path/to/ckpt_a.pt /path/to/ckpt_b.pt \
  --windows /root/autodl-tmp/data/wm/eval_a_k32.pt \
  --out experiment/koopman-world-model/shared_eval.json
```

数据固定为 `eval_a_k32.pt`（32 步、`a` split），不随训练 run 改变。

## 结果

| checkpoint | iteration | E1 | E8 | E16 | E32 |
|---|---:|---:|---:|---:|---:|
| `state_large_ratio2_gate/ckpt_latest.pt` | 1000 | 0.05416 | 0.24581 | 0.52513 | 1.37678 |
| `lafan1_state_large_ratio1_stab0_v2/ckpt_0004000.pt` | 4000 | 0.05840 | 0.30255 | 0.87842 | 4.33377 |
| `lafan1_state_large_ratio2_stab0_v2/ckpt_0004000.pt` | 4000 | **0.05120** | 0.25347 | 0.89637 | 8.75806 |

原始 JSON：`shared_eval_gate_ratio1_ratio2.json`。

## 解释与限制

- 同一固定窗口上，ratio1 从 1000 到 4000 的 E32 由 1.38 升到 4.33；
  ratio2 4000 的 E1/E8 优于 ratio1，但 E16/E32 更差（8.76），说明短时域
  拟合改善没有转化为长时域泛化。
- 在线日志中的 `eval_nse32` 约为 0.84–1.02，而共享评估的 E32 为 4.33/8.76。
  两者**不是同一分布**：在线值来自当前 replay，固定值来自较早策略采样的
  `eval_a_k32`。
- 因此不能把共享评估的退化直接解释为“世界模型变差”，更可能是模型向在线
  策略分布 specialization；但也不能再用在线 NSE 宣称跨 run 改进。
- 结论：后续必须同时报告 (a) 共享固定窗口 NSE，(b) 真实仿真策略 return。
  仅看在线 `eval_nse_*` 会系统性高估 ratio 带来的收益。

## 真实仿真配对评估

评估协议与论文 benchmark 一致（`scripts/eval_benchmark.py`）：40 个 LAFAN1
clip 各从第一帧开始，确定性策略，无观测/动作噪声，无 reset noise，无 domain
randomization，foot friction 固定 1.0，`seed=0`。所有 checkpoint 使用同一
clip 集合和同一初始状态。

| checkpoint | iteration | 日志 train_ret | mean norm return | step-weighted | 平均步数 | failed |
|---|---:|---:|---:|---:|---:|---:|
| `state_large_ratio2_gate` | 1000 | 0.335 | **0.6029** | **0.5837** | 907 | 40/40 |
| `lafan1_state_large_ratio1...` | 4000 | 0.266 | 0.5016 | 0.5119 | 1907 | 40/40 |
| `lafan1_state_large_ratio2...` | 4000 | 0.351 | 0.4892 | 0.4805 | 4491 | 37/40 |

原始 JSON：`benchmark_policy_gate.json`、`benchmark_policy_ratio1_4000.json`、
`benchmark_policy_ratio2_4000.json`。

关键现象：

- ratio2 的在线 `train_ret` 最高（0.351）、共享 E1/E8 最好，但真实 benchmark
  **最低**（0.4892）；ratio1 也比 gate 低。
- ratio2 平均存活 **4491** 步，远长于 gate 的 907 步，并且有 3/40 个 clip
  跑完；这是一种“更会活、但全局根节点跟丢”的策略。
- 奖励分解显示 ratio2 的 `motion_global_root_pos` 只有 **0.00016**，而
  `motion_body_pos/ori` 仍为 0.0183/0.0123：局部身体跟踪尚可，全局轨迹已崩溃。
- 因此在线 NSE 和训练 return 都不能代表真实策略变好；后续必须以固定
  benchmark return + 全局 root 误差为主指标。ratio2 的“优势”目前不能成立。

每步奖励分解（40 clips 的步数加权）：

| checkpoint | root_pos | root_ori | body_pos | body_ori |
|---|---:|---:|---:|---:|
| gate | 0.00308 | 0.00452 | 0.01776 | 0.00997 |
| ratio1 | 0.00121 | 0.00152 | 0.01821 | 0.01140 |
| ratio2 | 0.00016 | 0.00111 | 0.01830 | 0.01227 |

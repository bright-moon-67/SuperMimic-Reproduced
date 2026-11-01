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

原始 JSON：`shared_eval_gate_ratio1.json`。

## 解释与限制

- 同一固定窗口上，ratio1 从 1000 到 4000 的 E32 由 1.38 升到 4.33，而
  ratio1 在线日志中的 `eval_nse32` 约为 0.94–1.02。两者**不是同一分布**：
  在线值来自当前 replay，固定值来自较早策略采样的 `eval_a_k32`。
- 因此不能把共享评估的退化直接解释为“世界模型变差”，更可能是模型向在线
  策略分布 specialization；但也不能再用在线 NSE 宣称跨 run 改进。
- 结论：后续必须同时报告 (a) 共享固定窗口 NSE，(b) 真实仿真策略 return。
  仅看在线 `eval_nse_*` 会系统性高估 ratio 带来的收益。
- 该结果只覆盖 gate 与 ratio1；ratio2 的 4000 checkpoint 生成后应加入同一张表。

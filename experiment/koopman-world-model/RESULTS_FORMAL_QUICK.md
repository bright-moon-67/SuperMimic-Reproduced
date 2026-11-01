# 固定数据容量/重构对照（plan6/plan7 quick screen）

数据：`/root/autodl-tmp/data/wm/lafan1_paper_quick.pt`（40 LAFAN1 clips、
policy rollouts、8192 envs × 1000 steps）。
协议：固定 `a_8` / `a_32` evaluation windows；`batch 4096 × 1`，2000 updates，
lr 2e-4，compile default。

这仍是 **quick screen**，不是完整 paper budget；用于排除容量混杂和验证
`rec_true` 消融，不替代长训结果。

## 结果

| run | 参数量 | E1 | E8 | E32 | 说明 |
|---|---:|---:|---:|---:|---|
| MLP 512³ | 0.63M | 0.0643 | 0.3253 | 1.713 | 容量匹配 small |
| MLP 1280³ | 3.55M | 0.0497 | 0.2184 | 0.881 | 容量匹配 large |
| MLP 2048³ | 8.82M | **0.0390** | **0.1449** | **0.495** | 旧 quick baseline |
| state small | 0.53M | 0.0599 | 0.1908 | **0.676** | rec_true=0 |
| state small + rec_true=0.1 | 0.53M | 0.0604 | 0.1935 | 0.699 | rec_true=0.1 |
| state large + rec_true=0.1 | 3.46M | 0.0549 | 0.1826 | 0.680 | rec_true=0.1 |

## 结论

1. **容量确实重要**：MLP 512→1280 有显著提升，1280→2048 仍在提升；不能再用
   0.53M state small 直接对比 8.82M MLP。
2. **容量匹配后 state Koopman 的长时域有优势**：3.46M state large 的 E8/E32
   已优于 3.55M MLP 1280，但 E1 仍略差；说明结构不是全面落后，而是短时域拟合
   和优化速度仍有差距。
3. **`rec_true=0.1` 目前没有收益**：state small 和 large 的 E1/E8/E32 都没有
   改善，甚至略差。先不要默认开启；后续应尝试更小权重、更长训练，并单独报告
   `rec_pred` / `rec_true` 曲线。

## 相关脚本

- `run_formal_quick_parallel.sh`
- 结果 JSON：`experiment/koopman-world-model/formal_quick/`
- 日志：`/root/autodl-tmp/logs/formal_quick/`

## 稳定性消融（state small，2000 updates）

| λ_stab | E1 | E8 | E32 | cons_rel | erank | max σ(M) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.0563** | **0.1766** | **0.6141** | 0.572 | 21.41 | 2.003 |
| 1 | 0.0574 | 0.1808 | 0.6298 | 0.541 | 21.18 | 1.356 |
| 10 | 0.0599 | 0.1908 | 0.6760 | 0.528 | 20.64 | 1.10 |

稳定性惩罚确实把 `σ_max(M)` 从 2.00 压到 1.10，但 E1/E8/E32 反而单调变差。
这说明对 state-augmented 模型，内部潜空间谱半径不能代表状态积分后的闭环敏感度；
λ_stab=10 对这个结构过强。后续 Koopman 在线实验应优先测试 λ_stab=0 或 1，
而不是继续加稳定性约束。结果文件：`stab_ablation/*.json`。

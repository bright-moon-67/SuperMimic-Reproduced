# Koopman World Model 实验编排（plan_one.md 第 5/7 节）

基线资产：`logs/baseline_mlp_backup/ckpt_latest.pt`（MLP baseline checkpoint + 配置 + eval 结果）。

> **2026-10-31 参数对齐（重要）**：容量并行实验 c1–c4 已全部停止，不作为结果。
> 后续正式对照统一用 `run_paper_aligned.sh`，对齐 `origin/main` 的 `lafan1_a800`：
> LAFAN1 **40 条 clip**、预训练策略采样、`return` 采样、8192 envs、
> WM batch **4096 × 10 batches/update**、MLP hidden **2048³**、lr 2e-4、window 8、
> `compile=default`，以及 320k env steps × 0.4 UTD = **128k WM updates**。
> `run_p2b_sweep.sh` / `run_capacity_parallel.sh` 保留为显式 legacy 诊断
> （batch 2048、1 batch/update、`compile=none`），不是原仓库参数。
>
> ```bash
> # 完整对齐（默认 128k WM updates，耗时很长；先确认预算）
> bash experiment/koopman-world-model/run_paper_aligned.sh
>
> # 快速冒烟/诊断：只改 update 预算，其余参数仍按原仓库
> UPDATES=3000 EVAL_INTERVAL=500 bash experiment/koopman-world-model/run_paper_aligned.sh
> ```
>
> **在线 vs 离线**：上面 `run_paper_aligned.sh` 是**冻结 replay 快照的离线 WM 诊断**，
> 不能复现源仓库“策略持续产生新数据、WM 与 policy 共同适应”的分布。若要严格对齐
> 源仓库的 LAFAN1 实验，用 `run_lafan1_online.sh`（`sm-train`，8192 envs，
> 40 clips，320k iterations，UTD 0.4）；离线结果只用于快速筛选结构和损失。
>
> **增广状态 \([z;\xi]\)**：设置 `XI_STATE=1` 可打开 `koopman.use_xi_state`。
> 新增 `A_xi`、`V_xi`，零初始化；`z_{t+1}` 每步接收归一化 \(\xi\)，但 \(\xi\) 块保持
> 恒等，P2-B 的谱约束仍只作用于 \(z\) 块（`M_z(a)`），不作用于含 \(\xi\) 的增广矩阵。
>
> **状态增广 Koopman**：`STATE_KOOPMAN=1` 打开 `koopman.state_in_latent`；
> \(z=[\mathrm{canonical}(s);\phi(s,\xi)]\)，dynamics 输出
> \(u=(A z+\dots)\)，线性 decoder 把 \(u[:71]\) 映射为 70 维 delta，
> 再用 `StateSpec.integrate` 得到 \(s_{t+1}\)，rollout 不重新编码。
> 支持 domain-state、action-state、action-domain 低秩耦合；稳定性约束作用于
> \(M(a,\xi)=A+\sum a_i U_iV_i^\top+\sum_j\xi_jP_jQ_j^\top\)。
> 快速对照：`bash experiment/koopman-world-model/run_state_koopman_quick.sh`。

> **2026-10-31 P0 更新**：推荐先冻结数据，再做 MLP/Koopman 对照。旧命令（每次 run 重新采集）
> 只用于冒烟，不是 held-out 协议。
>
> ```bash
> # 1) 冻结数据 + A/B 评估窗口（一次）
> .venv/bin/python experiment/koopman-world-model/collect_wm_data.py \
>     --out /root/autodl-tmp/data/wm/lafan1.pt
>
> # 2) 同一份数据训练，保存权重
> .venv/bin/python experiment/koopman-world-model/train_wm_only.py \
>     --wm-type koopman --protocol a \
>     --data-in /root/autodl-tmp/data/wm/lafan1.pt \
>     --eval-modes latent,reencode \
>     --save-wm /root/autodl-tmp/logs/wm/k0r_latent.pt \
>     --out experiment/koopman-world-model/wm_k0r_latent.json
>
> # 3) 同权重诊断
> .venv/bin/python scripts/diag_koopman.py \
>     --wm-state /root/autodl-tmp/logs/wm/k0r_latent.pt \
>     --eval-windows-a8 /root/autodl-tmp/data/wm/eval_a_k8.pt \
>     --eval-windows-a32 /root/autodl-tmp/data/wm/eval_a_k32.pt \
>     --out experiment/koopman-world-model/diag_p1.json
> ```
>
> 详细阶段与门禁见 `docs/KOOPMAN_FIX_PLAN.md`。

## 实验 1：独立 WM 训练（阶段 B）——同一批真实转移对照 MLP vs Koopman

```bash
# 随机策略采样数据（快速起步）
.venv/bin/python experiment/koopman-world-model/train_wm_only.py \
    --wm-type koopman --collect-steps 2000 --updates 3000 --out experiment/koopman-world-model/wm_koopman.json
.venv/bin/python experiment/koopman-world-model/train_wm_only.py \
    --wm-type mlp     --collect-steps 2000 --updates 3000 --out experiment/koopman-world-model/wm_mlp.json

# 用已训练 Actor 采样数据（策略相关状态，推荐）
.venv/bin/python experiment/koopman-world-model/train_wm_only.py \
    --wm-type koopman --policy-ckpt /root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt \
    --out experiment/koopman-world-model/wm_koopman_actor.json
```

成功标准：Koopman 的 E1/E8 与 MLP 同量级，E32 不发散，consistency 不漂移。

## 实验 2：Koopman 联合训练（从头 Actor，实验 D）

```bash
.venv/bin/sm-train --run-name koopman_lafan1 \
    --data.npz-dir data/npz/lafan1 \
    --env.num-envs 8192 \
    --world-model.type koopman \
    --world-model.koopman-mode latent \
    --world-model.koopman.dz 128 \
    --world-model.koopman.rank 8 \
    --log-root /root/autodl-tmp/logs
```

## 实验 3：已有 Actor + Koopman 续训（实验 B，模型替换适应）

```bash
.venv/bin/sm-train --run-name koopman_adapt \
    --resume /root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt \
    --resume-skip wm \
    --world-model.type koopman \
    --world-model.koopman.dz 128 \
    --world-model.koopman.rank 8 \
    --log-root /root/autodl-tmp/logs
```

## 实验 4：MLP 对照（实验 A/C）

```bash
.venv/bin/sm-train --run-name mlp_baseline --log-root /root/autodl-tmp/logs          # 从头（C）
.venv/bin/sm-train --run-name mlp_continue --resume /root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt --log-root /root/autodl-tmp/logs  # 续训（A）
```

## 判定口径

- 唯一最终标准：真实 MuJoCo Warp 仿真 Return（`scripts/eval_benchmark.py`）。
- 同时记录 Koopman 想象 Return 与真实 Return 的差值，检测模型利用（plan 测试 4）。
- Koopman 诊断：`wm/lift_loss`、`wm/consistency`、`wm/rec_loss`、`wm/nse_kstep`（TensorBoard）。

## 其他文件

- `smoke_koopman.py`：WM 级冒烟（300 步训练 + BPTT + reencode）
- `smoke_train_koopman.py`：完整训练循环冒烟（含 torch.compile）
- `train_wm_only.py`：独立 WM 对照实验脚本

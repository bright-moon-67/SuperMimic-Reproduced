# Koopman World Model 实验编排（plan_one.md 第 5/7 节）

基线资产：`logs/baseline_mlp_backup/ckpt_latest.pt`（MLP baseline checkpoint + 配置 + eval 结果）。

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

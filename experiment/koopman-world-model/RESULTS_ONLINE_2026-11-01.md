# 在线 LAFAN1 训练结果（2026-10-31 / 2026-11-01）

本次记录的是 **在线联合训练**（`sm-train`，8192 envs，LAFAN1 40 clips），不是
`train_wm_only.py` 的离线 quick run。两个 run 使用同一个旧 Actor/Critic
checkpoint 初始化，训练数据、环境参数、policy trainer 和 schedule 均对齐，唯一
区别是世界模型：

- `lafan1_state_koopman_plan5_fix`：state-augmented Koopman，`z=[canonical(s); φ(s,ξ)]`，
  `dz=128`、`rank=8`、domain rank 8、action-domain rank 4、xi-state、P2-B stability；
- `lafan1_mlp_online_plan5`：MLP `2048×2048×2048`（8.81M），普通 delta 预测。

共同配置：`TOTAL_ITERATIONS=320000`、`MAX_HOURS=24`、checkpoint interval 4000、
`NUM_ENVS=8192`、policy batch 8192、WM batch `4096×10`、UTD 0.4。

## 运行结果

两个 run 都在 2026-11-01 02:00 左右因交互会话被清理而停止；没有 NaN/OOM。
停止时 GPU 为空，checkpoint 已正常保存。

| run | 最后 iteration | 训练时长 | wm nse1 | wm nse8 | imag_ret | survival | train_ret | 速度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| state Koopman | 9950 | 3.13 h | 0.1688 | 0.7383 | 0.792 | 0.917 | 0.089 | 0.9 it/s |
| MLP 2048³ | 12150 | 2.78 h | **0.0119** | **0.0221** | **2.614** | **1.000** | **0.755** | 1.3 it/s |

代表性曲线：

| model | it 250 | it 500 | it 1000 | it 5000 | it 8000 / 10000 | final |
|---|---:|---:|---:|---:|---:|---:|
| state nse1 | **0.1011** | 0.1120 | 0.1415 | 0.2286 | 0.1560 | 0.1688 |
| state nse8 | **0.4403** | 0.5862 | 0.7295 | 1.0752 | 0.6883 | 0.7383 |
| state imag_ret | 1.326 | 1.074 | 0.958 | 0.691 | 0.858 | 0.792 |
| MLP nse1 | 0.0747 | 0.0769 | 0.0599 | 0.0196 | 0.0132 | 0.0119 |
| MLP nse8 | 0.4363 | 0.4030 | 0.2275 | 0.0409 | 0.0284 | 0.0221 |
| MLP imag_ret | 1.327 | 1.431 | 1.876 | 2.504 | 2.604 | 2.614 |

## 结论

1. **MLP 在线训练全面优于当前 state-augmented Koopman**：最终 WM 单步误差低约
   14×，8 步误差低约 33×，imagination return 高约 3.3×。
2. state Koopman 的 WM 误差在 250 iteration 附近最低（nse1≈0.101，nse8≈0.440），
   之后随着策略分布漂移明显上升；MLP 则持续下降并稳定在 0.012/0.022。
3. 当前 state-augmented 结构在离线 quick run 中能把 E32 从经典 Koopman 的
   683 降到约 0.68，但**在线闭环下仍无法跟上策略分布**，暂不适合作为主训练模型。
4. 本次在线 NaN 已定位并修复：四元数积分中有限但过大的旋转增量导致 float32
   `angle²` 溢出，已在 `StateSpec.integrate` 增加 `delta_clip` / `max_rot_inc`
   限幅，并在 WM trainer 中跳过非有限 loss/grad。

## Checkpoints

- state：`/root/autodl-tmp/logs/lafan1_state_koopman_plan5_fix/ckpt_0004000.pt`、
  `ckpt_0008000.pt`、`ckpt_latest.pt`
- MLP：`/root/autodl-tmp/logs/lafan1_mlp_online_plan5/ckpt_0004000.pt`、
  `ckpt_0008000.pt`、`ckpt_0012000.pt`、`ckpt_latest.pt`
- 日志：`/root/autodl-tmp/logs/lafan1_state_koopman_plan5_fix.nohup.log`、
  `/root/autodl-tmp/logs/lafan1_mlp_online_plan5.nohup.log`

## 复现/续训

两个 run 的命令分别由 `run_lafan1_online.sh` 生成：

```bash
# state-augmented Koopman
NUM_ENVS=8192 TOTAL_ITERATIONS=320000 MAX_HOURS=24 \
RUN_NAME=lafan1_state_koopman_plan5_fix LOG_ROOT=/root/autodl-tmp/logs \
STATE_KOOPMAN=1 XI_STATE=1 STAB=1 \
RESUME=/root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt \
RESUME_MODE=transfer_policy \
bash experiment/koopman-world-model/run_lafan1_online.sh

# MLP baseline
NUM_ENVS=8192 TOTAL_ITERATIONS=320000 MAX_HOURS=24 \
RUN_NAME=lafan1_mlp_online_plan5 LOG_ROOT=/root/autodl-tmp/logs \
WM_TYPE=mlp STAB=0 XI_STATE=0 STATE_KOOPMAN=0 \
RESUME=/root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt \
RESUME_MODE=transfer_policy \
bash experiment/koopman-world-model/run_lafan1_online.sh
```

长时间运行必须使用 `setsid`（或 watchdog）脱离交互会话，否则 SSH/VS Code
会话清理时进程会被一起杀掉。`watch_lafan1_runs.sh` 提供了带 checkpoint 的
自动重启保护。

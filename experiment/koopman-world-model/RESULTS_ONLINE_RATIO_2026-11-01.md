# Koopman 在线 ratio 对照记录（2026-11-01）

## 运行配置

两个 run 都使用：

- 环境：LAFAN1 40 clips，8192 envs，10s episodes，action noise 0.1，
  return sampling，replay capacity 1000，obs/reset noise，domain randomization
- 世界模型：State-Augmented Koopman large
  - `dz=256`、`rank=16`、`rank_domain=8`、`rank_action_domain=4`
  - encoder/decoder `1024×1024`，WM 参数约 **3.46M**
  - `state_in_latent=True`、`use_xi_state=True`
  - `lambda_cons=0.1`、`cons_target_detach=True`
  - `lambda_lift_phys=0.0`、`lambda_lift_phi=0.1`
  - `lambda_rec_true=0.0`
  - `lambda_stab=0.0`、`a_rho_target=0.99`
- Policy：H=32、batch 8192、history 10、lr 2e-4
- WM：window 8、batch 4096×10、lr 2e-4、compile default
- Schedule：UTD 0.4，warmup 40，320k 名义总步数，24h 上限
- 两个 run 都从 `state_large_ratio2_gate/ckpt_latest.pt` 的 iteration 1000
  以 `RESUME_MODE=full` 恢复；policy 最初来自旧 `lafan1_a800` checkpoint，
  因此**不是从头训练**

运行：

| run | WM:Actor | log |
|---|---|---|
| `lafan1_state_large_ratio2_stab0_v2` | 2:1 | `/root/autodl-tmp/logs/lafan1_state_large_ratio2_stab0_v2.nohup.log` |
| `lafan1_state_large_ratio1_stab0_v2` | 1:1 | `/root/autodl-tmp/logs/lafan1_state_large_ratio1_stab0_v2.nohup.log` |

两个进程都用 `setsid nohup` 启动，已脱离交互会话。GPU 约 66.6 GB / 99%。

## 最新在线指标

| run | iteration | train_nse1 | eval_nse1 | eval_nse8 | eval_nse32 | imag_ret | surv | train_ret | it/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio2 | 2500 | **0.0557** | **0.0443** | **0.1444** | **0.6950** | **1.658** | **0.979** | **0.350** | 0.6 |
| ratio1 | 3400 | 0.0670 | 0.0506 | 0.1766 | 0.9241 | 1.599 | 0.977 | 0.316 | 0.9 |

同 iteration 对照（2500）：

| run | train_nse1 | eval_nse8 | eval_nse32 | imag_ret | surv | train_ret |
|---|---:|---:|---:|---:|---:|---:|
| ratio2 | **0.0557** | **0.1444** | **0.695** | **1.658** | **0.979** | **0.350** |
| ratio1 | 0.0944 | 0.2587 | 1.298 | 1.327 | 0.962 | 0.230 |

## Return 变化

ratio2：

| iteration | 1050 | 1500 | 2000 | 2300 | 2500 |
|---|---:|---:|---:|---:|---:|---:|
| train_ret | 0.161 | 0.205 | 0.320 | 0.353 | **0.350** |
| imag_ret | 1.414 | 1.093 | 1.365 | 1.527 | **1.658** |

ratio1：

| iteration | 1700 | 2000 | 2500 | 3000 | 3400 |
|---|---:|---:|---:|---:|---:|---:|
| train_ret | 0.146 | 0.221 | 0.230 | 0.241 | **0.316** |
| imag_ret | 0.958 | 1.123 | 1.327 | 1.487 | **1.599** |

判断：

- 两个 run 的 `train_ret` 和 `imag_ret` 总体都在上升；
- ratio2 在相同 iteration 明显更好，return 已达到 0.35 左右；
- ratio1 速度更快，但 return 上升更慢；
- 目前仍远低于已有 MLP 长实验：MLP 在 2000 iteration 约 0.592，
  12000 iteration 约 0.756。

## 梯度统计

单步动作 Jacobian（`scripts/diag_gradient.py`）：

| checkpoint | 方向数 | cosine mean | relative L2 | magnitude ratio |
|---|---:|---:|---:|---:|
| state large（gate 1000） | 32 | 0.935 | 0.413 | 0.720 |

方向基本一致，但 Koopman 的动作敏感度约为真实仿真的 **0.72×**。
这是单步诊断，不是多步 BPTT 或策略梯度误差。

## 离线固定数据对照

| 模型 | 参数量 | E1 | E8 | E32 |
|---|---:|---:|---:|---:|
| MLP 1280³ | 3.55M | 0.0497 | 0.2184 | 0.881 |
| State small λ0 | 0.53M | 0.0563 | 0.1766 | 0.614 |
| State large λ0 | 3.46M | **0.0520** | **0.1629** | **0.606** |
| MLP 2048³ | 8.82M | **0.0390** | **0.1449** | **0.495** |

稳定性消融（state small）：

| λ_stab | E1 | E8 | E32 |
|---:|---:|---:|---:|
| 0 | **0.0563** | **0.1766** | **0.6141** |
| 1 | 0.0574 | 0.1808 | 0.6298 |
| 10 | 0.0599 | 0.1908 | 0.6760 |

## 结论

1. ratio2 目前在线 WM 指标和 return 都优于 ratio1；
2. ratio2 的计算成本约为 ratio1 的 2 倍，速度约 0.6 vs 0.9 it/s；
3. 两个 Koopman run 都仍明显落后于已有 MLP 长实验；
4. 当前没有 4000 iteration checkpoint；第一个正式 checkpoint 预计在约
   2500/4000 → 还需要一段时间。

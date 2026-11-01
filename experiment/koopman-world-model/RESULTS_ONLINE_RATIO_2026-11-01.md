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
4. ratio1 已到 4000 iteration 并保存 `ckpt_0004000.pt`；ratio2 也已在 4000
   保存 checkpoint；
5. **plan9 复核后必须加限制**：以上结论只对“当前 replay 的在线指标”成立。
   同一固定窗口和真实仿真 benchmark 上，gate > ratio1 > ratio2（见文末
   “plan9 复核”），ratio2 的在线优势没有转化为真实策略优势。

## 后续更新（2026-11-01 11:05 CST）

| run | iteration | train_nse1 | eval_nse1 | eval_nse8 | eval_nse32 | imag_ret | surv | train_ret | wm_gn | pi_gn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio2 | 2900 | 0.0510 | 0.0434 | 0.1425 | 0.828 | 1.723 | 0.976 | 0.343 | 22.3 | 0.065 |
| ratio1 | 4000 | 0.0631 | 0.0497 | 0.1627 | 0.913 | 1.624 | 0.971 | 0.266 | 27.9 | 0.066 |

- ratio2 的 `train_ret` 最近 8 个点平均约 **0.337**，在 0.32–0.35 平台波动；
- ratio1 的 `train_ret` 最近 8 个点平均约 **0.279**，也在平台期；
- ratio2 的 `eval_nse8` 已到 **0.1425**，明显低于 ratio1 的 0.1627；
- 两者仍未追上 MLP；ratio1 已有 4000 checkpoint；ratio2 在 4000 也生成了
  checkpoint（见下方 plan9 复核，在线指标不能代表真实策略）。

## plan9 复核：共享固定窗口与真实仿真

上面的 `eval_nse_*` 是每个 run 从自己的 replay 采样，不是同一测试集。
plan9 改为在固定 `eval_a_k32.pt` 上评估所有 checkpoint：

| checkpoint | iteration | E1 | E8 | E16 | E32 |
|---|---:|---:|---:|---:|---:|
| gate | 1000 | 0.05416 | 0.24581 | 0.52513 | 1.37678 |
| ratio1 | 4000 | 0.05840 | 0.30255 | 0.87842 | 4.33377 |
| ratio2 | 4000 | **0.05120** | 0.25347 | 0.89637 | 8.75806 |

同时用 `scripts/eval_benchmark.py`（40 clips、确定性、无噪声/DR、seed 0）
做真实仿真配对评估：

| checkpoint | iteration | 日志 train_ret | mean norm return | step-weighted | 平均步数 | failed |
|---|---:|---:|---:|---:|---:|---:|
| gate | 1000 | 0.335 | **0.6029** | **0.5837** | 907 | 40/40 |
| ratio1 | 4000 | 0.266 | 0.5016 | 0.5119 | 1907 | 40/40 |
| ratio2 | 4000 | 0.351 | 0.4892 | 0.4805 | 4491 | 37/40 |

- 共享窗口上 ratio2 的 E1/E8 最好，但 E32 最差；在线 `eval_nse32` 的改善
  不能推广到固定数据。
- 真实 benchmark 上 gate > ratio1 > ratio2，和在线 `train_ret`/NSE 的顺序
  相反。ratio2 平均存活 4491 步且 3/40 跑完，但
  `motion_global_root_pos` 只有 0.00016（gate 为 0.00308）：策略学会了
  局部身体跟踪和存活，却丢失了全局根节点轨迹。
- 因此 ratio 结论必须改为：“ratio2 在当前 replay 的短时域拟合上更好，
  但没有带来真实 benchmark 提升；在线指标不能作为策略改进证据。”
  下一步应优先修全局 root 奖励/终止与真实 benchmark，而不是继续提高 ratio。


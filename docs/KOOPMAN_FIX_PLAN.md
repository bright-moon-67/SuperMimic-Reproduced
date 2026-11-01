# Koopman 世界模型修复计划 v2

**状态：P0–P2-B 已实施（2026-10-31）；容量并行 c1–c4 已中止；下一步统一按 upstream LAFAN1 参数重跑。**
代码基线：`6ea5b92`（`main`）。

> **2026-10-31 参数对齐**：停止所有正在进行的训练。后续正式对照使用
> `experiment/koopman-world-model/run_paper_aligned.sh`，对齐 `origin/main` 的
> `lafan1_a800`：40 条 LAFAN1、policy rollouts、return sampling、8192 envs、
> WM batch 4096×10、MLP 2048³、lr 2e-4、window 8、compile default、128k WM updates。
> 旧的 8-clip/2048-env/1-batch 配置只保留为 legacy 诊断。
> **在线严格对齐**用 `experiment/koopman-world-model/run_lafan1_online.sh`
> （`sm-train`，LAFAN1 only，320k iterations，UTD 0.4）；`run_paper_aligned.sh`
> 只是冻结快照的离线 WM 对照。
版本：v2（2026-10-31），根据 `reference/plan3.md` 的技术审查修订 v1。
目标：先证明 32 步失稳的机制，再选择修复方法；在通过门禁前冻结 Actor 联合训练。

配套文件：

- `reference/plan2.md`：第一轮外部审查（问题清单）。
- `reference/plan3.md`：第二轮技术审查（本版逐条响应，见附录 A）。
- `reference/plan_one.md`：Koopman 第一版设计与公式。
- `experiment/koopman-world-model/RESULTS.md`：已有实验结果（本计划会修正）。
- `docs/EXPERIMENTS.md`：实施记录。

### v2 相对 v1 的主要修订

| # | v1 | v2 |
|---|---|---|
| 1 | 认定 v1 reencode 结果“数学上不可能发散” | 改为“数值异常、不可用于结论、待复核”；列出 6 项复核项 |
| 2 | 把 Frobenius 当作 σ_max 的等价廉价替代 | 明确二者不等价；Frobenius 仅作 Matrix Weight Decay Baseline；训练用幂迭代需与精确 SVD 校准 |
| 3 | 认为 `detach` 可避免表示坍缩 | 明确 `detach` 只阻断共同追逐，不保证不坍缩；新增方差/有效秩监控与可选 anti-collapse 正则 |
| 4 | `B_xi` “始终注册，旧 checkpoint 可直接加载” | 改为显式迁移：`strict=False` + 缺失键白名单；不恢复不兼容的优化器状态；修正零初始化测试 |
| 5 | G5 用 `∂s_H/∂a_0` 代表策略梯度 | 拆成 J1/J2a/J2b/J3；G5b 才对应策略梯度 |
| 6 | 只有一套 train/val/test 划分 | 拆成协议 A（分布内）与协议 B（跨运动），分开报告 |
| 7 | 在线 `wm_gate` 自动跳过 policy update | 第一轮改为“离线门禁决定是否启动；在线只记录”，自动 skip 默认关闭 |
| 8 | ξ 条件化紧跟稳定性实验 | 推迟到根因与容量实验之后；P2 只做两项正交实验 |

### 对 `plan3.md` 的总体判断

**同意并采纳**：

- reencode 不是闭环稳定性的保证；`detach` 不是防坍缩保证；Frobenius 与 σ_max 不等价；
- `B_xi` 的旧 checkpoint 需要显式迁移；零初始化下的 ξ 测试必须拆成三部分；
- G5 混淆了“状态 Jacobian”和“策略梯度”，必须拆成 J1/J2a/J2b/J3；
- 在线 `wm_gate` 不应在第一轮自动 skip；P2 应拆成两项正交实验。

**进一步限定**：

1. plan3 说“不能仅凭 E8 与 `nse_kstep` 不同就证明数据自相矛盾”——同意；
   但在当前提交的代码下，reencode 训练与评估是同一组算子，所以该 JSON 仍然
   **不可用于性能结论**，应标注“异常、待复核”，而不是直接接受它的 E8。
2. plan3 建议保留 Frobenius 作为“更便宜对照”——本计划保留，但明确它是
   **matrix weight decay baseline**；若它有效，必须排除“只是把动力学压小”的解释。
3. plan3 的 J3 全参数方向差分成本高；本计划先用低维子空间/SPSA + 公共随机数，
   不要求第一轮做完整高维梯度。
4. plan3 建议 G3 同时报告方差/有效秩——同意；本计划额外报告 `erank(x_true)` 作为参照，
   避免把“数据分布本身很窄”误判为潜空间坍缩。

**暂不采纳**：

- 第一轮不引入 `A(ξ)`、EMA target encoder 或复杂稳定性约束；
- P2 之前不加入 ξ 条件化和大容量；
- 不把在线自动 skip 作为第一轮门禁，先用固定验证集做离线门禁。

---

## 0. 一页结论

`plan2.md` 的 P0 判断成立，但现有实验还缺三类关键证据，不能直接按它的表格做归因：

1. **没有保存 Koopman 权重。** `train_wm_only.py` 只写 JSON，导致 plan2 要求的
   “同一权重下 `latent` vs `reencode` 的 32 步对比”实际上没有做过。
2. **已有的 reencode 对照不是同权重对照。** `wm_koopman_reencode_v2.json` 是另一套训练目标
   （每步重编码），与 `wm_koopman.json`（latent 训练）是两个不同模型；
   两者的 E1/E8/E32 差异混合了“训练方式”和“评估方式”两个因素。
3. **`wm_koopman_reencode.json`（v1）内部数值不一致，暂不可用于任何结论。**
   同一文件里 `E8=1.28e6`，而训练指标 `nse_kstep=0.43`。按当前提交的代码，
   reencode 训练与 `rollout_error` 的 reencode 分支在代数上是同一组算子
   （`E → K → D → Integrate`），这组数字不太可能来自同一份权重/同一版本代码。
   但这**不表示** reencode 天生稳定：reencode 仍是闭环映射
   $s_{t+1}=\operatorname{Integrate}(s_t, D(K(E(s_t),a_t)))$，
   离开训练分布后仍可能发散。因此标注为“**异常、待复核**”，并检查：
   1. 训练与评估的 `koopman_mode` 是否一致；
   2. 训练窗口（8 步）与评估窗口（32 步）的输入分布是否不同；
   3. `nse_kstep` 是否确实是训练窗口最后一步；
   4. 评估窗口是否包含 NaN/极端长尾状态；
   5. `rollout_error()` 与训练是否使用相同的积分/裁剪路径；
   6. 结果 JSON 是否缺少当时的 config/commit（当前确实缺少）。

   其中第 3 项在当前代码中可确认：`nse_kstep` 是训练窗口**最后一步**的指标；
   第 1 项在当前代码中是同一个 `cfg.koopman_mode`，但 v1 的代码版本无法核对；
   第 6 项确实缺失。第 2 项（8 步 vs 32 步窗口分布）是主要差异来源之一。

本机数值复核（`dz=128`，`A = 0.99I + 0.01N`，seed 0/1/2）：

| seed | ρ(A) | max σ(M_t) | 32 步 ‖z‖ 放大（中位数 / 最大） |
|---|---|---|---|
| 0 | 1.1089 | 1.1507 | 9.46× / 16.8× |
| 1 | 1.0958 | 1.1512 | 9.34× / 13.1× |
| 2 | 1.0995 | 1.1549 | 9.95× / 15.0× |

即使双线性项按 `0.01` 初始化、动作取 `0.3·N(0,1)`，**初始化本身**就会在 32 步内把潜状态放大
约一个数量级。P0-1 的机制在初始化层面得到确认；训练后的 `ρ(A)`、`σ_max(M_t)` 仍需从
checkpoint 读取（当前没有 checkpoint，见 P0）。

已跑结果中真正有效的信息：

| 运行 | 模式 | E1 | E8 | E32 | consistency | 参数量 |
|---|---|---|---|---|---|---|
| `wm_mlp` | MLP | **0.0385** | **0.0588** | **0.247** | — | 8.81M |
| `wm_koopman` | latent | 0.2468 | 0.8037 | 9.99e5 | 2.835 | 0.443M |
| `wm_koopman_lift1` | latent，λ_lift=1.0 | 0.1938 | 0.6127 | 2.65e5 | 0.315 | 0.443M |
| `wm_koopman_reencode_v2` | reencode | 0.1127 | 0.4620 | 7.33 | 0.525 | 0.443M |
| `wm_koopman_reencode` | **异常（待复核）** | 0.1070 | 1.28e6 | 2.55e25 | 0.500 | 0.443M |

解读：

- λ_lift 提高 10× 只把 consistency 从 2.83 压到 0.315，E32 仍是 `1e5` 量级 →
  **“只加 lift 权重”不是有效修复**，与 plan2 §4 的判断一致。
- reencode 把 E32 从 `1e6` 压到 `7.3`，说明“每步把潜状态锚回物理状态”方向可能有效；
  但仍比 MLP 差约 30×，E1 也差 2.9× → **除了潜状态漂移，还有单步精度/容量问题**。
- 两个 run 的训练目标不同，所以上面的 reencode 优势只能作为弱证据；同权重 K1 仍是第一优先级。
- 另有一个容易被忽略的分布差异：训练 `nse_kstep` 来自 `window=8` 的窗口，而评估 `E8`
  来自 `window=32` 的窗口（要求 32 步都有效）。两者的第 8 步不是同一批状态。
  P0 会同时报告 `E8@8-step-windows` 与 `E8@32-step-windows`，避免把分布差异当成模型差异。

---

## 1. 验收门禁 v2

门禁分三类：**硬门禁**（必须通过才能进入下一步）、**诊断指标**（必须报告，不单独判死）、
**工程容忍**（人为设定的阶段性阈值，不是理论指标）。所有门禁都在
“固定数据 + 固定评估窗口 + 固定 checkpoint”下判定。

| 门禁 | 指标 | 通过标准 | 性质 |
|---|---|---|---|
| G0a 可复现 | 同一 checkpoint、同一设备、同一 seed 跑两次 `diag_koopman.py` | 关键数值相对误差 ≤1e-5（float32）/≤1e-10（float64）；排除 timing/路径字段 | 硬门禁 |
| G0b 模式一致性 | 同一权重下 `s_1^latent` 与 `s_1^reencode` | `max‖Δ‖ ≤ 1e-6`（float32）；若不等，先查实现，不继续归因 | 硬门禁 |
| G0c 结果溯源 | 每个结果 JSON / checkpoint | 含完整 config、git commit、数据 hash、评估窗口 hash、argv | 硬门禁 |
| G1 长时域稳定 | 固定 val/test 上的逐步 `E1..E32` 曲线 | 有限；无不正常的指数放大；最佳 checkpoint 的 E32 ≤ 3× MLP；报告平滑趋势/置信区间 | 硬门禁（3× 为工程容忍） |
| G2 精度 | E1 / E8 | E1 ≤ 2× MLP，E8 ≤ 3× MLP | 工程容忍 |
| G3 表示健康 | `cons_abs`、`cons_rel`、`rec_true`、逐维方差、有效秩 | 相对 K0r：`cons_rel` 至少下降 50%，且 `rec_true`/方差/有效秩不退化；`cons_rel ≤ 0.05` 仅作参考目标 | 硬门禁（多指标联合，阈值由 P0/P1 校准） |
| G4 潜状态幅度 | `‖z_t‖` 曲线、`max_t/median_t` | 不出现异常放大（相对初始化尺度的阈值由 P0 数据校准）；不只看“是否单调增长” | 诊断指标 |
| G5a 动力学敏感度 | J2b：WM 与仿真的固定动作 `(∂s_H/∂a)v` 方向 | 非接触段 cosine ≥ 0.5；接触段单独报告 | 诊断指标（P4 前不作硬门禁） |
| G5b 策略梯度 | J3：`(∇_θ J_WM)·v_θ` 与仿真回报中心差分方向 | 多个随机方向的平均 cosine ≥ 0.5，并给出置信区间 | 硬门禁（恢复 Actor 前） |
| G6 联合训练 | 真实仿真 Return | ≥ 95% MLP；同时报告失败动作数、样本效率、GPU-hours | 工程容忍 |

- G1–G4 未通过前，Koopman 配置不进入 `sm-train` 的 Actor 联合训练。
- **阶段性验收用相对改进**：P2-A/P2-B 先看“相对 K0r 是否改善”，不要求一次达到 G1/G2 的
  绝对阈值；只有最终候选（P3/P4 前）才用绝对阈值判定。
- **不要用单个标量作充分条件**：G3 的相对一致性可以被大的近常数分量“稀释”；
  G4 的范数增长也可能是合理动力学。必须同时看绝对误差、重构误差、方差/有效秩和物理状态误差。
- 每个模型至少报告 `E1..E32` 的完整曲线，而不是只报 E32。

---

## 2. P0：冻结基线与修复实验协议（不改模型行为）

> 2026-10-31 已落地：`ReplayBuffer` 快照/固定采样、`collect_wm_data.py`、
> `scripts/diag_koopman.py`、`train_wm_only.py --data-in/--save-wm/--eval-windows*`、
> 诊断测试。下方 `[ ]` 保留为验收清单。

这一阶段的目标是让后续每个结论都**可复现、可归因、可比较**。在默认 `--window 8` 下，
全部改动不改变 K0 的训练数值；任何改变窗口长度的实验都需要重新标定基线。

### 2.1 固定数据集、双评估协议、固定评估窗口

新增 `experiment/koopman-world-model/collect_wm_data.py`：

- **在一个 env 中加载 train + val + test 的全部 clip**（当前 `train_wm_only.py` 只加载
  `walk1_subject1` + `dance1_subject1`，无法从这两个运动里划出另外两个运动的验证集）。
- 采集一次并保存到 `/root/autodl-tmp/data/wm/<name>.pt`（`data/`、`*.pt` 已在 `.gitignore`）。
- 保存字段：`states (k+1,n,71)`、`a_exec (k,n,29)`、`a_clean (k,n,29)`、`xi (n,33)`、
  `done (k,n)`、`frame (k,n)`、`clip`、`ep_id`、`valid`。
- **窗口完整性**：只使用 `buffer.window_mask(k)` 允许的窗口，要求同一 `ep_id`、
  不跨终止、不跨写指针；导出后断言 `ep_id[0] == ep_id[-1]`。
- **保留可审计的原始行级索引**：除窗口外，保存 `ep_id`、`clip`、`frame`、`terminal`、
  `ep_start`、`ptr` 等行级元数据（或完整 episode 表），以便重新采样窗口、审计
  “是否跨 episode 拼接”；若磁盘受限，至少保存每个窗口的 `ep_id` 起止与 `frame`。
- 同时保存 `meta.json`：全部 npz 路径与 sha256、`MotionLibrary` 路径/哈希、采集步数、
  policy ckpt、seed、git commit、`argv`。
- **两套评估协议分开报告**：

  | 协议 | train | val | test | 回答的问题 |
  |---|---|---|---|---|
  | A 分布内 | `walk1_subject1` + `dance1_subject1` | `walk1_subject2` + `dance1_subject2` | `walk1_subject5` + `dance1_subject3` | 能否拟合训练涉及的动力学 |
  | B 跨运动 | 同 A | 同 A | `run2_subject1` + `sprint1_subject2` | 分布外泛化 |

  两个协议分别保存 `eval_{a,b}_k{8,32}.pt`；不得用 B 的结果代替 A，也不得只报平均值。
- **采集分布**：使用 `sampling_mode="start"` 或显式 uniform clip 采样，确保每个 clip 都有数据；
  不要用默认的 return 采样后再按 clip 划分（会让低回报 clip 样本过少）。
- 若用 `policy_ckpt` 采集：确认该 policy 未在 test clips 上训练过，否则协议 B 不再是分布外；
  A/B 必须使用同一采集策略，并记录 policy ckpt hash。
- 采集时 `replay_capacity` 要足够大（至少覆盖一个 episode 长度 + 32 步），否则长窗口会被
  写指针淘汰；对 40 条 clip，建议分 clip 采集或增大 capacity。
- 评估窗口用固定 seed 一次性采样，并同时保存 `k=8` 与 `k=32` 两套，用于区分
  “训练窗口 8 步”与“评估窗口 32 步”的分布差异。
- 每个评估窗口保存 `frame`，保证可以用 `library.ref(frame)` 重建参考帧。

`ReplayBuffer`：

- [ ] `_sample_from_mask(..., generator=None)`、`sample_windows(..., mask=None, generator=None)`，
      让评估采样与训练 RNG 解耦，并支持按 `clip`/`ep_id` 过滤。
- [ ] 导出时校验：无 NaN/Inf；`done` 与 `terminal` 一致；`frame` 在 clip 范围内。

### 2.2 训练脚本修正（`experiment/koopman-world-model/train_wm_only.py`）

- [ ] **修 `window` 未生效**：`WMTrainerCfg(window=window, batch_size=..., ...)`。
      目前 `--window` 只存在于 argparse，`WMTrainerCfg.window` 仍是默认 8。
- [ ] `--data-in PATH`：加载冻结数据集，完全跳过 env 采集；支持 `--protocol a|b`。
- [ ] `--save-wm PATH`：保存 `wm.state_dict()` + 完整构造参数 + config + 数据 hash。
- [ ] `--eval-windows-8` / `--eval-windows-32`：使用固定评估窗口（替代
      `rec.buffer.sample_windows(1024, 32)`）；保留 `--eval-windows` 作为两者的兼容别名。
- [ ] `--eval-modes latent,reencode`：同一权重两种模式都评估。
- [ ] **结果 JSON 写入完整溯源信息**：`config`（`asdict`）、`git_commit`、`argv`、
      `data_sha256`、`eval_windows_sha256`；缺任何一项的结果不得用于结论。
- [ ] 记录 `params`、`updates/s`、`peak_mem`、总训练时间，便于容量对照。
- [ ] 暴露全部消融开关（统一成嵌套 `--koopman.*`，保留 `--dz` / `--rank` 作为兼容别名）：
      `--hidden`、`--koopman.enc-hidden`、`--koopman.dec-hidden`、
      `--koopman.lambda-cons`、`--koopman.lambda-rec-true`、`--koopman.a-rho-target`、
      `--koopman.stab-mode`、`--koopman.lambda-stab`、`--koopman.xi-every-step`、
      `--koopman.b-init`、`--koopman.dec-last-gain`。
- [ ] 每次评估额外写：`E8@8step`、`E8@32step`、`E1..E32` 曲线、`consistency_abs`、
      `consistency_rel`、`rho_A`、`max_sigma_M`、`z_norm`、`latent_var`、`latent_erank`、
      六组通道 NSE。
- [ ] 评估前后检查 NaN/Inf；出现即标记该 run 无效。

### 2.3 模型侧诊断接口

`src/supermimic/world_model/koopman.py`：

- [ ] `transition_matrix(action) -> (B, dz, dz)`：
      `M_t = A + Σ_i a_{t,i} U_i V_iᵀ`，与 `propagate` 用同一个 einsum，测试对拍。
- [ ] `spectral_stats()`：精确 `ρ(A)`（`eigvals`）与 `σ_max(A)`（`svdvals`），仅诊断。
- [ ] `sigma_max_power(M, iters=10, detach_v=True)`：
      - 训练正则用低成本近似；`v` 在每次迭代后 detach，降低梯度方差；
      - **3 次迭代不够**：当最大/次大奇异值接近时会明显低估，导致“看似 σ≤1，实际 >1”；
      - 每次评估用精确 SVD 在子集上校准，报告 `σ̂_max / σ_max` 的误差。
- [ ] `prod_norm(M_list)`：逐步累积 `P ← M_t P`，在子集上用精确 SVD 报告 `‖P‖₂`；
      训练中只记录，不做正则。
      注意：`M_t` 只描述齐次项；控制项 `B a` 也会累积。同时记录 `‖B‖` 与动作尺度，
      必要时在 J2b 中直接测 `∂s_H/∂a`，不要只看乘积范数。

`src/supermimic/world_model/world_model.py`：

- [ ] `rollout_error(..., mode: str | None = None)`：`None` 时用 `cfg.koopman_mode`，保持旧调用兼容。
- [ ] `rollout_diagnostics(states, a_exec, xi, mode) -> dict[str, Tensor]`（`@torch.no_grad()`）：
  - 每步 `E1..E32`（`nse_total`），而不是只返回最后一个；
  - 六组通道 NSE：root pos / rotvec / lin vel / ang vel / joint pos / joint vel
    （按 `StateSpec.err_scale` 的分段求均值）；
  - `z_norm`、`cons_abs`、`cons_rel = cons_abs / (z_norm + ε)`；
  - **表示健康度**：`latent_var = mean_j Var_B(z_{:,j})`、
    `latent_erank = exp(−Σ p_i log p_i)`（`p_i` 为 `Cov(z)` 的归一化特征值）；
  - `sigma_max_M`、`prod_norm`：默认只取前 64 个窗口，诊断用精确 SVD；
  - `valid_mask` / NaN 计数，避免把无效窗口算进平均。
- [ ] **E1 一致性检查**：同一权重下 `step_z(z0, s0, a0, xi)` 与
      `step(s0, a0, xi)` 应在 float32 下逐元素一致（≤1e-6）。这是零成本实现检查，
      见 G0b。

### 2.4 新增 `scripts/diag_koopman.py`

输入：`--ckpt`（Trainer checkpoint）或 `--wm-state` + `--koopman-*`；
`--eval-windows-a8` / `--eval-windows-a32` / `--eval-windows-b8` / `--eval-windows-b32`；
`--horizon 32`；`--n-sigma 64`。
输出：JSON + 控制台表。必须回答：

1. 同一权重 `latent` vs `reencode` 的 `E1..E32`；**先断言 `E1` 在两种模式下一致**；
2. `ρ(A)`、精确 `σ_max(M_t)`、`‖M_{t−1}…M_0‖`、幂迭代近似误差；
3. `‖z_t‖`、`cons_abs/rel(t)`、`latent_var`、`latent_erank`、六组通道 NSE(t)；
4. `E8@8step` 与 `E8@32step` 的差异（输入分布检查）；
5. NaN/Inf 与长尾状态计数；
6. 机制归因（见 §3）与建议的下一步实验。

脚本必须把完整 config 与 git commit 写进 JSON；缺少这些字段的结果不进入结论。

### 2.5 测试修正与补强（`tests/test_koopman.py`）

- [ ] **修 `test_koopman_jacobian_finite_difference`**：
  - 现在比较的是“逐样本方向导数 `(j_auto·v).sum(-1)`”与“整个 batch 的总方向导数”；
  - batch 总和版改为 `dir_auto = (j_auto * v).sum()`；
  - 另加逐样本版：`f_per = m(...).sum(-1)`，与 `(j_auto*v).sum(-1)` 比较；
  - **不能在全零 decoder 下测**（两边都为零会假通过）：用 `dec_last_gain` 初始化或先训练 10 步。
- [ ] `test_e1_modes_equal`：同一权重下 `step_z(z0,s0,a0,xi)` 与 `step(s0,a0,xi)`
      逐元素一致（`atol=1e-6`）；这是 G0b 的单元测试版本。
- [ ] `test_transition_matrix_matches_propagate`：`M_t z + B a == propagate(z, a)`。
- [ ] `test_init_spectral_radius`：`a_rho_target=0.99`，多 seed 断言 `ρ(A) ≤ 0.99 + 1e-4`。
- [ ] `test_power_iteration_vs_svd`：随机矩阵上 `sigma_max_power` 与精确 SVD 的相对误差
      ≤5%；构造“最大/次大奇异值接近”的矩阵，验证 3 次迭代确实低估、10 次迭代误差可接受。
- [ ] `test_consistency_in_loss`：`λ_cons>0` 时 `loss == Σ 加权 parts`，且梯度能到 encoder/K。
- [ ] `test_rec_true_uses_true_state`：只对 `rec_true` backward，encoder 梯度非零；
      `rec_true != rec_pred`；潜变量方差/有效秩不降为 0。
- [ ] `test_latent_collapse_metrics`：构造 `E(x)=const` 的退化模型，
      断言 `latent_var`/`latent_erank` 检测到坍缩。
- [ ] `test_xi_conditioning`（三部分）：
      1. `B_xi=0` 时，`xi_every_step` 开/关的输出与旧模型逐位一致；
      2. 人为设置非零 `B_xi`，不同 ξ 输出不同；
      3. 只对包含 ξ 的项 backward，`B_xi` 获得非零梯度。
- [ ] `test_xi_migration`：旧 `state_dict`（无 `B_xi`）经白名单迁移后，
      关闭 `xi_every_step` 的输出与旧模型一致；`load_state_dict(strict=True)` 的失败被显式处理。
- [ ] `test_closed_loop_directional_derivative`：8 步与 32 步闭环方向导数
      （double、小 dz、非零 decoder）——J2a，验证 BPTT 实现。
- [ ] `test_training_window_used`：`WMTrainerCfg(window=3)` 时 `sample_windows` 的 k 为 3。
- [ ] `test_eval_sampling_generator`：固定 generator 两次采样完全一致。
- [ ] 更新 `test_koopman_window_loss_and_bptt_gradients` 的 metrics 键
      （`rec_true_loss`、`cons_loss`、`latent_var`、`latent_erank`、`stab_loss` 等）。

### 2.6 文档修正

- [ ] `experiment/koopman-world-model/RESULTS.md`：
  - 标注 `wm_koopman_reencode.json` 为“**异常、待复核**”，而不是“数学上不可能”；
  - MLP 参数量 “~13M” → **8.81M**（`wm_mlp.json` 实测 8,815,111）；
  - 补充 v2 / lift1 的结果、分布差异（8 步 vs 32 步窗口）与解释；
  - 所有结果表加一列“config/commit/data hash 是否完整”。

**P0 完成标志**：

- `pytest tests/test_koopman.py` 全绿；
- `diag_koopman.py` 在随机权重与真实 checkpoint 上都可复现（关键数值 ≤1e-5）；
- 同一权重下 `E1` 的 `latent`/`reencode` 两种模式一致（G0b）；
- `train_wm_only.py` 支持冻结数据、A/B 评估窗口、完整 config 溯源、保存权重；
- 至少一份 `diag_p0.json` 同时包含逐步曲线、谱、有效秩、通道误差与 NaN 检查。

---

## 3. P1：诊断根因（不改 Koopman 结构）

目的：先回答“32 步发散主要来自矩阵放大、潜空间—物理状态不闭合，还是单步欠拟合”，
再决定修什么。P1 不做任何结构或损失改动。

### 3.1 运行内容

1. K0r：原始 latent 配置，冻结数据 + 保存权重；
2. K1：在 K0r 权重上分别跑 `latent` / `reencode`，并断言 `E1` 一致；
3. `diag_p1.json`：逐步曲线、谱、有效秩、通道误差、`E8@8` vs `E8@32`。

```bash
# 采集一次，包含 A/B 两套协议所需的全部 clip
.venv/bin/python experiment/koopman-world-model/collect_wm_data.py \
    --out /root/autodl-tmp/data/wm/lafan1.pt

.venv/bin/python experiment/koopman-world-model/train_wm_only.py \
    --wm-type koopman --koopman-mode latent --protocol a \
    --data-in /root/autodl-tmp/data/wm/lafan1.pt \
    --eval-windows-8 /root/autodl-tmp/data/wm/eval_a_k8.pt \
    --eval-windows-32 /root/autodl-tmp/data/wm/eval_a_k32.pt \
    --save-wm /root/autodl-tmp/logs/wm/k0r_latent.pt \
    --out experiment/koopman-world-model/wm_k0r_latent.json

.venv/bin/python scripts/diag_koopman.py \
    --wm-state /root/autodl-tmp/logs/wm/k0r_latent.pt \
    --eval-windows-a8 /root/autodl-tmp/data/wm/eval_a_k8.pt \
    --eval-windows-a32 /root/autodl-tmp/data/wm/eval_a_k32.pt \
    --eval-windows-b8 /root/autodl-tmp/data/wm/eval_b_k8.pt \
    --eval-windows-b32 /root/autodl-tmp/data/wm/eval_b_k32.pt \
    --out experiment/koopman-world-model/diag_p1.json
```

### 3.2 机制归因表

| 观察 | 更可能的机制 | 下一步 |
|---|---|---|
| `‖M_{t−1}…M_0‖` 远大于 1，`‖z_t‖` 相应放大 | 传播矩阵放大 | P2-B 稳定性 |
| `cons_rel` 显著大于 0，且随步数增长 | 潜空间—物理状态不闭合 | P2-A 表示闭合 |
| `E1` 已远差于 MLP，但 `prod_norm≈1`、`cons_rel` 小 | 单步欠拟合/容量 | P3 容量 |
| reencode 也随步数发散 | 闭环映射本身不稳或分布外 | 先看通道误差与容量，再决定 |
| latent 明显好于 reencode | 编码器在预测分布上不稳 | 查 encoder 输入分布与重编码误差 |
| 两者都稳定但 Actor 差 | WM 不是瓶颈 | P4 梯度验证 |

### 3.3 额外检查

- **输入分布**：`E8@8step` 与 `E8@32step` 都报告；若差异大，说明 32 步窗口是不同分布子集，
  不能直接把 `nse_kstep` 与 `E8` 对比。
- **通道误差**：root pos / rotvec / lin vel / ang vel / joint pos / joint vel 分别画曲线，
  定位是“位置积分漂移”还是“关节动力学错误”。
- **谱与乘积范数**：`ρ(A)`、精确 `σ_max(M_t)`、`‖M_{t−1}…M_0‖` 在固定子集上逐 步报告；
  同时报告幂迭代近似与精确 SVD 的误差。
- **NaN/长尾**：统计无效窗口、极端 `‖s_t‖`、`done` 前后窗口，排除数据问题。
- **K1 的边界**：K1 只说明“同一权重下推理模式的影响”，不能单独证明“重新训练后应选哪种模式”。

注意：已有 `reencode_v2` 只是弱证据；K1 才是同权重结论。reencode 仍可能因闭环/分布外而不稳，
不能把“reencode 更好”当作问题已解决的证据。

### 3.4 P1 结果与决策（2026-10-31）

固定协议 K0r（dz=128, rank=8, 3000 updates，window=8）：

- `ρ(A)=1.248`，`σ_max(A)=1.374`（初始化约 1.10 / 1.15）；
- `σ_max(M_t)≈1.80`，32 步 `‖M₃₁…M₀‖≈4.5e3`；
- A 集 32 步：latent `E1=0.239 / E8=0.945 / E32=4.04e5`；
  reencode `E1=0.239 / E8=1.014 / E32=13.97`；
- `cons_rel`：latent 0.087 vs reencode 0.042；`erank`：27/128 vs 19/128；
- 同一冻结数据 MLP 对照：A32 `E1=0.0479 / E8=0.0751 / E32=0.345`；
  Koopman reencode 为 `0.239 / 1.014 / 13.97`（5× / 40×），latent 为 `0.239 / 0.945 / 4.04e5`；
- `E1` 两种模式完全一致（G0b pass）；A/B 与 8/32 窗口差异都很小。

**归因**：矩阵放大是 E32 发散的主因；潜状态闭合漂移是共同因素；单步容量/欠拟合是第三个因素。
**决策**：先做 P2-B（稳定性），再做 P2-A（闭合）；P3 容量不可省。

---

## 4. P2-A：表示闭合（K2A）

**只做这一项，不同时加入稳定性正则、ξ 条件化或容量改动。**

### 4.1 目标损失（P2-A 单独）

$$
\mathcal L_A =
\mathcal L_{\text{state}}
+ \mathcal L_{\text{term}}
+ \lambda_1 \mathcal L_{\text{lift}}
+ \lambda_2 \mathcal L_{\text{rec,true}}
+ \lambda_3 \mathcal L_{\text{cons}}
\;(\; + \lambda_5 \mathcal L_{\text{var}}\;)
$$

| 项 | 定义 | 目标侧 / 作用 |
|---|---|---|
| `L_lift` | `‖K(z_t,a_t) − E(x_{t+1}^true)‖²` | detach；动力学监督 |
| `L_rec_true` | `‖R(E(x_t^true)) − x_t^true‖²` | 自编码；迫使 encoder 保留状态信息 |
| `L_cons` | `‖z_{t+1} − E(x_{t+1}^{pred})‖²` | **目标 detach**；目标是**预测物理状态**的编码，不是下一真实状态 |
| `L_var` | 可选方差 hinge（见 4.5） | 仅在监控到坍缩时启用 |

旧的 `L_rec`（预测状态重构）保留为 `L_rec_pred`，权重仍走 `lambda_rec`，不删除；
建议 P2-A 中把它调到很小或 0，把“真实状态重构”作为主约束。

**关于 `detach` 的正确表述**：`detach` 只阻断目标分支的梯度，避免两个分支共同追逐；
它**不保证**不坍缩。若 `E(x)=0, K(0,a)=0`，则 `L_lift = L_cons = 0`。
真正的防坍缩来自 `L_rec_true` 的信息保留 + 方差/有效秩监控 + 必要时的方差正则。

另一个边界：`L_cons` 的目标 `E(s^{pred})` 本身可能落在 encoder 的训练分布之外。
如果 `reencode` 诊断显示 E 在预测状态上不稳定，`L_cons` 会把 z 拉向不可靠目标；
这时应先看 `rec_true` 与 `reencode` 结果，而不是盲目加大 `lambda_cons`。

### 4.2 表示健康度监控（必须）

- `latent_var = mean_j Var_B(z_{:,j})`；
- `latent_erank = exp(−Σ p_i log p_i)`，`p_i` 为 `Cov(z)` 的归一化特征值；
- 同时报告 `erank(canonical(s))` 作为参照，避免把“数据分布本身很窄”误判为潜空间坍缩；
- 与初始化/首个 checkpoint 比较：方差下降 >10× 或有效秩 < 20%·dz 时**告警**；
  这两个阈值由 P0 的基线分布校准，不能直接当作“理论充分条件”；
- 每个 eval 都写入 metrics，不能只看 `L_cons` 下降。

### 4.3 代码改动

`koopman.py` 的 `KoopmanCfg` 新增（**默认值 = K0 行为**，实验显式打开）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `lambda_cons` | `0.0` | P2-A 打开，建议先 0.1 |
| `lambda_rec_true` | `0.0` | P2-A 打开，建议先 0.1 |
| `lambda_var` | `0.0` | 仅坍缩时启用 |
| `lift_target` | `"online"` | `"online"` / `"ema"`（见 4.5） |

`world_model.py` 的 `_window_loss_koopman`：

- [ ] 循环前一次性计算 `x_true = canonical(states)`（`(k+1,n,71)`）与
      `z_true = encode(x_true, xi_expand)`（`(k+1,n,dz)`），**只用一次 encoder 前向**；
- [ ] `L_lift` 用 `z_true[1:].detach()` 作为目标，去掉现在每步一次的 encoder 前向；
- [ ] `L_rec_true = ‖recon(z_true) − x_true‖²`，只在真实状态上；
- [ ] 每步 `L_cons` 用当前 `z_{j+1}` 与 `encode(s_{j+1}).detach()`；
- [ ] `window_loss(..., with_parts=True)` 返回未加权各项，供梯度尺度工具与日志使用；
- [ ] 记录 `latent_var`、`latent_erank`；
- [ ] **等价性测试**：`lambda_cons=0`、`lambda_rec_true=0` 时，`loss` 与旧实现逐位接近
      （`rtol=1e-5`）。

### 4.4 权重不靠猜：`scripts/diag_loss_scales.py`

- 固定 batch、固定 seed；
- 对每项 loss 单独 `backward`，统计 `encoder / recon / decoder / A / B / U,V` 的梯度范数；
- 输出表并给出推荐 λ：例如让 `L_cons` 对 encoder 的梯度范数 ≥ `L_state` 的 10%；
- 每次改 λ 前先跑它，避免 plan2 §4 指出的“数值贡献 1.19%”问题。

### 4.5 可选 anti-collapse 与 EMA

- `L_var = mean_j relu(γ − std_B(z_{:,j}))²`（VICReg 风格），`γ` 由初始 `std` 设定；
  只在 4.2 监控到坍缩时启用，默认关闭。
- EMA target encoder（`lift_target="ema"`、`ema_decay=0.999`）只在
  “`rec_true` 下降但 `lift` 不降、`latent_var` 变小”时启用；默认不做。

### 4.6 验收

- G1/G2 不发散、精度不退化；
- G3 通过，且 `latent_var`/`latent_erank` 未坍缩；
- 若 P2-A 单独无效，不进入 P2-C（组合），先转 P2-B 单独测试。

---

## 5. P2-B：温和稳定性（K2B）

> 2026-10-31 已实现：`a_rho_target`、`stab_mode`（none/frobenius/power）、`lambda_stab`、
> `stab_iters`、`stab_batch`；`KoopmanDynamics.stability_penalty()`；训练脚本 CLI 与测试。
> λ 扫描脚本：`experiment/koopman-world-model/run_p2b_sweep.sh`。

**只做这一项；不要与 P2-A 同时开启。**

### 5.1 A 初始化

- [ ] `a_rho_target`：初始化时 `A = a_eye·I + a_noise·N`，若 `ρ(A) > target` 则整体乘
      `target/ρ(A)`。放在 `nn.Parameter` 之前，CPU 上一次 `eigvals` 即可。
- [ ] **不要默认把 target 压得很小**：`0.99` / `0.995` 两档即可；过度收缩会损失位置/姿态模式。
- [ ] 记录初始化前后的 `ρ(A)`、`σ_max(A)`，并断言 `ρ(A) ≤ target + 1e-4`。

### 5.2 σ_max hinge（训练正则）

$$
\mathcal L_{\text{stab}} = \mathbb E\left[\max(0,\hat\sigma_{\max}(M_t)-1)^2\right],
\qquad M_t = A+\sum_i a_{t,i}U_iV_i^\top
$$

- [ ] `sigma_max_power(M, iters=10, detach_v=True)`：
  - **3 次迭代不够**：当最大/次大奇异值接近时会明显低估，导致“看似 σ≤1，实际 >1”；
  - 每次迭代后 detach `v`，只让梯度经过 `M`，降低方差；
  - 训练用近似值，评估用精确 SVD 在子集上校准，报告 `σ̂_max/σ_max` 的误差分布。
- [ ] `lambda_stab` 由 `diag_loss_scales.py` 决定（建议先扫 `1e-3 ~ 1e-1`）。
- [ ] 只惩罚 `σ_max > 1` 的部分，不惩罚所有尺度。
- [ ] 正则采样动作要覆盖两类分布：Replay Buffer 动作 + 当前 policy 动作（若可用）；
      只对 buffer 动作正则，可能在想象分布上失效。
- [ ] 提供 `stab_exact_every`（例如每 50 次更新）在 8–16 个样本上做精确 SVD，
      校准近似值并更新惩罚；精确 SVD 在这个规模上可接受，不能只依赖 10 次幂迭代。

### 5.3 Frobenius 不是等价替代

`‖M_t‖_F²/dz` 只是 **Matrix Weight Decay Baseline**：它惩罚所有奇异值（包括需要保持的
近单位/慢变化模式），会带来系统性收缩偏差。保留它作为消融，但：

- 不把它描述为“更便宜的 σ_max 正则”；
- 单独报告正则化前后在近单位模式上的预测偏差；
- 若它降低了 E32 但抬高了 E1/E8，说明只是把动力学压小，不是修复。

### 5.4 乘积范数

第一轮不做乘积范数 hinge。只记录 `‖M_{t−1}…M_0‖`；只有当归因明确指向
“乘积放大”、且 σ_max hinge 不足时，才考虑对 `log‖P_t‖` 做 hinge。

### 5.5 日志与验收

- 训练日志新增：`wm/rho_A`、`wm/sigma_max_M_exact`、`wm/sigma_max_M_approx`、
  `wm/prod_norm`、`wm/z_norm`、`wm/cons_rel`、六组通道 NSE。
- 验收：G4 改善、G1 不发散、G2 不显著退化；若 E1/E8 变差，记录为“收缩偏差”。

**2026-10-31 λ 扫描结果**：

| λ_stab | ρ(A) | σ_max(M) | ‖M…M₀‖ | latent E32 | reencode E32 | E1 |
|---|---|---|---|---|---|---|
| 1 | 1.070 | 1.346 | 33.7 | 2.92e3 | 23.48 | 0.2563 |
| 10 | **0.981** | **1.161** | **2.45** | **324.1** | 24.19 | 0.2593 |

λ=10 在 E1/E8/reencode 几乎不变的情况下，把 latent E32 再降约 9×，因此选 λ=10 作为
P2-B 稳定性设置。`cons_rel` 必须与 `cons_abs`/`z_norm` 一起看：K0r 的 `cons_rel=0.087`
对应 `cons_abs=1.09e5`、`z_norm=1786`；λ=10 的 `cons_rel=0.318` 对应
`cons_abs=11.2`、`z_norm=9.47`。

**P2-B 后剩余瓶颈**：reencode E32 24.19 vs MLP 0.345（约 70×），E1 0.259 vs 0.048（约 5×）；
第 32 步误差主要是 `joint_pos=727`、`root_rot=206`，速度类已很小。
这说明矩阵放大已基本解决，下一步的主要矛盾是 **decoder/积分器/容量**，P3 容量优先于 P2-A。

### 5.6 P2-C：只在两项各自有效时组合

先分别得到 P2-A、P2-B 的独立结果，再决定是否做 2×2 组合（A only / B only / A+B / neither）。
如果只有一项有效，不做组合；不要用组合结果掩盖单项无效。

---

## 6. P3：容量与结构消融

容量实验放在 P2 之后，并且要分成两组：

- **C-base**：在 K0r（原始损失）上加容量，隔离“容量”本身；
- **C-best**：在 P2 中胜出的配置（A / B / A+B）上加容量，回答“修复后是否仍欠拟合”。

实测参数量（本机 `KoopmanDynamics` / `DynamicsMLP`）：

| 配置 | 参数量 |
|---|---|
| Koopman 当前 `dz=128, r=8, (256,256)` | **0.443M** |
| Koopman `dz=128, r=8, (512,512)` | 1.19M |
| Koopman `dz=256, r=16, (512,512)` | 1.62M |
| Koopman `dz=256, r=16, (1024,1024)` | 4.50M |
| MLP `(420,420,420)` | 0.439M（与当前 Koopman 匹配） |
| MLP `(2048,2048,2048)` | 8.81M（基线） |

执行顺序：

1. **C-base-reencode + 4.50M**：若 E1 从 0.113 明显下降 → 容量是主因；
2. **C-base-latent + 4.50M**：确认容量是否也能缓解长时域；
3. **C-base 1.62M**：中间点，判断是否单调收益；
4. **MLP (420,420,420)**：排除“Koopman 只是参数少”的解释；
5. 若 P2 有胜出配置，重复第 1–2 步（C-best）；
6. 同时记录吞吐、显存、`updates/s`、GPU-hours；参数量相近 ≠ 计算量相近；
7. 修正 `RESULTS.md` 的 MLP 参数量（13M → 8.81M）。

容量结论必须同时报告 E1/E8/E32 与训练成本，不能只看参数量。

> 2026-10-31 并行执行：`experiment/koopman-world-model/run_capacity_parallel.sh`
> 同时跑 c1（1.20M）、c2（1.62M）、c3（4.50M）、c4（MLP 0.44M）。
> 单进程约 2.45 GB 显存，4 并发实测约 14.5 GB / 99% GPU，A800 可承受。
>
> **2026-10-31 中止**：该容量 sweep 已停止，不作为正式结果。容量问题仍值得回答，
> 但必须先在 `run_paper_aligned.sh` 的 upstream 数据/训练参数下复现基线，
> 再按相同 batch 4096×10 预算做容量对照；否则参数量结论会被 1-batch/8-clip 配置污染。

---

## 7. P3b：物理上下文 ξ 条件化（可选，在 P2/P3 之后）

这是一个**可证伪假设**，不是默认修复：默认配置下，`xi` 只进入初始编码
（`WorldModel.step_z` 的 `xi` 参数实际未被使用，`_loss_step_z` 的 `xi_norm` 也未进入
`propagate`），因此 32 步传播必须把物理参数一直保存在 `z` 里。

> **2026-10-31 已实施（增广状态版）**：选择 \([z;\xi]\) 而不是只加 `B_xi` 偏移。
> `KoopmanCfg.use_xi_state=True` 时注册 `A_xi: (dz,33)` 与
> `V_xi: (act,33,r)`，零初始化；`propagate` 每步计算
> `z_next += A_xi xi + sum_i a_i U_i(V_i^T z + V_xi_i^T xi)`。
> 稳定性/`sigma_max` 仍只作用于 \(z\) 块 \(M_z(a)\)，不含恒等的 \(\xi\) 块。
> 已同步 `forward`、`WorldModel.step_z`、`_loss_step_z`、`rollout_diagnostics`、
> `imagination`；新增 `augmented_transition_matrix()` 供诊断。
> `use_xi_state=False` 时行为与旧模型完全一致，旧 checkpoint 可直接加载。

### 7.1 实现（增广状态版）

- `KoopmanCfg.use_xi_state: bool = False`；
- `KoopmanDynamics` 在开启时注册 `A_xi: (dz, 33)`、`V_xi: (act, 33, r)`，
  零初始化，因此开启后初始输出与旧模型一致；
- `propagate(z, action, xi_norm=None)`：开启时
  \[
  z_{t+1}=Az_t+A_\xi\xi_n+Ba_t+\sum_i a_i U_i(V_i^\top z_t+V_{i\xi}^\top\xi_n)
  \]
  等价于块矩阵
  \[
  \tilde A=\begin{bmatrix}A&A_\xi\\0&I\end{bmatrix},\quad
  \tilde U_i=\begin{bmatrix}U_i\\0\end{bmatrix},\quad
  \tilde V_i=\begin{bmatrix}V_i\\V_{i\xi}\end{bmatrix}
  \]
- `augmented_transition_matrix(action)` 返回完整块矩阵，仅用于诊断；
- P2-B 的 `stability_penalty` / `transition_matrix` 仍只使用 \(z\) 块，
  避免 \(\xi\) 的恒等块让 \(\sigma_{\max}\ge1\) 使 hinge 失效；
- 同步修改：`forward`、`WorldModel.step_z`、`_loss_step_z`、
  `rollout_diagnostics`、`imagination/compiled.py::ImagStep._step_z`；
- `XI_STATE=1` 在 `run_paper_aligned.sh` / `run_lafan1_online.sh` /
  `run_offline_quick_parallel.sh` 中打开。

### 7.2 Checkpoint 兼容性

`use_xi_state=False` 时不注册 `A_xi/V_xi`，旧 checkpoint 可直接 `strict=True` 加载；
开启后 state dict 多出 `dynamics.A_xi`、`dynamics.V_xi`，必须用对应 config 加载。
不同配置之间不要复用旧 WM 权重做 `strict=True` 加载。

- 严格加载旧 checkpoint：保持 `use_xi_state=False`；
- 从旧 WM 迁移到 xi-state：新增键零初始化即可保持初始输出一致，但 optimizer state
  不兼容，应重新建 optimizer；
- 若不能保证以上验证，拒绝加载旧 checkpoint，而不是静默 `strict=False`。

### 7.3 测试

1. `use_xi_state=True` 且 `A_xi=V_xi=0` 时，输出与旧模型逐位一致；
2. 人为设置非零 `A_xi/V_xi`，验证不同 ξ 会改变 `z` 和 `s`；
3. `window_loss.backward()` 能给 `A_xi/V_xi` 非零/有限梯度；
4. `augmented_transition_matrix` 与 `propagate` 一致；
5. `stability_penalty` 只使用 \(z\) 块，不受 `A_xi/V_xi` 影响。

### 7.4 验收与边界

- 与 K0r / P2-best 在**同数据、同容量、同预算**下对比；
- 只有 E1/E8/E32 至少一项有稳定改善且不损害其余项，才进入 K6（`A(ξ)`/`B(ξ)`）；
- 若 ξ 条件化无效，不继续加参数化矩阵，保持 latent 动力学简洁。

## 7.5 P3c：状态增广 Koopman（已实现）

用户选择的更完整结构：

\[
z_t=[\mathrm{canonical}(s_t);\phi(s_t,\xi)]
\]

\[
u_{t+1}=Az_t+Ba_t+C\xi+b
+\sum_i a_{t,i}U_iV_i^\top z_t
+\sum_j \xi_jP_jQ_j^\top z_t
+\sum_i a_{t,i}W_i\xi
\]

\[
\delta_t=W_d u_{t+1}[:71],\quad
s_{t+1}=\mathrm{integrate}(s_t,\delta_t)
\]

\[
z_{t+1}=[\mathrm{canonical}(s_{t+1});u_{t+1}[71:]]
\]

- `StateAugmentedKoopman` 实现于 `koopman.py`；
- `WorldModel.step_z` / `_loss_step_z` / `rollout_diagnostics` / imagination
  均走 `step_latent`，rollout 内不调用 encoder；
- `KoopmanCfg.state_in_latent=True`，`rank_domain`、`rank_action_domain` 可配；
- P2-B 约束完整 \(M(a,\xi)\)，而不是只约束 \(A\)；
- 多步训练：`window_loss` 的 k=8 state loss + lift/consistency；下一步可把
  consistency 显式改成窗口内每步的 \(\|z_{j+1}-E(s_{j+1},\xi)\|^2\)；
- 快速入口：`run_state_koopman_quick.sh`；在线入口：`STATE_KOOPMAN=1
  run_lafan1_online.sh`。
- **2026-10-31 P0 修复**：低秩因子改为“一方随机、一方为零”，避免 \(U=V=0\)
  双零梯度；`cons_rel` 改为 L2/L2，新增 `cons_mse`/`cons_l2`/`phi_cons_*`；
  `lambda_cons` 已加入 P2-A consistency loss，state-augmented 只监督 \(\phi\)。

---

## 8. P4：梯度验证与恢复 Actor

### 8.1 三级梯度测试（修正 v1 的 G5 混淆）

v1 把 `∂s_H/∂a_0` 当作“策略梯度方向”，这是错的：前者是固定动作序列的**状态 Jacobian**，
后者是 `∇_θ J(θ)`。两者必须分开。

| 级别 | 比较对象 | 回答的问题 | 用途 |
|---|---|---|---|
| J1 | 单步 `(∂s_1/∂a_0) v`（autograd） vs 中心差分 | 一步动作梯度是否正确 | 单元测试 |
| J2a | WM 的 `(∂s_H/∂a_{0:H-1}) v`（autograd） vs WM 自身中心差分，8/32 步 | BPTT 实现是否正确 | 单元测试 |
| J2b | WM 的 `(∂s_H/∂a_{0:H-1}) v` vs MuJoCo 同方向中心差分 | 长期动力学敏感度是否可信 | G5a 诊断 |
| J3 | `(∇_θ J_WM)·v_θ` vs `J_sim(θ+εv_θ) − J_sim(θ−εv_θ)` 的中心差分 | WM 是否给出有利于 Actor 的梯度 | G5b 硬门禁 |

- J1/J2a 用 double、小 dz、非零 decoder，避免零初始化假通过。
- J2b 必须固定同一初始状态、同一动作序列、同一积分步；接触切换附近的差分单独报告。
- J3 先用**随机方向 / 低维子空间**（如 policy 最后一层）做中心差分，避免对 1M+ 参数做全梯度；
  至少 8–16 个方向，报告平均 cosine 与置信区间。
- J3 的仿真差分必须用**公共随机数**：`θ±εv` 使用同一初始状态、同一 reset/DR seed、
  同一动作噪声；否则差分会被采样噪声淹没。参数维度太高时可用 SPSA 或分层随机方向。

### 8.2 `scripts/eval_koopman_bptt.py`

- `--level j1|j2a|j2b|j3`；
- 统一目标函数（reward + termination）、时间窗、初始状态、随机性；
- J2b/J3 分别输出 `cosine`、范数比、逐步曲线、接触/非接触分段统计；
- 结果写入 JSON，带 config/commit/data hash；
- J1/J2a 是正确性测试，应进入 `pytest`；J2b/J3 是模型保真度评估，放在脚本。

### 8.3 门禁与训练循环

- **离线门禁优先**：用固定 val/test 的 E1/E8/E32 + G5b 决定是否允许启动 Actor；
- 在线阶段同时监控两个量，但不要用一个阈值混合：
  - 固定验证集误差（模型能力）；
  - 当前 Replay Buffer 误差（当前状态分布）；
- `TrainConfig.wm_gate` 第一轮默认 `None`（只记录，不 skip）；等离线门禁稳定后再考虑自动 skip；
- `resume_skip` 含 `wm` 时，默认重置 `iteration` / `num_updates`（或提供 `--reset-schedule`），
  否则 cosine LR 从旧训练末期开始，而新 WM 才刚开始训练；
- 迁移 checkpoint 时不恢复不兼容的 optimizer state。

### 8.4 验收

- G5b 通过前，不恢复 Actor 联合训练；
- G6 只作为“可用性”门槛，不能用来证明 Koopman 优于 MLP；
- 同时报告失败动作数、样本效率、GPU-hours，避免只看 Return。

---

## 9. 消融矩阵 v2

| ID | 阶段 | 配置 | 关键开关 | 验证的假设 | 状态 |
|---|---|---|---|---|---|
| K0 | P0 | 原 latent | 默认 | 复现 E32≈1e6 | 已有 JSON，无权重 |
| K0r | P1 | 原 latent，冻结数据 + 存权重 | `--save-wm` | 可诊断基准 | 待跑 |
| K1 | P1 | K0r 权重分别跑 latent/reencode | `rollout_error(mode=)` | 同权重潜状态漂移；E1 必须一致 | 待跑 |
| K1b | P1 | reencode 训练 | `--koopman-mode reencode` | 每步重编码是否足够 | 部分（v2，待固定协议复跑） |
| K2A | P2-A | + 真实重构 + 闭合损失 | `lambda_rec_true`, `lambda_cons` | 表示闭合 | 待跑 |
| K2A2 | P2-A | + 方差正则（仅坍缩时） | `lambda_var` | 防坍缩 | 可选 |
| K2B | P2-B | + ρ 初始化 + σ_max hinge | `a_rho_target`, `lambda_stab` | 动力学矩阵放大 | 待跑 |
| K2B2 | P2-B | + Frobenius（matrix weight decay baseline） | `stab_mode=frobenius` | 对照，不等价 | 可选 |
| K2C | P2-C | K2A + K2B | 组合 | 是否互补 | 仅两项各自有效时 |
| K3 | P3 | 容量 1.62M / 4.50M（C-base / C-best） | `dz`, `rank`, hidden | 欠拟合 | 待跑 |
| K4 | P3 | 容量匹配 MLP 0.44M | `--hidden 420 420 420` | 公平对照 | 待跑 |
| K5 | P3b | 每步 ξ | `xi_every_step` | 物理上下文丢失 | 仅 P2/P3 后 |
| K6 | P3b | 参数化 `A(ξ)/B(ξ)` | — | 更强条件化 | 仅 K5 有效再做 |

规则：

- **一次只改一个因素**；
- P2-A、P2-B 分别训练、分别报告；不要一开始就组合；
- 每个 run 必须保存权重、数据 hash、评估窗口 hash、config、git commit；
- 每个结论至少对应一个 `diag_*.json`；
- 结果表里同时给出 `E1..E32` 曲线与训练成本。

---

## 10. 文件改动清单

| 文件 | 改动 | 阶段 |
|---|---|---|
| `src/supermimic/world_model/koopman.py` | `transition_matrix`、精确 `spectral_stats`、`sigma_max_power`、`prod_norm`、`a_rho_target`、`B_xi`、`xi_every_step`、初始化消融 | P0/P2-B/P3b |
| `src/supermimic/world_model/world_model.py` | `rollout_error(mode=)`、`rollout_diagnostics`、逐步曲线、方差/有效秩、损失拆分、`L_rec_true`/`L_cons`/`L_stab`、`with_parts` | P0/P2-A/P2-B |
| `src/supermimic/world_model/wm_trainer.py` | 记录诊断指标（`log_parts`、`latent_var/erank`、谱） | P0/P2 |
| `src/supermimic/world_model/checkpoint.py` | **新增**：`B_xi` 迁移、缺键白名单、优化器状态策略 | P3b |
| `src/supermimic/data/replay_buffer.py` | `mask`/`generator` 参数、按 clip/ep_id 采样 | P0 |
| `src/supermimic/imagination/compiled.py` | `_step_z` 传 `xi_norm`（P3b 后） | P3b |
| `src/supermimic/imagination/rollout.py` | 确认 reencode/latent 分支与 ξ 传递 | P3b |
| `src/supermimic/config.py` / `training/loop.py` | `wm_gate`（默认只记录）、resume 调度重置、固定 val 日志 | P4 |
| `experiment/koopman-world-model/train_wm_only.py` | `window` 修复、数据/权重/评估窗口、A/B 协议、config 溯源、消融开关、诊断日志 | P0 |
| `experiment/koopman-world-model/collect_wm_data.py` | **新增**：冻结数据集、A/B split、窗口完整性校验 | P0 |
| `experiment/koopman-world-model/run_ablation.sh` | **新增**：K0–K6 编排 | P2/P3 |
| `scripts/diag_koopman.py` | **新增**：同权重 latent/reencode、逐步曲线、谱、有效秩、通道诊断、E1 一致性 | P0/P1 |
| `scripts/diag_loss_scales.py` | **新增**：分项梯度范数 → λ 选择 | P2-A |
| `scripts/eval_koopman_bptt.py` | **新增**：J1/J2a/J2b/J3 | P4 |
| `tests/test_koopman.py` | Jacobian 修复、E1 一致性、幂迭代校准、坍缩检测、ξ 迁移、闭环方向导数等 | P0 |
| `tests/test_wm_protocol.py` | **新增**：冻结数据窗口完整性、固定评估采样、config 溯源 | P0 |
| `experiment/koopman-world-model/RESULTS.md` | 标注 v1 异常待复核、修正参数量、补 v2/lift1、加 config/commit 完整性列 | P0 |

---

## 11. 不做的事 / 放弃条件

- **reencode 只作为诊断**，不能作为 Koopman 最终结果；论文口径必须报告 latent。
- **不把 v1 reencode 结果当作“数学上不可能”**：它只是数值异常、待复核；
  reencode 仍是闭环映射，离开训练分布后可能发散。
- **不把 Frobenius 当作 σ_max 的等价替代**：它只是 matrix weight decay baseline，
  会收缩有用的近单位模式。
- **不把 `detach` 当作防坍缩保证**：必须监控 `latent_var` / `latent_erank`，
  必要时加方差正则。
- 不为了压 E32 把所有特征值强行压向 0（会牺牲位置/姿态模式）。
- 不同时改初始化、损失、容量、ξ；否则无法解释是哪一项有效。
- **P2 之前不加入 ξ 条件化和大容量**：先把“矩阵放大 / 闭合 / 单步欠拟合”分开。
- **第一轮不做在线自动 skip policy update**：先用固定验证集做离线门禁；
  在线只记录固定 val 与 Replay Buffer 两个误差。
- 不在门禁前继续 Actor 训练，也不通过“多训一会儿”掩盖 WM 失稳。

放弃条件（满足任一条即转路线 A/B，不再给当前结构加复杂度）：

1. P2-A、P2-B、P3 之后，latent 的 E32 仍 > 3× MLP 且 E1 > 2× MLP；
2. 表示坍缩无法通过 `L_rec_true` + 方差监控稳定控制；
3. J3（策略梯度方向）在多个随机方向上持续不通过，且不是由接触/噪声主导。

路线：

- **路线 A**：物理状态锚定的 Koopman，`z = [x; φ(x)]`，直接监督物理分量；
- **路线 B**：MLP + Koopman 残差混合，`Δ = f_MLP + α f_Koopman`；
- 科研写作中必须明确两条路线不再是纯 Koopman。

---

## 12. 提交切分（便于逐步 review）

| PR | 内容 | 对训练行为影响 |
|---|---|---|
| P0 | 冻结数据/双协议、固定评估窗口、诊断脚本、测试修复、`window`/`save-wm`/config 溯源 | 无（K0 数值不变） |
| P2-A | `L_rec_true`、`L_cons`、方差/有效秩监控（默认关闭） | 无（显式打开） |
| P2-B | `a_rho_target`、σ_max hinge、Frobenius baseline（默认关闭） | 无（显式打开） |
| P3 | 容量参数暴露、消融编排、结果更新 | 实验配置 |
| P3b | 每步 ξ、`B_xi` 迁移、checkpoint 兼容性（默认关闭） | 无（显式打开） |
| P4 | J1–J3 评估脚本、离线门禁、在线日志、resume 调度 | 训练行为变化（显式开启） |

注意：P1 是**分析阶段**，不产生新的模型代码；它的产出是 `diag_p1.json` 与机制归因，
用于决定 P2-A/P2-B 的优先级。

---

## 13. 时间与算力估算

- 单次 3000 更新（batch 2048，window 8）：按现有 JSON 文件时间戳约 **10–15 min / A800**；
- P0（代码 + 测试 + 数据冻结 + 双协议）：**1–1.5 天**；
- P1（K0r + K1 + 诊断）：**0.5–1 天**；
- P2-A / P2-B（各 2–4 次 run + 梯度尺度）：**1–2 天**；
- P3（容量 6–8 次 run）：**1 天**；
- P3b（ξ，可选）：**0.5–1 天**；
- P4（J1–J3 + 门禁 + 少量 Actor 试跑）：**2–3 天**；
- 总计约 **1–1.5 周**，其中 GPU 训练本身 < 8 GPU·h（不含 Actor 试跑）。

**建议的第一步**：只做 P0，不碰模型结构；跑通 K0r + K1 + `diag_p1.json`，
再决定先做 P2-A 还是 P2-B。

---

## 附录 A：对 `plan3.md` 的逐条响应

| # | plan3 意见 | 本计划处理 |
|---|---|---|
| 1 | reencode E8 不能断言“不可能发散” | §0 改为“数值异常、待复核”；列出 6 项复核；§3.2 增加“reencode 也发散”分支 |
| 2 | Frobenius 与 σ_max 不等价 | §5.3 明确标为 Matrix Weight Decay Baseline；不当作等价替代 |
| 3 | 3 次幂迭代可能低估 σ_max | §2.3/§5.2 改为 10 次、detach `v`、与精确 SVD 校准 |
| 4 | `detach` 不能排除坍缩 | §4.2 增加方差/有效秩监控；§4.5 增加可选方差正则 |
| 5 | `L_cons` 目标应是预测状态编码 | §4.1 明确目标为 `E(x_{t+1}^{pred})`，并 detach |
| 6 | `B_xi` 旧 checkpoint 无法 strict 加载 | §7.2 增加显式迁移、缺键白名单、optimizer 状态策略 |
| 7 | 零初始化下“不同 ξ 输出不同”测试矛盾 | §7.3 拆成三部分测试 |
| 8 | G5 的 `∂s_H/∂a_0` 不是策略梯度 | §8.1 拆成 J1/J2a/J2b/J3；G5b 才对应策略梯度 |
| 9 | G0 “JSON 完全一致”过严 | §1 改为关键数值容差 + 确定性测试 |
| 10 | G1 “最近 3 次不上升”有噪声 | §1 改为固定 val、最佳 checkpoint、平滑趋势/置信区间、逐步曲线 |
| 11 | G3 相对一致性可被常数分量稀释 | §1/§4.2 增加绝对误差、方差、有效秩、重构误差 |
| 12 | 数据划分需区分分布内/跨运动 | §2.1 增加协议 A/B，分别报告 |
| 13 | K1 应加 `E1 latent == E1 reencode` | §1 G0b、§2.5 测试、§2.4 脚本断言 |
| 14 | 在线 `wm_gate` 不宜过早 | §8.3 改为离线门禁优先；在线只记录，自动 skip 默认关闭 |
| 15 | 实施顺序应减少同时引入的机制 | §9 改为 P1 诊断 → P2-A/P2-B 正交 → P3 容量 → P3b ξ → P4 梯度 |

---

## 附录 B：最小诊断清单（每次 run 必存）

- `config`（完整 `asdict`）、`git_commit`、`argv`；
- `data_sha256`、`eval_windows_sha256`、`library_hash`；
- `params`、`updates/s`、`peak_mem`、`GPU-hours`；
- `E1..E32` 曲线、`E8@8step`、`E8@32step`；
- `rho_A`、`sigma_max_M`（exact/approx）、`prod_norm`、`z_norm`；
- `cons_abs`、`cons_rel`、`rec_true`、`rec_pred`、`lift`；
- `latent_var`、`latent_erank`；
- 六组通道 NSE、NaN/Inf 计数；
- 权重文件 + 加载所需的迁移信息。

---

## 14. plan5 修复记录（2026-10-31）

本轮针对 `reference/plan5.md` 的 P0/P1 问题落地，不改变原始 Koopman 基线默认行为：

| 问题 | 修复 | 开关/默认 |
|---|---|---|
| `L_cons` 目标未 detach，梯度会穿过目标编码器 | `cons_target_detach` | `--cons-target-detach`，默认 False |
| state-augmented `rec_loss` 恒为 0（直接返回显式 state block） | 新增 `recon` 头，只从 `phi` 重建 canonical state | `reconstruct(z)=recon(z[...,71:])` |
| `lift_loss` 对 state/phi 使用同一权重，phi 被物理块梯度淹没 | `lambda_lift_phys` / `lambda_lift_phi` 分块 | None 时回退 `lambda_lift` |
| `lift_loss` 被乘了两次 | loss 改为 `+ total_lift/k`，`total_lift` 内已含权重 | 经典路径数值不变 |
| resume 无法区分“续训/迁移策略/只恢复 WM” | `resume_mode=full/transfer_policy/wm_only` | 默认 full |
| 评估 val/test 混在 a/b | 新增 `val_*` / `test_a_*` / `test_b_*` 冻结集 | 保留 a/b 兼容 |
| 只看 `sigma_max(M)`，看不到闭环放大 | `diag_koopman.py` 增加 `closed_loop_fd_1step/all` | 自动输出 |

验证：

- `pytest -q tests/test_koopman.py tests/test_wm_protocol.py`：24 passed；
- 20-update smoke：`rec_loss` 从 0 变为 0.206/0.191，`lift_phys`/`lift_phi` 均输出；
- 状态增广 2000-update quick run 启动：`/root/autodl-tmp/logs/wm_plan5/`。

---

## 15. plan7/plan8 执行记录（2026-11-01）

- 在线评估日志拆分为 `wm/train_nse_1step`、`wm/eval_nse_1/8/32step` 和
  `wm/fixed_nse_*`，保证 E1/E8/E32 来自同一批窗口；
- 增加 `schedule.wm_updates_per_policy_update`（方案 B，支持 WM:Actor=2:1/4:1）；
- 增加可选 `schedule.wm_only_iterations`，默认 0，不改变现有行为；
- `PolicyTrainer` 对非有限 rollout/objective/grad 跳过 actor step；环境 action/obs
  非有限时 fail-fast，避免污染 replay buffer；
- 固定数据消融结果见 `experiment/koopman-world-model/RESULTS_FORMAL_QUICK.md`。

关键结论：state small 在 λ_stab=0 时 E1=0.0563、E8=0.1766、E32=0.6141；
λ_stab=10 时变为 0.0599/0.1908/0.6760，**强稳定性惩罚反而变差**。

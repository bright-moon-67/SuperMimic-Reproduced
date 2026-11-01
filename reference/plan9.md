# 对 commit `ed8c12e` 的分析：Ratio2 确实更好，但尚未解决在线策略优化的核心问题

我已读取你指定的 github.com  和 github.com 。

先给出我的总体判断：

这轮实验支持 `WM_RATIO=2` 优于 `WM_RATIO=1`，但还不能证明“增加 World Model 更新次数就能解决 Koopman 在线训练问题”。更值得关注的是：预测精度已经明显提高，真实 Return 却开始出现平台期，而且 32 步预测没有持续同步改善。

这意味着接下来需要从单纯提高拟合频率，转向检查 长时域闭环误差、Actor 梯度可靠性、训练数据分布和训练效率。

## 一、最有说服力的结果：同为 iteration 2500

两个实验从相同的 iteration 1000 checkpoint 恢复，使用 State-Augmented Koopman Large、关闭稳定性正则，主要区别是 WM:Actor 更新比例。

|
指标

|

Ratio1

|

Ratio2

|

Ratio2 改善

|
| --- | --- | --- | --- |
|

Train NSE1

|

0.0944

|

0.0557

|

下降 41.0%

|
|

Eval E8

|

0.2587

|

0.1444

|

下降 44.2%

|
|

Eval E32

|

1.298

|

0.695

|

下降 46.5%

|
|

Imagined Return

|

1.327

|

1.658

|

上升 24.9%

|
|

真实 Train Return

|

0.230

|

0.350

|

上升 52.2%

|
|

预测 Survival

|

0.962

|

0.979

|

上升 1.7 个百分点

|

图表选项

Ratio ablation at iteration 2500

Lower error is better. Both runs resumed from iteration 1000.



这是一个非常有价值的结果：World Model 的多步预测改善与真实环境 Return 改善同时出现。

不过，它只能说明两者相关，还不能证明 Return 的提升完全由预测精度改善造成。两组训练中的 Replay Buffer 和策略轨迹也在不断变化，不是严格固定的数据对照。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

## 二、后续数据显示：Ratio2 的优势还在，但两组都出现了平台迹象

图表选项

Online normalized return progression

Reported checkpoints; higher is better. Curves have different logged iteration points.



最新报告：

|
指标

|

Ratio2，2900 iter

|

Ratio1，4000 iter

|
| --- | --- | --- |
|

Eval E1

|

0.0434

|

0.0497

|
|

Eval E8

|

0.1425

|

0.1627

|
|

Eval E32

|

0.828

|

0.913

|
|

Train Return

|

0.343

|

0.266

|
|

最近 8 点平均 Return

|

0.337

|

0.279

|
|

Imagined Return

|

1.723

|

1.624

|
|

WM Gradient Norm

|

22.3

|

27.9

|
|

Actor Gradient Norm

|

0.065

|

0.066

|

Ratio2 用更少的环境迭代仍然获得了更好的当前指标，但这里不是同 iteration 比较，不能直接作为严格效果差距。

更重要的是两个细节：

* Ratio2 的 Return 从 2300 iteration 的 0.353，到 2900 iteration 的 0.343，暂时没有持续突破，最近也在约 0.32–0.35 之间波动。

* Ratio2 的 E8 从 2500 iteration 的 0.1444 改善到 2900 的 0.1425，但 E32 从 0.695 上升到 0.828。

后者提示：短期预测变好时，长时域预测未必同步改善。 但这些在线评估是在变化的 Replay Buffer 上重新采样的，因此还不能确定是模型长时域能力退化，还是评估数据变难。

这恰好是接下来必须区分的两种情况。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

## 三、从源码看，有四个问题会直接影响这轮实验的解释

这部分我认为比继续调 Koopman 结构更紧迫。

### 1. `WM_RATIO=2` 实际增加的是完整的 World Model 优化次数

在 github.com  中，每次达到 UTD 更新条件，会执行：

WMUpdate⁡()×R⏟R=WM_RATIO⟶ActorCriticUpdate⁡()\underbrace{\operatorname{WMUpdate}() \times R}_{R=\texttt{WM\_RATIO}} \quad\longrightarrow\quad \operatorname{ActorCriticUpdate}()R=WM_RATIOWMUpdate()×R⟶ActorCriticUpdate()

而 github.com  中，一次 `WMUpdate()` 包含 10 个 minibatch，各自执行一次 Adam 更新。

因此：

|
单次 Actor 更新对应的工作量

|

Ratio1

|

Ratio2

|
| --- | --- | --- |
|

WM 优化器步数

|

10

|

20

|
|

Actor 更新调用

|

1

|

1

|
|

Critic 更新阶段

|

1

|

1

|
|

WM 单 minibatch 样本数

|

4096

|

4096

|

这意味着 Ratio2 确实进行了更充分的模型拟合，而不仅仅是改变一个日志频率。

考虑 UTD=0.4，在足够长的训练区间，平均每个向量化仿真迭代对应：

NWM optimizer=0.4×10×R\boxed{ N_{\mathrm{WM\ optimizer}}=0.4\times10\times R }NWM optimizer=0.4×10×R

即 Ratio1 约为 4 次 WM Adam 更新，Ratio2 约为 8 次。这里的 iteration 是 8192 个并行环境共同前进一步，并不是单个机器人发生一次状态转移。

因此 Ratio2 在同 iteration 更好，至少有一个直接解释：它消耗了更多的梯度更新和计算资源。

要区分“模型训练频率更合理”与“单纯训练预算更多”，必须增加相同 WM 优化器步数或相同 GPU-hours 的对照。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

### 2. 发现一个具体的日志统计问题：Ratio2 只记录第二次 WM Update 的指标

当前训练循环的关键逻辑是：

Python

运行

```
m_wm = {}
for _ in range(sch.wm_updates_per_policy_update):
    m_wm = self.wm_trainer.update(self.buffer)
```

`m_wm` 在每次循环中被覆盖。

因此 Ratio2 的 `wm/train_nse_1step`、`wm/grad_norm` 等指标，只代表第二次完整 WM Update，而不是这两次 Update 的平均值。

Ratio1 只有一次，不受这个问题影响。

这不会改变已经执行的梯度更新，因此不是造成 Ratio2 性能差异的训练 bug；但它使部分训练指标的统计口径不完全对齐。

我的建议是：

* 将每次 WM Update 的指标进行累计平均。

* 额外记录 `wm/first_nse1` 和 `wm/last_nse1`，观察重复拟合是否真的降低误差。

* 分别记录 `wm_optimizer_steps` 和 `actor_optimizer_steps`，不要仅使用现有的 `time/updates`。

当前 `num_updates` 每次达到 UTD 更新条件只加一，不会因为 `WM_RATIO=2` 额外加一，因此它并不等于实际的 WM 优化器步数。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

### 3. `full resume` 没有完整恢复 Replay Buffer 和仿真环境状态

这是另一个值得注意的地方。

你的记录说明两个 run 都从 iteration 1000 的同一 Checkpoint，以 `RESUME_MODE=full` 恢复。

源码确实恢复了：

* World Model、Actor、Critic 权重；

* 三套优化器状态；

* iteration 和部分更新计数；

* Motion Sampler 的历史统计。

但是，当前 `save()` 并没有保存 Replay Buffer、完整仿真器状态、RNG 状态，也没有恢复用于在线 NSE 评估的固定窗口。

而 `run()` 开始时还会重新执行：

env.reset⁡()\operatorname{env.reset}()env.reset()

因此它是模型和优化器状态上的完整恢复，并不是整个在线训练过程的逐比特连续恢复。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这意味着即使两个 run 起点的网络参数相同，后续采样的数据也不保证完全相同。

这不否定 Ratio 实验的意义，但应把它解释为：

> 相同网络与优化器起点下，两种在线训练调度的比较。

而不是：

> 两组在每一个时间点使用完全相同的数据，唯一差异就是多训练一次 WM。

对于以后更严格的机制消融，可以在分叉点保存并恢复 Replay Buffer、环境状态和随机数状态。

不过我建议先做共享固定评估集，它的成本更低、收益更直接。

### 4. 当前 `fixed_nse_*` 可以看趋势，但并不是跨 run 共享的测试集

这是我认为优先级最高的评价协议问题。

当前代码在每个 Trainer 中第一次满足条件时执行：

Python

运行

```
self._fixed_eval = self.buffer.sample_windows(1024, 32)
```

这批窗口随后在本次运行中保持不变。

所以它可以用于观察同一 run 的模型对这批窗口的预测误差是否改善。

但 Ratio1 和 Ratio2 是两个独立进程，各自采集数据、各自采样。因此两者的 `_fixed_eval` 不保证相同。

而且这些窗口来自训练 Replay Buffer，本身不是严格独立的 Held-out Test Set。

建议增加一份离线保存的共享固定在线评估数据，让 Ratio1、Ratio2 和 MLP 的所有 Checkpoint 都使用同一批 32 步窗口：

Ek(R=1)(Dshared)vsEk(R=2)(Dshared)\boxed{ E_k^{(R=1)}(\mathcal D_{\mathrm{shared}}) \quad\text{vs}\quad E_k^{(R=2)}(\mathcal D_{\mathrm{shared}}) }Ek(R=1)(Dshared)vsEk(R=2)(Dshared)

其中 k=1,8,16,32k=1,8,16,32k=1,8,16,32。

这样才能回答：

* Ratio2 是否真的具有更好的动力学泛化能力？

* 还是它当前的 Replay Buffer 比较容易预测？

* 32 步误差上升，是否只是数据分布改变？

目前你的 `eval_nse_*` 是在当前 Replay Buffer 中重新采样的，所以这一改动尤其重要。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

## 四、单步动作 Jacobian：结果值得关注，但目前还不足以证明 BPTT 梯度可靠

你记录的 gate 1000 Checkpoint：

|
指标

|

结果

|
| --- | --- |
|

Cosine Similarity

|

0.935

|
|

Relative L2 Error

|

0.413

|
|

Magnitude Ratio

|

0.720

|
|

测试方向数

|

32

|

数学上，它比较的是：

JWMv与JsimvJ_{\mathrm{WM}}v \quad\text{与}\quad J_{\mathrm{sim}}vJWMv与Jsimv

其中：

JWM=∂FWM∂a,Jsim=∂Fsim∂aJ_{\mathrm{WM}}=\frac{\partial F_{\mathrm{WM}}}{\partial a}, \qquad J_{\mathrm{sim}}=\frac{\partial F_{\mathrm{sim}}}{\partial a}JWM=∂a∂FWM,Jsim=∂a∂Fsim

0.935 的平均余弦相似度是一个积极信号，意味着这些测试方向上的响应通常有较好的方向一致性。

但 Magnitude Ratio 为 0.720，说明在这组采样方向上，模型响应范数平均偏小。

按这组平均值粗略理解，约对应 21∘21^\circ21∘ 的方向夹角；但各指标是逐方向求均值，不能把它们当成同一个平均向量的精确几何关系。

### 更重要的是，我发现这份诊断本身存在一个需要先核实的物理条件不一致风险

在 github.com  中：

* World Model 使用数据集提供的 `xi`。

* 用于真实仿真有限差分的环境设置了 `domain_randomization=False`。

* `StateIO.write()` 只回写 `qpos/qvel`，没有把数据集中的摩擦、COM 偏移等域参数重新设置进仿真器。

所以，如果输入数据集是在域随机化条件下采集的，WM 和真实仿真器可能并不处在相同的 ξ\xiξ 下。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

这是一个实质性的可比性问题。

另外，当前诊断只复位 `qpos/qvel` 和 `qacc_warmstart`，还需要确认执行器内部状态、动作管理器历史及接触缓存是否影响相邻有限差分试验。

因此，我不会直接把 `0.720` 解释为“Koopman 的真实动作梯度系统性低估了 28%”，更不会立即建议给模型 Jacobian 乘一个补偿系数。

正确的顺序是：先保证仿真域参数和内部状态严格对齐，再重新测量单步 Jacobian。

其次，这一结果来自 gate 1000，而不是 Ratio1/Ratio2 最新 Checkpoint，所以它也不能证明 Ratio2 的梯度质量现在是否优于 Ratio1。

## 五、为什么模型预测已经改善，Actor Return 仍然明显落后于 MLP？

我认为有三个值得检验的机制，按目前证据强弱排序。

### 原因 A：Koopman 需要更多在线拟合，但目前还不能完全跟上策略变化

证据强度：中等偏强

Ratio2 对比 Ratio1，增加 WM 更新后：

E8↓,E32↓,Rreal↑E_8\downarrow,\quad E_{32}\downarrow,\quad R_{\mathrm{real}}\uparrowE8↓,E32↓,Rreal↑

这支持“World Model 在线拟合不足是性能瓶颈之一”的判断。

不过，即使 Ratio2 已把 E8 降到 0.14 左右，真实 Return 仍在 0.35 附近，而文档报告的既有 MLP 长实验在 iteration 2000 已达到约 0.592。

这表明单纯增加 WM 更新次数，目前尚未消除在线策略性能差距。同时，MLP 实验的训练起点和数据轨迹不一定与当前两组 Koopman 完全一致，不能把这组 Return 差距全部归因于动力学结构。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

### 原因 B：8 步开环误差不足以评价 32 步 Actor BPTT

证据强度：合理假设，尚待验证

当前 World Model 训练窗口是：

k=8k=8k=8

而 Actor 优化时采用：

H=32H=32H=32

并且两种展开方式不同。

WM 评估时使用 Replay Buffer 记录的动作：

s^t+1=Fω(s^t,atrecorded,ξ)\hat s_{t+1}=F_\omega(\hat s_t,a_t^{\mathrm{recorded}},\xi)s^t+1=Fω(s^t,atrecorded,ξ)

Actor BPTT 时使用根据预测状态不断重新计算的动作：

at=πθ(o^t−9:t,gt)s^t+1=Fω(s^t,at,ξ)\boxed{ \begin{aligned} a_t&=\pi_\theta(\hat o_{t-9:t},g_t)\\ \hat s_{t+1}&=F_\omega(\hat s_t,a_t,\xi) \end{aligned} }ats^t+1=πθ(o^t−9:t,gt)=Fω(s^t,at,ξ)

因此，Actor 的梯度不仅依赖动力学的状态预测误差，还依赖：

∂Fω∂st,∂Fω∂at\boxed{ \frac{\partial F_\omega}{\partial s_t}, \qquad \frac{\partial F_\omega}{\partial a_t} }∂st∂Fω,∂at∂Fω

在 32 步内如何连续传播。

你当前的 E8 已经有明显改善，但 E32 在新近一次评估中上升，说明长时域部分仍值得特别关注。

但这里不能据此直接宣称 BPTT 梯度错误。必须通过闭环和梯度实验确认。

### 原因 C：Reward、Termination 和 Critic 可能存在模型—真实环境偏差

证据强度：目前不足，需要新增日志

最新 Ratio2：

Imagined Return=1.723\text{Imagined Return}=1.723Imagined Return=1.723

Predicted Survival=0.976\text{Predicted Survival}=0.976Predicted Survival=0.976

Train Return=0.343\text{Train Return}=0.343Train Return=0.343

这三个指标不能直接相减或作倍数比较。

代码中 `imag_return_undisc` 是 32 步想象轨迹中，按生存权重累计的未折扣奖励；`train_ret` 则是对真实仿真环境已结束 Episode 的奖励按 Episode 长度和最大奖励进行归一化后的结果。

也就是说，两个 Return 并不是同一度量。

但它们仍值得做配对评估，尤其是：

* World Model 是否高估身体姿态跟踪奖励？

* 软终止概率是否低估真正的摔倒风险？

* 32 步终端 Critic 是否给出过于乐观的价值？

现有 `train/episode_len` 指标也应与 Return 一起分析，因为仅查看已结束 Episode 的归一化回报，不足以反映完整的长期存活能力。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

## 六、关于梯度范数，我还有一个具体发现

最新的两个结果：

WM GradNorm=22.3, 27.9\mathrm{WM\ GradNorm}=22.3,\ 27.9WM GradNorm=22.3, 27.9

而训练配置设置：

WM GradClip=1.0\mathrm{WM\ GradClip}=1.0WM GradClip=1.0

在 `wm_trainer.py` 中，`clip_grad_norm_()` 返回的是裁剪前的梯度范数。

因此这些日志说明 WM 梯度尺度明显大于裁剪阈值，相关更新很可能经常受到梯度裁剪的影响。

但不能仅凭平均值断言所有更新都在裁剪，也不能因为 GradNorm 大就认定发生了梯度爆炸。

还有两个细节：

1. Ratio2 目前只汇总第二次 WM Update 的 GradNorm。

2. World Model 的 State Loss 使用 70 维误差的求和，而 NSE 日志使用平均，两者天然具有不同的数值尺度。

我建议添加：

Gradient Clip Fraction=#(∥∇ωL∥>1)#(WM optimizer steps)\boxed{ \text{Gradient Clip Fraction} = \frac{\#(\|\nabla_\omega\mathcal L\|>1)} {\#(\text{WM optimizer steps})} }Gradient Clip Fraction=#(WM optimizer steps)#(∥∇ωL∥>1)

以及不同损失项的梯度范数。

如果 State Loss、Lift Loss、Consistency Loss 的梯度尺度差异很大，就可能需要进一步检查辅助损失对主任务梯度的影响。

Actor 的 `pi_gn=0.065/0.066` 较小，也不意味着梯度方向正确，只能说明它们的总体尺度接近。

另一个较小但明确的源码细节：`a_rho_target=0.99` 在经典 Koopman 的初始化中使用；当前 `StateAugmentedKoopman` 则直接把 AAA 初始化为零矩阵，没有使用这个配置值进行谱半径缩放。因此，记录中的该配置项不代表当前 State-Augmented 模型在训练后满足 ρ(A)≤0.99\rho(A)\leq0.99ρ(A)≤0.99。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

## 七、我建议你接下来按这个顺序开展实验

这次我不建议立即做大量结构改造。你现在已经有比较清楚的结果，下一步应该尽可能让每个实验只回答一个问题。

|
优先级

|

实验或改动

|

具体内容

|

要判断的事情

|
| --- | --- | --- | --- |
|

P0

|

修复训练指标统计

|

Ratio2 的两次 WM Update 全部参与统计，增加实际优化器步数

|

排除日志偏差

|
|

P0

|

共享固定验证集

|

同一批 32 步窗口评估两个 run 的 Checkpoint

|

Ratio2 是否真正改善模型预测

|
|

P0

|

同一策略初始状态的真实仿真评估

|

固定参考动作、初始状态和随机化条件，评估 Actor

|

排除 Train Return 的采样波动

|
|

P1

|

重新测量单步 Jacobian

|

正确恢复域参数与执行器状态，比较 Ratio1/Ratio2/MLP

|

梯度方向是否真的更好

|
|

P1

|

BPTT Horizon 消融

|

在表现较好的 Checkpoint 上测试 H=8、16、32

|

长时域梯度是否影响策略训练

|
|

P1

|

WM-only 阶段

|

延迟 Actor 更新，进一步拟合当前数据

|

Actor 是否过早使用不准确的 WM

|
|

P2

|

重新检查 Loss 权重

|

统计 State、Lift、Consistency、Termination 的梯度贡献

|

是否存在辅助目标冲突

|
|

P2

|

结构性残差改造

|

只有确认模型表达能力仍不足后再做

|

Koopman 结构是否限制预测精度

|

### 我特别推荐三个具体实验

实验 1：Ratio1 和 Ratio2 的 Checkpoint 交叉评估

在共享数据集上评估：

E1,E8,E16,E32Root Position / Rotation ErrorJoint Position / Velocity ErrorTermination Calibration\begin{aligned} &E_1,E_8,E_{16},E_{32}\\ &\text{Root Position / Rotation Error}\\ &\text{Joint Position / Velocity Error}\\ &\text{Termination Calibration} \end{aligned}E1,E8,E16,E32Root Position / Rotation ErrorJoint Position / Velocity ErrorTermination Calibration

至少比较 iteration 2500 的两个模型。

若 Ratio2 在共享数据上仍明显优于 Ratio1，就可以更加确信其预测优势是真实存在的，而非不同 Replay Buffer 造成的假象。

实验 2：从同一个 Checkpoint 缩短 Actor BPTT Horizon

保持 World Model、数据采集配置和 WM_RATIO 不变，单独修改：

H∈{8,16,32}H\in\{8,16,32\}H∈{8,16,32}

我的建议是先测试 H=16，而不是立即缩到 H=8，因为缩短 Horizon 会让策略更依赖 Critic 的终端价值。

如果 H=16 的真实 Return 明显改善，而 WM E8/E32 相近，那么就支持“过长的模型梯度传播限制了 Actor”的假设。

但这需要真正训练并对照，不能只看同一个 Checkpoint 的 Imagined Return。

实验 3：比较多步策略目标的方向导数

这是研究价值最高的一项。

在固定 Actor 参数 θ\thetaθ 和扰动方向 vvv 下，比较：

DvJWM=∇θJWM⊤v\boxed{ D_vJ_{\mathrm{WM}} = \nabla_\theta J_{\mathrm{WM}}^\top v }DvJWM=∇θJWM⊤v

与真实仿真器有限差分：

DvJsim≈Jsim(θ+ϵv)−Jsim(θ−ϵv)2ϵ\boxed{ D_vJ_{\mathrm{sim}} \approx \frac{ J_{\mathrm{sim}}(\theta+\epsilon v) - J_{\mathrm{sim}}(\theta-\epsilon v) }{2\epsilon} }DvJsim≈2ϵJsim(θ+ϵv)−Jsim(θ−ϵv)

对 H=8、16、32 分别测量。

需要对齐随机化参数和初始状态，尽量使用相同外生随机数；真实接触或硬终止发生切换时，应单独统计这些样本，因为有限差分可能不平滑。

这样就可以验证：

Koopman 的 BPTT 梯度究竟是否沿着能改善真实回报的方向传播。

## 八、计算预算：Ratio2 更慢，但不能简单说整体成本翻倍

报告里的速度：

0.6 it/svs0.9 it/s0.6\ \mathrm{it/s}\quad\text{vs}\quad0.9\ \mathrm{it/s}0.6 it/svs0.9 it/s

对应 Ratio2 的单 iteration 耗时约为 Ratio1 的 1.5 倍，而不是完整训练耗时的 2 倍。

另外，文档记录两组训练进程都在运行，GPU 使用率约为 99%。如果它们确实同时竞争同一张 A800，那么当前吞吐量还受到进程间资源竞争的影响。

因此需要区分三种效率：

1. 环境交互效率：达到某个 Return 用多少真实环境步？

2. 训练数据效率：达到某个 Return 用多少 WM 梯度更新？

3. 计算效率：达到某个 Return 用多少 GPU-hours？

Ratio2 目前在第一项上表现更好，但在后两项上还需要更严格的测量。

我建议后续绘制：

Real Returnvs{Environment StepsWM Optimizer StepsGPU-hours\boxed{ \text{Real Return} \quad\text{vs}\quad \begin{cases} \text{Environment Steps}\\ \text{WM Optimizer Steps}\\ \text{GPU-hours} \end{cases} }Real Returnvs⎩⎨⎧Environment StepsWM Optimizer StepsGPU-hours

这比只比较 `it/s` 更能体现 Koopman 的训练效率。

## 九、现在要不要继续把 WM_RATIO 提到 4？

我暂时不建议直接运行完整的 Ratio4 长训。

原因并不是 Ratio4 一定无效，而是：

* Ratio2 虽然改善预测和回报，但真实 Return 已出现短期平台迹象。

* 更高 Ratio 会显著增加 WM 优化成本。

* 尚未排除长时域 Actor 梯度、固定评估分布和 Termination/Critic 的影响。

* Ratio2 的优势是否仍存在于相同 GPU-hours 下，目前尚不清楚。

如果你想测试 Ratio4，可以先用共享 Replay 数据做短期 WM-only 对照，检查增加优化次数是否继续改善 E8/E32；有明显收益后再决定是否投入在线长训。

同样，现在没有必要恢复 `lambda_stab=10`，也没有充分证据支持默认开启 `rec_true`。你的既有固定数据消融已经不支持优先投入这两个方向。

## 十、最终判断

结合这次提交，我的判断可以浓缩成三个层次：

已经有证据支持：增加 WM 更新次数有实际收益。

Ratio2 在相同 iteration 的 E8、E32 和真实 Return 上均明显优于 Ratio1。这不是仅仅改善训练损失。

仍然不确定：为什么更好的 World Model 没有带来接近 MLP 的在线回报。

可能是在线数据拟合仍不充分，也可能是 32 步 BPTT、终止预测或 Critic 的偏差。当前数据还不能区分。

下一步最值得解决：预测精度与策略梯度质量之间的关系。

需要从同一数据上的开环预测，进一步走向闭环轨迹与真实策略目标方向导数的比较。

如果我是按照你现在的 AutoDL 实验成本安排工作，我会：

先保留并观察当前 Ratio2，等它产生与 Ratio1 对齐的 4000 iteration Checkpoint；同时优先修复 WM 指标聚合、建立共享固定验证集，并纠正 `diag_gradient.py` 的仿真域参数对齐问题。

这三项都不需要再次修改 Koopman 的核心架构，却能显著提升后续实验结论的可信度。

之后如果确认：Ratio2 已在同一数据上达到不错的预测精度，但真实 Return 仍明显落后，那么下一阶段最有价值的研究就不再是继续拟合状态，而是直接优化和验证 Koopman 提供的 BPTT 梯度。

这也最贴近 SuperMimic 原论文的核心优势——利用可微 World Model 的一阶梯度，提高策略优化效率，而不仅仅是实现准确的状态预测。


Ratio ablation at iteration 2500

Lower error is better. Both runs resumed from iteration 1000.

metric	ratio1	ratio2
E8	0.2587	0.1444
E32	1.298	0.695

Online normalized return progression

Reported checkpoints; higher is better. Curves have different logged iteration points.

iteration	ratio2	ratio1
1050	0.161	
1500	0.205	
1700		0.146
2000	0.32	0.221
2300	0.353	
2500	0.35	0.23
2900	0.343	
3000		0.241
3400		0.316
4000		0.266
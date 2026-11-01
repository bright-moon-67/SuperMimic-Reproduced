# Koopman-SuperMimic：当前代码修改建议

我重新检查了 github.com  当前公开的 `main` 分支，重点阅读了 `koopman.py`、`world_model.py`、`rollout.py`、`policy_trainer.py`、`training/loop.py`、实验脚本和相关测试。

结论：不需要推翻现有架构，但有几处代码应该优先修改。我认为目前最需要解决的不是 Actor 或 Critic，而是 Koopman 的训练目标、物理状态闭环与稳定性诊断之间的不一致。

先明确两点：

* 之前的低秩双零初始化和 `cons_rel` RMS/L2 混用问题，在当前源码中都已经修复，不应重复修改。

* `lambda_cons` 现在确实进入了 World Model 总损失，并且 `run_state_koopman_quick.sh` 和在线脚本的 State-Augmented 分支会显式启用它。真正仍需审查的是一致性损失的梯度方向和效果，而不是它是否存在。

  ![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

  GitHub+3

以下区分可以从源码确定的问题与需要消融实验验证的设计风险。

## 一、最应该修改的六个位置

|
优先级

|

文件及函数

|

当前问题

|

建议

|
| --- | --- | --- | --- |
|

P0

|

`world_model.py` → `_window_loss_koopman`

|

一致性目标端也参与反向传播

|

增加 `detach_target` 开关，先做梯度对照

|
|

P0

|

`koopman.py` → `StateAugmentedKoopman`

|

重构损失对显式状态块恒为零

|

改为真实状态辅助监督或有信息量的潜状态监督

|
|

P0

|

`koopman.py` → `stability_penalty`

|

约束的是中间传播矩阵，而非完整物理闭环 Jacobian

|

增加闭环敏感度诊断

|
|

P0

|

`training/loop.py` → `Trainer.load`

|

迁移旧 Actor 时仍继承旧训练迭代数

|

把模型迁移和完整续训拆开

|
|

P1

|

`collect_wm_data.py`

|

验证集和测试集被合并成同一评估集合

|

分离 `val/test`，避免测试集参与调参

|
|

P1

|

`world_model.py` → `window_loss`

|

Koopman 辅助损失与物理状态误差存在重复监督和尺度差异

|

分通道、分潜变量块记录梯度与误差

|

我会优先处理前三项，再修训练与评估协议，最后才考虑扩大 Koopman 的网络容量。

## 二、P0-1：一致性损失应当明确控制梯度流向

文件：`src/supermimic/world_model/world_model.py`，`_window_loss_koopman()`

当前代码的核心是：

Python

运行

```
cons_step = z - self.encode_state(s, xi)
if state_in_latent:
    cons_step = cons_step[..., 71:]
total_cons = total_cons + cons_step.pow(2).mean()
```

对应：

Lcons=∥zt+1−Eϕ(s^t+1,ξ)∥2\mathcal L_{\mathrm{cons}} = \left\| z_{t+1} - E_\phi(\hat s_{t+1},\xi) \right\|^2Lcons=∥zt+1−Eϕ(s^t+1,ξ)∥2

当前代码中，右侧的 Encoder 输出没有停止梯度。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这会产生两类梯度：

Lcons→∇zt+1\mathcal L_{\mathrm{cons}} \xrightarrow{\nabla} z_{t+1}Lcons∇zt+1

以及：

Lcons→∇Eϕ(s^t+1,ξ)\mathcal L_{\mathrm{cons}} \xrightarrow{\nabla} E_\phi(\hat s_{t+1},\xi)Lcons∇Eϕ(s^t+1,ξ)

第二条路径又包括：

Eϕ⟶s^t+1⟶Decoder / Integration / DynamicsE_\phi \longrightarrow \hat s_{t+1} \longrightarrow \text{Decoder / Integration / Dynamics}Eϕ⟶s^t+1⟶Decoder / Integration / Dynamics

因此，目前的 `consistency loss` 不仅在约束传播潜状态靠近重新编码的物理状态，也同时允许重新编码结果朝传播潜状态移动。

这不是必然错误，但与你之前修复计划中采用停止梯度目标的设计不同。

### 建议修改

增加配置：

`cons_target_detach: bool = True`

将一致性目标明确拆开：

ztarget=sg⁡(Eϕ(s^t+1,ξ))z_{\mathrm{target}} = \operatorname{sg} \left(E_\phi(\hat s_{t+1},\xi)\right)ztarget=sg(Eϕ(s^t+1,ξ))

然后：

Lcons=∥zt+1−ztarget∥2\boxed{ \mathcal L_{\mathrm{cons}} = \left\| z_{t+1}-z_{\mathrm{target}} \right\|^2 }Lcons=∥zt+1−ztarget∥2

其中 sg⁡\operatorname{sg}sg 代表停止梯度。

这样，当前这项损失主要推动预测潜状态及其前向生成路径去匹配目标，不再通过目标 Encoder 产生额外梯度。

但不要把 `detach=True` 当成必然更好的版本。它也存在目标 Encoder 训练不足、目标随其他损失变化等问题。

我建议先比较两个版本：

|
版本

|

目标 Encoder 反传

|

作用

|
| --- | --- | --- |
|

C0：当前实现

|

是

|

双侧一致性优化

|
|

C1：`detach_target=True`

|

否

|

单侧潜状态对齐

|

除了 E1/E8/E32，还应检查 Encoder 的梯度范数、潜状态有效秩，以及 `consistency loss` 是否下降。

我在独立 PyTorch 小测试中验证了这两种计算图的区别：不使用 `detach` 时目标 Tensor 会获得梯度，使用 `detach` 后目标 Tensor 不获得该损失的梯度，而预测 Tensor 仍获得梯度。

验收标准： 不只是 `L_cons` 下降，而是长时域状态误差下降，且潜空间没有坍缩。

## 三、P0-2：State-Augmented Koopman 的重构损失存在退化

文件：`src/supermimic/world_model/koopman.py`，`StateAugmentedKoopman.reconstruct()`

这项问题可以直接从代码结构确认。

你现在定义：

zt=\(xtϕt\)z_t= \begin{bmatrix} x_t\\ \phi_t \end{bmatrix}zt=\(xt​ϕt​​\)

其中：

xt=Canonical⁡(st)x_t=\operatorname{Canonical}(s_t)xt=Canonical(st)

而 `reconstruct()` 的实现是直接返回潜状态的物理特征块：

Python

运行

```
def reconstruct(self, z):
    return z[..., : self.feat_dim]
```

与此同时，状态增广分支的每次传播会把预测物理状态重新写入该块：

zt+1=\(Canonical⁡(s^t+1)ϕt+1\)z_{t+1} = \begin{bmatrix} \operatorname{Canonical}(\hat s_{t+1})\\ \phi_{t+1} \end{bmatrix}zt+1=\(Canonical(s^t+1​)ϕt+1​​\)

因此正常情况下：

R(zt)=xtR(z_t)=x_tR(zt)=xt

于是：

Lrec=∥R(zt)−xt∥2=0\boxed{ \mathcal L_{\mathrm{rec}} = \|R(z_t)-x_t\|^2 = 0 }Lrec=∥R(zt)−xt∥2=0

这个损失基本不提供学习信号。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+2

这是一个需要修改的确定性问题。

### 我建议如何修改？

对于 State-Augmented 分支，直接返回物理状态块是合理的，但它不应再被当作有意义的辅助重构学习目标。

你可以选择：

方案 A：禁用当前退化的重构损失。

保持状态块直接复制，但 `L_rec` 不再参与该分支优化，避免日志产生误导。

方案 B：新增由学习特征 ϕt\phi_tϕt 重建真实状态的辅助 Head。

x~t=Rη(ϕt)\tilde x_t=R_\eta(\phi_t)x~t=Rη(ϕt)

Lrec,true=∥Rη(Eϕ(xttrue,ξ))−xttrue∥2\boxed{ \mathcal L_{\mathrm{rec,true}} = \|R_\eta(E_\phi(x_t^{\mathrm{true}},\xi)) -x_t^{\mathrm{true}}\|^2 }Lrec,true=∥Rη(Eϕ(xttrue,ξ))−xttrue∥2

这是我更倾向于做的研究版本，因为它要求学习得到的 ϕt\phi_tϕt 保留真实状态信息，而不是依赖直接复制的物理特征块获得零损失。

不过，ϕt\phi_tϕt 是否需要完整重构物理状态，本身仍是建模选择；过强的重构约束可能占用表示能力。因此建议先将它作为辅助消融，而不是强制主损失。

另外，我建议你在日志中分别记录：

Lrec,physical\mathcal L_{\mathrm{rec,physical}}Lrec,physical

Lrec,learned\mathcal L_{\mathrm{rec,learned}}Lrec,learned

避免把一个恒为零的损失误认为编码器已经具备重构能力。

## 四、P0-3：State-Augmented 的稳定性约束没有覆盖真实的闭环动力学

文件：`src/supermimic/world_model/koopman.py`，`transition_matrix()`、`stability_penalty()`、`step_latent()`

这是我认为对后续训练效果影响最大的结构性问题，但它属于设计风险，不是代码执行错误。

当前 State-Augmented 模型先计算：

ut+1=M(at,ξ)zt+Bat+Cξ+bu_{t+1}=M(a_t,\xi)z_t+B a_t+C\xi+but+1=M(at,ξ)zt+Bat+Cξ+b

接着：

Δt=D(ut+1phys)\Delta_t=D(u_{t+1}^{\mathrm{phys}})Δt=D(ut+1phys)

st+1=Integrate⁡(st,Δt)s_{t+1}=\operatorname{Integrate}(s_t,\Delta_t)st+1=Integrate(st,Δt)

最后：

zt+1=\(Canonical⁡(st+1)ut+1latent\)z_{t+1} = \begin{bmatrix} \operatorname{Canonical}(s_{t+1})\\ u_{t+1}^{\mathrm{latent}} \end{bmatrix}zt+1=\(Canonical(st+1​)ut+1latent​​\)

但你当前的稳定性正则只约束：

M(at,ξ)\boxed{ M(a_t,\xi) }M(at,ξ)

即：

Lstab=max⁡(0,σmax⁡(M)−1)2\mathcal L_{\mathrm{stab}} = \max(0,\sigma_{\max}(M)-1)^2Lstab=max(0,σmax(M)−1)2

问题在于：

Actor 进行 BPTT 时，实际经过的不是单独的 MMM，而是完整的状态转移函数。

真正需要关心的是：

Jclosed=∂(st+1,ϕt+1)∂(st,ϕt)\boxed{ J_{\mathrm{closed}} = \frac{\partial(s_{t+1},\phi_{t+1})} {\partial(s_t,\phi_t)} }Jclosed=∂(st,ϕt)∂(st+1,ϕt+1)

它还包含 Decoder、积分器以及 `Canonical` 操作的 Jacobian。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

### 一个能够说明问题的简单例子

假设内部矩阵：

M=0M=0M=0

同时 Decoder 预测零增量，而物理积分采用：

st+1=st+Δts_{t+1}=s_t+\Delta_tst+1=st+Δt

则：

Δt=0\Delta_t=0Δt=0

所以：

st+1=sts_{t+1}=s_tst+1=st

此时：

∥M∥2=0\|M\|_2=0∥M∥2=0

但：

∂st+1∂st=1\boxed{ \frac{\partial s_{t+1}}{\partial s_t}=1 }∂st∂st+1=1

也就是说，内部 Koopman 矩阵完全收缩，不代表完整物理状态映射也收缩。

我用一个独立的 PyTorch 自动微分小实验验证了这个例子。

### 建议修改

不要立刻删除已有 `stability_penalty()`。

它仍可用于分析中间潜空间动力学。

但应新增一项：

`closed_loop_jacobian_diagnostics()`

用于在少量固定验证状态上估计：

∥Jclosedv∥\left\|J_{\mathrm{closed}}v\right\|∥Jclosedv∥

以及连续多步：

∥Jclosed,H−1⋯Jclosed,0v∥\left\| J_{\mathrm{closed},H-1}\cdots J_{\mathrm{closed},0}v \right\|∥Jclosed,H−1⋯Jclosed,0v∥

建议优先使用 Jacobian-Vector Product（JVP）或者有限差分方向导数，不需要每次显式构造完整 Jacobian。

第一次只做诊断，不要马上加入新的闭环 Jacobian 正则，否则会同时改变训练目标和模型行为。

如果发现中间矩阵很稳定，但完整闭环依然放大状态扰动，说明继续增大 λstab\lambda_{\mathrm{stab}}λstab 未必有效。

这是区分「Koopman 矩阵不稳定」和「物理积分闭环不稳定」最直接的方法。

## 五、P0-4：Checkpoint 恢复需要区分完整续训与模型迁移

文件：`src/supermimic/training/loop.py`，`Trainer.load()`

当前代码使用：

Python

运行

```
self.iteration = ck["iteration"]
self.start_iteration = self.iteration
self.num_updates = ck["num_updates"]
```

而新的训练过程中，World Model 的学习率依然根据 `self.iteration` 计算。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

在 `run_lafan1_online.sh` 中：

Bash

```
RESUME_ARGS=(--resume "$RESUME" --resume-skip wm)
```

这会产生一个问题：

当你把旧 MLP 实验中已经训练好的 Actor/Critic 迁移过来，使用一个新的 Koopman World Model 时：

* Actor 可能处于训练中后期；

* Koopman World Model 从初始化开始；

* 学习率调度仍继承旧迭代数。

虽然 `start_iteration` 的处理允许重新经过 warm-up，但它并没有重置学习率调度的时间坐标。

### 建议修改

在训练配置中增加明确的恢复模式：

|
模式

|

加载内容

|

调度行为

|
| --- | --- | --- |
|

`resume_full`

|

全部模型和优化器

|

保留迭代数

|
|

`transfer_policy`

|

Actor、可选 Critic

|

新建 WM 和优化器，重置训练进度

|
|

`resume_wm`

|

Koopman WM 及其优化器

|

恢复 WM 对应训练进度

|

同时避免把「跳过 WM 权重」和「恢复完整训练进度」默认捆绑。

对于你当前的实验，我更推荐 `transfer_policy`。

这能明确比较：

Pretrained Actor + New Koopman\text{Pretrained Actor + New Koopman}Pretrained Actor + New Koopman

而不是在不知不觉中继续执行旧实验的学习率调度。

另外，新的 Koopman 尚未通过长时域验证时，不应该仅凭已经过了 warm-up 就自动更新 Actor。

## 六、P1：`lift_loss` 与物理状态损失需要分块处理

文件：`src/supermimic/world_model/world_model.py`，`_window_loss_koopman()`

当前 State-Augmented 模型中：

zt=\(xtϕt\)z_t= \begin{bmatrix} x_t\\ \phi_t \end{bmatrix}zt=\(xt​ϕt​​\)

而 `lift_loss` 直接比较：

Llift=∥zt+1−Eaug(st+1true)∥2\mathcal L_{\mathrm{lift}} = \|z_{t+1}-E_{\mathrm{aug}}(s_{t+1}^{\mathrm{true}})\|^2Llift=∥zt+1−Eaug(st+1true)∥2

因此它实际上包含：

Llift=Lphysical block+Llearned block\mathcal L_{\mathrm{lift}} = \mathcal L_{\mathrm{physical\ block}} + \mathcal L_{\mathrm{learned\ block}}Llift=Lphysical block+Llearned block

这里省略了不同维度数量和均值归一化产生的比例系数。

但你已经有：

Lstate\mathcal L_{\mathrm{state}}Lstate

监督预测物理状态。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这使物理状态误差在两个损失中被重复监督，而且两个损失使用的状态尺度不一样：

* State Loss 使用 `StateSpec.error_vector()` 计算归一化物理误差；

* Lift Loss 使用潜变量和 canonical feature 的直接均方误差。

不一定错误，但它可能使训练梯度的分配偏离你期望的方向。

### 建议修改

将 Lift Loss 拆为：

Llift=λphysLphys+λϕLϕ\boxed{ \mathcal L_{\mathrm{lift}} = \lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}} + \lambda_\phi\mathcal L_\phi }Llift=λphysLphys+λϕLϕ

其中：

Lphys=∥xt+1pred−xt+1true∥2\mathcal L_{\mathrm{phys}} = \|x_{t+1}^{\mathrm{pred}}-x_{t+1}^{\mathrm{true}}\|^2Lphys=∥xt+1pred−xt+1true∥2

Lϕ=∥ϕt+1pred−sg⁡(Eϕ(xt+1true,ξ))∥2\mathcal L_\phi = \|\phi_{t+1}^{\mathrm{pred}} - \operatorname{sg}(E_\phi(x_{t+1}^{\mathrm{true}},\xi))\|^2Lϕ=∥ϕt+1pred−sg(Eϕ(xt+1true,ξ))∥2

对于 State-Augmented 分支，我建议优先测试：

λphys=0\lambda_{\mathrm{phys}}=0λphys=0

只用原有 `L_state` 监督物理状态，让 `L_phi` 专门约束学习的潜变量演化。

然后与当前全维 Lift Loss 做对照。

这样更容易判断辅助 Koopman 损失本身是否有效。

## 七、P1：固定数据实验仍然混用了 Validation 与 Test

文件：`experiment/koopman-world-model/collect_wm_data.py`

当前脚本构造评估窗口时使用：

Python

运行

```
("a", (*val_clips, *test_a_clips))
("b", (*val_clips, *test_b_clips))
```

也就是将验证动作和测试动作一起放进评估集。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这意味着，如果你根据 `eval_a` 或 `eval_b` 的结果不断调整损失权重、容量和结构，那么测试集实际上也参与了模型选择。

建议改成独立的：

* `val_a`：验证与超参数选择；

* `test_a`：分布内最终测试；

* `test_b`：跨动作最终测试。

只有选定最佳 Checkpoint 后才使用 Test 集进行报告。

除此之外，建议正式实验至少保存每个状态通道的 E1/E8/E32，而不只保存全状态平均误差。

因为你的旧实验已经显示，关节位置和根部姿态的长时域误差比较突出。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

## 八、暂时不建议修改的地方

有几处我反而建议保持原样。

Actor 的 10 帧历史观测机制不需要改。 当前 `imagination/rollout.py` 正确维护了历史队列，并在每一步更新预测观测。

Critic 暂时不需要改结构。 当前 Actor 更新时冻结 Critic 参数，但终端 Value 仍在可微计算图里；Critic 随后使用停止梯度的 GAE Target 独立训练。这一基本逻辑没有发现需要因 Koopman 而修改的地方。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+2

暂时不要给 Koopman Encoder 强行增加 10 帧历史输入。 原始 MLP World Model 也是基于当前完整仿真状态进行预测。当前 Koopman 的效果差，尚不能归因于缺少显式历史。

也不建议立即增加更多层的 MLP 或更大的 dzd_zdz。 如果没有先修复训练目标和诊断协议，扩大模型只会进一步增加实验成本。

## 九、建议交给代码 Agent 的修改顺序

我建议你按以下顺序提交，每次只改一个独立模块：

1. PR 1：修辅助损失

   在 `world_model.py` 中增加 `cons_target_detach` 开关；对 State-Augmented 分支禁用退化重构损失；将 `lift_loss` 拆分为物理块和学习特征块。

   验收：默认配置不改变旧模型数值；开启新配置后，梯度到达预期参数。

2. PR 2：补全闭环动力学诊断

   在 `scripts/diag_koopman.py` 中新增完整状态转移的方向导数测试，同时保留现有 `sigma_max_M` 和矩阵乘积监控。

   验收：能够清楚区分内部潜动力学放大与物理状态闭环放大。

3. PR 3：修 Checkpoint 迁移

   在 `training/loop.py` 中区分 `resume_full` 与 `transfer_policy`，迁移新 WM 时重新初始化相应优化器和训练调度。

   验收：旧 Actor 权重正确迁移，新 Koopman 从独立的学习率起点训练。

4. PR 4：修评估协议

   分离 Validation/Test，固定窗口、数据哈希及随机种子，重跑 MLP、当前 Koopman 和修改后的 Koopman。

   验收：同一数据与评估协议下得到可直接对比的结果。

5. PR 5：恢复 BPTT 验证

   世界模型预测通过检查后，先验证 Actor 梯度能否穿过 Koopman、Decoder 和积分器，再与真实仿真器的策略扰动方向比较。

   验收：真实仿真 Return 改善，而不仅仅是想象 Return 改善。

## 十、如果只能先改一处，我选哪一处？

我会先改 `world_model.py` 中的 State-Augmented 辅助损失。

具体是：

LWM=Lstate+Lterm+λϕLlift,ϕ+λconsLcons,detach+λrecLrec,true+λstabLstab\boxed{ \begin{aligned} \mathcal L_{\mathrm{WM}} ={}& \mathcal L_{\mathrm{state}} + \mathcal L_{\mathrm{term}}\\ &+ \lambda_\phi\mathcal L_{\mathrm{lift},\phi}\\ &+ \lambda_{\mathrm{cons}}\mathcal L_{\mathrm{cons,detach}}\\ &+ \lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec,true}} + \lambda_{\mathrm{stab}}\mathcal L_{\mathrm{stab}} \end{aligned} }LWM=Lstate+Lterm+λϕLlift,ϕ+λconsLcons,detach+λrecLrec,true+λstabLstab

其中 `L_rec,true` 应使用真正的学习式辅助重构 Head；如果暂时不实现，就先禁用该项，而不是继续报告恒为零的重构损失。

这只是建议优先验证的新损失组合，尚不能认为它一定优于当前实现。

同时保留原版本做对照，确保能回答究竟是哪一项改变改善了性能。

我对当前代码的总体判断是：BPTT 主路径基本完整，明显需要加强的是 Koopman 内部的有效监督，以及验证真实物理闭环的梯度质量。

需要说明：本次是公开源码的静态审查，并使用独立 PyTorch 小实验验证了关键梯度性质；我没有实际下载并运行你的完整仓库，也没有访问 AutoDL Checkpoint。因此，以上可以确认代码结构和部分数学问题，但不能声称修改后已实测改善训练。

我建议先完成 PR 1 和 PR 2，不扩大网络，也不继续大规模 Actor 联合训练。这样最有机会以较低成本判断，当前失败究竟来自辅助损失设计，还是 Koopman 结构本身不足以拟合复杂人形动力学。

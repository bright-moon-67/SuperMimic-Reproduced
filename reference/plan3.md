# 《KOOPMAN_FIX_PLAN.md》技术审查报告

审查结论：建议有条件通过，但不建议按原文直接实施全部阶段。

这份计划的总体方向是正确的：先冻结现有结果、建立可复现的诊断协议，再分别研究潜空间一致性、动力学稳定性、模型容量和 BPTT 梯度质量。特别值得肯定的是，你已经明确区分了同权重评估与不同训练目标之间的比较，避免了之前实验中较严重的因果归因问题。

但是，我认为当前版本仍有几处需要在编码前修正的技术问题，主要涉及：

* 对 `reencode` 实验异常结果的判断过于绝对；

* 稳定性正则化的数学含义不够严谨；

* 潜空间一致性损失仍存在表示坍缩风险；

* Checkpoint 兼容性与配置开关存在工程冲突；

* BPTT 梯度验收标准与拟实施的梯度测试尚未严格对应。

我结合上传文件和目前公开的仓库源码进行了交叉核对。以下审查针对你文件中注明的 `6ea5b92` 基线；公开仓库中能够读取到的相关实现与计划描述基本对应，但我没有运行你的 AutoDL 训练或读取未公开的模型权重。

KOOPMAN_FIX_PLAN.md

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+2

## 一、整体评审

方案完整性

较好

实验可复现性设计

较好

数学与损失设计

需修订

BPTT 验证方法

需修订

直接进入联合训练

暂不建议

这份计划最正确的决定，是把阶段 0 放在所有模型修改之前。

原脚本确实没有保存 Koopman 权重，并且评估窗口来自当前 Replay Buffer 的即时采样。`window` 参数也没有传入 `WMTrainerCfg`。因此，先修复实验基础设施，不仅合理，而且必要。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

我建议保留原计划的整体分阶段结构，但调整一些验收指标和实施细节。

## 二、必须修改的问题

### 1. 关于异常的 `reencode` 实验：不能认定 E8 在数学上“不可能发散”

必须修改 · P0

文件第 25–27 行写道：

> `reencode` 评估走 `WorldModel.step`，与 MLP 相同的重编码路径，8 步不可能发散。

我不同意这个推断。

KOOPMAN_FIX_PLAN.md

`reencode` 仅意味着每步重新编码物理状态：

zt=Eϕ(xt,ξ)s^t+1=Integrate⁡(st,Dψ(K(zt,at)))\begin{aligned} z_t &= E_\phi(x_t,\xi)\\ \hat s_{t+1} &=\operatorname{Integrate} \left(s_t,D_\psi(K(z_t,a_t))\right) \end{aligned}zts^t+1=Eϕ(xt,ξ)=Integrate(st,Dψ(K(zt,at)))

它没有消除自回归动力学：

s^t+1=Fω(s^t,at,ξ)\hat s_{t+1}=F_\omega(\hat s_t,a_t,\xi)s^t+1=Fω(s^t,at,ξ)

如果 s^t\hat s_ts^t 偏离训练分布，下一步重新编码的输入也会偏离训练分布。Encoder、Koopman 动力学、Decoder 和物理积分器组合起来，依然可以产生不稳定的闭环映射。

因此：

Reencode⇏Closed-loop Stability\boxed{ \text{Reencode} \not\Rightarrow \text{Closed-loop Stability} }Reencode⇒Closed-loop Stability

`E8` 与训练日志里的 `nse_kstep` 差异极大，的确是一个需要严查的异常，但仅凭两者不同，不能证明数据自相矛盾。

需要先核实：

1. 训练与评估是否使用完全相同的动力学模式；

2. 两项指标是否使用相同的输入窗口分布；

3. `nse_kstep` 是否只表示训练窗口的最后一步误差；

4. 评估数据是否存在异常状态、极端长尾样本或者非法值；

5. `rollout_error()` 是否使用了与训练相同的状态积分流程。

建议修订： 将 v1 标注为“异常、暂不可用于性能结论，待复核”，而不是认定“重编码八步不可能发散”。

这个修正很重要，否则后续可能忽略一个真实存在的 `reencode` 闭环稳定性问题。

### 2. 稳定性正则的两种实现不能视为等价替代

必须修改 · P0

计划第 287–297 行提出：

Lpower=E\(max⁡(0,σmax⁡(Mt)−1)2\)\mathcal L_{\mathrm{power}} = \mathbb E \left\( \\max(0,\\sigma\_{\\max}(M\_t)-1)^2 \\right\)Lpower=E\(max(0,σmax​(Mt​)−1)2\)

同时给出 Frobenius 正则：

LF=∥Mt∥F2dz\mathcal L_F = \frac{\|M_t\|_F^2}{d_z}LF=dz∥Mt∥F2

作为更便宜的对照。

KOOPMAN_FIX_PLAN.md

两者并不等价。

第一项有明确的阈值意义：惩罚最大奇异值超过 1 的矩阵。

第二项只是惩罚全部元素的平方和。它会持续将矩阵压小，包括动力学中可能需要保持的单位模长或慢变化模式。

尤其对于机器人惯性、位置与姿态的积分特性，这种无阈值收缩可能明显损害预测。

因此，我建议：

* 保留 Frobenius 作为消融实验；

* 但不要将其视为更便宜、语义相同的稳定性正则；

* 将它单独称为 Matrix Weight Decay Baseline；

* 对重要的近单位特征值模式，记录正则化前后的预测偏差。

此外，计划中的 `sigma_max_power(M, iters=3)` 还有一个数值问题：三次幂迭代可能严重低估实际最大奇异值，特别是在最大与次大奇异值很接近时。

在某些情况下，这会使模型看起来满足：

σ^max⁡(M)≤1\hat\sigma_{\max}(M)\leq1σ^max(M)≤1

但实际：

σmax⁡(M)>1\sigma_{\max}(M)>1σmax(M)>1

因此建议把训练用的低成本近似和诊断用的准确 SVD 分开，并定期测量近似误差。

### 3. 仅靠 `detach` 不能排除潜空间表示坍缩

必须修改 · P0

计划第 237–241 行提出：

Lcons=∥zt+1−sg⁡(E(xt+1pred))∥2\mathcal L_{\mathrm{cons}} = \left\| z_{t+1} - \operatorname{sg}(E(x_{t+1}^{\mathrm{pred}})) \right\|^2Lcons=zt+1−sg(E(xt+1pred))2

并注明 `detach` 用于避免两边一起坍缩。

KOOPMAN_FIX_PLAN.md

这里需要纠正：

停止目标分支的梯度，可以降低两个分支同时互相追逐的问题，但不能从数学上保证不会表示坍缩。

例如，如果：

E(x)=0,K(0,a)=0E(x)=0,\qquad K(0,a)=0E(x)=0,K(0,a)=0

则：

Llift=0\mathcal L_{\mathrm{lift}}=0Llift=0

Lcons=0\mathcal L_{\mathrm{cons}}=0Lcons=0

这两个损失本身无法排除常数表示。

你新增的真实状态重构损失是正确方向：

Lrec,true=∥R(E(xttrue))−xttrue∥2\mathcal L_{\mathrm{rec,true}} = \|R(E(x_t^{\mathrm{true}}))-x_t^{\mathrm{true}}\|^2Lrec,true=∥R(E(xttrue))−xttrue∥2

因为它能够在重构准确的前提下迫使 Encoder 保留状态信息。

但仍建议监控潜变量的跨样本方差和有效秩：

Var⁡B(z:,j)\operatorname{Var}_{\mathcal B}(z_{:,j})VarB(z:,j)

以及：

EffectiveRank⁡(Cov⁡(z))\operatorname{EffectiveRank}(\operatorname{Cov}(z))EffectiveRank(Cov(z))

否则可能出现：潜空间一致性损失下降，但有效动力学信息也在减少。

还有一个值得修正的细节：`L_cons` 中使用的目标，应明确是预测物理状态经过编码后的潜变量，而不是下一真实状态的潜变量。前者是内部状态闭合约束，后者是动力学监督，两者的意义不同。

### 4. `B_xi` 的 Checkpoint 兼容性描述有误

必须修改 · P1

第 303–310 行提出：

> `B_xi` 始终注册、零初始化，保证不同配置的 `state_dict` 键一致，旧 checkpoint 可直接加载。

前半句只对采用新代码创建的模型成立。

旧模型的 `state_dict` 中不存在 `B_xi`。新模型增加这个 `nn.Parameter` 后：

Python

运行

```
new_model.load_state_dict(old_state, strict=True)
```

原则上会因为缺少 `B_xi` 而报错。

`strict=False` 可以允许缺失键，但必须明确检查允许缺失的参数名，避免把其他不兼容错误悄悄忽略。

因此，推荐：

* 新代码不同配置之间统一注册 `B_xi`；

* 对旧 Checkpoint 使用显式迁移逻辑；

* 迁移后验证旧模型输出与新模型关闭 `xi_every_step` 时数值一致；

* 不默认恢复不兼容的旧 Optimizer State。

还有另一个问题：计划同时要求 `B_xi` 零初始化，并测试同一 (z,a)(z,a)(z,a) 输入不同 ξ\xiξ 时，输出应该不同。

KOOPMAN_FIX_PLAN.md

零初始化时，这个测试应该得到相同输出，而不是不同输出。

因为：

Bξ=0  ⟹  Bξξ1=Bξξ2=0B_\xi=0 \implies B_\xi\xi_1=B_\xi\xi_2=0Bξ=0⟹Bξξ1=Bξξ2=0

正确的测试应该分成三部分：

1. 零初始化时，打开或关闭条件化仍保持原模型输出。

2. 人为设置非零 `B_xi`，验证不同 ξ\xiξ 会改变输出。

3. 验证训练后 `B_xi` 能获得非零梯度，并且不同 ξ\xiξ 的响应可以被学习。

这是一个明确的工程测试矛盾，建议在实现 K4 之前解决。

### 5. BPTT 梯度的验收目标与测试目标不一致

必须修改 · P0

这是我认为整份报告中最需要进一步完善的科学验证设计。

你设定了：

cos⁡(gWM,gsim)≥0.5\cos(g_{\mathrm{WM}},g_{\mathrm{sim}})\geq0.5cos(gWM,gsim)≥0.5

作为 G5 门禁。

但阶段 6 的具体实现写成：

∂sH∂a0\frac{\partial s_H}{\partial a_0}∂a0∂sH

与真实仿真器有限差分比较。

KOOPMAN_FIX_PLAN.md

KOOPMAN_FIX_PLAN.md

这两个对象不是一回事。

前者如果是我们真正关心的策略梯度，应该指：

gWM=∇θJ^WM(θ)\boxed{ g_{\mathrm{WM}}=\nabla_\theta\hat J_{\mathrm{WM}}(\theta) }gWM=∇θJ^WM(θ)

后者是：

JH,0=∂sH∂a0\boxed{ J_{H,0} = \frac{\partial s_H}{\partial a_0} }JH,0=∂a0∂sH

它是关于初始动作的状态 Jacobian，不是关于 Actor 参数的回报梯度。

此外，计划还要求保持同一动作序列：

(a0,…,aH)(a_0,\ldots,a_H)(a0,…,aH)

这测量的是固定动作序列的开环动力学敏感度。

真实的 SuperMimic BPTT 则是：

at=πθ(ot,gt)a_t=\pi_\theta(o_t,g_t)at=πθ(ot,gt)

状态改变会进一步影响未来动作，因此属于闭环梯度。

两者必须分开评估。

我建议改成三级测试：

|
级别

|

比较对象

|

回答的问题

|
| --- | --- | --- |
|

J1

|

单步 Js,avJ_{s,a}vJs,av

|

世界模型的一步动作梯度是否正确？

|
|

J2

|

多步 ∂sH/∂a0:H−1\partial s_H/\partial a_{0:H-1}∂sH/∂a0:H−1 的方向导数

|

固定动作下的长期动力学敏感度是否正确？

|
|

J3

|

∇θJ\nabla_\theta J∇θJ 的方向一致性

|

世界模型是否真正提供有利于 Actor 的梯度？

|

其中 J3 才应该对应 G5。

在仿真器不可自动微分的情况下，可以选择 Actor 参数空间中的随机方向 vθv_\thetavθ，计算真实回报的中心差分：

DvJsim≈Jsim(θ+ϵvθ)−Jsim(θ−ϵvθ)2ϵD_vJ_{\mathrm{sim}} \approx \frac{ J_{\mathrm{sim}}(\theta+\epsilon v_\theta) - J_{\mathrm{sim}}(\theta-\epsilon v_\theta) }{ 2\epsilon }DvJsim≈2ϵJsim(θ+ϵvθ)−Jsim(θ−ϵvθ)

再与世界模型的自动微分方向导数比较：

DvJ^WM=∇θJ^WM⋅vθD_v\hat J_{\mathrm{WM}} = \nabla_\theta\hat J_{\mathrm{WM}}\cdot v_\thetaDvJ^WM=∇θJ^WM⋅vθ

这可以检验模型预测的策略优化方向是否与真实仿真一致。

不过，要保证两边优化的是可比的目标函数、初始状态和时间窗口，并尽量固定随机性。接触切换附近的差分结果还需要单独处理。

建议将 G5 拆成 G5a（动力学 Jacobian）和 G5b（策略梯度）。 没有必要在早期就对完整高维 Actor 参数进行昂贵的全梯度有限差分。

## 三、需要修改的验收门禁

目前 G0–G6 的思路正确，但有些标准容易导致误判。

|
门禁

|

当前问题

|

建议

|
| --- | --- | --- |
|

G0：JSON 完全一致

|

不同 GPU、执行路径或计时字段可能不完全一致

|

固定环境后比较关键数值，允许明确的浮点容差；另设确定性测试

|
|

G1：E32 最近 3 次不上升

|

SGD 训练存在自然波动，连续三次单调下降不是必要条件

|

使用固定验证集、最佳 Checkpoint、平滑趋势和置信区间

|
|

G2：E1/E8 相对 MLP

|

可保留，但 2×/3× 是人为设定的工程门槛

|

明确是阶段性容忍标准，不是理论指标

|
|

G3：相对一致性 ≤0.05

|

可能被潜变量尺度、偏移和表示坍缩干扰

|

同时报告绝对误差、重构误差、潜变量方差及有效秩

|
|

G4：范数不单调增长

|

真实系统的合理潜动态也可能增长

|

重点判断是否出现异常放大和物理预测发散

|
|

G5：Cosine ≥0.5

|

梯度目标定义尚不一致

|

拆为开环 Jacobian 与闭环策略梯度测试

|
|

G6：Return ≥95% MLP

|

适合作为初步可用性门槛，但不是证明 Koopman 优势

|

同时要求报告失败动作数、样本效率和 GPU-hours

|

G1–G4 尤其需要小心：数学上的潜空间稳定，不等于物理状态预测准确；物理状态预测稳定，也不等于策略梯度正确。

例如，潜状态的一致性指标：

∥z−E(s)∥∥z∥+ϵ\frac{\|z-E(s)\|}{\|z\|+\epsilon}∥z∥+ϵ∥z−E(s)∥

可以因潜空间包含较大的近常数分量而变小，并不一定说明有效动力学坐标准确。

所以最好不要把单个相对误差阈值当作充分条件。

另外，对于 G1，建议同时报告每步误差曲线，而不是仅报告最后一步：

E1,E2,…,E32E_1,E_2,\ldots,E_{32}E1,E2,…,E32

因为不同模型可能在不同时间位置开始发散。

## 四、计划中还有三个容易被忽略的问题

### 1. 固定数据集方案需要明确数据来源与分布

你在阶段 0 中设计了：

|
数据划分

|

运动

|
| --- | --- |
|

Train

|

`walk1_subject1`、`dance1_subject1`

|
|

Validation

|

`walk2_subject1`

|
|

Test

|

`run2_subject1`

|

这是一个合理的跨运动泛化测试起点，但还不能同时代表训练分布内的预测能力。

KOOPMAN_FIX_PLAN.md

我建议增加两个不同的评估协议：

协议 A：训练分布内测试。 使用相同运动类别、不同轨迹或独立采样状态，考察是否能够拟合训练涉及的动力学。

协议 B：跨运动测试。 使用 `walk2`、`run2` 等保留运动，考察分布外泛化。

这两种结果必须分开报告。

否则可能出现某个 Koopman 模型在训练分布内已经稳定，却因为测试集运动分布不同而未通过 G1；反过来，也可能平均值良好，但关键接触模式仍然完全失败。

还要明确 `collect_wm_data.py` 如何为 Validation/Test 收集对应运动的数据。当前公开的 `train_wm_only.py` 默认只构建 `walk1_subject1` 和 `dance1_subject1` 的环境，因此不能简单地从这两个运动的数据里划分出另外两个运动的验证轨迹。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

另外，固定数据要保存完整的轨迹、动作执行值、环境参数、终止信息，以及能够重新构造参考动作帧的信息，确保不会跨 Episode 拼接错误窗口。

### 2. K1 有一个非常有用的零成本正确性检查

你计划在同一份 Koopman 权重下比较：

latentvsreencode\text{latent} \quad\text{vs}\quad \text{reencode}latentvsreencode

我建议增加一个明确断言：

s^1latent=s^1reencode\boxed{ \hat s_1^{\mathrm{latent}} = \hat s_1^{\mathrm{reencode}} }s^1latent=s^1reencode

因为两个模式都从同一个真实初始状态开始：

z0=E(s0,ξ)z_0=E(s_0,\xi)z0=E(s0,ξ)

并使用相同的：

z1=K(z0,a0)z_1=K(z_0,a_0)z1=K(z0,a0)

然后经过同一个 Decoder 和积分器。

它们第一次应该在进入第二步时，才可能因为是否重新编码而产生差异。

所以：

E1latent≈E1reencode\boxed{ E_1^{\mathrm{latent}} \approx E_1^{\mathrm{reencode}} }E1latent≈E1reencode

如果第一步结果已经不同，就应先查实现错误、模型模式、数据预处理和状态初始化，而不是继续研究潜空间漂移。

这个检查应该加入 G0。

此外，K1 只能说明同一权重下不同推理模式的效果，不能单独证明重新训练后的最佳 Koopman 架构应当采用哪一种模式。

### 3. 不应过早引入自动化联合训练门禁

计划提出在 `loop.py` 中加入：

WM NSE8>τ  ⟹  skip policy update\mathrm{WM\ NSE}_8>\tau \implies \text{skip policy update}WM NSE8>τ⟹skip policy update

这个想法有价值，但我不建议放在第一轮修复里。

KOOPMAN_FIX_PLAN.md

原因是在线 Replay Buffer 的分布本身会随着 Actor 变化：

Dt=D(πθt)\mathcal D_t=\mathcal D(\pi_{\theta_t})Dt=D(πθt)

如果 Actor 一直被冻结，Replay Buffer 的状态分布就可能缺乏新探索；反过来，Actor 一旦更新，数据分布又会变化。

因此，基于当前在线 NSE 的单一阈值，容易把两个因素混在一起：

* 世界模型的预测能力变化；

* 采样状态分布发生变化。

更稳妥的方案是：

第一阶段先使用离线固定验证集决定是否允许启动 Actor 训练；进入在线联合训练后，同时监控固定验证集误差和当前 Replay Buffer 误差。

这两者解决不同问题，不应该互相替代。

## 五、建议保留的部分

尽管上面指出了不少问题，我认为计划中一些设计确实值得完整保留。

第一，优先补齐 Checkpoint 和固定评估协议。 这是所有进一步研究的基础。没有同一份权重的对照，无法严谨区分模型结构和评估方式带来的变化。

第二，把真实状态重构和潜状态闭合分开。 两者对应不同约束，应该独立做消融，而不是都归入一个笼统的 `consistency_loss`。

第三，增加真实物理上下文 ξ\xiξ 的逐步条件化实验。 这是合理的可证伪假设，但不能预设它一定有效。

第四，引入容量匹配的 MLP。 你现在的 Koopman 与基线 MLP 参数量相差很大。增加小型 MLP 作为对照，对于解释研究结论非常重要。

KOOPMAN_FIX_PLAN.md

第五，明确放弃条件。 如果当前双线性 Koopman 经过一致性、稳定性和容量实验后依然明显失败，就转向物理状态锚定或者残差混合，不继续无限增加结构复杂度。这是良好的科研决策。

KOOPMAN_FIX_PLAN.md

## 六、我建议的修订版实施顺序

不需要推翻你的计划。我建议只调整成下面五个阶段。

1. P0：冻结基线与修复实验协议

   完成数据冻结、Checkpoint 保存、固定验证窗口、窗口参数修复、同权重模式切换和 Jacobian 单元测试。

   验收： 同一权重下 `latent` 与 `reencode` 的 E1 一致，数值结果可重复，MLP 旧行为不变。

2. P1：诊断根因，不修改 Koopman 结构

   测试 32 步逐步误差、潜空间范数、相对一致性、潜变量有效秩、实际矩阵乘积增益以及六组物理状态误差。

   验收： 能明确区分潜空间放大、物理状态积分漂移、单步欠拟合等不同情况。

3. P2：先做两项正交实验

   实验 A：真实状态重构 + 潜状态一致性。

   实验 B：温和的潜空间稳定性约束。

   分别训练，再评估组合是否必要，不立即加入 ξ\xiξ 条件化和大模型容量。

4. P3：容量与结构消融

   比较当前小型 Koopman、扩大后的 Koopman、参数量匹配的小型 MLP，以及原始大 MLP。

   同时记录 E1/E8/E32、实际吞吐和显存。

5. P4：严格验证梯度后再恢复 Actor

   先做单步方向导数，再做多步开环敏感度，最后比较策略参数扰动带来的真实回报变化。

   只有模型在真实仿真评估中表现合理，才进入联合训练。

我认为这个顺序比原计划更有优势，因为它减少了同时引入多个新机制带来的解释困难。

特别是，在 P2 之前，我建议暂时不要增加参数化 A(ξ)A(\xi)A(ξ)、复杂稳定性约束或 EMA Encoder。这些方法都可能有用，但它们也会让问题更难定位。

## 七、最终审查意见

|
审查项目

|

结论

|
| --- | --- |
|

是否抓住当前 Koopman 失败的主要方向

|

是

|
|

是否应该先做阶段 0

|

是，强烈支持

|
|

能否直接按当前计划全部编码

|

不建议

|
|

当前最重要的修订

|

reencode 结论、稳定性数学、梯度门禁、Checkpoint 兼容性

|
|

是否需要推翻 Koopman 架构

|

目前没有足够证据

|
|

是否应该立即恢复 Actor 训练

|

不应该

|

我还有一个比具体代码更重要的建议。

你目前应首先验证的是：32 步发散究竟来自 Koopman 传播矩阵，还是来自潜空间与物理状态之间不闭合的双状态系统。

这是两个不同的机制。

如果主要是：

∥M31⋯M0∥≫1\left\|M_{31}\cdots M_0\right\| \gg1∥M31⋯M0∥≫1

那么动力学矩阵的长期增益控制可能有效。

如果主要是：

∥zt−E(s^t)∥≫0\left\|z_t-E(\hat s_t)\right\| \gg0∥zt−E(s^t)∥≫0

那么仅约束 AAA 的谱不一定解决问题，甚至可能让预测更差。

如果两者都不明显，而 E1 仍远差于 MLP，那么需要优先研究的是模型容量、Decoder 表达能力和状态增量目标的适配性。

我的最终意见是：这份报告的研究思路可以通过，实施方案需要修订后再执行。尤其应该把“先证明失稳机制，再选择修复方法”作为贯穿整个项目的原则。

这比直接堆叠损失函数或增强 Koopman 网络，更有可能让你得到可靠、可发表且能解释失败原因的实验结果。

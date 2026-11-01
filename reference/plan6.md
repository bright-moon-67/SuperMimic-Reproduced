# 你的 SuperMimic-Reproduced：最新一轮代码复查

我重新检查了 github.com  当前公开的 `main` 分支，重点对照了上次审查后新增的 `plan5` 修复、State-Augmented Koopman、世界模型损失、诊断脚本和训练入口。

结论：这次确实修复了多处以前存在的问题。但我认为现在最重要的不是继续增加 Koopman 结构，而是修正闭环梯度诊断，并完成有说服力的预测精度对照实验。

## 一、上次的问题，哪些已经解决？

|
原先的问题

|

当前状态

|

我的评价

|
| --- | --- | --- |
|

低秩因子同时为零，梯度无法更新

|

已修复

|

正确

|
|

`cons_rel` 的 RMS/L2 混用

|

已修复

|

正确

|
|

State-Augmented 重构损失恒为零

|

已修复

|

需要进一步验证监督目标

|
|

一致性目标无法停止梯度

|

已增加 `cons_target_detach`

|

正确

|
|

Lift Loss 物理块与潜变量块混在一起

|

已拆分权重

|

正确

|
|

Lift Loss 重复乘权重

|

已修复

|

重要修复

|
|

Checkpoint 迁移继承旧学习率进度

|

已新增 `transfer_policy`

|

明显改善

|
|

验证集与测试集混合

|

已新增独立划分

|

改善

|
|

只诊断潜空间矩阵，不诊断闭环

|

已新增有限差分诊断

|

方法仍需修改

|

这些修改可以在当前 `koopman.py`、`world_model.py`、`training/loop.py` 中找到。仓库的修复记录也报告了相关单元测试通过，但我没有在本地独立运行这些测试。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+3

特别是 State-Augmented 模型现在使用：

zt=\(xt;ϕt\)z_t=\(x\_t;\\phi\_t\)zt=\(xt​;ϕt​\)

并通过：

s^t+1=Integrate⁡(st,D(ut+1))\hat s_{t+1}=\operatorname{Integrate}(s_t,D(u_{t+1}))s^t+1=Integrate(st,D(ut+1))

再将物理状态写回潜状态。

这条路径已经比较明确，Actor 与世界模型之间的基本接口没有发现需要整体重写的理由。

## 二、目前最需要关注的三个问题

### 1. P0：新增加的闭环梯度诊断仍然不够严谨

文件：github.com 

这是我这次最想指出的问题。

当前 `_closed_loop_fd()` 首先初始化：

z0=E(s0,ξ)z_0=E(s_0,\xi)z0=E(s0,ξ)

但在计算扰动时，分别给物理状态和潜状态添加随机扰动：

s0′=s0+ϵvss'_0=s_0+\epsilon v_ss0′=s0+ϵvs

z0′=z0+ϵvzz'_0=z_0+\epsilon v_zz0′=z0+ϵvz

随后通过有限差分计算输出变化。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这样得到的确实是扩展状态空间某个随机方向上的敏感度，但存在两个问题。

首先，扰动后的 s0′,z0′s'_0,z'_0s0′,z0′ 不一定满足潜状态与物理状态之间的一致性关系。

例如：

z0′≠E(s0′,ξ)z'_0\neq E(s'_0,\xi)z0′=E(s0′,ξ)

对于 State-Augmented Koopman：

zt=\(Canonical⁡(st);ϕt\)z_t=\(\\operatorname{Canonical}(s\_t);\\phi\_t\)zt=\(Canonical(st​);ϕt​\)

随意独立扰动两者，会破坏物理状态块和潜状态块之间的对应关系。

其次，物理状态中包含四元数。

直接对四元数四个分量添加随机扰动，可能破坏：

∥q∥=1\|q\|=1∥q∥=1

也就是把初始状态推到不合法的姿态表示上。

因此，当前这个数值不能直接解释成机器人真实闭环的动力学敏感度。

我建议将诊断拆分为：

* 物理一致扰动： 在合法状态空间内扰动位置、速度和旋转，再重新生成初始潜状态。

* 潜变量扰动： 固定物理状态，只研究学习潜变量 ϕt\phi_tϕt 的敏感度。

如果你要研究真正的状态 Jacobian，建议使用：

Jphysicalv≈F(s⊕ϵv,a,ξ)−F(s⊕(−ϵv),a,ξ)2ϵ\boxed{ J_{\mathrm{physical}}v \approx \frac{ F(s\oplus\epsilon v,a,\xi) - F(s\oplus(-\epsilon v),a,\xi) }{2\epsilon} }Jphysicalv≈2ϵF(s⊕ϵv,a,ξ)−F(s⊕(−ϵv),a,ξ)

其中 ⊕\oplus⊕ 表示尊重姿态流形约束的状态扰动，而不是简单地对所有状态分量相加。

另外，当前只用一个随机方向做前向有限差分，不能得到最大奇异值，也不能证明整个闭环稳定。

应把它称为 Directional Sensitivity，不要直接当作闭环稳定性的充分指标。

这是我建议优先修复的地方。

### 2. P0：重构损失已经有效，但还不是真实状态重构监督

文件：github.com 

你已经修复 State-Augmented 模型中原先退化的重构 Head：

Rη(ϕt)R_\eta(\phi_t)Rη(ϕt)

不再直接返回显式物理状态块。

这是正确的。

但是，目前 `_window_loss_koopman()` 中仍采用：

xtpred=Canonical⁡(s^t)x_t^{\mathrm{pred}} = \operatorname{Canonical}(\hat s_t)xtpred=Canonical(s^t)

然后：

Lrec=∥Rη(ϕt)−xtpred∥2\boxed{ \mathcal L_{\mathrm{rec}} = \|R_\eta(\phi_t)-x_t^{\mathrm{pred}}\|^2 }Lrec=∥Rη(ϕt)−xtpred∥2

而不是直接使用真实状态：

xttrue=Canonical⁡(sttrue)x_t^{\mathrm{true}} = \operatorname{Canonical}(s_t^{\mathrm{true}})xttrue=Canonical(sttrue)

代码中的核心是：

Python

运行

```
x_hat = self.dynamics.reconstruct(z)
x_cur = self.spec.canonical(s)
total_rec = total_rec + (x_hat - x_cur).pow(2).mean(-1).mean()
```

其中 `s` 是正在自回归传播的预测状态。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这意味着：

重构损失可以保证潜变量与模型自己预测的物理状态一致，但不能单独保证二者接近真实物理状态。

更糟糕的情况是，潜变量和预测物理状态可能一起偏离真实轨迹，却仍然满足较低的内部重构误差。

当然，你已经有：

Lstate\mathcal L_{\mathrm{state}}Lstate

用于监督真实状态，所以这不是缺少监督的致命 bug。

但如果希望单独评价学习潜变量的信息保留能力，我建议增加：

Lrec,true=∥Rη(Eϕ(xttrue,ξ))−xttrue∥2\boxed{ \mathcal L_{\mathrm{rec,true}} = \left\| R_\eta(E_\phi(x_t^{\mathrm{true}},\xi)) - x_t^{\mathrm{true}} \right\|^2 }Lrec,true=Rη(Eϕ(xttrue,ξ))−xttrue2

然后把真实状态重构和预测状态一致性拆开记录。

不要直接删除现有损失，两者应先做消融。

### 3. P0：还没有看到新版本的完整预测效果证明

这个问题可能比前两项更加实际。

仓库最近的修复记录提到：

* State-Augmented Koopman 的重构损失已经产生非零数值；

* 相关测试通过；

* 新的 2000-update Quick Run 已启动。

但我在当前公开的 `RESULTS.md` 中，还没有看到这轮修复完成后的系统性 E1/E8/E32 结果。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

换句话说：

代码修复已经发生，但这些修复是否真正改善预测精度，目前公开结果还不足以判断。

此前已经完成的稳定性实验表明：

|
模型

|

32-step NSE

|
| --- | --- |
|

原始 MLP

|

0.345

|
|

原始 Koopman latent

|

约 404,000

|
|

Koopman + 强稳定性正则

|

324.1

|
|

Koopman reencode

|

13.97

|

这些来自旧固定数据协议，不能直接视为最新 State-Augmented 模型的性能。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

现在需要一组真正的新对照：

|
模型

|

必须记录

|
| --- | --- |
|

MLP Baseline

|

E1、E8、E32、真实仿真 Return

|
|

Bilinear Koopman

|

E1、E8、E32、梯度敏感度

|
|

State-Augmented Koopman

|

E1、E8、E32、潜状态一致性

|
|

State-Augmented + 新辅助损失

|

同样指标，比较新增损失的实际效果

|

这里最重要的是同一份固定数据、相同评估窗口，以及记录准确的模型配置。

## 三、还有两个工程问题值得留意

第一，你已经新增 `run_paper_aligned.sh`，这是一个正确方向。

它采用更接近 CuboiLeo 原始 LAFAN1 配置的世界模型训练预算，能够减少之前小数据、短训练带来的混杂因素。

但它仍然是固定 Replay Buffer 的离线训练实验，不能等同于原仓库在线 Actor–World Model 共同适应的训练过程。脚本本身也明确说明了这一点。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

因此，离线实验回答的是模型能否拟合指定数据分布，不能独立证明最终 Actor 训练会更好。

第二，J1/J2/J3 梯度验证还没有完整落地。

当前 `tests/test_koopman.py` 已有自动微分、低秩参数梯度和部分数值测试，但修复计划提出的完整 J2b/J3——与真实 MuJoCo 动力学和 Actor 策略梯度的对照——尚未看到对应的独立评估脚本实现。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

这并不是目前必须先完成才能继续做 World Model 实验的阻塞项，但在恢复大规模 Actor BPTT 前，应该补齐。

## 四、现在我建议怎么推进？

我的优先顺序如下。

P0

先完成新 State-Augmented 模型的 E1/E8/E32 正式实验

暂停继续扩展架构，固定数据、Checkpoint 和评估窗口，确认最新修复到底带来了什么提升。

P0

修正闭环有限差分诊断

保证状态扰动遵守物理表示和潜状态一致性约束，将物理敏感度与纯潜变量敏感度分开。

P1

增加真实状态重构消融

分别测试预测状态重构、真实状态重构和两者组合，检查对潜变量有效秩和长时域误差的影响。

P1

继续进行容量匹配实验

将小型 MLP、原始大型 MLP 和不同容量的 Koopman 进行对照，避免将容量不足直接误判为结构缺陷。

P2

世界模型达标后再做 J2b/J3 和 Actor 联合训练

不要只依据想象轨迹的回报提升，来证明 Koopman 提供了正确的控制梯度。

## 五、我的总体判断

你的仓库目前已经从早期的“尝试接入 Koopman”，进入了一个更成熟的阶段：有多个动力学候选架构、明确的消融配置、固定数据协议，以及对谱性质和潜状态闭合性的诊断。

但我仍然认为，最核心的科学问题尚未解决：

结构更稳定的 Koopman  ⟹?  更准确的长期预测与 BPTT 梯度\boxed{ \text{结构更稳定的 Koopman} \;\stackrel{?}{\Longrightarrow}\; \text{更准确的长期预测与 BPTT 梯度} }结构更稳定的 Koopman⟹?更准确的长期预测与 BPTT 梯度

前面的实验已经证明，降低 Koopman 内部矩阵的放大倍数，并不会自动把状态预测误差降到 MLP 水平。

现在应该把注意力放到：

State Prediction FidelityPhysical Closed-loop SensitivityActor Gradient Fidelity\boxed{ \begin{aligned} &\text{State Prediction Fidelity}\\ &\text{Physical Closed-loop Sensitivity}\\ &\text{Actor Gradient Fidelity} \end{aligned} }State Prediction FidelityPhysical Closed-loop SensitivityActor Gradient Fidelity

这三个指标上。

我的建议是：先不要继续增加损失项和模型结构；把最新 State-Augmented Koopman 的完整预测结果跑出来，再决定下一步是改动力学结构、改状态表示，还是改训练目标。

目前这轮复查基于公开 `main` 源码和仓库文档；我没有访问到 AutoDL 本地权重，也没有独立运行整套训练，因此对于尚未公开的实验效果，不作已经改善的判断。

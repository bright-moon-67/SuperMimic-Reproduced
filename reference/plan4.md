# bright-moon-67/SuperMimic-Reproduced 最新复查

复查日期：2026 年 10 月 31 日

我重新查看了 github.com ，重点检查了最新的 Koopman 实现、实验脚本、测试和 `docs/KOOPMAN_FIX_PLAN.md` v2。

## 一、先说结论：有明显进展，但还没解决核心的预测精度问题

这次仓库相比我们上次讨论时，已经有了实质性改动。

你已经完成或部分完成了固定 Replay Buffer 数据、保存 World Model Checkpoint、同权重 `latent/reencode` 诊断、动力学矩阵的谱分析、稳定性正则化，并且新增了两条结构探索路线：

* `use_xi_state`：让物理参数 ξ\xiξ 显式参与每一步潜状态传播。

* `state_in_latent`：把物理状态特征直接放入潜状态，尝试解决潜状态与物理状态不一致的问题。

这些不是单纯的文档更新，当前代码中能找到对应实现。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+3

不过，最新实验显示：

你已经显著抑制了最极端的潜状态数值爆炸，但没有让 Koopman 世界模型具备接近 MLP 的预测能力。

### 仓库最新记录的实验结果

下表来自 `docs/KOOPMAN_FIX_PLAN.md` v2 中的 2026 年 10 月 31 日实验记录，而不是我独立重新训练的结果。

|
模型

|

32 步 NSE

|

评价

|
| --- | --- | --- |
|

MLP Baseline

|

0.345

|

当前对照基准

|
|

原始 Koopman latent

|

约 404,000

|

严重发散

|
|

原始 Koopman reencode

|

13.97

|

明显改善，但误差仍高

|
|

Koopman 稳定性正则 λ=1\lambda=1λ=1

|

2,920

|

潜状态发散减弱

|
|

Koopman 稳定性正则 λ=10\lambda=10λ=10

|

324.1

|

进一步抑制发散

|

图表选项

32-step prediction error after stability regularization

Lower is better. Same documented diagnostic family; logarithmic differences described in text.



同时，λ=10\lambda=10λ=10 时：

ρ(A)≈0.981,∥M31⋯M0∥2≈2.45\rho(A)\approx0.981,\qquad \|M_{31}\cdots M_0\|_2\approx2.45ρ(A)≈0.981,∥M31⋯M0∥2≈2.45

相较于原始模型记录的约 4.5×1034.5\times10^34.5×103 矩阵乘积范数，潜空间放大确实得到了明显控制。

但是：

E32Koopman=324.1≫E32MLP=0.345\boxed{ E_{32}^{\mathrm{Koopman}}=324.1 \gg E_{32}^{\mathrm{MLP}}=0.345 }E32Koopman=324.1≫E32MLP=0.345

即使在 `reencode` 模式下，稳定性正则版本记录的 E32=24.19E_{32}=24.19E32=24.19，也仍然约是 MLP 的 70 倍。

与此同时，单步误差约为 0.2590.2590.259，而 MLP 约为 0.0480.0480.048。

这些结果提示：问题已经不能仅仅归结为 AAA 的谱半径过大。 模型的基础状态转移拟合能力、动作响应、解码与物理积分之间的配合，仍存在明显差距。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

## 二、发现一个新的实质性代码问题：State-Augmented Koopman 的双线性分支可能永远无法学习

P0 · 可以从代码直接确认

对应文件：koopman.py 

你新增了 `StateAugmentedKoopman`，模型大致是：

zt+1=Azt+Bat+Cξ+b+∑iaiUiVi⊤zt+∑jξjPjQj⊤zt+∑iaiWr,iWs,i⊤ξz_{t+1} = Az_t+Ba_t+C\xi+b + \sum_i a_iU_iV_i^\top z_t + \sum_j\xi_jP_jQ_j^\top z_t + \sum_i a_iW_{r,i}W_{s,i}^\top\xizt+1=Azt+Bat+Cξ+b+i∑aiUiVi⊤zt+j∑ξjPjQj⊤zt+i∑aiWr,iWs,i⊤ξ

这个方向值得尝试。但当前初始化存在一个明确的问题。

在 `StateAugmentedKoopman.__init__()` 中：

U=V=0,P=Q=0,Wr=Ws=0U=V=0,\qquad P=Q=0,\qquad W_r=W_s=0U=V=0,P=Q=0,Wr=Ws=0

即每一对低秩矩阵的两个因子同时初始化为零。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

以 UV⊤zUV^\top zUV⊤z 为例，其梯度满足：

∂L∂U∝V,∂L∂V∝U\frac{\partial\mathcal L}{\partial U} \propto V,\qquad \frac{\partial\mathcal L}{\partial V} \propto U∂U∂L∝V,∂V∂L∝U

当：

U=V=0U=V=0U=V=0

就得到：

∇UL=0,∇VL=0\boxed{ \nabla_U\mathcal L=0,\qquad \nabla_V\mathcal L=0 }∇UL=0,∇VL=0

这不是一般意义上的“训练初期梯度偏小”，而是：如果没有其他直接作用于这些参数的机制，它们会一直停留在零值。

我使用 PyTorch 构建了与仓库相同形式的低秩双线性项，并执行反向传播，确认两个因子同时为零时，两者的梯度都精确为零。

因此，若你启用了 `state_in_latent=True`，当前模型虽然定义了动作—状态双线性项、物理参数—状态耦合项和动作—物理参数交互项，但这些低秩交互项在现有初始化下实际上无法被普通损失训练起来。

这可能使你以为正在训练一个复杂的 Koopman 模型，实际上主要训练的是其中的加性线性部分。

### 建议怎么改？

保留“初始预测增量为零”的先验，但不要同时将每对低秩因子置零。

例如：

Ui∼N(0,σ2),Vi=0U_i\sim\mathcal N(0,\sigma^2), \qquad V_i=0Ui∼N(0,σ2),Vi=0

这样初始低秩输出仍然为零，但 ViV_iVi 可以立即获得梯度。随后 ViV_iVi 离开零点，UiU_iUi 也能够参与学习。

对 P,QP,QP,Q 和 Wr,WsW_r,W_sWr,Ws 采用同样原则。

修复后应增加一个单元测试：执行一次包含有效监督的反向传播，验证至少每对低秩因子中的一个具有非零梯度，并在后续更新中验证两个因子都能变化。

适用范围说明： 这个问题专门针对新增加的 `StateAugmentedKoopman` 分支。默认 `KoopmanDynamics` 中的 U,VU,VU,V 是小随机初始化，不存在同样的双零问题。你目前公布的主要 P1/P2-B 实验也不能直接归因于这个新问题。

## 三、第二个明确问题：潜空间相对一致性指标计算尺度有误

P0 · 影响诊断结论

对应文件：world_model.py 

你计划中定义的潜空间相对误差是：

r=∥zt−E(xt)∥2∥zt∥2+ϵ\boxed{ r= \frac{\|z_t-E(x_t)\|_2} {\|z_t\|_2+\epsilon} }r=∥zt∥2+ϵ∥zt−E(xt)∥2

但当前 `rollout_diagnostics()` 实际计算的是：

rcode=mean⁡j(zj−E(x)j)2∥z∥2+ϵ\boxed{ r_{\mathrm{code}} = \frac{ \sqrt{\operatorname{mean}_j(z_j-E(x)_j)^2} }{ \|z\|_2+\epsilon } }rcode=∥z∥2+ϵmeanj(zj−E(x)j)2

分子是 RMS，分母却是 L2 范数。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

对 dzd_zdz 维潜变量，在分母非零且忽略 ϵ\epsilonϵ 时：

rcode=rdz\boxed{ r_{\mathrm{code}} = \frac{r}{\sqrt{d_z}} }rcode=dzr

这会使你的相对一致性指标系统性偏小。

例如在 128 维潜空间：

r=0.5r=0.5r=0.5

代码却会报告约：

rcode=0.0442r_{\mathrm{code}}=0.0442rcode=0.0442

这两个数字差别很大。

我已用 Python 对 128 维和 199 维向量核对这一比例。

这意味着计划中的 `cons_rel ≤ 0.05` 门禁，不能直接用于当前代码输出的 `cons_rel`。

我建议要么使用分子和分母都为 L2 范数，要么都采用 RMS；重点是保持定义一致，并在修改后重新计算历史指标。

另外，当前的 `cons_abs` 实际存储的是均方误差，不是绝对误差或 L2 范数，名称也建议改为 `cons_mse`。

这是诊断指标的错误，不代表模型的梯度计算因此错误，但它会直接影响你对潜空间闭合程度的判断。

## 四、第三个问题：新增的物理状态锚定结构，并不完全满足线性 Koopman 演化

P1 · 架构与理论表述问题

这是我对新模型最重要的结构性保留意见。

你目前设计：

zt=\(xtϕt\)z_t= \begin{bmatrix} x_t\\ \phi_t \end{bmatrix}zt=\(xt​ϕt​​\)

先使用双线性动力学产生：

ut+1=K(zt,at,ξ)u_{t+1}=K(z_t,a_t,\xi)ut+1=K(zt,at,ξ)

然后使用 ut+1u_{t+1}ut+1 的前一部分预测物理状态增量：

Δt=D(ut+1phys)\Delta_t=D(u_{t+1}^{\mathrm{phys}})Δt=D(ut+1phys)

通过显式积分：

st+1=Integrate⁡(st,Δt)s_{t+1}=\operatorname{Integrate}(s_t,\Delta_t)st+1=Integrate(st,Δt)

最后重新构建潜状态：

zt+1=\(Canonical⁡(st+1)ut+1latent\)\boxed{ z_{t+1} = \begin{bmatrix} \operatorname{Canonical}(s_{t+1})\\ u_{t+1}^{\mathrm{latent}} \end{bmatrix} }zt+1=\(Canonical(st+1​)ut+1latent​​\)

这些步骤在 `StateAugmentedKoopman.step_latent()` 中可以确认。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

好处非常明显：物理状态部分每一步都与预测物理状态同步，不会发生原先那种独立传播带来的物理状态—潜状态脱节。

但是，这里的实际状态更新已经包含：

Canonical⁡∘Integrate⁡\operatorname{Canonical}\circ\operatorname{Integrate}Canonical∘Integrate

这两个一般具有非线性的操作。

因此，严格来说：

zt+1≠K(zt,at,ξ)\boxed{ z_{t+1}\neq K(z_t,a_t,\xi) }zt+1=K(zt,at,ξ)

完整潜状态更新不是单纯由一个线性或双线性算子完成。

我的判断是：这可以作为一种物理状态锚定的结构化世界模型，但不应直接声称整个提升状态满足双线性 Koopman 演化。

这不是说模型不能有效，而是理论表述需要精确。

还有一个伴随问题：`reconstruct()` 直接返回潜状态中的物理特征部分：

R(zt)=ztphysR(z_t)=z_t^{\mathrm{phys}}R(zt)=ztphys

由于这部分本来就是从当前物理状态构建的，原有重构损失在正常连续传播中可能天然为零。它不能再像原始 Deep Koopman 那样，对学习得到的特征映射 ϕ\phiϕ 提供有效的重构监督。

因此，新结构尤其需要检查：

ϕt+1与E(xt+1,ξ)\phi_{t+1} \quad\text{与}\quad E(x_{t+1},\xi)ϕt+1与E(xt+1,ξ)

之间的一致性，而不能依赖物理状态块的“天然闭合”来判断整个 Koopman 表示已经学好。

## 五、还有四处需要重点处理的工程问题

|
问题

|

当前代码状态

|

我的判断

|
| --- | --- | --- |
|

一致性损失仍未进入优化目标

|

`consistency` 仍主要作为监控指标

|

尚未修复

|
|

在线 Actor 训练缺乏模型质量门禁

|

`run_lafan1_online.sh` 启动完整联合训练

|

不宜直接长时间运行

|
|

`resume_skip=wm` 的学习率进度问题

|

仍恢复旧 `iteration/num_updates`

|

旧问题仍在

|
|

验证集与测试集合并评估

|

Protocol A/B 都包含 `val_clips`

|

实验协议需要完善

|

### 1. P2-A 还没有真正落实

目前 `_window_loss_koopman()` 的实际损失为：

L=Lstate+Lterm+λliftLlift+λrecLrec+λstabLstab\boxed{ \mathcal L= \mathcal L_{\mathrm{state}} +\mathcal L_{\mathrm{term}} +\lambda_{\mathrm{lift}}\mathcal L_{\mathrm{lift}} +\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}} +\lambda_{\mathrm{stab}}\mathcal L_{\mathrm{stab}} }L=Lstate+Lterm+λliftLlift+λrecLrec+λstabLstab

`consistency` 被记录，但没有参与这个表达式。计划中的 `L_rec_true`、`L_cons` 也还没有完整落地。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

所以，现阶段只能认为 P2-B 稳定性修复已经实施，P2-A 表示闭合修复仍在计划阶段。

尤其对于默认的连续潜状态模型，我仍然认为 P2-A 值得做，而且优先级高于盲目扩大模型。

### 2. 在线训练仍然会过早更新 Actor

你新增了 `run_lafan1_online.sh`，这是好事，因为它明确区分了冻结数据的离线世界模型训练与真正在线的 SuperMimic 联合训练。

但是，当前 `Trainer.run()` 的逻辑仍是：

Collect→WM Update→Policy Update\text{Collect} \rightarrow \text{WM Update} \rightarrow \text{Policy Update}Collect→WM Update→Policy Update

经过配置的 warm-up 后，Actor 就会开始更新，没有检查 Koopman 是否通过 32 步预测门禁。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

因此，即使你的文档强调「G5b 通过之前不恢复 Actor」，实际脚本仍然允许用户直接进入联合训练。

这两者目前没有通过代码进行强制关联。

我建议在调用在线脚本前，先用离线实验明确选出合格 Checkpoint，避免用一个 E32 仍然很差的 Koopman 模型更新已经表现良好的 Actor。

### 3. Checkpoint 恢复调度问题仍然存在

`Trainer.load()` 在跳过旧 World Model 时，仍恢复：

iteration=ck\("iteration"\)\texttt{iteration}=\texttt{ck\("iteration"\)}iteration=ck\("iteration"\)

以及：

num_updates=ck\("num\_updates"\)\texttt{num\_updates}=\texttt{ck\("num\\\_updates"\)}num_updates=ck\("num\_updates"\)

后续 Cosine Learning Rate 又直接依赖 `iteration`。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

因此如果使用：

`RESUME=<旧 MLP checkpoint>`

并跳过旧 WM 权重，新 Koopman 可能在旧训练后期的学习率状态下开始学习。

建议提供明确的迁移模式，区分：

* 完整断点续训；

* 只迁移 Actor/Critic 权重，重新开始 Koopman 训练与学习率调度。

这两种操作的语义不一样，不应依靠同一套默认逻辑。

### 4. Protocol A/B 仍需把验证集与最终测试集分开

`collect_wm_data.py` 现在能够保存 Replay Buffer 和固定评估窗口，这比之前有很大改善。

但 Protocol A 构造评估窗口时，合并了：

Validation Clips+Test A Clips\text{Validation Clips}+\text{Test A Clips}Validation Clips+Test A Clips

Protocol B 也类似。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这样没有必然导致训练集泄漏，但如果你依据这些合并结果调节超参数，最终测试集就已经参与了模型选择。

建议将评估分别输出为 `val_a`、`test_a`、`test_b`，最终的测试集只在模型选择完成后使用。

另外，`scripts/diag_koopman.py` 暴露了 `n_sigma` 参数，但调用 `rollout_diagnostics()` 时没有传递它，因此目前修改命令行参数不会改变实际用于谱诊断的窗口数量。这是一个容易修复的小问题。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

## 六、我现在建议的下一步

结合最新实验，我不建议马上继续大规模容量扫描。

虽然原始 Koopman 容量明显小于 MLP，但现在存在几个还没有排除的结构性问题。直接增加潜空间维度和 Encoder 宽度，可能只会增加计算成本。

建议按下面顺序处理：

|
优先级

|

具体行动

|

判断标准

|
| --- | --- | --- |
|

P0

|

修复 State-Augmented 双零低秩初始化

|

双线性参数能获得有效梯度

|
|

P0

|

纠正 `cons_rel` 的 RMS/L2 混用

|

诊断指标与数学定义一致

|
|

P0

|

完成 P2-A 一致性损失

|

`latent` 32 步预测得到可重复的改善

|
|

P1

|

单独评估 State-Augmented 新分支

|

与原始 latent、reencode 使用同一数据预算

|
|

P1

|

增加小型 MLP 容量匹配基线

|

区分容量不足与 Koopman 结构限制

|
|

P1

|

修复在线训练门禁与迁移调度

|

不会误用不合格 WM 更新 Actor

|
|

P2

|

完成 J1/J2/J3 梯度测试

|

在真实仿真器上验证控制梯度方向

|

特别是 State-Augmented 分支，我认为值得给它一次修复初始化后的独立实验，因为它直接针对原始方案最明显的潜状态—物理状态分离问题。

但这不应与原始 Deep Koopman 混为一个模型：需要分开报告，并保留各自的结构定义。

## 七、我的最终判断

现在这个仓库比上次更成熟，尤其在实验可复现性和失稳原因诊断方面，进步是实质性的。

不过我也看到了一个新的信号：

你的研究重心可能需要从“如何把 Koopman 矩阵变稳定”，转向“当前模型是否学到了真实的关节动作响应与物理状态变化”。

因为最新结果已经说明：

∥M31⋯M0∥2\|M_{31}\cdots M_0\|_2∥M31⋯M0∥2

可以被明显压低，但 E32E_{32}E32 依然远高于 MLP，单步误差也没有明显改善。

所以，真正值得问的是：

Koopman 预测的状态增量是否准确？动作对关节动力学的影响是否准确？解码输出与显式积分是否匹配？这些误差怎样沿 32 步累积？\boxed{ \begin{aligned} &\text{Koopman 预测的状态增量是否准确？}\\ &\text{动作对关节动力学的影响是否准确？}\\ &\text{解码输出与显式积分是否匹配？}\\ &\text{这些误差怎样沿 32 步累积？} \end{aligned} }Koopman 预测的状态增量是否准确？动作对关节动力学的影响是否准确？解码输出与显式积分是否匹配？这些误差怎样沿 32 步累积？

当前日志里，`joint_pos` 和 `root_rot` 的长时域误差尤其突出。我建议进一步将这两个通道分离出来，做真实状态输入的单步预测误差、自由递推误差，以及动作 Jacobian 误差的逐步对照。

如果它们在单步情况下就明显差于 MLP，继续加强谱约束恐怕不会解决根本问题。

最终建议：保留当前的 Koopman 路线，但先修复发现的两个明确问题（低秩双零初始化、相对一致性指标定义），落实 P2-A，再对物理状态锚定分支进行严格的独立验证。暂时不要恢复大规模 Actor BPTT 联合训练。

本次结论基于 10 月 31 日可访问的公开源码与仓库报告；低秩零梯度问题已用独立 PyTorch 小实验验证，其余训练性能结论仍需你保存的 Checkpoint 和 AutoDL 实际运行结果确认。


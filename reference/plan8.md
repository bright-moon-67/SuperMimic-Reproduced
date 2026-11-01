# 新一轮实验分析：State-Augmented Koopman 已经有竞争力，但在线训练仍是主要瓶颈

我对这轮结果的判断比之前更积极一些，但仍然保持谨慎：

你现在已经有证据表明，State-Augmented Koopman 可以在更小的模型容量下，获得优于同容量 MLP 的多步预测性能。 这是实质性进展。

但它还不能证明 Koopman 更适合 SuperMimic 的 BPTT，也不能解释之前在线训练失败的全部原因。

我认为这轮数据支持三个判断：

1. State-Augmented 结构有效，值得保留。

2. 对该结构施加较强的内部矩阵稳定性惩罚，可能适得其反。

3. 下一阶段最值得优化的是在线学习过程，而不是继续扩容或增加辅助损失。

## 一、重新比较模型：你的 Koopman 实际上已经优于同容量 MLP

先以相近参数量进行比较，避免把小 Koopman 直接与大 MLP 混为一谈。

小模型对照：State Koopman λ_stab=0 vs MLP 512³

E1 下降

# 12.4%

E8 下降

# 45.7%

E32 下降

# 64.2%

图表选项

Matched-size models: offline prediction error

MLP 0.63M parameters; state Koopman 0.53M. Lower is better.



这里有一个值得纠正的认识：在小模型、相近容量的比较中，去掉稳定性正则后的 State Koopman 连 E1 也优于 MLP。 并不是所有配置下 Koopman 的单步预测都更差。

再看大一些的模型：

|
指标

|

MLP 1280³（3.55M）

|

State Large + rec_true（3.46M）

|

结果

|
| --- | --- | --- | --- |
|

E1

|

0.0497

|

0.0549

|

Koopman 高 10.5%

|
|

E8

|

0.2184

|

0.1826

|

Koopman 低 16.4%

|
|

E32

|

0.881

|

0.680

|

Koopman 低 22.8%

|

这表明 State-Augmented Koopman 的优势可能更多体现在多步预测误差的控制，而不只是单步拟合。

但这个结论目前局限于你的固定数据、当前训练预算和单次实验结果。还需要重复种子及统计波动范围才能确认优势是否稳健。

与 8.82M 参数的最大 MLP 比较，State Koopman 的 E32 仍然较高；因此还不能宣称它在绝对预测精度上超过原始大 MLP。

## 二、稳定性消融：现在不建议默认启用 λ_stab=10

这是这轮实验最清楚的结果。

图表选项

Stability regularization vs prediction accuracy

State-augmented small; 2000 offline updates. Lower E32 is better.



你得到：

λstab↑⟹σmax⁡(M)↓\lambda_{\mathrm{stab}}\uparrow \quad\Longrightarrow\quad \sigma_{\max}(M)\downarrowλstab↑⟹σmax(M)↓

同时：

E1↑,E8↑,E32↑E_1\uparrow,\qquad E_8\uparrow,\qquad E_{32}\uparrowE1↑,E8↑,E32↑

在这组实验中，强正则并没有改善预测，反而让三个时域指标都变差。

### 为什么可能发生？

State-Augmented Koopman 的实际传播不是单纯：

zt+1=Mtzt+Batz_{t+1}=M_tz_t+B a_tzt+1=Mtzt+Bat

它还包括物理状态积分及状态回写：

ut+1=K(zt,at,ξ)Δt=D(ut+1)st+1=Integrate⁡(st,Δt)zt+1=\(Canonical⁡(st+1);ϕt+1\)\boxed{ \begin{aligned} u_{t+1}&=K(z_t,a_t,\xi)\\ \Delta_t&=D(u_{t+1})\\ s_{t+1}&=\operatorname{Integrate}(s_t,\Delta_t)\\ z_{t+1}&=\(\\operatorname{Canonical}(s\_{t+1});\\phi\_{t+1}\) \end{aligned} }ut+1Δtst+1zt+1=K(zt,at,ξ)=D(ut+1)=Integrate(st,Δt)=\(Canonical(st+1​);ϕt+1​\)

因此，约束内部矩阵的最大奇异值，不等于约束完整物理状态传播的敏感度。

而且你可能正在抑制完成某些运动所需要的动力学增益。

但需要澄清一个术语：你表中的 `max σ(M)` 是最大奇异值，而不是谱半径。 两者不能互换。

同时，若实验使用了仓库的 `STAB` 开关，开启稳定性正则还可能同时设置 `a_rho_target=0.99`，这会改变矩阵 AAA 的初始化。严格归因时需要确保不同 λstab\lambda_{\mathrm{stab}}λstab 组具有相同的 AAA 初始化条件。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

就当前结果而言，我支持你正在进行的：

λstab=0\boxed{\lambda_{\mathrm{stab}}=0}λstab=0

但它应被视为当前最优实验候选，而不是已经证明所有稳定性约束都不适用。

## 三、离线表现好、在线表现差，最可能的原因是什么？

我现在最怀疑的不是 BPTT 的梯度公式写错了，而是离线数据与在线训练早期数据的分布差异，加上 World Model 尚未拟合好就开始更新 Actor。

### 1. 你的离线数据来自一个已有策略

公开仓库的 `run_paper_aligned.sh` 使用已有 Actor Checkpoint 采集固定数据，然后单独训练 World Model。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

这意味着，离线数据很可能主要覆盖已有策略比较熟悉的运动状态。

但在线从头训练时，初始 Actor 的动作分布可能完全不同：

Dpretrained≠Dearly policy\mathcal D_{\mathrm{pretrained}} \neq \mathcal D_{\mathrm{early\ policy}}Dpretrained=Dearly policy

而且在线训练期间：

Dπθ0→Dπθ1→⋯\mathcal D_{\pi_{\theta_0}} \rightarrow \mathcal D_{\pi_{\theta_1}} \rightarrow \cdotsDπθ0→Dπθ1→⋯

不断变化。

因此，离线实验得到：

E8K=0.1766E_8^{K}=0.1766E8K=0.1766

并不意味着早期在线训练时也应该获得相同的 E8。

你此前在线记录的：

E8K=0.738E_8^{K}=0.738E8K=0.738

也不一定意味着同一个 Koopman 网络在训练过程中退化了。它可能正在拟合更困难的状态分布。

### 2. 现有 Warm-up 不等于 World Model 预训练

我核对了公开仓库的训练循环：

Collect→WM Update→Actor Update\text{Collect} \rightarrow \text{WM Update} \rightarrow \text{Actor Update}Collect→WM Update→Actor Update

Warm-up 阶段主要用于积累仿真数据；达到条件后，代码在每轮 World Model 更新之后立即执行 Actor 更新。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

也就是说，原来的 warm-up 并不能确保：

E8K≤EacceptableE_8^{K}\leq E_{\mathrm{acceptable}}E8K≤Eacceptable

如果 Koopman 比 MLP 收敛得慢，那么它可能还没有建立可靠的动力学预测，Actor 就开始通过它反向传播。

这能产生一种负反馈：

WM 不准确→BPTT 梯度偏差→Actor 改变数据分布→WM 更难拟合\boxed{ \text{WM 不准确} \rightarrow \text{BPTT 梯度偏差} \rightarrow \text{Actor 改变数据分布} \rightarrow \text{WM 更难拟合} }WM 不准确→BPTT 梯度偏差→Actor 改变数据分布→WM 更难拟合

目前没有直接证据证明这个循环就是你在线失败的主因，但它与已有结果一致，而且非常值得优先验证。

### 3. 我建议新增一个真正的 WM-only 阶段

保留当前模拟器和数据采集，只暂停 Actor/Critic 优化：

收集真实数据↓更新 Koopman WM↓固定验证窗口评价 WM↓预测达标后开启 Actor BPTT\boxed{ \begin{aligned} &\text{收集真实数据}\\ &\downarrow\\ &\text{更新 Koopman WM}\\ &\downarrow\\ &\text{固定验证窗口评价 WM}\\ &\downarrow\\ &\text{预测达标后开启 Actor BPTT} \end{aligned} }收集真实数据↓更新 Koopman WM↓固定验证窗口评价 WM↓预测达标后开启 Actor BPTT

这里不能只根据训练 minibatch 的损失决定是否达标。

需要同时参考：

* 固定验证窗口 E1/E8；

* 当前策略分布下的 E1/E8；

* 不同状态通道的误差；

* 软终止预测是否可靠。

我认为，增加一个独立的 World Model 预训练阶段，比把 WM_RATIO 从 2 直接加到 4 更值得优先测试。

原因是 WM_RATIO 主要改变更新频率，却不能保证第一次 Actor 更新前已经有一个足够好的世界模型。

## 四、正在进行的 `state large + ratio2 + stab0`，能验证什么？

你的新实验方向是合理的，但我希望把它的解释边界说清楚。

它同时包含三个因素：

$$boxed{ \text{Large Capacity} + \text{WM_RATIO}=2 + \lambda_{\mathrm{stab}}=0 }

相较之前的在线方案，模型容量、更新比例和稳定性约束都可能发生变化。

所以：

如果这次在线 Return 有明显提升，可以说明这一组合有效，但不能单独归因于 WM\_RATIO=2。

反过来，如果仍然失败，也不能说明增加 WM 更新频率无效。

### 我建议这次长训重点记录的指标

| 
指标

 | 

要回答的问题

 |
| --- | --- |
| 

`wm/fixed_nse_1step`

 | 

模型在固定分布上是否持续改善？

 |
| 

`wm/fixed_nse_8step`

 | 

多步预测是否稳定？

 |
| 

`wm/eval_nse_1step`

 | 

模型能否适应在线数据？

 |
| 

`wm/eval_nse_8step`

 | 

在线短期自回归是否可靠？

 |
| 

`wm/eval_nse_32step`

 | 

是否出现长时域漂移？

 |
| 

`train/episode_norm_return`

 | 

真实仿真运动能力有没有改善？

 |
| 

`policy/imag_survival_H`

 | 

想象轨迹是否预测了过多终止？

 |
| 

`policy/policy_grad_norm`

 | 

策略更新是否长期受到异常梯度影响？

 |
| 

`time/wm_s_per_iter`

 | 

增加 WM 更新是否过于昂贵？

 |

我建议重点观察以下三种情况。

情况 A：E1/E8 持续下降，Train Return 随后上升。

说明在线模型拟合可能是主要瓶颈。可以继续研究 Warm-up 和 WM 更新比例。

情况 B：E1/E8 已经很低，但 Train Return 仍然不提升。

这时应该把重点转向 BPTT 梯度质量、Critic、Soft Termination 和闭环模型偏差。

情况 C：固定评估 E1/E8 较低，但在线 E1/E8 一直很高。

这是在线分布覆盖不足的强烈信号，应进一步检查新数据分布、Replay Buffer 采样及不同运动模式，而不是继续盲目扩容。

### 还有一个工程核验点

你反馈已经实现了 `WM_RATIO=2/4`，我以你的 AutoDL 本地改动为准。

不过目前我能访问的 GitHub `main` 版本中，训练循环仍然是每次触发时执行一次 `wm_trainer.update()`，随后执行策略更新；公开的在线启动脚本也尚未包含 `WM_RATIO` 变量。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com&sz=32)

GitHub+1

所以我还不能独立确认本地 `ratio2` 的实际实现是否等价于：

两次完整 WM Update:一次 Actor Update\\text{两次完整 WM Update} : \\text{一次 Actor Update}两次完整 WM Update:一次 Actor Update

请以正在运行的代码版本和 Checkpoint 配置为准，特别确认新增代码没有意外改变 Actor 更新次数或学习率调度。

当前日志尚未产生第一个 iteration 指标，因此现在只能确认实验配置方向，不能判断训练成效。初始化和编译阶段的 GPU 占用本身也不能作为模型已经开始正常训练的证据。

五、另外两项结论：`rec_true` 和扩大容量都不值得优先投入
---------------------------------

### 1\. `rec_true=0.1`：暂时没有证据支持默认开启

在 small 模型中：

| 
指标

 | 

不加 rec\_true

 | 

加 rec\_true=0.1

 |
| --- | --- | --- |
| 

E1

 | 

0.0599

 | 

0.0604

 |
| 

E8

 | 

0.1908

 | 

0.1935

 |
| 

E32

 | 

0.676

 | 

0.699

 |

如果其余配置相同，那么 `rec_true` 加入后，三个指标分别变差约 0.8%、1.4% 和 3.4%。

但差距并不大，单次训练不足以证明这个损失普遍有害。

我的建议：

当前在线模型先不增加 `rec_true`，保留它作为后续消融项。

真实状态重构在理论上可以约束潜空间的信息保留能力，但你的 State-Augmented 模型已经显式包含物理状态，再要求学习潜变量完整重构这些信息，不一定对下一步动力学预测有帮助。

### 2\. 当前 Large 没有体现明显的容量收益

Small + rec\_true 与 Large + rec\_true 比较：

| 
指标

 | 

Small

 | 

Large

 |
| --- | --- | --- |
| 

参数量

 | 

0.53M

 | 

3.46M

 |
| 

E1

 | 

0.0604

 | 

0.0549

 |
| 

E8

 | 

0.1935

 | 

0.1826

 |
| 

E32

 | 

0.699

 | 

0.680

 |

扩大容量改善了 E1/E8，但 E32 改善有限。

因此：

如果 Large 在线训练没有明显优于 Small，我建议以后主要使用 Small 做结构和训练策略消融。

它参数更少、实验成本通常更低，而且从固定数据实验看已经有相当不错的多步预测能力。

不过，更少参数不保证更少 GPU 时间。你之前的在线实验中，Koopman 的实际 iteration 吞吐已经低于 MLP，所以还要实测完整训练成本。

六、我会怎样安排下一轮实验？
--------------

你已经启动了 Large + Ratio2 + Stab0。我建议保留这个实验，不再同时启动多个变化很大的版本。

拿到第一批有效曲线后，我会优先安排下面的实验。

| 
次序

 | 

实验

 | 

只改变什么

 | 

能回答什么

 |
| --- | --- | --- | --- |
| 

1

 | 

当前 Large + Ratio2 + Stab0

 | 

已在运行

 | 

组合方案是否有效？

 |
| 

2

 | 

Large + Ratio1 + Stab0

 | 

仅 WM 更新比例

 | 

Ratio2 是否真正有帮助？

 |
| 

3

 | 

Small + Ratio2 + Stab0

 | 

仅模型容量

 | 

Large 在线是否值得？

 |
| 

4

 | 

Large + Ratio2 + WM-only 预训练

 | 

仅延迟 Actor 更新

 | 

是否因 Actor 过早使用不准确的 WM 而失败？

 |
| 

5

 | 

在表现较好的配置上测试 H=8/16/32

 | 

仅 BPTT Horizon

 | 

是否存在长时域策略梯度问题？

 |

我会特别优先考虑第 4 项，因为它可以直接检验我们目前最重要的根因假设。

### 进一步的模型梯度检查

当 Koopman 在线 E1/E8 已经接近同数据下的 MLP，真实 Return 却仍然明显更低时，再研究：

JK,av\=∂FK(s,a)∂av\\boxed{ J\_{K,a}v = \\frac{\\partial F\_K(s,a)}{\\partial a}v }JK,a​v\=∂a∂FK​(s,a)​v​

与真实仿真器有限差分：

Jsim,av≈Fsim(s,a+ϵv)−Fsim(s,a−ϵv)2ϵ\\boxed{ J\_{\\mathrm{sim},a}v \\approx \\frac{ F\_{\\mathrm{sim}}(s,a+\\epsilon v) - F\_{\\mathrm{sim}}(s,a-\\epsilon v) }{2\\epsilon} }Jsim,a​v≈2ϵFsim​(s,a+ϵv)−Fsim​(s,a−ϵv)​​

之间的差异。

应同时比较梯度方向和大小，并在接触切换附近单独统计。

这才是判断 Koopman 是否真正适合 SuperMimic BPTT 的关键实验。

此外，如果出现“预测准确但策略不进步”，还应检查 Critic 目标和 Soft Termination 的校准情况，不能把所有失败都归因于动力学 Jacobian。

七、我认为这轮结果最重要的科研意义
-----------------

你现在实际上得到两个不同层次的结论。

第一层已经有较强的实验支持：

State-Augmented Koopman\>同容量 MLP\\boxed{ \\text{State-Augmented Koopman} > \\text{同容量 MLP} }State-Augmented Koopman\>同容量 MLP​

这里的“大于”仅指你当前固定数据实验中观察到的预测精度，而不是所有控制指标。

这是值得保留的结果。

第二层尚未被证明：

更好的离线多步预测⟹?更好的在线 BPTT 训练\\boxed{ \\text{更好的离线多步预测} \\stackrel{?}{\\Longrightarrow} \\text{更好的在线 BPTT 训练} }更好的离线多步预测⟹?​更好的在线 BPTT 训练​

目前你的在线结果尚不支持这个推断。

我认为这两层之间的差距，恰恰可能是你接下来最值得研究的方向。

如果后续通过适当的预训练和更新比例，State Koopman 能在在线训练中达到接近 MLP 的真实回报，那么你的工作可以强调：

> 使用更小、具有结构化动力学表示的世界模型，在维持多步预测与控制性能的同时降低模型参数量。

如果在线策略仍然失败，但共享数据上的预测误差已经相近，那么重点可以转向：

> 为什么准确的世界模型预测不足以保证基于 BPTT 的策略优化成功？

这同样是有意义的研究问题，不过需要梯度实验来支撑。

### 我的最终建议

目前先维持 `λ_stab=0`，不默认开启 `rec_true`，不继续扩大网络；让当前 Large + Ratio2 长训提供第一组完整在线证据。

在这之前，我会把最重要的改进工作放在两个方面：

一是加入真正的 WM-only 预训练/解冻机制，避免 Actor 在 Koopman 尚不准确时就开始 BPTT；二是建立 MLP 与 Koopman 在同一在线数据分布上的预测误差对照。

这两项比再增加一个 Koopman 结构、再堆一项辅助 Loss，更有可能解决你现在的核心问题。
$$


Matched-size models: offline prediction error

MLP 0.63M parameters; state Koopman 0.53M. Lower is better.

horizon	koopman	mlp
E1	0.0563	0.0643
E8	0.1766	0.3253
E32	0.6141	1.713

Stability regularization vs prediction accuracy

State-augmented small; 2000 offline updates. Lower E32 is better.

e32	lambda
0.6141	0
0.6298	1
0.676	10
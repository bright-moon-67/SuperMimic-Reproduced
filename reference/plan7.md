# Koopman-SuperMimic 实验结果分析与优化建议

这组数据非常有价值，因为它揭示了一个比之前更明确的问题：

State-Augmented Koopman 已经显著改善了离线世界模型预测，但进入在线 Actor–World Model 联合训练后，优势没有保持下来。

我重新核对了你仓库目前的 `koopman.py`、`world_model.py`、`training/loop.py`、`policy_trainer.py` 和在线训练脚本。结合这些代码，我认为现阶段最应该优化的不是模型容量，而是在线训练流程、模型在新状态分布上的适应能力，以及 BPTT 梯度质量。

但首先有一个必须排除的配置风险：你仓库在线脚本默认没有开启 State-Augmented Koopman，这可能直接影响对当前结果的判断。

## 一、先从数字判断问题在哪里

### 1. 离线实验：State-Augmented 已经基本解决严重发散

以原始 MLP 为基准，误差越低越好。

|
模型

|

E1 / MLP

|

E8 / MLP

|

E32 / MLP

|
| --- | --- | --- | --- |
|

经典 Koopman

|

2.58×

|

5.59×

|

1379×

|
|

经典 Koopman large

|

2.03×

|

2.45×

|

1010×

|
|

State-Augmented small

|

1.54×

|

1.32×

|

1.38×

|
|

State-Augmented large

|

1.41×

|

1.23×

|

1.39×

|

图表选项

Offline E32: MLP vs state-augmented Koopman

Lower is better; classic Koopman excluded from this chart because of its very large error.



从中可以提取三个重要结论。

第一，State-Augmented 是目前真正有效的架构改进。

经典 Koopman 的 E32 存在灾难性误差，而 State-Augmented 已将其控制在与 MLP 同一数量级。这说明物理状态锚定可能确实解决了之前的重要失稳机制。

第二，扩大 State-Augmented 容量几乎没有改善 E32。

从 small 到 large：

* E1 降低约 8.3%；

* E8 降低约 6.5%；

* E32 反而增加约 0.4%。

这说明当前长时域预测瓶颈不能简单归因于模型参数太少。继续扩大 Encoder、潜空间维度或低秩矩阵，不是我推荐的首要方向。

第三，离线 E32 已经不再差得离谱。

State-Augmented small：

E32=0.683E_{32}=0.683E32=0.683

MLP：

E32=0.495E_{32}=0.495E32=0.495

相差约 38%。

这是仍需改善的差距，但与经典 Koopman 相比已经完全不同。若两者使用相同的固定数据和评估协议，现在有理由进入控制性能验证。

### 2. 在线实验：差距又显著扩大

你提供的在线结果：

|
指标

|

State Koopman

|

MLP

|

判断

|
| --- | --- | --- | --- |
|

Iteration

|

9950

|

12150

|

训练进度不同

|
|

WM NSE1

|

0.169

|

0.012

|

Koopman 高约 14.1 倍

|
|

WM NSE8

|

0.738

|

0.022

|

Koopman 高约 33.5 倍

|
|

Imagined Return

|

0.792

|

2.61

|

Koopman 更低

|
|

Predicted Survival

|

0.917

|

1.000

|

Koopman 更低

|
|

Train Return

|

0.089

|

0.755

|

Koopman 明显更差

|
|

速度

|

0.9 it/s

|

1.3 it/s

|

Koopman 慢约 31%

|

这里必须注意：在线 `wm nse1` 和 `wm nse8` 来自训练期间的当前 Replay Buffer，而离线实验来自固定数据集，两套 NSE 不应该直接视为同分布测试结果。

此外，两个在线模型的 iteration 不同，需要对齐训练步数或环境交互步数后再作严格性能比较。

但即使考虑这些限制，当前在线训练差距仍然很明显。

这给出了我的核心诊断假设：

在线数据分布变化+Koopman 适应不充分  ⟶  Actor 获得较差的 BPTT 梯度  ⟶  更差的新数据\boxed{ \text{在线数据分布变化} + \text{Koopman 适应不充分} \;\longrightarrow\; \text{Actor 获得较差的 BPTT 梯度} \;\longrightarrow\; \text{更差的新数据} }在线数据分布变化+Koopman 适应不充分⟶Actor 获得较差的 BPTT 梯度⟶更差的新数据

这是可能形成的负反馈循环，目前还不是已证实的唯一原因。

## 二、最高优先级：先确认在线训练真的启用了 State-Augmented

P0 · 先排查

这次查看代码时，我注意到一个很重要的配置细节。

在 github.com  中：

Bash

```
STATE_KOOPMAN=${STATE_KOOPMAN:-0}
```

意味着：如果没有显式设置环境变量，脚本默认使用的是普通 Koopman 动力学，而不是 State-Augmented 分支。

只有当 `STATE_KOOPMAN=1` 时，才会添加：

Bash

```
--world-model.koopman.state-in-latent
```

并启用相应的双线性耦合与辅助损失配置。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

请优先检查你在线训练 Checkpoint 中保存的完整配置：

world_model.koopman.state_in_latent=True\boxed{ \texttt{world\_model.koopman.state\_in\_latent}=\mathrm{True} }world_model.koopman.state_in_latent=True

同时检查启动日志的 World Model 参数量是否符合你预期的 State-Augmented 配置。

为什么我特别强调这个问题？

因为你报告的在线 Koopman：

E8=0.738E_8=0.738E8=0.738

与离线经典 Koopman 的：

E8=0.8102E_8=0.8102E8=0.8102

比较接近，但明显高于离线 State-Augmented 的：

E8=0.1912E_8=0.1912E8=0.1912

这不能证明在线训练使用了错误模型——因为线上与线下数据分布不同——但足以让配置核验成为第一项工作。

如果实际跑的是经典 Koopman，那么我们就不应该用这组在线结果来否定 State-Augmented 的效果。

## 三、一个已确认的日志问题：在线 NSE1 与 NSE8 并非同一批数据

建议优先修正

我发现 github.com  中打印这两个值时，使用了不同的数据来源。

目前的逻辑相当于：

Python

运行

```
nse1 = agg["wm/nse_1step"] / agg_n
nse8 = float(err[-1])
```

其中：

* `nse1` 是最近多次 World Model 更新时，训练 minibatch 的平均第一步误差；

* `nse8` 则由 `online_error()` 从 Replay Buffer 重新采样窗口进行评估。

它们不是同一批窗口上的 E1 和 E8。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1

因此，不能直接用你这一行日志计算：

0.7380.169\frac{0.738}{0.169}0.1690.738

来推断 Koopman 八步预测中的误差放大倍数。

### 建议修改

同一次评估统一获取：

(E1,E2,…,E8)=EvaluateWM⁡(Bfixed)(E_1,E_2,\ldots,E_8) = \operatorname{EvaluateWM}(\mathcal B_{\mathrm{fixed}})(E1,E2,…,E8)=EvaluateWM(Bfixed)

然后分开记录：

* `wm/train_nse1`：训练 minibatch 的误差；

* `wm/eval_nse1`：固定评估窗口上的第一步误差；

* `wm/eval_nse8`：同一批窗口上的第八步误差；

* `wm/eval_nse32`：固定长时域窗口上的预测误差。

同时把评估来源区分为固定数据集和当前在线 Replay Buffer。后者的分布会随策略变化，所以不适合单独用于跨时间判断模型是否退步。

这是低成本、确定应该做的修改。

## 四、我最怀疑的根因：World Model 的在线学习跟不上 Actor

查看当前 github.com ，在线训练的更新顺序是：

Simulator Collect→World Model Update→Actor BPTT Update\boxed{ \text{Simulator Collect} \rightarrow \text{World Model Update} \rightarrow \text{Actor BPTT Update} }Simulator Collect→World Model Update→Actor BPTT Update

代码允许在 warm-up 结束后，每次世界模型更新都紧接着执行 Actor 更新。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

而你的 github.com  配置了：

warmup_iterations=40\texttt{warmup\_iterations}=40warmup_iterations=40

并采用：

updates_per_env_step=0.4\texttt{updates\_per\_env\_step}=0.4updates_per_env_step=0.4

以及每次 World Model 更新对应一次策略更新。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

这里的关键问题是：

40 步 warm-up 主要是在积累仿真数据，不代表新的 Koopman World Model 已经经过充分预训练。

当 Koopman 还不能准确预测未来时，Actor 已经开始使用其梯度优化。

可以产生以下循环：

Koopman 预测偏差↓BPTT 更新方向不准确↓Actor 进入新的、难以预测的状态区域↓Replay Buffer 分布变化↓Koopman 继续面临新的拟合难度\boxed{ \begin{aligned} &\text{Koopman 预测偏差}\\ &\downarrow\\ &\text{BPTT 更新方向不准确}\\ &\downarrow\\ &\text{Actor 进入新的、难以预测的状态区域}\\ &\downarrow\\ &\text{Replay Buffer 分布变化}\\ &\downarrow\\ &\text{Koopman 继续面临新的拟合难度} \end{aligned} }Koopman 预测偏差↓BPTT 更新方向不准确↓Actor 进入新的、难以预测的状态区域↓Replay Buffer 分布变化↓Koopman 继续面临新的拟合难度

MLP 可以在同样的训练流程下成功，不代表 Koopman 也具有相同的在线收敛速度。

### 优化方案 A：先让 Koopman 学会预测，再开启 Actor 更新

建议把训练分为三个阶段：

阶段 A

World Model 预训练

使用固定或持续收集的数据，仅更新 ω\omegaω，冻结 Actor。

目标是当前策略分布上的 E1/E8 足够低，而且连续评估不出现明显恶化。

阶段 B

保守联合训练

先使用较短的 BPTT Horizon，降低 Actor 学习率，增加 World Model 相对更新次数。

逐步放开策略，而不是立即采用完整 32 步优化。

阶段 C

恢复完整 SuperMimic

只有当模型在在线状态分布上达到稳定的预测水平，才恢复 H=32H=32H=32 和原始更新比例。

### 优化方案 B：调整 World Model 与 Actor 的更新比例

当前：

NWM:NActor=1:1N_{\mathrm{WM}}:N_{\mathrm{Actor}}=1:1NWM:NActor=1:1

这里每个 WM Update 内部包含多个训练 minibatch，而不是单次梯度下降。

我建议首先测试：

NWM:NActor=2:1与4:1\boxed{ N_{\mathrm{WM}}:N_{\mathrm{Actor}} = 2:1 \quad\text{与}\quad 4:1 }NWM:NActor=2:1与4:1

这可以通过跳过一部分 Actor 更新实现，但要注意比较时同时报告总计算预算与 GPU-hours。

不要仅因为模型更新次数增加就认为算法改进成功。

优先级判断：高。 这项修改比继续增加 Koopman 参数量更值得尝试，但是否有效，需要通过固定数据和在线实验验证。

## 五、优化 BPTT：我建议首先缩短 Horizon，而不是继续扩大 Koopman

这是第二个值得优先开展的实验。

当前你的代码使用：

H=32H=32H=32

Actor 更新时：

∇θJ^=∇θ\(∑t\=031γtctr^t+1+γ32c32Vψ(s^32)\)\nabla_\theta\hat J = \nabla_\theta \left\( \\sum\_{t=0}^{31}\\gamma^tc\_t\\hat r\_{t+1} + \\gamma^{32}c\_{32}V\_\\psi(\\hat s\_{32}) \\right\)∇θJ^=∇θ\(t\=0∑31​γtct​r^t+1​+γ32c32​Vψ​(s^32​)\)

这里的问题是：

离线 32 步状态误差尚可，不代表在线 BPTT 的策略梯度一定准确。

原因是离线 E32 使用 Replay Buffer 记录的动作序列，属于开环动力学预测。

但在线 BPTT 中：

at=πθ(o^t−9:t,gt)a_t=\pi_\theta(\hat o_{t-9:t},g_t)at=πθ(o^t−9:t,gt)

每一步的预测状态都会影响未来动作。

所以：

开环预测准确⇏闭环策略梯度准确\boxed{ \text{开环预测准确} \not\Rightarrow \text{闭环策略梯度准确} }开环预测准确⇒闭环策略梯度准确

尤其是关节动作、接触状态或根部姿态出现少量偏差时，Actor 可能连续改变后续动作，造成与真实机器人不同的轨迹。

### 建议进行 Horizon 消融

保持 Koopman 模型不变，分别测试：

|
Horizon

|

目的

|

预期代价

|
| --- | --- | --- |
|

H=8H=8H=8

|

验证短时域 BPTT 是否能够训练 Actor

|

更依赖 Critic

|
|

H=16H=16H=16

|

检查扩大时域是否改善回报

|

中等计算成本

|
|

H=32H=32H=32

|

与原始 SuperMimic 对齐

|

最大的模型误差累积风险

|

我的建议不是永久将 Horizon 改为 8，而是尝试：

H:8→16→32\boxed{ H:8\rightarrow16\rightarrow32 }H:8→16→32

只有当较短 Horizon 下真实仿真 Return 已经开始提高、世界模型误差稳定时，才扩大展开长度。

不过这也有一个前提：Critic 必须能够提供有用的终端价值。若 Critic 还不准确，缩短 Horizon 也可能降低训练质量。

因此要同时记录 `critic_loss`、预测 Value、GAE Target 和真实 Return，而不是只观察 Actor Loss。

从研究角度，我认为这个实验很有价值：它能检验 Koopman 的主要障碍究竟是短期动力学拟合，还是长时间尺度上的控制梯度传播。

## 六、World Model 的损失函数还有优化空间

当前 github.com  对 State-Augmented Koopman 使用了：

LWM=Lstate+Ltermination+λlift,ϕLlift,ϕ+λconsLcons+λrecLrec+λstabLstab\begin{aligned} \mathcal L_{\mathrm{WM}} ={}& \mathcal L_{\mathrm{state}} +\mathcal L_{\mathrm{termination}}\\ &+\lambda_{\mathrm{lift},\phi}\mathcal L_{\mathrm{lift},\phi} +\lambda_{\mathrm{cons}}\mathcal L_{\mathrm{cons}}\\ &+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}} +\lambda_{\mathrm{stab}}\mathcal L_{\mathrm{stab}} \end{aligned}LWM=Lstate+Ltermination+λlift,ϕLlift,ϕ+λconsLcons+λrecLrec+λstabLstab

其中物理状态的 Lift Loss 也可通过配置单独开启。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

我认为有三个方向值得考虑。

### 1. 增加真实状态重构，而不是只重构预测状态

当前重构损失主要是：

Lrec,pred=∥Rη(ϕt)−Canonical⁡(s^t)∥2\mathcal L_{\mathrm{rec,pred}} = \left\| R_\eta(\phi_t) - \operatorname{Canonical}(\hat s_t) \right\|^2Lrec,pred=∥Rη(ϕt)−Canonical(s^t)∥2

其中 s^t\hat s_ts^t 是模型自身预测的状态。

这有助于潜变量与预测物理状态保持一致，但两者可能一起偏离真实状态。

建议增加独立辅助目标：

Lrec,true=∥Rη(Eϕ(streal,ξ))−Canonical⁡(streal)∥2\boxed{ \mathcal L_{\mathrm{rec,true}} = \left\| R_\eta(E_\phi(s_t^{\mathrm{real}},\xi)) - \operatorname{Canonical}(s_t^{\mathrm{real}}) \right\|^2 }Lrec,true=Rη(Eϕ(streal,ξ))−Canonical(streal)2

这样可以检验真实状态监督是否改善潜空间的信息保留能力。

不过不要直接堆叠所有损失。建议先分别训练 `rec_pred only`、`rec_true only` 和两者结合的版本。

### 2. 检查 Stability Loss 是否过强

在线脚本默认：

Bash

```
STAB=${STAB:-1}
```

开启后设置：

λstab=10\lambda_{\mathrm{stab}}=10λstab=10

通过对内部传播矩阵的最大奇异值进行惩罚，抑制其放大。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

这对于之前会爆炸的经典 Koopman 很有意义。

但对于已经具有显式物理状态积分的 State-Augmented 模型，约束内部矩阵：

MtM_tMt

并不能完整代表：

∂s^t+1∂s^t\frac{\partial \hat s_{t+1}}{\partial \hat s_t}∂s^t∂s^t+1

的动力学结构。

过强的约束还可能抑制真实的高敏感度控制模式。

我建议单独比较：

λstab∈{0,1,10}\lambda_{\mathrm{stab}}\in\{0,1,10\}λstab∈{0,1,10}

并同时观察 E1、E8、E32、动作 Jacobian 和真实 Return。

不要因为内部谱半径更小，就直接认为梯度更适合机器人控制。

### 3. 分析哪类物理状态预测最差

建议将当前在线世界模型的误差分成：

Lstate=Lroot,pos+Lroot,rot+Lroot,vel+Ljoint,pos+Ljoint,vel\mathcal L_{\mathrm{state}} = \mathcal L_{\mathrm{root,pos}} + \mathcal L_{\mathrm{root,rot}} + \mathcal L_{\mathrm{root,vel}} + \mathcal L_{\mathrm{joint,pos}} + \mathcal L_{\mathrm{joint,vel}}Lstate=Lroot,pos+Lroot,rot+Lroot,vel+Ljoint,pos+Ljoint,vel

再按照机器人运动模式分别统计。

比如，在同一个固定验证集上比较：

* 普通行走；

* 跑步；

* 跳跃和腾空；

* 足部落地；

* 大姿态转动。

如果 Koopman 主要在接触切换附近失准，那么下一步应该改进接触相关特征或采用分模式动力学。

如果所有运动的关节速度都普遍预测较差，则更应该检查状态增量预测器、动作输入和执行器响应的建模。

先定位错误通道，再修改模型结构。

## 七、为什么 Koopman 的在线速度反而更慢？

这是一个值得调查的工程问题。

你目前的测量是：

Koopman=0.9 it/s\text{Koopman}=0.9\ \mathrm{it/s}Koopman=0.9 it/s

MLP=1.3 it/s\text{MLP}=1.3\ \mathrm{it/s}MLP=1.3 it/s

即 Koopman 每个 iteration 的耗时更高。

这不一定矛盾：参数量更少，并不保证执行速度更快。

你的 State-Augmented Koopman 包含：

* Encoder；

* 低秩状态—动作耦合；

* 物理上下文交互；

* 状态积分；

* 辅助重构；

* 一致性损失；

* 稳定性正则。

其中稳定性正则可能是一个值得优先测量的成本来源。

`stability_penalty()` 通过 `transition_matrix()` 构建动力学矩阵，再使用幂迭代估计最大奇异值。它在 World Model 训练窗口中反复调用。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1

### 建议立即拆分耗时

你的 `training/loop.py` 已经记录：

* `time/sim_s_per_iter`

* `time/wm_s_per_iter`

* `time/policy_s_per_iter`

因此不需要重写性能分析器，先比较这些指标即可。

如果主要慢在 World Model Training，重点检查 Stability Loss、潜空间辅助损失及矩阵构造。

如果主要慢在 Policy Update，重点检查 Koopman 的 32 步前向传播与反向传播成本，以及 `torch.compile` 是否产生额外计算开销。

我会首先比较同一 State-Augmented 模型下：

STAB=0vsSTAB=1\texttt{STAB=0} \quad\text{vs}\quad \texttt{STAB=1}STAB=0vsSTAB=1

的吞吐量与显存占用。

如果关闭 Stability Loss 能明显加速，但预测性能没有明显恶化，那么比直接优化 CUDA Kernel 更值得优先处理。

## 八、最有诊断价值的实验：在同一份在线数据上比较 MLP 与 Koopman

这是我认为下一轮实验信息价值最高的一项。

因为目前最不清楚的是：

Koopman 在在线训练中表现差，是模型本身无法拟合当前数据，还是 Actor–World Model 的联合更新造成了负反馈？

建议做一个交叉实验。

先收集两份在线数据：

DMLP\mathcal D_{\mathrm{MLP}}DMLP

由 MLP Actor 在真实仿真器中收集；

DKoopman\mathcal D_{\mathrm{Koopman}}DKoopman

由当前 Koopman Actor 收集。

然后冻结数据，不再更新 Actor，训练和评估两种世界模型：

|
训练与评估数据

|

MLP World Model

|

State-Augmented Koopman

|
| --- | --- | --- |
|

DMLP\mathcal D_{\mathrm{MLP}}DMLP

|

实验 A

|

实验 B

|
|

DKoopman\mathcal D_{\mathrm{Koopman}}DKoopman

|

实验 C

|

实验 D

|

所有实验采用相同数据预算，并另设固定验证窗口。

### 结果该如何解释？

情况一：Koopman 在两份数据上都明显差于 MLP。

说明主要问题可能仍然在 Koopman 模型表示、损失设计或优化能力。

此时应该优先修改 World Model，而不是 Actor。

情况二：Koopman 能在两份固定数据上拟合得不错，但联合训练始终失败。

说明更可能是联合训练中的分布变化、策略梯度质量、训练时序或 Critic 问题。

此时优先修改 warm-up、更新比例和 Horizon。

情况三：Koopman 只在其自身 Actor 收集的数据上严重失败。

说明这些状态可能对 Koopman 特别困难，也可能是策略导致了特殊的动作分布或接触分布。

应进一步检查接触模式、动作幅度、关节速度、状态分布覆盖范围，以及是否存在模型梯度利用。

这比继续盲目调整 Latent Dimension 更容易得到清楚的结论。

## 九、关于 Imagined Return 和 Survival，不能直接下结论

你在线记录的：

Imagined Return=0.792\text{Imagined Return}=0.792Imagined Return=0.792

Predicted Survival=0.917\text{Predicted Survival}=0.917Predicted Survival=0.917

Train Return=0.089\text{Train Return}=0.089Train Return=0.089

值得关注，但目前不能直接用 `0.792` 与 `0.089` 的差异证明 Model Exploitation。

因为代码中的：

`imag_return_undisc`

是 32 步想象轨迹中，生存权重乘奖励之后的累计值：

E\(∑t\=031ctr^t+1\)\mathbb E\left\(\\sum\_{t=0}^{31}c\_t\\hat r\_{t+1}\\right\)E\(t\=0∑31​ct​r^t+1​\)

而 `train/episode_norm_return` 是真实环境完成 Episode 后，按照最大奖励与 Episode 长度归一化的平均回报。

这两个数值不是同一种尺度，也不来自完全相同的初始状态集合。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1

同样，`survival=0.917` 是 World Model 在想象轨迹中的预测生存概率，不是实际仿真器中机器人完成动作的比例。

### 建议增加配对评估

从完全相同的真实初始状态出发：

s0,g0:Hs_0,\quad g_{0:H}s0,g0:H

分别运行：

Actor + Real Simulator\text{Actor + Real Simulator}Actor + Real Simulator

以及：

Actor + Koopman World Model\text{Actor + Koopman World Model}Actor + Koopman World Model

比较相同时间范围内的：

Rsim,RWMR_{\mathrm{sim}},\quad R_{\mathrm{WM}}Rsim,RWM

以及：

dsim,d^WMd_{\mathrm{sim}},\quad \hat d_{\mathrm{WM}}dsim,d^WM

这样才能判断世界模型是否系统性高估奖励、低估跌倒风险，或者两者兼有。

还可以单独检查 Termination Head 的 BCE、校准误差及真实终止比例。

## 十、如果仍然不理想，再考虑结构上的改进

在上述诊断之前，我不建议直接引入更复杂的 Contact-Aware Koopman。

但如果已经确认当前 State-Augmented 模型存在表达瓶颈，我更倾向于尝试小规模的 Residual Dynamics。

例如，保持现有 Koopman 主干：

ΔtK=D(K(zt,at,ξ))\Delta_t^{K}=D(K(z_t,a_t,\xi))ΔtK=D(K(zt,at,ξ))

另外学习一个小型残差：

ΔtR=hη(xt,at,ξ)\Delta_t^{R}=h_\eta(x_t,a_t,\xi)ΔtR=hη(xt,at,ξ)

最终：

Δt=ΔtK+αΔtR\boxed{ \Delta_t=\Delta_t^K+\alpha\Delta_t^R }Δt=ΔtK+αΔtR

它的优势在于：

* Koopman 继续负责结构化动力学；

* 残差网络补偿难以用低秩双线性表示的局部非线性；

* 可以单独分析 Koopman 与 Residual 对预测的贡献。

但这已经属于混合动力学模型，不能再称为纯粹的双线性 Koopman。

而且如果 Residual MLP 最终承担了几乎全部预测能力，那么这种方案的科研价值可能有限。应报告去掉 Koopman 分支后的性能，证明它确实提供额外贡献。

## 十一、我建议的下一轮实验清单

为了节约 AutoDL 的实验成本，我会按照这个顺序做，而不是一次性大规模网格搜索。

|
优先级

|

实验

|

改动

|

核心目的

|
| --- | --- | --- | --- |
|

P0

|

配置与日志校验

|

检查 `state_in_latent`；统一在线 E1/E8 数据源

|

排除模型配置及指标问题

|
|

P0

|

固定在线数据交叉实验

|

MLP / Koopman 分别拟合同一在线数据

|

区分模型能力与联合训练问题

|
|

P0

|

WM-only Warm-up

|

延迟 Actor 更新直到 WM 达到验证门槛

|

检查早期错误 BPTT 是否破坏训练

|
|

P1

|

Horizon 消融

|

H=8,16,32H=8,16,32H=8,16,32

|

检查长时域策略梯度是否有害

|
|

P1

|

更新比例消融

|

WM:Actor = 1:1、2:1、4:1

|

检查模型能否及时适应策略分布

|
|

P1

|

Stability Loss 消融

|

λ=0,1,10\lambda=0,1,10λ=0,1,10

|

检查是否过度限制真实动力学

|
|

P1

|

Actor LR 消融

|

从 2×10−42\times10^{-4}2×10−4 降至 5×10−55\times10^{-5}5×10−5 或 10−510^{-5}10−5

|

降低不准确梯度的破坏性

|
|

P2

|

梯度方向测试

|

比较 WM 与真实仿真器的方向导数

|

验证 BPTT 可靠性

|
|

P2

|

Residual Koopman

|

加轻量非线性修正 Head

|

解决确认存在的表达瓶颈

|

其中学习率、Horizon 和更新比例都是待验证的实验候选值，并非已经证明有效的配置。

每个实验至少保存固定验证集上的 E1/E8/E32、真实仿真 Return、失败率、显存和 GPU-hours。尤其要将不同模型对齐到相同环境交互步数，必要时另做相同计算预算的对照。

## 十二、最终判断：最有希望的优化方向是什么？

我会按重要性这样排序：

第一：确认配置与评价协议。

在线脚本默认没有开启 State-Augmented，这是必须立即排除的风险；在线 NSE1/NSE8 的不同采样来源也必须修正。

第二：改善在线 World Model 与 Actor 的训练协调。

这是我目前最看好的方向。延迟 Actor 更新、提高 WM 的相对更新频率、逐步扩大 BPTT Horizon，可能比扩大模型容量更有效。

第三：验证真实策略梯度，而不只验证状态预测。

即使 State-Augmented 离线 E32 已接近 MLP，也不能直接推断：

∇θJ^Koopman≈∇θJsim\nabla_\theta \hat J_{\mathrm{Koopman}} \approx \nabla_\theta J_{\mathrm{sim}}∇θJ^Koopman≈∇θJsim

这是决定 Koopman 能否真正替代原始 MLP World Model 的关键。

### 一个更深层的研究判断

你现在的结果很有意思：

Offline E32: Koopman 接近 MLPOnline Policy Return: Koopman 远差于 MLP\boxed{ \begin{aligned} &\text{Offline E32: Koopman 接近 MLP}\\ &\text{Online Policy Return: Koopman 远差于 MLP} \end{aligned} }Offline E32: Koopman 接近 MLPOnline Policy Return: Koopman 远差于 MLP

这意味着：

在一个固定数据分布上学到较好的多步预测，并不一定足以支持稳定的在线可微策略优化。

但目前还不能断言主要问题就是 Jacobian 错误，因为模型配置、在线数据分布与训练时序尚未完全排除。

如果后续确认：两种世界模型在同一在线状态分布下预测误差接近，而 Koopman 的真实策略梯度方向显著更差，那么就可以形成一个很有价值的科研问题：

> World Model 的状态预测精度，是否足以衡量它对于 BPTT 策略优化的适用性？

[text](../.ruff_cache)而你提出的 Koopman 结构，恰好能够成为研究这一问题的切入点。

我当前最推荐的下一步，是先核查在线 Checkpoint 中的 `state_in_latent` 配置，然后用 MLP 和 State-Augmented Koopman 对同一份在线 Replay Buffer 做固定数据对照。 这两件事最可能快速揭示你现在的性能差距究竟来自哪里，也能避免继续浪费算力在不必要的网络扩容上。


Offline E32: MLP vs state-augmented Koopman

Lower is better; classic Koopman excluded from this chart because of its very large error.

E32	model
0.495	MLP
0.683	State small
0.686	State large
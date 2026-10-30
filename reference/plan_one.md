[CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced?utm_source=chatgpt.com)([GitHub](https://github.com/CuboiLeo/SuperMimic-Reproduced))

# 在 CuboiLeo/SuperMimic-Reproduced 中接入 Koopman World Model：详细实施方案

**可以，而且你已经完成了部分训练、指标也比较理想，这正是开展 Koopman 对照实验的有利起点。**

但我建议你采取一种比较保守的工程策略：

> **保留已经验证有效的 Actor、Critic、仿真环境、奖励函数、数据收集和 BPTT 训练流程，只替换 World Model 的动力学核心。**

不要一开始就重写整个项目，也不要直接把已经训练好的 MLP World Model 权重替换成 Koopman 权重后继续训练 Actor。

原因是两种动力学模型的参数化方式不同。直接替换会使 Actor 面对一个尚未验证的新动力学模型，已有的训练成果反而可能被破坏。

## 一、先确定具体要改哪些代码

我重新核对了 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced?utm_source=chatgpt.com) 的公开目录。

这个仓库的模块划分对我们的改造比较友好：

| 模块                | 仓库位置                             | Koopman 改造策略   |
| ------------------- | ------------------------------------ | ------------------ |
| World Model         | `src/supermimic/world_model/`        | **主要修改位置**   |
| World Model Trainer | `src/supermimic/world_model/`        | 增加潜空间损失     |
| BPTT Imagination    | `src/supermimic/imagination/`        | 尽量不改           |
| Actor / Critic      | `src/supermimic/policy/`             | 不改               |
| 主训练循环          | `src/supermimic/training/loop.py`    | 只改模型构建和配置 |
| 训练配置            | `src/supermimic/config.py`           | 增加 Koopman 参数  |
| 仿真器              | `src/supermimic/envs/`               | 不改               |
| 可微前向运动学      | `src/supermimic/kinematics/g1_fk.py` | 不改               |
| 测试                | `tests/`                             | 增加 Koopman 测试  |

仓库 README 明确说明，现有世界模型采用**朝向规范化状态特征上的 MLP 动力学、显式状态积分、学习式终止 Head，以及通过可微 FK 实现的解析奖励和观测 Head**。

这意味着，我们不应该直接替换整个 World Model 对象，而应优先替换其中的**动力学预测子模块**，保留状态积分、Reward、Observation 和 Termination 的既有逻辑。([GitHub](https://github.com/CuboiLeo/SuperMimic-Reproduced?utm_source=chatgpt.com))

------

## 二、我建议的第一版模型：低秩双线性 Deep Koopman

先明确：下面是我为你的研究提出的模型设计，**不是原论文或 CuboiLeo 仓库已经实现的结构**。

### 2.1 保留原始物理状态

原项目的数据流可以抽象成：

(st,at,ξ)→Canonical Featuresxt→MLP DynamicsΔt→State Integrations^t+1(s_t,a_t,\xi) \xrightarrow{\text{Canonical Features}} x_t \xrightarrow{\text{MLP Dynamics}} \Delta_t \xrightarrow{\text{State Integration}} \hat s_{t+1}

其中：

- sts_t：完整仿真状态；
- ata_t：Actor 输出的关节控制指令；
- ξ\xi：物理随机化参数；
- xtx_t：朝向规范化的动力学输入特征；
- Δt\Delta_t：用于显式积分的状态增量。

**第一版 Koopman 不要直接预测完整物理状态。**

我建议让 Koopman 模型仍然输出原项目所需的 Δt\Delta_t，这样能最大限度保留现有积分逻辑和坐标系处理。

### 2.2 Koopman 结构

定义：

zt=Eϕ(xt)z_t=E_\phi(x_t)

其中 EϕE_\phi 是神经网络编码器。

采用：

zt+1=Azt+Bat+∑i=129at,iNizt(1)\boxed{ z_{t+1} = Az_t+Ba_t+ \sum_{i=1}^{29}a_{t,i}N_i z_t } \tag{1}

然后：

Δ^t=Dψ(zt+1)(2)\boxed{ \hat\Delta_t=D_\psi(z_{t+1}) } \tag{2}

最终：

s^t+1=Integrate⁡(st,Δ^t)(3)\boxed{ \hat s_{t+1} = \operatorname{Integrate}(s_t,\hat\Delta_t) } \tag{3}

但是这里有一个必须说明的理论问题。

如果我们把 Eϕ(xt)E_\phi(x_t) 当作当前提升状态，又在每一步根据完整预测物理状态重新编码，那么：

zt+1=Eϕ(xt+1)z_{t+1}=E_\phi(x_{t+1})

与：

zt+1=K(zt,at)z_{t+1}=K(z_t,a_t)

不一定一致。

因此，需要增加 Koopman Consistency Loss，约束两种潜状态更新方式。

否则，这个模型可能只是一个带矩阵层的普通神经网络，无法保证学到了有意义的潜空间动力学。

### 2.3 为什么选择低秩双线性？

普通线性 Koopman：

zt+1=Azt+Batz_{t+1}=Az_t+Ba_t

动作 Jacobian 为：

∂zt+1∂at=B\frac{\partial z_{t+1}}{\partial a_t}=B

它不依赖状态。

但人形机器人动作效果具有明显的状态依赖性。

因此，采用：

Ni=UiVi⊤N_i=U_iV_i^\top

得到：

zt+1=Azt+Bat+∑iat,iUiVi⊤zt(4)\boxed{ z_{t+1} = Az_t+Ba_t+ \sum_i a_{t,i}U_iV_i^\top z_t } \tag{4}

这种形式允许动作对潜状态的影响随 ztz_t 变化，同时避免保存大量完整 NiN_i 矩阵。

**第一版建议：**

dz=128,r=8d_z=128,\qquad r=8

编码器和解码器先使用较小的 MLP，避免一开始就引入比原模型更大的计算负担。

这只是起始实验配置，不是已经验证的最优超参数。

------

## 三、训练目标如何修改？

### 3.1 原来的 World Model Loss 必须保留

SuperMimic 的世界模型训练采用八步自回归预测：

LWM=∑j=18[∥s^j−sj∥2+(r^j−rj)2+∥o^j−oj∥2+BCE⁡(d^j,dj)]\mathcal L_{\mathrm{WM}} = \sum_{j=1}^{8} \left[ \|\hat s_j-s_j\|^2+ (\hat r_j-r_j)^2+ \|\hat o_j-o_j\|^2+ \operatorname{BCE}(\hat d_j,d_j) \right]

实际仓库中，部分 Head 是解析实现，所以具体启用哪些损失项，应保持仓库当前设置不变。

**不要因为接入 Koopman 就修改 Reward、Termination 或 Observation Loss。**

这样才能把性能变化归因于动力学模型。

### 3.2 增加潜空间一致性损失

对于真实仿真转移：

(st,at,st+1)(s_t,a_t,s_{t+1})

定义：

zt=Eϕ(xt)z_t=E_\phi(x_t)

zt+1pred=Kω(zt,at)z_{t+1}^{\mathrm{pred}}=K_\omega(z_t,a_t)

zt+1target=Eϕ(xt+1)z_{t+1}^{\mathrm{target}}=E_\phi(x_{t+1})

然后：

Llift=∥zt+1pred−zt+1target∥2(5)\boxed{ \mathcal L_{\mathrm{lift}} = \left\| z_{t+1}^{\mathrm{pred}} - z_{t+1}^{\mathrm{target}} \right\|^2 } \tag{5}

注意，xt+1x_{t+1} 必须由真实下一状态及对应物理上下文构建，而不是随意把预测增量作为编码器输入。

### 3.3 增加重构损失

如果编码器输入为 xtx_t，可以训练一个辅助重构 Head：

x~t=Rη(Eϕ(xt))\tilde x_t=R_\eta(E_\phi(x_t))

并定义：

Lrec=∥x~t−xt∥2(6)\boxed{ \mathcal L_{\mathrm{rec}} = \|\tilde x_t-x_t\|^2 } \tag{6}

这里的 RηR_\eta 与用于预测状态增量的 DψD_\psi 可以是两个不同的 Head。

我更推荐分开，因为：

- RηR_\eta 负责重构当前特征；
- DψD_\psi 负责输出下一步动力学增量。

它们的监督目标不同。

### 3.4 第一阶段的总损失

L=LWM+λliftLlift+λrecLrec(7)\boxed{ \mathcal L = \mathcal L_{\mathrm{WM}} + \lambda_{\mathrm{lift}}\mathcal L_{\mathrm{lift}} + \lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}} } \tag{7}

第一轮实验可以先设置：

λlift=0.1,λrec=0.1\lambda_{\mathrm{lift}}=0.1,\qquad \lambda_{\mathrm{rec}}=0.1

但这两个权重仅用于起步，必须观察各项损失的实际量级后再调整，不能直接认为它们适用于所有状态归一化方案。

------

## 四、最关键的工程设计：保证原有 BPTT 不需要重写

你已经有一套正在工作的训练系统，因此最重要的是维持原有 World Model 的调用接口。

建议采用这样的结构：

```
SuperMimic Training Loop
│
├── Simulator
│   └── Replay Buffer
│
├── World Model Trainer
│   ├── MLP Dynamics             ← 原始 Baseline
│   └── Koopman Dynamics         ← 新增
│
├── Imagination / BPTT
│   ├── Actor
│   ├── World Model.forward()
│   ├── Reward
│   ├── Soft Termination
│   └── Critic
│
└── Evaluation
    ├── World Model Prediction Error
    ├── Jacobian Error
    └── Policy Return
```

### 4.1 先建立一个统一的动力学接口

理想情况下，两种模型都接受：

(st,at,ξ)(s_t,a_t,\xi)

并返回：

s^t+1\hat s_{t+1}

例如：

```
class DynamicsInterface:
    def forward(self, state, action, sim_context):
        """
        state:       [B, state_dim]
        action:      [B, action_dim]
        sim_context: [B, context_dim]

        Returns:
            next_state: [B, state_dim]
        """
        raise NotImplementedError
```

这是接口设计示意，**不是仓库现有类的原样代码**。实际接入时，应以现有模型的 `forward()` 参数、状态数据结构和返回值为准。

尤其不要为了统一接口，把项目的结构化状态强行展平成一个 Tensor，再绕过现有的姿态积分与可微 FK。

### 4.2 Koopman 模块内部的计算

建议分成四步：

xt=Canonicalize⁡(st,ξ)zt=Eϕ(xt)zt+1=Kω(zt,at)s^t+1=Integrate⁡(st,Dψ(zt+1))\boxed{ \begin{aligned} x_t&=\operatorname{Canonicalize}(s_t,\xi)\\ z_t&=E_\phi(x_t)\\ z_{t+1}&=K_\omega(z_t,a_t)\\ \hat s_{t+1} &=\operatorname{Integrate} \left(s_t,D_\psi(z_{t+1})\right) \end{aligned} }

但是要注意：**如果你的目标是严格研究 Koopman 潜空间演化，最好同时实现一个“潜状态连续传播”的版本。**

也就是说，在 8 步 World Model 训练时：

z0=Eϕ(x0)z_0=E_\phi(x_0)

之后：

z1=K(z0,a0)z_1=K(z_0,a_0)

z2=K(z1,a1)z_2=K(z_1,a_1)

⋯\cdots

而不是每一步都完全重新编码：

zt+1=Eϕ(x^t+1)z_{t+1}=E_\phi(\hat x_{t+1})

后者可以作为工程上的替代模型，但会削弱对真正线性潜空间演化的检验。

因此我建议：

**第一版实现时，同时保留两种模式，但把潜状态连续传播作为正式 Koopman 实验。**

### 4.3 BPTT 的梯度路径

对于双线性模型：

zt+1=Azt+Bat+∑iat,iNiztz_{t+1} = Az_t+Ba_t+ \sum_i a_{t,i}N_i z_t

有：

∂zt+1∂zt=A+∑iat,iNi\boxed{ \frac{\partial z_{t+1}}{\partial z_t} = A+\sum_i a_{t,i}N_i }

以及：

∂zt+1∂at=B+[N1zt,…,N29zt]\boxed{ \frac{\partial z_{t+1}}{\partial a_t} = B+[N_1z_t,\ldots,N_{29}z_t] }

因此 Actor 的梯度路径变成：

J^→∇s^t+1→∇Dψ→∇zt+1→∇Kω→∇at→∇πθ→∇θ\boxed{ \hat J \xrightarrow{\nabla} \hat s_{t+1} \xrightarrow{\nabla} D_\psi \xrightarrow{\nabla} z_{t+1} \xrightarrow{\nabla} K_\omega \xrightarrow{\nabla} a_t \xrightarrow{\nabla} \pi_\theta \xrightarrow{\nabla} \theta }

在连续潜状态传播的版本中，梯度还会穿过前面所有时间步的 Koopman 更新。

**特别注意：更新 Actor 时可以冻结 Koopman 模型参数，但不能对其输出执行 detach()。**

否则 BPTT 会被切断。

------

## 五、怎样利用你已经训练好的模型？

这是你当前最有优势的地方。

我不建议从头开始采样，也不建议马上让 Koopman 和 Actor 联合训练。

建议采用三个阶段。

### 阶段 A：保存并固定当前 Baseline

先保留当前已经训练好的：

- Actor Checkpoint；
- Critic Checkpoint；
- World Model Checkpoint；
- Replay Buffer（如果项目当前支持保存）；
- 完整训练配置；
- 随机种子；
- TensorBoard 日志。

建议新建独立实验分支，例如：

`experiment/koopman-world-model`

**不要覆盖当前已经获得较好结果的 Checkpoint。**

### 阶段 B：固定数据，单独训练 Koopman

先让原始 MLP 和 Koopman 使用同一批真实仿真转移：

D={(st,at,st+1,ξ)}\mathcal D= \{(s_t,a_t,s_{t+1},\xi)\}

进行训练。

这一阶段只更新：

ϕ,ω,ψ,η\phi,\omega,\psi,\eta

这里分别表示 Koopman 编码器、潜动力学、预测解码器和辅助重构网络的参数。

Actor 和 Critic 都保持不变。

目的不是马上提高机器人控制性能，而是先回答：

**Koopman 能否在相同数据条件下学到足够准确的世界模型？**

建议先检查：

E1,E8,E32E_1,\quad E_8,\quad E_{32}

其中 EkE_k 表示第 kk 步预测误差。

不过 E32E_{32} 应作为独立验证指标，不要直接把它加入训练损失，否则会改变与原论文八步训练窗口的比较条件。

### 阶段 C：接入 Actor BPTT

只有在 Koopman 的预测误差、梯度和数值稳定性通过检查后，再接入策略训练。

这一阶段建议设置两组实验：

| 实验 | Actor 初始化                | World Model | 目的                         |
| ---- | --------------------------- | ----------- | ---------------------------- |
| A    | 当前已训练 Actor            | 原始 MLP    | 已有训练基准                 |
| B    | 当前已训练 Actor 的相同权重 | Koopman     | 比较模型替换后的继续训练表现 |
| C    | 从头初始化 Actor            | 原始 MLP    | 标准训练基准                 |
| D    | 从头初始化 Actor            | Koopman     | 比较完整训练过程             |

A、B 是**模型替换后的适应能力实验**。

C、D 才更适合回答：

> 从头训练时，Koopman 是否优于原始 MLP？

不能只比较 A、B 就声称 Koopman 提高了整个算法的训练效率。

此外，A、B 应尽量使用相同的 Critic 初始化，并记录模型切换初期的 Return 变化。

------

## 六、我认为必须增加的四项测试

### 测试 1：单步梯度检查

给定固定状态和动作：

s^t+1=FK(st,at)\hat s_{t+1}=F_K(s_t,a_t)

检查：

∂FK∂at\frac{\partial F_K}{\partial a_t}

是否能通过 PyTorch 自动微分正确获得。

可以使用有限差分：

FK(s,a+ϵv)−FK(s,a−ϵv)2ϵ\frac{ F_K(s,a+\epsilon v) - F_K(s,a-\epsilon v) }{2\epsilon}

与：

JFK,avJ_{F_K,a}v

比较。

这个测试验证的是**模型内部自动微分是否正确**，还不能证明模型梯度符合真实仿真器。

### 测试 2：真实动力学 Jacobian 对照

再用真实 MuJoCo Warp 仿真器：

Fsim(s,a+ϵv)−Fsim(s,a−ϵv)2ϵ\frac{ F_{\mathrm{sim}}(s,a+\epsilon v) - F_{\mathrm{sim}}(s,a-\epsilon v) }{2\epsilon}

估计动作方向导数。

然后比较：

JFK,av与JFsim,avJ_{F_K,a}v \quad\text{与}\quad J_{F_{\mathrm{sim}},a}v

这里要确保两个扰动实验从完全相同的状态开始，且使用相同的物理上下文。

在接触切换附近，应单独报告结果，因为真实动力学可能不光滑。

### 测试 3：32 步 BPTT 梯度检查

让 Koopman 参与完整的 32 步 Imagination。

检查：

∥∇θJ^∥\|\nabla_\theta\hat J\|

是否出现：

- NaN；
- Inf；
- 梯度突然爆炸；
- 梯度长期接近零；
- 明显异常的动作输出。

同时与原始 MLP 的梯度范数进行对照。

但梯度范数接近并不意味着梯度方向正确，因此还需要比较梯度方向的一致性。

### 测试 4：真实仿真器中的策略表现

这是最终判断标准。

必须在真实 MuJoCo Warp 环境中评估，而不能只看 Koopman 想象轨迹中的 Return。

记录：

RsimR_{\mathrm{sim}}

而不是只记录：

R^Koopman\hat R_{\mathrm{Koopman}}

如果 Koopman 想象回报不断提高，但真实仿真回报下降，说明可能发生了模型利用问题。

------

## 七、建议的实验顺序

我建议不要一次性实现所有研究想法，而是按照下面顺序逐步推进：

| 顺序 | 实验                            | 成功标准                     |
| ---- | ------------------------------- | ---------------------------- |
| 0    | 原始 MLP Baseline               | 已有结果可稳定复现           |
| 1    | Koopman 单步预测                | 输出维度、状态积分和梯度正确 |
| 2    | Koopman 八步预测                | 误差可接受、训练稳定         |
| 3    | Koopman 32 步预测               | 无明显发散                   |
| 4    | Koopman Jacobian 检查           | 与有限差分结果一致           |
| 5    | Koopman + BPTT                  | Actor 梯度可正常传播         |
| 6    | Koopman + Actor–Critic 联合训练 | 真实仿真 Return 能持续改善   |
| 7    | 与原始 MLP 对照                 | 同等预算下比较性能与成本     |
| 8    | 加入 Gradient Fidelity Loss     | 检查是否进一步改善策略表现   |

这里特别强调：

**不要在第一版就加入 Gradient Fidelity Loss。**

因为你首先需要知道，性能变化到底来自 Koopman 的结构，还是来自额外的梯度监督。

如果两个改动同时加入，实验结论会不清楚。

------

## 八、AutoDL A800 上如何安排？

你前面选择的 A800，如果确实是 80 GB 显存，适合开展这类实验。

不过我建议先使用 LAFAN1，而不是立即进行全量 BONES-SEED。

推荐初期配置：

| 参数                     | 建议起点                      |
| ------------------------ | ----------------------------- |
| 数据集                   | LAFAN1                        |
| Koopman Latent Dimension | 128                           |
| Bilinear Rank            | 8                             |
| World Model Window       | 8                             |
| Imagination Horizon      | 32                            |
| Encoder                  | 两层较小 MLP                  |
| Decoder                  | 两层较小 MLP                  |
| Actor / Critic           | 保持原项目                    |
| Reward / Termination     | 保持原项目                    |
| 数据收集                 | 使用原项目 Replay Buffer      |
| 优化方式                 | 先独立训练 WM，再接入联合训练 |

潜空间维度和低秩参数需要消融，不应直接当作最终配置。

例如 dz=128,r=8d_z=128,r=8 时：

完整双线性参数量=29×1282=475136\text{完整双线性参数量} = 29\times128^2 = 475136

低秩双线性参数量=29×2×128×8=59392\text{低秩双线性参数量} = 29\times2\times128\times8 = 59392

这里只计算双线性项，不包括 Encoder、Decoder、A、B 等其他参数。

**第一轮实验的目标不是立即超过 MLP，而是确认 Koopman 在这个复杂系统中能否稳定工作。**

------

## 九、一个必须提前处理的难点：潜状态与物理状态如何同步？

我认为这是实际接入时最容易出问题的地方。

SuperMimic 的 Actor 需要物理观测：

ot=gω(st,ξ)o_t=g_\omega(s_t,\xi)

Reward 和 Critic 也需要物理状态。

但是 Koopman 的动力学在潜空间中传播：

zt+1=K(zt,at)z_{t+1}=K(z_t,a_t)

因此每个时间步都必须得到：

s^t+1=Integrate⁡(s^t,D(zt+1))\hat s_{t+1} = \operatorname{Integrate} \left( \hat s_t,D(z_{t+1}) \right)

然后：

o^t+1=gω(s^t+1,ξ)\hat o_{t+1} = g_\omega(\hat s_{t+1},\xi)

再将 o^t+1\hat o_{t+1} 输入 Actor。

完整闭环为：

zt→K(⋅,at)zt+1zt+1→DΔt(st,Δt)→Integratest+1st+1→Observationot+1(ot+1,gt+1)→Actorat+1\boxed{ \begin{aligned} z_t &\xrightarrow{K(\cdot,a_t)} z_{t+1}\\ z_{t+1} &\xrightarrow{D} \Delta_t\\ (s_t,\Delta_t) &\xrightarrow{\mathrm{Integrate}} s_{t+1}\\ s_{t+1} &\xrightarrow{\mathrm{Observation}} o_{t+1}\\ (o_{t+1},g_{t+1}) &\xrightarrow{\mathrm{Actor}} a_{t+1} \end{aligned} }

这里存在两套状态：

1. 物理状态 sts_t；
2. Koopman 潜状态 ztz_t。

**两者必须始终保持一致。**

这也是为什么需要潜空间一致性损失。

此外，原项目的 Actor 具有观测历史，因此想象轨迹还必须同步维护历史观测和前一次动作。

不能因为引入 Koopman，就省略这些输入。

### 我特别建议做一个一致性诊断

在每个预测时间步计算：

etconsistency=∥zt−Eϕ(xt)∥\boxed{ e_t^{\mathrm{consistency}} = \left\| z_t-E_\phi(x_t) \right\| }

其中 xtx_t 是由当前预测物理状态构建的规范化特征。

如果连续传播的 ztz_t 与重新编码物理状态得到的潜变量差距越来越大，就说明模型内部出现了潜状态—物理状态不一致。

这会影响下一步预测，也可能影响 BPTT 的梯度。

------

## 十、最终建议：先做一个最小可行的 Koopman 版本

综合考虑你已经训练出较好指标、使用 AutoDL、希望进行科研实验，我建议：

**第一版只做四项改动：**

1. 新增 Koopman Dynamics 模块。
2. 在配置中加入 `world_model.type = "koopman"` 之类的模型选择项。
3. 在 World Model Trainer 中加入 Koopman Consistency Loss。
4. 增加 Koopman 的预测误差、Jacobian 和 BPTT 测试。

其余模块全部保持原样。

这样即使 Koopman 的性能不如 MLP，也不会影响你已有的训练结果，而且可以清楚判断失败原因。

### 还有一个重要的工程提醒

目前我核对到的是仓库公开的目录结构、README 和训练设计。**你的本地版本可能已经发生修改，尤其是你已经完成部分训练。**

因此，以上是基于公开工程结构的接入方案，而不是已经与你本地代码逐行对齐的补丁。

**下一步最合适的做法，是先检查你当前使用的这几个文件：**

- `src/supermimic/world_model/` 下的动力学模型实现；
- World Model Trainer；
- `src/supermimic/imagination/` 下的 Rollout 实现；
- `src/supermimic/config.py`。

**我建议优先实现的不是最复杂的 Contact-Aware Koopman，而是一个可以与原始 MLP 公平对照的 Low-Rank Bilinear Deep Koopman。** 先证明它在当前 SuperMimic 工程里具有可用的预测精度和 BPTT 梯度，再讨论更复杂的模型结构。
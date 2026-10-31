# 你的 Koopman-SuperMimic：代码审查与问题诊断

我已经检查了你公开仓库中的 Koopman 实现、世界模型损失、BPTT 接口，并找到了你提交的 **MLP 与 Koopman 独立训练实验结果**。

先给出一个明确判断：

**目前的结果不是 Koopman 比 MLP 略差，而是 Koopman 的长时域动力学预测已经出现严重失稳。在解决这一点以前，不建议继续投入大量 GPU 时间训练 Actor。**

## 一、先看仓库中真正记录的实验结果

你的仓库保存了两份很有价值的实验记录：

* [Koopman 世界模型训练结果](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/experiment/koopman-world-model/wm_koopman.json)
* [原始 MLP 世界模型训练结果](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/experiment/koopman-world-model/wm_mlp.json)

在第 3000 次更新时：

| 指标          | 原始 MLP | 你的 Koopman | 差异                |
| ----------- | ------ | ---------- | ----------------- |
| 1-step NSE  | 0.0385 | 0.2468     | Koopman 约高 6.4 倍  |
| 8-step NSE  | 0.0588 | 0.8037     | Koopman 约高 13.7 倍 |
| 32-step NSE | 0.2471 | 999,174    | Koopman 约高 404 万倍 |

32-step prediction error

MLP

# 0.247

Koopman

# 999,174

这些数字不是同一个模型训练初期的正常波动。你的 Koopman 从第 250 次更新到第 3000 次更新，单步预测逐渐改善，但 32 步误差长期维持在极高的水平，而且有剧烈波动。([github.com][1], [github.com][1], [github.com][2])

这意味着出现了一个很典型的现象：

$$
\boxed{
\text{短期预测逐渐变好}
\quad\text{但}\quad
\text{长时域自回归动力学不稳定}
}
$$

对 SuperMimic 而言尤其致命，因为 Actor 的 BPTT 正好需要沿着世界模型预测 32 步。

**我目前将问题的优先级排为：潜空间稳定性与一致性 → 损失函数约束 → 训练和评估协议 → 模型容量与结构 → Actor 联合训练。**

下面会具体指出对应文件、代码和修复思路。

## 二、问题一：Koopman 潜空间可能存在明显的动力学不稳定

(P0 · 最高优先级)

对应文件：[src/supermimic/world\_model/koopman.py](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/src/supermimic/world_model/koopman.py)

你实现的是低秩双线性模型：

$$
z_{t+1}
=
Az_t+Ba_t+
\sum_i a_{t,i}U_iV_i^\top z_t
$$

我们可以把它整理成：

$$
\boxed{
z_{t+1}=M_tz_t+Ba_t
}
$$

其中：

$$
M_t=A+\sum_i a_{t,i}U_iV_i^\top
$$

这意味着连续多步预测时，潜状态会受到一系列矩阵乘积的影响：

$$
z_H=
M_{H-1}\cdots M_0z_0
+
\text{控制输入贡献}
$$

### 代码中的具体风险

你的初始化采用：

$$
A=0.99I+0.01N
$$

其中 \(N\) 是随机矩阵。

这个初始化**不保证**：

$$
\rho(A)<1
$$

我用同样形式生成了几组 128 维随机矩阵，得到的初始谱半径约在 1.09～1.11 之间。

这只是独立数值模拟，不代表你保存的 Koopman Checkpoint 的实际谱半径，但它足以说明这种初始化没有稳定性保证。

更重要的是，哪怕约束了 \(A\)，也不能保证：

$$
M_t=A+\sum_i a_{t,i}N_i
$$

稳定。

因为动作本身会改变状态传播矩阵。

### 为什么它可能解释你的结果？

短时间预测可能仍然合理，但长时域误差沿矩阵乘积放大：

$$
\|\delta z_H\|
\lesssim
\left(
\prod_{t=0}^{H-1}\|M_t\|
\right)
\|\delta z_0\|
+\text{中间误差}
$$

当某些方向持续被放大，32 步预测就可能发生严重漂移。

**这与你实验中“1 步、8 步逐渐改善，但 32 步极端发散”的现象高度吻合。**

不过，目前公开实验日志没有记录训练后 \(A\)、\(M_t\) 的谱或奇异值。因此，我把它判定为**强烈怀疑的主要机制**，而不是已经证实的唯一原因。([github.com][3])

### 我建议先做的检查

在同一批固定轨迹上记录：

$$
\rho(A)
$$

$$
\sigma_{\max}(M_t)
$$

$$
\|z_t\|_2,\quad t=0,\ldots,32
$$

以及：

$$
\left\|
M_{31}M_{30}\cdots M_0
\right\|_2
$$

注意，单独看特征值还不够，因为非正规矩阵可能在谱半径不大的情况下仍然发生显著的暂态放大。

**修复建议：**

先对 \(A\) 进行保守初始化，并限制动作相关双线性项的增益；增加针对实际采样动作的多步 Jacobian 范数监控。

但我不建议简单地把所有特征值强制压到远小于 1。机器人位置、姿态和速度存在需要保持的动态模式，过度收缩会造成预测衰减和系统性偏差。

---

## 三、问题二：连续潜状态与物理状态没有被充分约束为一致

(P0 · 很可能是核心结构问题)

对应文件：

* [world\_model.py](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/src/supermimic/world_model/world_model.py)
* [imagination/rollout.py](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/src/supermimic/imagination/rollout.py)

这是我认为最值得你优先修改的设计。

你的 Koopman 模型同时维护两套状态：

**第一套：潜状态**

$$
z_{t+1}=K(z_t,a_t)
$$

**第二套：物理状态**

$$
\hat s_{t+1}
=
\operatorname{Integrate}
\left(
\hat s_t,D(z_{t+1})
\right)
$$

潜状态在连续传播，而物理状态使用 Decoder 输出的增量积分。

理论上，你希望：

$$
\boxed{
z_t\approx E(\operatorname{Canonical}(\hat s_t),\xi)
}
$$

但是，在连续潜状态模式下，这个关系并没有在每一步被强制满足。

### 为什么这是一个大问题？

假设某一步出现：

$$
z_t
\neq
E(\operatorname{Canonical}(\hat s_t),\xi)
$$

此时可能发生：

* Actor 根据物理状态 \(\hat s_t\) 生成动作；
* Koopman 根据已经漂移的潜状态 \(z_t\) 预测未来；
* Decoder 再将潜状态转成物理状态增量。

也就是：

$$
\boxed{
\underbrace{\hat s_t}_{\text{Actor 看见的状态}}
\neq
\underbrace{D_{\mathrm{state}}(z_t)}_{\text{潜状态隐含的状态}}
}
$$

这里 \(D_{\mathrm{state}}\) 表示潜状态对应的物理状态重建关系，并不等同于你现有的状态增量 Decoder。

结果就是 Actor 和动力学模型可能在使用**不一致的状态表示**。

### 代码中已经记录了这个问题，但没有真正优化它

你在 `window_loss_koopman()` 中计算了：

$$
\mathcal L_{\mathrm{cons}}
=
\|z_t-E(\operatorname{Canonical}(\hat s_t),\xi)\|^2
$$

但从当前损失组合来看，`consistency` 只是记录的指标，**没有加入最终优化目标**。

这非常关键。

更值得注意的是，你的训练记录中，`consistency` 从训练初期约 0.042 增加到末期约 2.835。

这说明两种表示之间的偏离在增长，与你的长时域预测恶化具有一致性。

不过，相关性不能直接证明因果关系。我们还需要消融实验确认。([github.com][4], [github.com][1], [github.com][1])

### 我建议马上做一个低成本判别实验

你的代码已经支持两种模式：

* `latent`：潜状态连续传播；
* `reencode`：每一步根据预测物理状态重新编码。

在**同一份已经训练好的 Koopman 权重**上，不重新训练，分别评估两种模式的 32 步误差。

| 结果                       | 更可能说明什么                |
| ------------------------ | ---------------------- |
| `reencode` 显著好于 `latent` | 潜状态—物理状态不一致可能是主要问题     |
| 两者都严重发散                  | 还存在单步模型精度、积分或模型容量方面的问题 |
| `latent` 好于 `reencode`   | 编码器在预测状态分布上可能不稳定       |
| 两者都稳定但 Actor 差           | 应转向 Jacobian 和策略梯度质量   |

这是我认为**目前性价比最高的一项实验**。

要特别注意，这项测试仅用于定位问题。不能因为 `reencode` 表现较好，就声称已经得到了严格的 Koopman 潜空间模型。

---

## 四、问题三：损失权重实际上没有提供足够强的潜空间约束

(P0 · 代码中可以确认)

你的总损失大致是：

$$
\mathcal L
=
\mathcal L_{\mathrm{state}}
+
\mathcal L_{\mathrm{termination}}
+
0.1\mathcal L_{\mathrm{lift}}
+
0.1\mathcal L_{\mathrm{rec}}
$$

但这里存在一个重要的数值尺度问题。

在第 3000 次更新时，你记录：

$$
\mathcal L_{\mathrm{state}}\approx24.24
$$

$$
\mathcal L_{\mathrm{lift}}\approx2.879
$$

$$
\mathcal L_{\mathrm{rec}}\approx0.0413
$$

因此，进入总损失的提升一致性项实际约为：

$$
0.1\times2.879=0.288
$$

只有状态损失的：

$$
\frac{0.288}{24.24}\approx1.19\%
$$

这里的 1.19% 是损失**数值贡献比例**，不是梯度贡献比例。不同损失对各参数的梯度范数可能不同，不能仅凭这个百分比认定约束必然不足。

但考虑到你已经观测到潜状态不一致和长时域失稳，我认为目前的损失尺度至少需要认真检查。([github.com][4], [github.com][1])

### 更深层的问题：重构损失的目标

目前重构损失所用的特征来自：

$$
x_t=\operatorname{Canonical}(\hat s_t)
$$

而这里的 \(\hat s_t\) 是模型预测出来的物理状态。

也就是说，你在训练：

$$
R(z_t)\approx
\operatorname{Canonical}(\hat s_t)
$$

这可以促进模型内部的一致性，但它并不能单独保证：

$$
R(E(\operatorname{Canonical}(s_t^{\mathrm{real}})))
\approx
\operatorname{Canonical}(s_t^{\mathrm{real}})
$$

**我建议将这两个目标分开。**

#### A. 真实状态重构

$$
\boxed{
\mathcal L_{\mathrm{rec,true}}
=
\left\|
R(E(x_t^{\mathrm{real}},\xi))
-
x_t^{\mathrm{real}}
\right\|^2
}
$$

#### B. 连续潜状态与预测物理状态一致性

$$
\boxed{
\mathcal L_{\mathrm{cons,pred}}
=
\left\|
z_t-
E(x_t^{\mathrm{pred}},\xi)
\right\|^2
}
$$

#### C. 真实转移的 Koopman 一致性

$$
\boxed{
\mathcal L_{\mathrm{lift}}
=
\left\|
K(E(x_t^{\mathrm{real}},\xi),a_t)
-
\operatorname{sg}\big(E_{\mathrm{target}}(x_{t+1}^{\mathrm{real}},\xi)\big)
\right\|^2
}
$$

其中 \(\operatorname{sg}\) 表示停止梯度。

我倾向于先采用一个缓慢更新的 EMA Target Encoder，让训练目标不要随在线编码器一起快速漂移。

它不是必需的修复，但值得作为稳定性消融项。

另外，还需要避免潜空间表示坍缩：即所有状态都被编码成近似相同的 \(z\)。

因此不能仅靠降低 `lift_loss` 判断 Koopman 是否学得更好了。

### 建议的总损失

$$
\boxed{
\begin{aligned}
\mathcal L={}&
\mathcal L_{\mathrm{WM}}
+\lambda_1\mathcal L_{\mathrm{lift}}\\
&+\lambda_2\mathcal L_{\mathrm{rec,true}}
+\lambda_3\mathcal L_{\mathrm{cons,pred}}
\end{aligned}
}
$$

具体权重不要立即全部设为 1，而应先测量每项损失对 Encoder、Dynamics 和 Decoder 的**梯度范数**，再调整。

---

## 五、问题四：物理上下文 \(\xi\) 只在初始编码时注入

(P1 · 高风险设计，但不是确定的错误)

你的 Encoder 接收状态特征和物理上下文：

$$
z_0=E(x_0,\xi)
$$

但是连续潜状态传播使用：

$$
z_{t+1}=K(z_t,a_t)
$$

传播函数本身没有重新接收 \(\xi\)。

因此，在 32 步连续传播中，模型必须把摩擦、质量和关节校准等物理参数信息**一直保存在潜状态里**。

理论上，网络可以学会这样做。

但当前模型没有显式约束某些潜变量维度必须保持静态物理参数。

这与原始 MLP 每一步都可以直接接收 \(\xi\) 的情况不同。([github.com][3], [github.com][4])

### 我建议怎么改？

最稳妥的方法不是让整个潜状态都负责保存物理上下文，而是显式分离：

$$
z_t=
\begin{bmatrix}
z_t^{\mathrm{dyn}}\\
\xi
\end{bmatrix}
$$

其中 \(\xi\) 不参与自由演化。

或者改成：

$$
\boxed{
z_{t+1}
=
A(\xi)z_t+B(\xi)a_t
+
\sum_i a_{t,i}N_i(\xi)z_t
}
$$

第二种更加灵活，但复杂度明显提高。

**对于第一版修复，我更推荐在每一步传播时显式输入归一化的 \(\xi\)，而不是直接学习复杂的参数化矩阵。**

这可以让你先验证“物理上下文在长时域中丢失”是不是问题来源。

---

## 六、问题五：模型初始化使初期的动作敏感度非常弱

(P1 · 可以从代码结构确认)

你的 Koopman 模型采用了几个比较保守的初始化：

* Encoder 最后一层较小的增益；
* \(B\) 初始化为零；
* 双线性低秩矩阵 \(U,V\) 初始化得很小；
* Decoder 最后一层初始化为零。

这些设置分别看都有一定道理。

例如，Decoder 输出零增量可以使世界模型在训练初期接近一个简单基线。

但组合起来会有一个问题：

**在 Decoder 最后一层为零的初始时刻，输出的状态增量对动作的导数也是零。**

即：

$$
\left.
\frac{\partial \hat\Delta}{\partial a}
\right|_{\mathrm{init}}
=0
$$

这不意味着所有参数都完全无法训练。

因为 Decoder 会首先从状态预测误差获得梯度，Encoder、Koopman 矩阵也可能通过辅助损失更新。

但在训练最初阶段，真实状态预测误差传向 Koopman 动作相关参数的路径受到明显限制。

而且即使 Decoder 开始更新，双线性项的动作影响也可能非常小：

$$
N_i z=U_iV_i^\top z
$$

当 \(U_i,V_i,z\) 的初始尺度都很小时，这一项尤其弱。

你的测试文件也显式涉及零初始化导致初始动作梯度为零的情况。([github.com][3], [github.com][5])

### 怎么改？

我建议不要一下子删除所有稳定性初始化。

可以分阶段测试：

1. 保留原有初始化，观察前期 \(\|\partial\hat\Delta/\partial a\|\) 的增长曲线。
2. 仅将 Decoder 输出层改成小幅非零初始化。
3. 再测试 \(B\) 的小幅非零初始化。
4. 保持其他配置不变，比较单步学习速度和 32 步稳定性。

这样能够确认初始化是否真正造成训练瓶颈，而不是凭经验一起修改。

---

## 七、问题六：MLP 与 Koopman 的模型容量差异太大

(P1 · 实验对照需要修正)

公开实验记录中：

原始 MLP 参数量

# 8.82M

Koopman 参数量

# 0.443M

MLP 的参数量约为 Koopman 的 **19.9 倍**。([github.com][1], [github.com][2])

所以，目前实验其实同时改变了两个因素：

$$
\boxed{
\text{模型结构}
+
\text{模型容量}
}
$$

这会使结论变得不清楚。

你不能仅从目前的结果推断：

> Koopman 结构不适合 SuperMimic。

也可能是：

> 当前容量过小的 Koopman 模型不适合这套复杂运动数据。

当然，容量差异也不能解释或合理化如此严重的 32 步失稳；后者仍需要优先解决。

### 建议增加两组对照

| 实验                           | 目的                   |
| ---------------------------- | -------------------- |
| 小型 MLP（与 Koopman 参数量接近）      | 排除主要由模型容量导致的性能差异     |
| 扩大 Encoder/Decoder 的 Koopman | 检验 Koopman 是否存在明显欠拟合 |

还要增加计算量对照。

因为参数量相近，不代表 GPU 训练成本相同。

---

## 八、问题七：单元测试中存在一个重要的 Jacobian 检查缺陷

(P1 · 可确认的测试代码问题)

对应文件：[tests/test\_koopman.py](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/tests/test_koopman.py)

你在一个方向导数测试中，对整个 Batch 的模型输出求和，得到一个标量函数：

$$
f(a)=\sum_{b,j}F(a)_{b,j}
$$

随后用有限差分计算：

$$
\frac{f(a+\epsilon v)-f(a-\epsilon v)}
{2\epsilon}
$$

这是**整个 Batch 的标量方向导数**。

但测试的自动微分侧采用了类似：

$$
(g\odot v).\operatorname{sum}(-1)
$$

它保留了 Batch 维度，因此得到的是每个样本各自的方向导数。

也就是说，目前比较的是：

$$
\underbrace{\text{每个样本的方向导数}}_{[B]}
$$

与：

$$
\underbrace{\text{整个 Batch 的方向导数}}_{\text{标量}}
$$

二者不是同一个数学量。

如果要比较总和，应该对全部维度求和：

$$
\boxed{
D_{\mathrm{AD}}=(g\odot v).\operatorname{sum}()
}
$$

然后才与标量有限差分比较。

还有一个更隐蔽的问题：

如果在 Decoder 输出全零的初始化时测试，那么：

$$
f(a+\epsilon v)=f(a-\epsilon v)
$$

自动微分和有限差分都可能得到零。

**即使 Jacobian 测试设计有问题，也可能因为两边都为零而通过。**

所以，我建议：

* 修复 Batch 维度上的求和；
* 使用一个已经产生非零动作敏感度的模型进行测试；
* 分别测试单步 `forward()` 和连续 `step_z()`；
* 对 8 步、32 步闭环做方向导数检查；
* 增加与真实仿真器方向导数的对照。

这属于**测试可靠性问题**。目前没有证据说明 PyTorch Autograd 或你实际的双线性矩阵计算本身是错误的。([github.com][5], [github.com][5])

---

## 九、问题八：现在的实验还不是严格的公平对照

(P1 · 实验协议问题)

对应文件：[experiment/koopman-world-model/train\_wm\_only.py](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/experiment/koopman-world-model/train_wm_only.py)

这里有两个值得修改的地方。

### 1. 所谓验证误差来自训练 Replay Buffer

脚本记录的 32 步评估，直接从当前 Replay Buffer 中抽取窗口。

因此它不是严格意义上的独立测试集。

这并不妨碍你发现长时域失稳——毕竟连训练分布上的预测都已经严重发散。

但如果未来要比较 Koopman 的泛化能力，就必须建立独立验证集。

我建议：

$$
\mathcal D
=
\mathcal D_{\mathrm{train}}
\cup
\mathcal D_{\mathrm{val}}
\cup
\mathcal D_{\mathrm{test}}
$$

最好按运动片段或独立轨迹划分，而不是将同一段轨迹中相邻、重叠的窗口随机放入训练集和测试集。

### 2. 两次独立收集数据并不等于完全相同的数据

当前实验脚本会为不同模型运行各自的数据收集过程。

即使配置和随机种子相同，也需要确认最终得到的是完全相同的状态—动作—下一状态序列。

特别是在并行物理仿真和随机重置的情况下，不应默认所有数据逐项相同。

我建议：

**先保存一份固定的数据集，再让 MLP 和 Koopman 从完全相同的数据训练。**

同时冻结固定的验证窗口，保证每次检查时使用同一组样本。

另外，脚本虽然暴露了 `window` 参数，但当前 World Model Trainer 的构建没有把它完整传入相应配置。默认使用 8 步时不一定造成错误，但如果你修改命令行窗口长度，可能不会得到预期效果。

这是一个可以直接修正的小型配置问题。([github.com][6], [github.com][6])

---

## 十、问题九：如果你通过 `resume-skip wm` 接入新模型，要特别小心

(P1 · 取决于你实际采用的训练命令)

对应文件：[training/loop.py](https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/src/supermimic/training/loop.py)

我注意到项目支持跳过旧 World Model 权重，从既有 Checkpoint 恢复其他模块。

这对于从 MLP 切换到 Koopman 很方便，但可能产生危险的训练状态：

$$
\boxed{
\text{已经训练好的 Actor}
+
\text{尚未训练好的 Koopman}
}
$$

如果立即开始 Actor 的 BPTT，Actor 就会使用一个预测能力很差的新世界模型更新参数。

即使之后 Koopman 慢慢学习正确动力学，Actor 也可能已经朝着错误方向优化。

另外，恢复旧的总训练迭代数可能让学习率调度处于训练后期，而新的 Koopman 实际上才刚开始训练。

**我没有证据确认你实际采用了这种恢复方式，所以这是一项条件性风险，而不是已经确认的实验错误。**([github.com][7])

推荐采用：

$$
\boxed{
\text{预训练 Koopman}
\rightarrow
\text{验证长时域预测}
\rightarrow
\text{恢复 Actor/Critic}
\rightarrow
\text{恢复联合训练}
}
$$

其中 Koopman 的学习率调度应独立于旧 Actor 的训练进度。

---

## 十一、我建议你按照这个顺序修复，而不是同时修改所有东西

**第一阶段：定位 32 步失稳**

固定现有 Checkpoint，比较 `latent` 与 `reencode` 的 32 步误差，同时记录 \(\|z_t\|\)、\(\|s_t\|\)、\(\sigma_{\max}(M_t)\) 和各状态通道误差。

这一阶段先不要继续训练 Actor。

---

**第二阶段：修复潜空间约束**

增加真实状态重构损失、连续潜状态一致性损失；检查各损失的梯度尺度，并加入对 \(M_t\) 长时域放大的监控或温和约束。

---

**第三阶段：修复模型表示与初始化**

显式提供每一步的物理上下文 \(\xi\)；测试 Decoder 与 \(B\) 的非零初始化；必要时扩大潜空间或网络容量。

---

**第四阶段：建立公平的 MLP 对照**

固定训练数据、独立验证轨迹，同时比较容量匹配的 MLP 与 Koopman，排除数据和参数量的混杂因素。

---

**第五阶段：恢复 BPTT**

只有世界模型的 8 步与 32 步预测达到可接受水平后，再检查策略梯度方向，最后恢复 Actor 联合训练。

### 一个我特别推荐的最小消融矩阵

| 版本 | 改动                 | 要验证的假设           |
| -- | ------------------ | ---------------- |
| K0 | 你目前的 Koopman       | 原始基准             |
| K1 | K0 + `reencode` 评估 | 是否主要是潜状态漂移       |
| K2 | K0 + 真实状态重构及一致性损失  | 是否主要缺乏闭环一致性      |
| K3 | K2 + 稳定性正则         | 是否主要是动力学矩阵导致误差放大 |
| K4 | K3 + 每步显式 \(\xi\)  | 是否存在物理上下文丢失      |
| K5 | K4 + 更大模型容量        | 是否存在明显欠拟合        |

一定要逐步比较，不要一次性把 K1～K5 的所有修改打进去，否则很难解释究竟是哪项改动有效。

---

## 十二、是否应该继续坚持 Bilinear Koopman？

这里我的判断比较保守。

你当前的动力学形式是：

$$
z_{t+1}
=
Az_t+Ba_t+\sum_i a_{t,i}N_i z_t
$$

但机器人仿真动力学包含高度非线性的关节运动、执行器响应和复杂接触切换。

即使修复了数值稳定性与训练损失，**一个全局固定的双线性潜空间模型也不一定具有足够的表达能力**。

尤其是你的 Decoder 输出状态增量，而不是直接重建下一状态，这使潜空间模型与原始 Koopman 理论中的闭合观测函数演化仍然存在距离。

因此，我建议给当前 Bilinear Koopman 一个明确的实验边界：

**如果经过潜状态一致性、稳定性和容量匹配后，它仍然无法达到 MLP 的可用预测精度，那么不应该无限增加复杂度来挽救这个架构。**

下一步可以考虑两条路线：

**路线 A：物理状态锚定的 Koopman。**

将部分物理状态特征显式保留在提升向量中：

$$
z_t=
\begin{bmatrix}
x_t\\
\phi(x_t)
\end{bmatrix}
$$

并直接监督潜空间中的物理特征分量，减少 Decoder 和状态积分之间的表示脱节。

但姿态、坐标系和接触信息仍需谨慎处理。

**路线 B：MLP + Koopman 混合动力学。**

保留你已经证明有效的 MLP 作为基线：

$$
\hat\Delta_t
=
f_{\mathrm{MLP}}(x_t,a_t,\xi)
+
\alpha f_{\mathrm{Koopman}}(x_t,a_t,\xi)
$$

让 Koopman 只学习结构化的补充动力学，先通过蒸馏或残差学习验证它是否确实提供有用信息。

这条路线可能更稳妥，但它不再是纯粹的 Koopman 世界模型。科研写作中应明确区分。

---

## 十三、最终诊断与优先级

| 问题                    | 证据状态         | 严重程度   | 我的建议                     |
| --------------------- | ------------ | ------ | ------------------------ |
| 32 步预测极端失稳            | **实验数据已证实**  | 极高     | 暂停 Actor 联合训练            |
| 潜状态与物理状态持续分离          | 结构风险明确，指标支持  | 极高     | 优先对比 `latent`/`reencode` |
| `consistency` 未加入优化损失 | **代码已确认**    | 高      | 增加对应损失并消融                |
| \(A\) 初始化不能保证稳定       | **代码与数学已确认** | 高      | 检查实际谱和矩阵乘积               |
| 重构损失没有直接锚定真实状态        | **代码已确认**    | 高      | 增加真实状态重构                 |
| \(\xi\) 只在初始编码中注入     | **代码已确认**    | 中高     | 测试每步条件化                  |
| 初始动作敏感度为零             | **初始化结构已确认** | 中      | 做非零初始化消融                 |
| Koopman 容量显著小于 MLP    | **实验数据已确认**  | 中高     | 增加容量匹配实验                 |
| Jacobian 测试维度不匹配      | **测试代码已确认**  | 中高     | 修复测试，排除虚假通过              |
| 缺乏严格独立验证集             | **脚本已确认**    | 中      | 固定数据、独立测试                |
| Checkpoint 切换可能导致错误训练 | 取决于实际命令      | 高（若发生） | 先预训练 WM 再联训              |

### 我认为最值得立刻执行的三件事

**第一，拿你现有的 Koopman Checkpoint，直接做 `latent` 和 `reencode` 的同权重 32 步对比。** 这很可能快速定位主要失稳机制。

**第二，把 `consistency` 从纯监控指标变成真正的优化约束，并增加真实状态重构目标。** 不过不要直接使用很大的权重，应先检查分项梯度范数。

**第三，检查实际训练后 \(A\) 和 \(M_t\) 的谱、奇异值以及潜状态范数随预测步数的变化。** 这能够判断是否存在动力学矩阵放大，而不是仅仅猜测。

我目前没有在你的 AutoDL 环境实际运行训练，也没有拿到完整 Checkpoint，因此尚不能判断上述机制各自贡献了多少误差。公开记录只足以确认 World Model 的严重长时域失稳，**还不足以确认你完整 Koopman-Actor 训练失败的全部原因**。

但从代码和已有结果看，我认为当前的首要问题**不是 Koopman 理论不适合机器人，而是这套具体实现尚未建立稳定、闭合的潜状态—物理状态演化关系**。

在证明这一关系可以稳定维持 32 步以前，继续增加 Actor 训练量、调整 Critic 或奖励函数，恐怕都不是最有效的方向。

[1]: https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/experiment/koopman-world-model/wm_koopman.json "SuperMimic-Reproduced/experiment/koopman-world-model/wm_koopman.json at main · bright-moon-67/SuperMimic-Reproduced · GitHub"
[2]: https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/experiment/koopman-world-model/wm_mlp.json "SuperMimic-Reproduced/experiment/koopman-world-model/wm_mlp.json at main · bright-moon-67/SuperMimic-Reproduced · GitHub"
[3]: https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/src/supermimic/world_model/koopman.py "SuperMimic-Reproduced/src/supermimic/world_model/koopman.py at main · bright-moon-67/SuperMimic-Reproduced · GitHub"
[4]: https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/src/supermimic/world_model/world_model.py "SuperMimic-Reproduced/src/supermimic/world_model/world_model.py at main · bright-moon-67/SuperMimic-Reproduced · GitHub"
[5]: https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/tests/test_koopman.py "SuperMimic-Reproduced/tests/test_koopman.py at main · bright-moon-67/SuperMimic-Reproduced · GitHub"
[6]: https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/experiment/koopman-world-model/train_wm_only.py "SuperMimic-Reproduced/experiment/koopman-world-model/train_wm_only.py at main · bright-moon-67/SuperMimic-Reproduced · GitHub"
[7]: https://github.com/bright-moon-67/SuperMimic-Reproduced/blob/main/src/supermimic/training/loop.py "SuperMimic-Reproduced/src/supermimic/training/loop.py at main · bright-moon-67/SuperMimic-Reproduced · GitHub"


Prediction errors at update 3000

Log-scaled presentation is not used; the enormous Koopman 32-step error is shown separately below.

horizon	koopman	mlp
1 step	0.24680415	0.03851255
8 steps	0.80372524	0.05879686
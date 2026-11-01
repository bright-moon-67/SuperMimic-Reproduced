# Koopman 单步动作 Jacobian 对照

脚本：`scripts/diag_gradient.py`

对同一个 `(s, a)`，比较 one-step 状态映射的动作方向导数：

$$
J_{\mathrm{wm}}v \approx \frac{F_{\mathrm{wm}}(s,a+\epsilon v)-F_{\mathrm{wm}}(s,a-\epsilon v)}{2\epsilon}
$$

$$
J_{\mathrm{sim}}v \approx \frac{F_{\mathrm{sim}}(s,a+\epsilon v)-F_{\mathrm{sim}}(s,a-\epsilon v)}{2\epsilon}
$$

真实仿真使用 MuJoCo/Warp，通过 `StateIO.write` 恢复同一状态，并通过
`SimContext.write` 恢复数据中的 `xi`（摩擦、encoder bias、COM offset）；
动作噪声关闭。这是**单步**诊断，不是多步 BPTT 或策略梯度诊断。

## 结果

| checkpoint | valid directions | cosine mean | cosine min | relative L2 | magnitude ratio |
|---|---:|---:|---:|---:|---:|
| state small（旧） | 4 | 0.919 | 0.887 | 0.634 | 0.422 |
| state large（gate 1000） | 32 | **0.935** | 0.866 | **0.413** | 0.720 |

解释：

- 方向余弦约 0.93，说明动作对状态变化的主要方向基本一致；
- relative L2 约 0.41，幅值比约 0.72，说明 Koopman 的动作敏感度仍偏小；
- 该结果已在恢复 `xi` 域参数后重新测量，但仍只覆盖 gate 1000 checkpoint；
- 这只能说明单步方向一致性，不能直接证明多步 BPTT 或策略梯度正确。

原始 JSON：

- `experiment/koopman-world-model/diag_gradient_state_large.json`
- small 测试结果在 `/tmp/diag_gradient_test.json`

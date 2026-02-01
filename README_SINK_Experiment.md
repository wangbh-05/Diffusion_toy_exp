# Diffusion Sink Point 实验分析指南

本文档用于解读 `sink_optimization.py` 生成的实验结果。该实验旨在探究扩散模型（Diffusion Model）内在的“流形吸附”特性，对比分析模型预测的**一阶方向（Score Field）**与能量最小化的**二阶方向（Sink Gradient）**之间的差异。

## 1. 核心概念

我们在训练好的 Baseline DDPM 模型上执行了 Test-time Optimization：
*   **初始状态**: 正常的 DDIM 采样结果 $x_{sample}$。虽然看起来像螺旋线，但并不完美位于流形中心。
*   **优化目标**: 寻找模型眼中“最真实”的点。即使得预测噪声模长最小化：
    $$x^* = \arg\min_x \| \epsilon_\theta(x, t^*) \|^2$$
    此处 $t^*$ 通常取一个小值（如 10），代表流形结构最清晰且数值较稳定的时刻。

## 2. 实验图表解读

### 图表 1: Sink 优化轨迹 (Sink Optimization Trajectory)
*   **文件名**: `1_sink_trajectory.png`
*   **内容**: 展示了点云在优化过程中的移动路径。
*   **图例**:
    *   **红点 (Initial)**: 原始扩散生成的点。它们可能略显松散，有一定的“厚度”。
    *   **蓝点 (Sink)**: 优化 $K$ 步后的终点。
    *   **灰线**: 移动轨迹。
*   **如何分析**:
    *   你看到的应该是一个**“吸附” (Projection)** 的过程。
    *   原本有厚度的螺旋带，会被迅速压缩成一条极细的线（脊线 Ridge）。这条线就是模型定义的**隐式流形 (Implicit Manifold)** 的确切位置。
    *   这证明了扩散模型不仅学到了分布，还学到了一个非常精确的距离场。

### 图表 2 & 3: 两种力场 (Score vs Gradient)
*   **文件名**:
    *   `2_score_field.png`: **Score Field** (预测场), $V = -\epsilon_\theta(x)$
    *   `3_gradient_field.png`: **Gradient Field** (优化场), $V = -\nabla_x \|\epsilon_\theta(x)\|^2$
*   **如何分析**:
    *   **Score Field**: 类似于“风向标”。它告诉粒子顺着概率密度增加的方向走（切向流动）。在流形上，它倾向于推动粒子沿着螺旋线延伸的方向移动（扩散随机游走）。
    *   **Gradient Field**: 类似于“重力场”。它告诉粒子往能量最低（噪声最小）的地方掉。它的方向几乎总是**垂直于流形**的。

### 图表 4: 场对比叠加图 (Field Comparison) —— **最为关键**
*   **文件名**: `4_field_comparison.png`
*   **内容**: 将上述两个场画在同一张图上进行角力。
    *   **绿色箭头**: Score (Diffusion)
    *   **紫色箭头**: Gradient (Sink)
*   **物理直觉验证**:
    1.  **在流形（螺旋线）附近**: 绿色和紫色箭头可能方向一致（都指向流形中心），或者绿色箭头带有沿着流形切向的分量。
    2.  **在空隙（Voids/两臂之间）**: 这是一个**分水岭 (Watershed)** 区域。
        *   **紫色箭头 (Gradient)** 应该展现出极强的**二分性**：在空隙中轴线上，一侧指向上臂，一侧指向下臂。它非常果断。
        *   **绿色箭头 (Score)** 在这里往往表现得“犹豫”或者平滑过渡，甚至可能指向错误的平均方向。
*   **结论**: Gradient Field 提供了比原始 Score Field 更强的**几里得流形纠正能力**。这解释了为什么 Test-time Optimization 可以进一步提升生成质量。

## 3. 下一步建议
如果你发现 `sink_optimization.py` 的效果很好，可以尝试：
*   调节 `--t_star` 参数（如 50 或 100），观察随着噪音增加，Sink Point 对应的流形是如何平滑/退化的。
*   调节 `--lr` 学习率，观察优化过程的稳定性。

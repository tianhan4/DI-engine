

#### 1. 双重不确定性估计与安全探索 (Uncertainty-Aware Exploration)

- **Epistemic Uncertainty (认知不确定性)：** 通过 **多头集成网络 (Ensemble Heads)** 实现。模型配置了多个独立的预测头（如 5 个），通过计算这几个头==预测结果的方差== (`logit_std`)，衡量“==模型对环境的不了解程度==”。这种==不确定性是由数据不足==引起的，随着模型在环境中不断探索并收集数据，这种不确定性会逐渐降低。代码中通过**多头网络（Ensemble Networks）**的预测分歧来量化它。
  **数学推导**：
  1. **多头预测**：模型包含 $E$ 个独立初始化的预测头（代码中设定 `ensemble_head_num=5`），给定状态 $s$ 和偏好权重 $w$，每个头 $e \in 1, 2, \dots, E$ 独立输出动作 $a$ 的预期效用预测值 $Q_e(s, a)$。
  2. **均值计算**：模型对动作 $a$ 的最终效用期望是所有头预测值的平均值：
    $$\mu(s, a) = \frac{1}{E} \sum_{e=1}^{E} Q_e(s, a)$$
  3. **标准差计算（认知不确定性）**：通过计算各个头预测值之间的标准差来衡量分歧程度。
    $$\sigma_{epistemic}(s, a) = \sqrt{\frac{1}{E} \sum_{e=1}^{E} (Q_e(s, a) - \mu(s, a))^2}$$
  *在代码中对应：* `logit_std = logits.std(dim=0, unbiased=False)`。
- **Aleatoric Uncertainty (偶然不确定性)：** ==通过预测高斯分布实现。==模型不仅预测指标的均值，还通过一个专门的网络分支 (`self._out_log_std`) ==预测指标对数标准差==，用来衡量“环境本身固有的随机性和风险”。比如某些动作必然带来高收益波动，这是环境机制决定的，给再多的数据也无法消除。
  **数学推导**：
  1. **指标方差预测**：网络针对每个动作 $a$，输出 $D$ 维（$D=4$）==具体指标的对数标准差 $\log \sigma_{m_i}(s, a)$==，其中 $i \in 1, \dots, D$。
  2. **还原标准差**：经过温度校准后，通过指数函数还原出每个指标的真实标准差：
    $$\sigma_{m_i}(s, a) = \exp(\log \sigma_{m_i}(s, a))$$
  3. **效用方差的线性组合**：在该 Bandit 设定中，总效用 $u$ 是动作产生的各项指标 $m_i$ 与用户偏好权重 $w_i$ 的内积：
    $$u(s, a) = \sum_{i=1}^{D} w_i \cdot m_i(s, a)$$
     假设各项指标的内在波动是相互独立的，根据方差的线性性质（常数乘以随机变量，其方差乘以常数的平方），总效用 $u$ 的方差为：
     $$Var(u) = \sum_{i=1}^{D} Var(w_i \cdot m_i) = \sum_{i=1}^{D} w_i^2 \cdot \sigma_{m_i}^2(s, a)$$
  4. **效用标准差（偶然不确定性）**：对上述方差开平方，并加上一个极小数 $\epsilon$（如 $10^{-8}$）防止梯度下溢：
    $$\sigma_{aleatoric}(s, a) = \sqrt{\sum_{i=1}^{D} (w_i \cdot \sigma_{m_i}(s, a))^2 + \epsilon}$$
- **动作打分公式 (Action Scoring)：在收集数据阶段，模型会将均值、认知不确定性和偶然不确定性三者结合起来，计算最终的决策分数（Score）。这一步实现了“在探索未知的同时规避风险”**。

  设 $\beta_{ucb}$ 为探索系数（`ucb_beta`），$\alpha_{risk}$ 为风险惩罚系数（`risk_alpha`），则动作的最终评分为：
  $$
  Score(s, a) = \mu(s, a) + \beta_{ucb} \cdot \sigma_{epistemic}(s, a) - \alpha_{risk} \cdot \sigma_{aleatoric}(s, a)
  $$

- **$\mu(s, a)$**：基础期望收益（贪心项）。
- **$+ \beta_{ucb} \cdot \sigma_{epistemic}(s, a)$**：**探索奖励 (UCB)**。模型越不懂的地方（多头分歧大），这项加分越高，鼓励模型去试错。
- **$- \alpha_{risk} \cdot \sigma_{aleatoric}(s, a)$**：**风险惩罚 (Risk Aversion)**。环境本身波动越大的地方，这项扣分越多，保护模型不去做极端危险的动作。模型最终会选择 $Score$ 最大的动作执行 (`greedy_action = score.argmax(dim=-1)`)。

#### 2. 指标监督学习与自适应校准 (Metrics Supervised Learning & Calibration)

模型引入了强监督信号：

- **多维指标预测：**模型配置中使用了 `gaussian_nll_huber` 作为指标的损失函数类型。结合了高斯负对数似然 (Gaussian NLL) 和 Huber 损失的混合函数，旨在同时优化方差预测并增强对异常值的鲁棒性。
  假设==真实的指标向量为 $y$==，模型预测的均值为 $\mu$ (`pred`)，模型预测的对数标准差为 $\log \sigma$ (`pred_log_std`)。
  - **方差计算**：预测的方差为 $V = \sigma^2 = \exp(2 \log \sigma)$。
  - **高斯 NLL 部分**：假设误差服从高斯分布，负对数似然损失定义为：
    $$L_{NLL} = \frac{(y - \mu)^2}{2V} + \log \sigma$$
  - **Huber 部分**：设绝对误差 $\Delta = |y - \mu|$，给定阈值 $\beta$ (`metrics_huber_beta`)，Huber 损失为：
    $$L_{Huber} = \begin{cases} \frac{\Delta^2}{2\beta}, & \text{if } \Delta < \beta \\ \Delta - \frac{\beta}{2}, & \text{otherwise} \end{cases}$$
  - **总损失**：最终的每个样本的损失是这两者的加权和：
    $$L_{total} = L_{NLL} + \lambda \cdot L_{Huber}$$
    其中 $\lambda$ 是 Huber 损失的权重 (`metrics_huber_weight`，默认为 0.1)。



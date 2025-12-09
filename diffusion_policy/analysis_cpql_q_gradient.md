# CPQL Q-Loss 梯度链实验分析

## 实验设置

- **任务**: LiftPegUpright-v1
- **算法**: CPQL (Consistency Policy Q-Learning)
- **变量**:
  - `alpha` ∈ {0.01, 0.05, 0.2} (Q-loss 权重)
  - `q_grad_mode` ∈ {single_step, last_5, whole_grad} (梯度链长度)

## 实验结果摘要

### 定性观察

| 现象 | 描述 |
|------|------|
| **Critic Loss** | 梯度链越长 / alpha 越大 → critic loss 越不稳定 (炸) |
| **Policy Loss** | 梯度链越长 / alpha 越大 → policy loss 负得越多 |
| **Q Mean** | 梯度链越长 / alpha 越大 → Q 值估计越高 |
| **Performance** | alpha 越小、梯度链越短 → success rate 越高 |

### 关键发现

**alpha=0.01 + single_step 表现最好** (success_once ≈ 0.32-0.42)，而更强的 Q-learning 信号反而损害性能。

---

## 问题诊断

### 1. Q 值过高估计 (Q Overestimation)

从图表可以看到:
- `q_mean` 随训练持续上升 (5 → 25+)
- 梯度链越长，Q 值上升越快
- alpha 越大，Q 值上升越快

**原因分析**:

在 offline RL 中，策略只能在有限的数据分布上训练。当策略生成 OOD (Out-of-Distribution) 动作时:

```
Q(s, a_generated) 可能被错误地高估
↓
策略梯度 ∇_θ Q(s, π_θ(s)) 将策略推向这些高估区域
↓
Q 网络继续高估这些区域
↓
恶性循环导致 Q 值爆炸
```

### 2. 梯度链长度的影响

**为什么更长的梯度链反而更差？**

```python
# 当前实现 (_sample_actions_with_grad)
for i in range(no_grad_steps, self.num_flow_steps):
    t = torch.full((B,), i * dt, device=device)
    v = self.velocity_net(x, t, global_cond=obs_cond)
    x = x + v * dt  # 梯度累积
```

假设 flow matching 有 10 步 (num_flow_steps=10):
- **whole_grad**: 梯度通过 10 次 `velocity_net` 调用累积
- **last_5**: 梯度通过 5 次调用累积
- **single_step**: 梯度只通过 1 次调用

更长的梯度链意味着:
1. **梯度幅度更大**: 链式法则导致梯度累乘
2. **更强的 Q 信号**: 策略更激进地追求高 Q 值
3. **更快偏离数据分布**: 策略生成更多 OOD 动作

### 3. 与原版 CPQL 的关键差异

查看原版 CPQL 代码 ([cccedric/cpql](https://github.com/cccedric/cpql)):

```python
# agents/ql_cm.py lines 151-163
new_action = self.diffusion.sample(model=self.actor, state=state)

q1_new_action, q2_new_action = self.critic(state, new_action)
if self.rl_type == "offline":
    if np.random.uniform() > 0.5:
        q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
    else:
        q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
```

原版 CPQL 使用了 **Q-loss 归一化技巧**:
```
q_loss = -Q1.mean() / |Q2|.mean().detach()  # 随机交替 Q1/Q2
```

这个技巧的作用:
1. **自适应缩放**: Q-loss 被当前 Q 值的尺度归一化
2. **防止 Q 值爆炸**: 当 Q 值增大时，loss 自动缩小
3. **稳定训练**: 保持 Q-loss 在合理范围内

**我们当前的实现**:
```python
q_loss = -q_value.mean()  # 直接取负均值，无归一化
```

---

## 根本原因分析

### Offline RL 的 Distribution Shift 问题

```
┌─────────────────────────────────────────────────────────────┐
│                    Offline RL 的困境                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  数据分布 D_data:    █████████████                          │
│                                                             │
│  策略分布 π_θ:       ░░░░░░░██████████░░░░░░░               │
│                              ↑                              │
│                         OOD 区域                            │
│                                                             │
│  Q 高估区域:                    ████████████                │
│                                  ↑                          │
│                           策略被吸引                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

在 offline RL 中:
1. Critic 只在数据分布 D_data 上被训练
2. 对于 OOD 动作，Q 值估计不可靠 (可能高估)
3. Q-learning 的 `max_a Q(s,a)` 会放大高估
4. 策略被吸引到高估区域，进一步偏离数据分布

### 为什么更强的 Q 信号更有害?

| 参数 | 效果 | 问题 |
|------|------|------|
| 更大 alpha | 更强 Q-loss 权重 | 更快偏离数据分布 |
| 更长梯度链 | 更大梯度幅度 | 更激进的策略更新 |
| 两者结合 | 最强 Q 信号 | 最快崩溃 |

---

## 为什么 BC-only 表现更好?

当 alpha=0.01 (近似纯 BC) 时表现最好，说明:

1. **数据质量足够好**: 演示数据已经包含了成功的策略
2. **BC 足够学习**: 模仿学习已经能达到不错的性能
3. **Q-learning 是噪声**: 在这个设置下，Q-learning 的信号是负面的

这揭示了一个重要问题: **Q-learning 在这个任务上可能是有害的**

---

## 与文献的对比

### Diffusion-QL 原版
- 使用 T=5 diffusion steps (我们用 10 flow steps)
- **完整梯度链** (所有步骤有梯度)
- 但有额外的保守估计技巧

### CPQL 原版
- 使用 **one-step sampling** (非 antmaze 任务)
- 有 **Q-loss 归一化**: `q_loss = -Q1 / |Q2|.detach()`
- 有 **gradient clipping**: `grad_norm=9.0` 等

### 我们的实现缺失
1. ❌ Q-loss 归一化
2. ❌ 保守 Q 估计 (CQL-style)
3. ❌ 足够强的正则化

---

## 建议的改进方向

### 1. 添加 Q-Loss 归一化 (推荐)

```python
# 参考原版 CPQL
if np.random.uniform() > 0.5:
    q_loss = -q1.mean() / q2.abs().mean().detach()
else:
    q_loss = -q2.mean() / q1.abs().mean().detach()
```

### 2. 添加保守 Q 估计 (CQL-style)

```python
# 对 OOD 动作惩罚
random_actions = torch.rand_like(actions) * 2 - 1
q_random = critic(obs, random_actions)
cql_loss = (q_random - q_data).mean()  # 惩罚随机动作的 Q 值
```

### 3. 使用更强的 BC 正则化

```python
# 增加 bc_weight，减少 alpha
bc_weight = 2.0  # 或更高
alpha = 0.001    # 或更低
```

### 4. Gradient Clipping

```python
# 限制 Q-loss 的梯度
torch.nn.utils.clip_grad_norm_(velocity_net.parameters(), max_norm=1.0)
```

### 5. Q 值裁剪

```python
# 限制 Q 值范围，防止爆炸
q_value = torch.clamp(q_value, -100, 100)
```

---

## 结论

当前实验结果表明:

1. **Q-learning 信号过强会损害性能** (alpha 越大越差)
2. **梯度链越长，Q 值过估越严重** (whole_grad 最差)
3. **缺乏原版 CPQL 的稳定化技巧** (Q-loss 归一化)

**核心问题**: 在 offline RL 设置下，未经约束的 Q-maximization 会导致策略偏离数据分布，产生 Q 值过估，最终损害性能。

**推荐**: 
1. 短期: 使用 `alpha=0.01, q_grad_mode=single_step`
2. 长期: 实现 Q-loss 归一化或 CQL-style 保守估计

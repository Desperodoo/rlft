# AWSC: Advantage-Weighted ShortCut Flow 项目总结

## 📋 项目概述

本项目实现了一个完整的 **Offline→Online RL 训练流程**，专为高维图像输入（RGB）和动作分块（Action Chunking）场景设计。核心算法是 **AWSC (Advantage-Weighted ShortCut Flow)**，结合了 Flow Matching 策略和优势加权（Advantage Weighting）机制。

### 为什么选择 AWSC？

在 **图像输入 + Action Chunking** 条件下，传统的 Q-Learning 类方案存在严重的稳定性问题：

| 方案 | 问题 |
|------|------|
| **直接 Q 最大化** (CPQL, DQL) | Policy 易跑到 OOD 区域，Critic 估计爆炸 |
| **Diffusion-QL** | Action chunking 下 Q-gradient 不稳定 |
| **SAC + Action Chunking** | 高维动作空间 (action_horizon × action_dim) 探索困难 |

因此，我们采用 **Advantage-Weighted BC** 的思路：
- Policy 保持在 demo 分布附近（不做 Q 最大化）
- Q 值仅用于加权 BC 样本
- Critic 在数据支撑区域学习，估计误差有界

---

## 🏗️ 三阶段训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: ShortCut Flow BC                     │
│  纯 BC 预训练，学习 demo 分布的 flow matching                      │
│  脚本: train_offline_rl.py --algorithm shortcut_flow             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                Stage 2: AW-ShortCut Flow Offline RL              │
│  离线 RL 微调，Q-weighted BC 初始化 Critic                        │
│  脚本: train_offline_rl.py --algorithm aw_shortcut_flow          │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Stage 3: AWSC Online RL (RLPD)                  │
│  在线 RL 微调，50% online + 50% offline 数据混合                   │
│  脚本: train_rlpd_online.py --algorithm awsc                     │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: ShortCut Flow BC
- **目标**：学习专家演示的动作分布
- **算法**：Flow Matching with ShortCut（局部 ODE 求解器近似）
- **输出**：预训练的 `velocity_net`

### Stage 2: AW-ShortCut Flow Offline RL
- **目标**：引入 Q 函数，在 demo 分布内进行优势加权
- **算法**：AWAC-style Q-weighted BC + SMDP Bellman Critic
- **关键**：使用 **EnsembleQNetwork** 确保与 Stage 3 兼容
- **输出**：微调的 `velocity_net` + 校准的 `critic`

### Stage 3: AWSC Online RL
- **目标**：环境交互，在线数据增强
- **算法**：RLPD (Reinforcement Learning with Prior Data)
- **数据混合**：`online_ratio=0.5`（50% 在线 + 50% 离线）
- **关键特性**：Policy-Critic 数据分离

---

## 🔧 核心算法组件

### 1. ShortCut Flow Policy

ShortCut Flow 是一种高效的 Flow Matching 策略，通过学习局部 ODE 求解器来加速采样：

```python
class ShortCutVelocityUNet1D(nn.Module):
    """
    Velocity network with step size conditioning.
    
    输入:
        - sample: (B, pred_horizon, action_dim) 当前噪声动作
        - timestep: (B,) 扩散时间步 t ∈ [0, 1]
        - step_size: (B,) 步长 d ∈ [0, 1]
        - global_cond: (B, obs_dim) 观测条件
        
    输出:
        - velocity: (B, pred_horizon, action_dim) 预测的速度场
    """
```

**推理过程**（ODE 积分）：
```python
x = torch.randn(B, pred_horizon, action_dim)  # 从噪声开始
dt = 1.0 / num_inference_steps

for i in range(num_inference_steps):
    t = torch.full((B,), i * dt)
    v = velocity_net(x, t, dt, obs_cond)
    x = x + dt * v  # Euler 积分
    
return x  # 生成的动作序列
```

### 2. EnsembleQNetwork

为了解决 Q 估计的方差问题，我们使用 **Ensemble Q-Network**：

```python
class EnsembleQNetwork(nn.Module):
    """
    可配置数量的 Q 网络集成。
    
    参数:
        num_qs: Q 网络数量（默认 10）
        num_min_qs: 子采样取最小的网络数（默认 2）
        
    关键方法:
        forward(actions, obs) → (num_qs, B, 1)  # 所有 Q 值
        get_min_q(actions, obs) → (B, 1)        # 保守估计
        get_mean_q(actions, obs) → (B, 1)       # 均值估计
    """
```

**RLPD 风格的保守 Q 估计**：
```python
def get_min_q(self, action_seq, obs_cond, random_subset=True):
    q_all = self.forward(action_seq, obs_cond)  # (num_qs, B, 1)
    
    if random_subset and self.num_min_qs < self.num_qs:
        # 随机选择 num_min_qs 个网络
        indices = torch.randperm(self.num_qs)[:self.num_min_qs]
        q_subset = q_all[indices]
    else:
        q_subset = q_all
    
    return q_subset.min(dim=0).values  # 取最小值
```

**为什么用 Ensemble？**
- 减少 Q 函数的过估计
- 随机子采样增加训练多样性
- 与 RLPD 原论文推荐的 `num_qs=10, num_min_qs=2` 一致

### 3. LayerNorm 稳定性

Q 网络中使用 **LayerNorm** 而非 BatchNorm：

```python
# EnsembleQNetwork 内部结构
for hidden_dim in hidden_dims:
    q_layers.extend([
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),  # 关键：稳定训练
        nn.Mish(),
    ])
```

**LayerNorm 的优势**：
- 不依赖 batch 统计量，小 batch size 下稳定
- 与 Transformer 架构一致（UNet 内部也用 GroupNorm）
- 防止 Q 值爆炸

### 4. Advantage Weighting

核心的优势加权机制（AWAC-style）：

```python
def _compute_advantage_weights(self, actions_for_q, obs_cond):
    with torch.no_grad():
        # 获取保守 Q 估计
        q_data = self.critic.get_min_q(actions_for_q, obs_cond, random_subset=True)
        
        # 计算优势: A(s,a) = Q(s,a) - V(s)
        baseline = q_data.mean()  # 用 batch 均值近似 V(s)
        advantage = q_data - baseline
        
        # 指数加权 + 裁剪
        weights = torch.clamp(
            torch.exp(self.beta * advantage),
            max=self.weight_clip
        )
        weights = weights / weights.mean()  # 归一化
        
    return weights
```

**关键参数**：
- `beta`: 温度参数，控制 Q 差异的敏感度（推荐 10~100）
- `weight_clip`: 防止权重过大（推荐 100~200）

---

## 🔄 SMDP Action Chunking

Action Chunking 将标准 MDP 转换为 **Semi-MDP (SMDP)**：

```
标准 MDP: s_t → a_t → r_t → s_{t+1}

Action Chunking SMDP:
s_t → [a_t, a_{t+1}, ..., a_{t+H-1}] → R_cum → s_{t+H}
      ─────────────────────────────
              act_horizon 步
```

**SMDP Bellman 方程**：
```python
# 累积奖励和折扣因子
R_cum = Σ_{i=0}^{H-1} γ^i * r_{t+i}
γ_H = γ^H if not done else 0

# TD Target
target_Q = R_cum + γ_H * min_q(s_{t+H}, π(s_{t+H}))
```

**关键维度**：
```python
obs_horizon = 2      # 观测历史长度
pred_horizon = 16    # 预测的动作序列长度
act_horizon = 8      # 实际执行的动作步数（用于 Q-learning）
```

---

## 🛡️ Policy-Critic 数据分离

在 Online RL 阶段，为了避免失败样本污染 Policy 训练：

```python
# 数据分离策略
if self.filter_policy_data and is_demo is not None:
    with torch.no_grad():
        q_values = self.critic.get_min_q(actions_for_q, obs_cond)
        baseline = q_values.mean()
        advantage = q_values - baseline
    
    # 保留: demo 样本 + 高 advantage 的在线样本
    keep_mask = is_demo | (advantage.squeeze() > self.advantage_threshold)
    
    if keep_mask.sum() > 0:
        # 使用过滤后的数据训练 Policy
        obs_features_filtered = obs_features[keep_mask]
        actions_filtered = actions[keep_mask]
        ...
```

**效果**：
- **Critic**: 使用所有数据，学习完整的 Q 函数
- **Policy**: 只使用高质量数据（demo + 成功探索）

---

## 📊 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `beta` | 100.0 | Advantage 温度，越大权重差异越明显 |
| `weight_clip` | 200.0 | 防止权重爆炸 |
| `num_qs` | 10 | Ensemble Q 网络数量 |
| `num_min_qs` | 2 | 保守估计的子采样数 |
| `gamma` | 0.9 | 折扣因子（action chunking 下用较小值） |
| `tau` | 0.005 | Target network 软更新系数 |
| `online_ratio` | 0.5 | 在线/离线数据混合比例 |
| `utd_ratio` | 8 | Update-to-Data ratio |
| `lr_actor` | 3e-4 | Policy 学习率 |
| `lr_critic` | 3e-4 | Critic 学习率 |

---

## 🚀 使用示例

### Stage 2: Offline RL 预训练
```bash
python train_offline_rl.py \
    --algorithm aw_shortcut_flow \
    --env_id LiftPegUpright-v1 \
    --obs_mode rgb \
    --demo_path demos/LiftPegUpright-v1.h5 \
    --use_ensemble_q \
    --num_qs 10 \
    --num_min_qs 2 \
    --beta 100.0 \
    --weight_clip 200.0 \
    --total_iters 100000
```

### Stage 3: Online RL 微调
```bash
python train_rlpd_online.py \
    --algorithm awsc \
    --env_id LiftPegUpright-v1 \
    --obs_mode rgb \
    --pretrained_path runs/awsc-offline/checkpoints/best.pt \
    --load_critic \
    --demo_path demos/LiftPegUpright-v1.h5 \
    --num_qs 10 \
    --num_min_qs 2 \
    --gamma 0.9 \
    --beta 100.0 \
    --utd_ratio 8 \
    --filter_policy_data \
    --total_timesteps 500000
```



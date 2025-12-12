# AW-ShortCut Flow Pipeline: 从Offline BC到Online RL的完整三阶段设计

## 🎯 核心目标与思路

### 为什么需要中间层？

直接从 ShortCut Flow BC 跳到 ReinFlow Online RL 会面临两个问题：

1. **Critic 未校准**：BC 阶段没有 critic，突然加入 RL 时，critic 的值函数估计可能不可靠
2. **Policy 未准备**：如果直接用 reward 驱动 ShortCut 的 step size，可能导致 OOD exploration + critic 爆炸

### 中间 Offline RL 层的作用

**不是**要把 policy 变成"离线最优"，而是：

- ✅ 让 policy 在 demo 分布附近**稍微朝高Q样本靠拢**（但不跑出分布）
- ✅ 把 critic 初步**校准**到数据分布，防止后续爆炸
- ✅ 保持 **ODE/shortcut 的几何结构完整**，为 online RL 阶段做好铺垫

### 为什么选择 Advantage-Weighted 而非直接 Q-Learning？

你的 AWCP 实验已经证明了：

| 方法 | Critic 稳定性 | Success Rate | 问题 |
|------|-------------|--------------|------|
| **直接 Q 最大化** | ❌ 易爆炸 | 低 | Policy 跑到 OOD 区域，Critic 无数据 |
| **Advantage-Weighted BC** | ✅ 稳定 | 高 | Policy stay in distribution，Critic 只学数据区域 |

因此中间层一定要用 **IQL/AWAC 风格的加权 BC**，而不是 CPQL 式的直接 Q maximization。

---

## 1️⃣ AW-ShortCut Flow (AW-SCF) 的蓝图

### 1.1 核心架构

```
输入：专家演示轨迹 (s, a_sequence)
      ↓
┌─────────────────────────────────────┐
│ Policy Network: ShortCutFlowAgent    │
│  - velocity_net (with step size d)   │
│  - EMA velocity_net (for consistency)│
└─────────────────────────────────────┘
      ↓ sample actions
      ↓
┌─────────────────────────────────────┐
│ Critic Network: DoubleQNetwork       │
│  - Q1, Q2 for conservative estimate  │
│  - Target Critic (EMA update)        │
└─────────────────────────────────────┘
      ↓ compute advantage
      ↓
┌─────────────────────────────────────┐
│ Loss Combination:                    │
│  1. AW-Flow Loss (Q-weighted)        │
│  2. AW-Shortcut Loss (Q-weighted)    │
│  3. Critic Loss (SMDP Bellman)       │
└─────────────────────────────────────┘
```

### 1.2 Policy 网络：继承 ShortCutFlowAgent

**不需要改动 ShortCutFlowAgent 的结构**，只是在 `compute_loss()` 中加入 Q-weighting：

```python
class AWShortCutFlowAgent(nn.Module):
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        critic: DoubleQNetwork,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        # ShortCut parameters
        max_denoising_steps: int = 8,
        step_size_mode: str = "fixed",      # 推荐用 fixed 或 uniform
        fixed_step_size: float = 0.0625,    # 1/16
        target_mode: str = "velocity",       # 必须用 velocity
        teacher_steps: int = 1,              # 小步 local approximation
        use_ema_teacher: bool = True,
        # Offline RL parameters
        beta: float = 0.5,                   # Advantage weighting
        weight_clip: float = 10.0,           # Prevent outliers
        bc_weight: float = 1.0,              # Flow matching weight
        consistency_weight: float = 0.3,     # Shortcut weight (light)
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        tau: float = 0.005,                  # Soft update for target critic
        gamma: float = 0.99,
        ema_decay: float = 0.999,
        device: str = "cuda",
    ):
        super().__init__()
        # ShortCut Flow components
        self.velocity_net = velocity_net
        self.velocity_net_ema = copy.deepcopy(velocity_net)
        for p in self.velocity_net_ema.parameters():
            p.requires_grad = False
        
        # Critic components
        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Offline RL hyperparameters
        self.beta = beta
        self.weight_clip = weight_clip
        self.bc_weight = bc_weight
        self.consistency_weight = consistency_weight
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.tau = tau
        self.gamma = gamma
        self.ema_decay = ema_decay
        
        # Store ShortCut parameters for later use
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.step_size_mode = step_size_mode
        self.fixed_step_size = fixed_step_size
        self.target_mode = target_mode
        self.teacher_steps = teacher_steps
```

### 1.3 关键 Loss 函数设计

#### Loss 1: Advantage-Weighted Flow Loss

```python
def _compute_flow_loss(self, obs_cond, actions, actions_for_q):
    """Flow matching loss with Q-based weighting."""
    
    # Step 1: 计算 advantage-based 权重（直接借鉴 AWCP）
    with torch.no_grad():
        q1, q2 = self.critic(actions_for_q, obs_cond)
        q_data = torch.min(q1, q2)
        baseline = q_data.mean()
        advantage = q_data - baseline
        
        # AWAC-style exponential weighting
        weights = torch.clamp(
            torch.exp(self.beta * advantage),
            max=self.weight_clip
        )
        weights = weights / weights.mean()  # Normalize
        weights = weights.squeeze(-1)
    
    # Step 2: ShortCut Flow 的标准采样（来自 ShortCutFlowAgent）
    B = actions.shape[0]
    device = actions.device
    
    # Sample noise and time
    x_0 = torch.randn_like(actions)
    t = torch.rand(B, device=device)
    
    # Interpolate
    t_expand = t.view(-1, 1, 1)
    x_t = (1 - t_expand) * x_0 + t_expand * actions
    
    # Target velocity
    v_target = actions - x_0
    
    # Sample step size d (来自 ShortCutFlowAgent._sample_step_size)
    d = self._sample_step_size_fixed(B, device)  # 推荐用 fixed 或 small uniform
    
    # Predict velocity with step size d
    v_pred = self.velocity_net(x_t, t, d, obs_cond)
    
    # Step 3: 计算 per-sample loss，然后用权重加权
    flow_loss_per_sample = F.mse_loss(v_pred, v_target, reduction="none")
    flow_loss_per_sample = flow_loss_per_sample.mean(dim=(1, 2))  # [B]
    
    # Weighted average
    flow_loss = (weights * flow_loss_per_sample).mean()
    
    return flow_loss, weights
```

**核心思想**：
- 高 Q 样本的 flow loss 获得更高权重
- 低 Q 样本的 flow loss 获得更低权重
- Policy 不跑出 demo 分布（因为仍然在最小化与所有 demo 的距离）

#### Loss 2: Advantage-Weighted Shortcut Consistency Loss

```python
def _compute_shortcut_loss(self, obs_cond, actions, weights):
    """Shortcut consistency loss with Q-based weighting."""
    
    B = actions.shape[0]
    device = actions.device
    
    # 从 ShortCutFlowAgent 中抠出来的逻辑
    x_0 = torch.randn_like(actions)
    
    # Sample time and step size for consistency
    t = self.t_min + torch.rand(B, device=device) * (self.t_max - self.t_min)
    delta_t = 0.02 + torch.rand(B, device=device) * (0.15 - 0.02)
    t_plus = torch.clamp(t + delta_t, max=self.t_max)
    
    # Interpolate
    t_expand = t.view(-1, 1, 1)
    t_plus_expand = t_plus.view(-1, 1, 1)
    x_t = (1 - t_expand) * x_0 + t_expand * actions
    x_t_plus = (1 - t_plus_expand) * x_0 + t_plus_expand * actions
    
    # Teacher target (from EMA network)
    with torch.no_grad():
        x_teacher = x_t_plus.clone()
        current_t = t_plus.clone()
        remaining_time = 1.0 - current_t
        dt_teacher = remaining_time / self.teacher_steps
        
        for _ in range(self.teacher_steps):
            v_teacher = self.velocity_net_ema(
                x_teacher, current_t, 
                self.fixed_step_size,  # 或从 step size head 采样
                obs_cond
            )
            dt_expand = dt_teacher.view(-1, 1, 1)
            x_teacher = x_teacher + v_teacher * dt_expand
            current_t = current_t + dt_teacher
        
        target = x_teacher if self.target_mode == "endpoint" else (x_teacher - x_0)
    
    # Student consistency prediction
    if self.target_mode == "endpoint":
        v_pred = self.velocity_net(x_t_plus, t_plus, self.fixed_step_size, obs_cond)
        # 这里实际上应该是用 2*d 的步长预测，但为了与 offline 阶段的保守性对齐，
        # 先用固定小步
        consistency_loss_per_sample = F.mse_loss(
            x_t_plus + v_pred * 0.0625,  # 一步小步
            target,
            reduction="none"
        ).mean(dim=(1, 2))
    else:  # velocity mode
        v_target = target
        v_pred = self.velocity_net(x_t_plus, t_plus, self.fixed_step_size, obs_cond)
        consistency_loss_per_sample = F.mse_loss(
            v_pred, v_target, reduction="none"
        ).mean(dim=(1, 2))
    
    # 关键：用同样的权重乘
    consistency_loss = (weights * consistency_loss_per_sample).mean()
    
    return consistency_loss
```

**关键**：shortcut loss 也用同样的权重乘，这样确保高 Q 样本的"学会大步"也被强化，低 Q 样本被去权重化。

#### Loss 3: Critic Loss（SMDP Bellman，直接用 AWCP 的逻辑）

```python
def _compute_critic_loss(self, obs_cond, next_obs_cond, actions_for_q, 
                         rewards, dones, cumulative_reward=None, 
                         chunk_done=None, discount_factor=None):
    """Critic loss using SMDP Bellman equation."""
    
    # Use SMDP fields if provided
    if cumulative_reward is not None:
        r = cumulative_reward
        d = chunk_done if chunk_done is not None else dones
        gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
    else:
        r = rewards
        d = dones
        gamma_tau = torch.full_like(r if r.dim() == 1 else r.squeeze(-1), self.gamma)
    
    # Ensure proper shape
    if r.dim() == 1:
        r = r.unsqueeze(-1)
    if d.dim() == 1:
        d = d.unsqueeze(-1)
    if gamma_tau.dim() == 1:
        gamma_tau = gamma_tau.unsqueeze(-1)
    
    # Scale rewards
    scaled_rewards = r * self.reward_scale
    
    with torch.no_grad():
        # Sample next actions using EMA policy
        next_actions_full = self._sample_actions_batch(next_obs_cond, use_ema=True)
        next_actions = next_actions_full[:, :self.act_horizon, :]
        
        # Compute target Q-values (conservative double Q)
        target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
        target_q = torch.min(target_q1, target_q2)
        
        # TD target
        target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
        
        if self.q_target_clip is not None:
            target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
    
    # Current Q-values
    current_q1, current_q2 = self.critic(actions_for_q, obs_cond)
    
    # MSE loss
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    
    return critic_loss
```

### 1.4 完整的 `compute_loss()` 方法

```python
def compute_loss(self, obs_features, actions, rewards, next_obs_features, dones,
                 actions_for_q=None, cumulative_reward=None, 
                 chunk_done=None, discount_factor=None):
    """Compute AW-ShortCut Flow loss: AW-Flow + AW-Shortcut + Critic."""
    
    if actions_for_q is None:
        actions_for_q = actions
    
    # Flatten obs_features if needed
    if obs_features.dim() == 3:
        obs_cond = obs_features.reshape(obs_features.shape[0], -1)
    else:
        obs_cond = obs_features
    
    if next_obs_features.dim() == 3:
        next_obs_cond = next_obs_features.reshape(next_obs_features.shape[0], -1)
    else:
        next_obs_cond = next_obs_features
    
    # Compute AW-Flow loss (includes weights)
    flow_loss, weights = self._compute_flow_loss(obs_cond, actions, actions_for_q)
    
    # Compute AW-Shortcut loss (uses same weights)
    shortcut_loss = self._compute_shortcut_loss(obs_cond, actions, weights)
    
    # Compute Critic loss
    critic_loss = self._compute_critic_loss(
        obs_cond, next_obs_cond, actions_for_q, rewards, dones,
        cumulative_reward=cumulative_reward,
        chunk_done=chunk_done,
        discount_factor=discount_factor,
    )
    
    # Total loss
    policy_loss = self.bc_weight * flow_loss + self.consistency_weight * shortcut_loss
    total_loss = policy_loss + critic_loss
    
    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "flow_loss": flow_loss,
        "shortcut_loss": shortcut_loss,
        "critic_loss": critic_loss,
        "weight_mean": weights.mean(),
        "weight_std": weights.std(),
    }
```

---

## 2️⃣ 为什么这个设计能解决 CPQL 的问题？

### CPQL 失败的原因回顾

你之前实现 CPQL 时发现：

```
直接做 Q-learning 最大化:
  policy_loss = bc_loss + alpha * (-Q(s, π(s)))
  ↓
  Policy 快速移向 OOD 动作区域
  ↓
  Critic 在那些区域没见过数据，只能 bootstrap 胡猜
  ↓
  Q 值爆炸 + Policy 学的是 critic 的幻觉
```

### AW-ShortCut Flow 如何避免

```
Advantage-Weighted BC:
  w_i = exp(β * (Q_i - baseline))
  policy_loss = Σ w_i * (flow_loss_i + consistency_loss_i)
  ↓
  Policy 始终在 demo 轨迹线上做加权 BC
  ↓
  不会跑到 OOD 区域
  ↓
  Critic 只需在数据分布上 self-consistent（用 SMDP Bellman）
  ↓
  更稳定，Q 不爆炸
```

### 理论支持

这个思想来自 **IQL (Implicit Q-Learning)** 和 **AWAC**：

| 特性 | CPQL-style | AWAC/IQL-style |
|------|-----------|-----------------|
| Policy 约束 | ❌ 无约束，最大化Q | ✅ 限制在数据分布，用Q加权 |
| Q 的作用 | ❌ 驱动 policy 离开分布 | ✅ 区分"好样本"vs"差样本" |
| Critic 学习 | ❌ 需要 generalize 到 OOD | ✅ 只需在数据区域 self-consistent |
| 稳定性 | ❌ 易爆炸 | ✅ 保守且稳定 |
| 适合场景 | ❌ 在线RL | ✅ 离线RL + 到在线的过渡 |

---

## 3️⃣ 完整的三阶段管道设计

### Stage 1: ShortCut Flow预训练（纯 BC，~100k steps）

**配置**（基于 sweep 结果的推荐）：

```yaml
# Model
algorithm: "shortcut_flow"

# ShortCut Flow parameters (from sweep best practices)
sc_target_mode: "velocity"                    # ✓ 学 local solver
sc_use_ema_teacher: true                      # ✓ EMA 稳定
sc_teacher_steps: 1                           # ✓ 保留局部性
sc_step_size_mode: "fixed"                    # ✓ 可靠性优先
sc_fixed_step_size: 0.0625                    # ✓ 1/16 小步
sc_t_sampling_mode: "uniform"                 # ✓ 全覆盖无偏
sc_inference_mode: "uniform"                  # ✓ 分布匹配
sc_num_inference_steps: 8

# Loss weights
flow_weight: 1.0
consistency_weight: 0.3                       # ✓ shortcut 轻量
self_consistency_k: 0.1                       # ✓ 低比例采样

# Training
total_iters: 100000
eval_freq: 2000
batch_size: 128
learning_rate: 1e-4
```

**目标**：
- ✅ 学好 local ODE solver（velocity 精确）
- ✅ shortcut 能力温和但可靠
- ✅ 为 offline RL 做好基础

**监控指标**：
- `flow_loss`: 应该单调下降
- `shortcut_loss`: 可能振荡，但整体趋势是下降
- `success_once`: 最终 > 0.40

### Stage 2: AW-ShortCut Flow 离线 RL（~20k-50k steps）

**配置**（基于此文档的设计）：

```yaml
# Model
algorithm: "aw_shortcut_flow"

# 继承 Stage 1 的 ShortCut 参数（freeze 或轻微调整）
sc_target_mode: "velocity"
sc_use_ema_teacher: true
sc_teacher_steps: 1
sc_step_size_mode: "fixed"
sc_fixed_step_size: 0.0625
sc_t_sampling_mode: "uniform"
sc_inference_mode: "uniform"                  # 保持分布匹配
sc_num_inference_steps: 8

# Offline RL parameters (新增)
beta: 0.5                                     # Advantage weighting temperature
weight_clip: 10.0                             # Prevent outliers
reward_scale: 0.1                             # Scale rewards
q_target_clip: 100.0                          # Clip critic targets
tau: 0.005                                    # Soft update for target critic
gamma: 0.99                                   # Discount factor

# Loss weights (与 Stage 1 相比：flow 仍主导)
flow_weight: 1.0
consistency_weight: 0.3
bc_weight: 1.0                                # 新增：保证 policy 不漂离分布

# Training
total_iters: 50000
eval_freq: 1000
batch_size: 128
learning_rate: 1e-4                           # 可能略降 (1e-5 ~ 1e-4)
warmup_iters: 2000                            # 前 2k steps 用纯 BC warm-up
```

**重点超参**：

| 超参 | 推荐值 | 说明 |
|------|--------|------|
| `beta` | 0.5 ~ 1.0 | 0.5 较保守，1.0 较激进。如果 critic 不稳定，用 0.5 |
| `weight_clip` | 5.0 ~ 10.0 | 防止少数高Q样本主导。10.0 较宽松 |
| `reward_scale` | 0.05 ~ 0.2 | 较小的值保证 Q 不要长得太快 |
| `bc_weight` | 1.0 固定 | 这个不动，保证 BC loss 始终主导 |
| `consistency_weight` | 0.3 固定 | 继承 Stage 1 |

**目标**：
- ✅ Critic 在数据分布上校准，loss 稳定下降（不爆炸）
- ✅ Policy 微微向高Q样本靠拢，success 有 5-10% 提升
- ✅ 保持 ODE/shortcut 结构完整
- ✅ 为 online RL 提供 warm-start policy + critic

**监控指标**：
- `critic_loss`: 应该单调下降，且不应该超过初值的 100 倍
- `flow_loss + shortcut_loss`: 应该保持相对稳定（Policy 不发散）
- `weight_mean / weight_std`: 权重应该合理分布（不应该某个样本权重 >>1）
- `success_once`: 应该有 5-10% 的提升（e.g., 0.40 → 0.44-0.45）

**如果出问题的调试**：

| 问题 | 原因 | 调整 |
|------|------|------|
| Critic loss 炸掉 | reward_scale 太大 或 Q bootstrap 不稳定 | ↓ reward_scale 到 0.05，或↓ beta 到 0.3 |
| Success 没有提升 | Advantage weighting 过弱 或 reward signal 本身问题 | ↑ beta 到 1.0，或检查 reward 计算 |
| Policy loss 大幅振荡 | 权重分布不均 | ↓ weight_clip，或↑ warmup_iters |
| Actor-Critic diverge | Policy 和 Critic 更新不同步 | ↓ learning_rate，或交替更新 |

### Stage 3: ReinFlow 在线 RL（从 Stage 2 初始化，~100k steps）

**配置**（以 Stage 2 的 checkpoint 初始化）：

```yaml
# Model
algorithm: "reinflow"

# 从 Stage 2 加载 checkpoint
pretrained_velocity_net: "stage2_checkpoint.pt"
pretrained_critic: "stage2_critic.pt"

# ReinFlow 特定参数
num_flow_steps: 8
ema_decay: 0.999
gamma: 0.99
gae_lambda: 0.95
clip_ratio: 0.2
entropy_coef: 0.01
value_coef: 0.5

# 在线阶段的 shortcut 配置（逐渐放松）
sc_inference_mode: "uniform"                  # 初期仍用 uniform
sc_num_inference_steps: 8

# On-policy 采样
num_envs: 10
num_steps_per_env: 200                        # Rollout length
num_epochs: 4                                 # PPO epochs
num_minibatches: 4

# Training
total_iters: 100000
eval_freq: 1000
```

**三子阶段策略**（可选 curriculum）：

1. **阶段 3a** (~25k steps)：冻结 critic，只用 BC loss + PPO policy loss
   - 目的：让 policy 轻微适应 online 环境

2. **阶段 3b** (~50k steps)：解冻 critic，引入 Q-learning
   - 目的：Critic 学习在线轨迹的价值

3. **阶段 3c** (~25k steps)：逐步开启 adaptive inference
   - 目的：让 shortcut 学会在高 reward 区域跳大步

**监控指标**：
- 标准 PPO 指标：policy_loss, value_loss, entropy
- Critic 指标：critic_loss, target_q std（不应该爆炸）
- Exploration 指标：episode_reward, success_rate（应该单调上升）

---

## 4️⃣ 代码集成检查清单

### 必需的新增/修改

#### 新文件：`algorithms/aw_shortcut_flow.py`

```python
# 主要内容：AWShortCutFlowAgent 类
# - 继承 ShortCutFlowAgent 的 step size / time sampling / shortcut target 逻辑
# - 新增 critic（DoubleQNetwork）和 target_critic
# - 新增 _compute_flow_loss、_compute_shortcut_loss、_compute_critic_loss
# - 实现 compute_loss、update_ema、update_target、get_action 等方法
```

#### 修改：`train_offline_rl.py`

添加 AW-ShortCut Flow 的命令行参数：

```python
# 既有的 ShortCut Flow 参数继续支持
parser.add_argument("--sc_target_mode", type=str, default="velocity")
parser.add_argument("--sc_use_ema_teacher", action="store_true", default=True)
# ... 其他 sc_ 参数

# 新增 AW-ShortCut Flow 参数
parser.add_argument("--beta", type=float, default=0.5, 
                    help="Advantage weighting temperature")
parser.add_argument("--weight_clip", type=float, default=10.0,
                    help="Max weight to prevent outliers")
parser.add_argument("--reward_scale", type=float, default=0.1)
parser.add_argument("--q_target_clip", type=float, default=100.0)

# 修改 create_agent 函数
if algorithm == "aw_shortcut_flow":
    agent = AWShortCutFlowAgent(
        velocity_net=velocity_net,
        critic=critic,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        beta=args.beta,
        weight_clip=args.weight_clip,
        reward_scale=args.reward_scale,
        q_target_clip=args.q_target_clip,
        # ... 其他参数
    )
```

#### 修改：`algorithms/__init__.py`

```python
from .aw_shortcut_flow import AWShortCutFlowAgent

__all__ = [
    # ...
    "AWShortCutFlowAgent",
    # ...
]
```

### Sweep 脚本

可参考现有的 `sweep_awcp_beta_parallel.sh` 的结构，创建 `sweep_aw_scf_offline_rl.sh`：

```bash
#!/bin/bash
# AW-ShortCut Flow Offline RL 超参扫描

BETAS=(0.3 0.5 1.0)
WEIGHT_CLIPS=(5.0 10.0)
REWARD_SCALES=(0.05 0.1 0.2)

for beta in "${BETAS[@]}"; do
    for wc in "${WEIGHT_CLIPS[@]}"; do
        for rs in "${REWARD_SCALES[@]}"; do
            # 从 Stage 1 checkpoint 初始化
            python train_offline_rl.py \
                --algorithm aw_shortcut_flow \
                --pretrained_velocity_net "stage1_best.pt" \
                --beta $beta \
                --weight_clip $wc \
                --reward_scale $rs \
                --exp_name "aw_scf_offline_beta${beta}_wc${wc}_rs${rs}" \
                # ... 其他参数
        done
    done
done
```

---

## 5️⃣ 理论总结

### 为什么这个管道在理论上是合理的？

#### 1. **Off-policy 数据的稳定学习** (Offline RL 共识)

在离线设置中，直接做 on-policy 梯度（如 REINFORCE 或 PPO）是不安全的，因为：
- 新采样的轨迹会远离数据分布
- Critic 没有在那些区域的数据，估计不可靠

**Advantage-Weighted BC** 通过以下机制解决：
- Policy 始终做 BC（生成接近 demo 的动作）
- 用 Q 来选择性强化"更好的 demo"
- 结果：policy 在数据分布附近做微小调整，Critic 的估计误差有界

#### 2. **ShortCut Flow 的 Local Approximation 性质** (from sweep)

Sweep 结论：ShortCut Flow 本质上在学"局部 ODE solver"，因此：
- 小步长 + EMA teacher + velocity target 最优
- 不适合激进的大步探索（会破坏局部结构）

**AW-ShortCut Flow** 的优势：
- Policy 保持在 demo 分布，不会学奇怪的大步
- Shortcut 能力被温和地强化（通过权重），而不是被激进地追求
- Online ReinFlow 可以在这个"健康的 shortcut"基础上，用真实 reward 指导大步探索

#### 3. **梯度流向的精心设计** (Critical!)

```
Stage 1: Flow matching（纯 BC）
  ∇_θ L_flow → 只优化 velocity 网络，无 Q 反馈

Stage 2: AW-ShortCut（离线 RL）
  ∇_θ L_AW_flow = ∇_θ (w * L_flow)
             = ∂w/∂Q * ∂Q/∂a * ∇_a L_flow + w * ∇_θ L_flow
                    ↑ 无梯度（detach）      ↑ 主梯度（权重调整）
  
  结果：Policy 参数 θ 的梯度完全来自：
    1. BC loss（主力）+ 权重调制（Q信息）
    2. 不直接最大化 Q（避免 CPQL 的问题）

Stage 3: ReinFlow（在线 RL）
  ∇_θ L_PPO + ∇_ψ L_actor_critic
  
  此时 Policy 已经是一个"稳定的、几何正确的"ODE solver，
  RL 可以安全地用真实 reward 做微调
```

这个设计确保了：
- **Stage 1 → 2 的连续性**：Shortcut 结构被保留
- **Stage 2 的稳定性**：Critic 学习有界，不爆炸
- **Stage 2 → 3 的平顺过渡**：Policy 已经是好的初始化，RL 微调容易成功

---

## 6️⃣ 失败案例与调试指南

### Case 1: Critic Loss 爆炸

**症状**：
```
Iter 1000:  critic_loss = 0.5
Iter 2000:  critic_loss = 2.1
Iter 3000:  critic_loss = 45.0  ← 爆炸
```

**原因分析**：
1. `reward_scale` 太大 → Q target 增长太快 → MSE loss 很大
2. `q_target_clip` 不够严格 → target Q 有离群值
3. Policy 跑出分布 → next actions 很奇怪 → target 估计差

**调整**：
```python
# 试试这个顺序：
1. ↓ reward_scale:  0.1 → 0.05
2. ↓ beta:          0.5 → 0.3  (弱化 AW weighting)
3. ↑ warmup_iters:  2000 → 5000 (纯 BC 预热更长)
4. ↓ q_target_clip: 100 → 50
```

### Case 2: Success 无提升（甚至下降）

**症状**：
```
Stage 1 baseline: success = 0.42
Stage 2 after 50k steps: success = 0.40  ← 变差了
```

**原因分析**：
1. AW weighting 太弱 → Policy 没有感受到 Q 的指导
2. Reward 信号错误 → demo 的 Q 估计本身就不对
3. Critic 还没校准好 → Q 估计噪声太大

**调整**：
```python
# 先确认 reward/Q 的合理性：
- 打印出 demo 的 Q 分布
- 检查：max(Q) / min(Q) 是否合理（应该是 5-20 倍而非 100+ 倍）
- 如果不对，检查 reward 计算逻辑

# 如果 reward 正确，调参：
1. ↑ beta: 0.5 → 1.0 (加强 AW weighting)
2. ↓ weight_clip: 10.0 → 5.0 (让高Q样本的权重限制更紧)
3. 延长 stage 2 的 iters: 50k → 100k
```

### Case 3: Policy 和 Critic 产生 "Adversarial" 动态

**症状**：
```
Iter 5000:  policy_loss ↓, critic_loss ↑
Iter 10000: policy_loss ↑, critic_loss ↓
→ 两个 loss 互相抵消，都无法收敛
```

**原因分析**：
- Policy 和 Critic 学习率不平衡
- 或者 critic 的 target 更新（soft update）太快

**调整**：
```python
# 方案 1: 降低学习率，同步更新
lr_policy = 5e-5 (↓ from 1e-4)
lr_critic = 5e-5

# 方案 2: 修改更新频率
update_critic_freq = 2  # 每 2 个 policy step 更新一次 critic
soft_update_tau = 0.001 (↓ from 0.005)

# 方案 3: 交替冻结
# 前 10k steps: 只更新 critic，冻结 policy
# 中间 20k steps: 同时更新
# 后 20k steps: 只更新 policy，冻结 critic
```

---

## 🎓 论文写作指引

### 章节组织

```markdown
## Offline RL 过渡层：Advantage-Weighted ShortCut Flow

### 4.1 Motivation
- 直接从 BC 跳到 Online RL 面临两个挑战：critic 未校准，policy 可能被 RL 推出分布
- Advantage-weighted BC 是离线 RL 的既往最佳实践（AWAC, IQL）
- 我们将其与 ShortCut Flow 结合，得到 AW-ShortCut Flow

### 4.2 Method
#### 4.2.1 Policy Loss: Advantage-Weighted Flow Matching
- Equation: w_i = clip(exp(β A_i), w_max), where A_i = Q_i - baseline
- Loss: L_AW_flow = Σ w_i ||v_pred(x_t^i, t^i, d^i) - v_target^i||^2

#### 4.2.2 Consistency Loss: Advantage-Weighted Shortcut
- 同样的权重应用到 shortcut consistency loss

#### 4.2.3 Critic Loss: SMDP Bellman
- Double Q-learning with soft target updates

### 4.3 Experimental Setup
- Stage 1 baseline: ShortCut Flow BC (best config from sweep)
- Stage 2: AW-ShortCut Flow (50k iterations)
- Hyperparameters: β, weight_clip, reward_scale

### 4.4 Results
- Offline RL stage 提升 5-10% success rate
- Critic 稳定学习，无爆炸现象
- 为 online ReinFlow 提供 warm-start

### 4.5 Ablation (Optional)
- Effect of β
- Effect of weight_clip
- Effect of reward_scale
```

### 关键论述

> Unlike naive Q-learning approaches (CPQL) which risk distributional shift and critic instability, our Advantage-Weighted ShortCut Flow keeps the policy on the offline demonstration manifold while using Q-values to modulate which samples to prioritize. This design leverages the mathematical structure of ShortCut Flow as a local ODE solver approximation: small steps and velocity targets ensure the policy remains in a reliable regime, while EMA teachers and conservative double-Q learning keep critic estimates bounded.

---

## 📋 实现清单

- [ ] 创建 `algorithms/aw_shortcut_flow.py` 文件
  - [ ] 实现 `AWShortCutFlowAgent.__init__`
  - [ ] 实现 `_compute_flow_loss`（AW-weighted）
  - [ ] 实现 `_compute_shortcut_loss`（AW-weighted）
  - [ ] 实现 `_compute_critic_loss`（SMDP Bellman）
  - [ ] 实现 `compute_loss` 方法汇总
  - [ ] 实现 `get_action` 方法
  - [ ] 实现 `update_ema` 和 `update_target` 方法
- [ ] 修改 `train_offline_rl.py`
  - [ ] 添加 AW-SCF 相关命令行参数（beta, weight_clip, etc.)
  - [ ] 在 `create_agent` 中添加 aw_shortcut_flow case
  - [ ] 验证数据加载和 reward 计算
- [ ] 修改 `algorithms/__init__.py` 导出 `AWShortCutFlowAgent`
- [ ] 创建 sweep 脚本 `sweep_aw_scf_offline_rl.sh`
- [ ] 测试三阶段管道
  - [ ] Stage 1：运行 ShortCut Flow BC 到收敛，保存 checkpoint
  - [ ] Stage 2：从 Stage 1 checkpoint 加载，运行 AW-SCF
  - [ ] 验证 success rate 有提升，critic 不爆炸
  - [ ] Stage 3：从 Stage 2 checkpoint 初始化 ReinFlow（future work）

---

## 参考文献与相关工作

- **AWAC**: Advantage-Weighted Actor-Critic (https://arxiv.org/abs/2006.09359)
- **IQL**: Implicit Q-Learning (https://arxiv.org/abs/2110.06169)
- **CPQL**: 你之前的实验中发现直接 Q-learning 不稳定
- **ShortCut Flow / ReinFlow**: 本工作的主要方法
- **SMDP 学习**: 处理变长 action chunks 的标准框架


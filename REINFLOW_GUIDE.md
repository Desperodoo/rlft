# ReinFlow: Online RL Fine-tuning for Flow Matching Policies

## 概述

ReinFlow 是一种基于 Flow Matching 的在线强化学习微调算法，用于提升预训练的 imitation learning 策略的性能。它是三阶段训练流程的最后一个阶段：

```
Stage 1: ShortCut Flow BC      → 纯行为克隆预训练
Stage 2: AW-ShortCut Flow      → 离线 RL（Q-weighted BC）
Stage 3: ReinFlow              → 在线 RL（PPO fine-tuning）
```

## 硬件需求与显存估计

### 4090 单卡 (24GB) 测试结果

基于 `test_reinflow_memory_real.py` 测试，使用 PickCube-v1 环境：

| num_envs | batch_size | Peak Memory | 推荐程度 |
|----------|------------|-------------|---------|
| 64 | 128 | ~650 MB | 保守 |
| 128 | 256 | ~1.2 GB | 稳健 |
| 256 | 256 | ~1.3 GB | **推荐** |
| 512 | 256 | ~2.3 GB | 激进 |
| 1024 | 512 | ~4.5 GB | 最大 |

**观察结论**：
- ManiSkill 的 `physx_cuda` 后端显存占用非常低
- 即使 1024 个并行环境也只用 ~4.5GB
- 瓶颈不在显存，可以大胆增加 `num_envs` 提高采样效率

**推荐配置**（单卡 4090）：
```bash
--num_envs 256 --minibatch_size 256 --num_inference_steps 8
```

### 显存测试命令

```bash
# 测试 RGB 模式
CUDA_VISIBLE_DEVICES=0 python test_reinflow_memory_real.py --gpu 0 --env PickCube-v1

# 测试 RGB+Depth 模式  
CUDA_VISIBLE_DEVICES=0 python test_reinflow_memory_real.py --gpu 0 --env PickCube-v1 --obs-mode "rgb+depth"
```

---

## 核心设计

### 1. Denoising MDP 架构

ReinFlow 的核心创新在于将 denoising 过程建模为 MDP：

```
标准 RL MDP:       s_t → a_t → r_t → s_{t+1}
                   (环境状态)

Denoising MDP:     s_k = (x_k, obs)  → v_k  → ... → s_K = (x_1, obs)
                   (denoising 状态)   (velocity)      (最终动作)
```

**关键特点：**
- **State**: `s_k = (x_k, obs)` - 当前 denoising 步骤的噪声状态 + 环境观测
- **Action**: `v_k` - 预测的 velocity field
- **Reward**: 稀疏奖励，只有最终执行的动作获得环境奖励
- **Critic**: `V(s_0)` 预测从初始噪声执行整个 action chunk 的期望回报

### 2. SMDP Action Chunk 执行

采用 Semi-MDP 框架执行完整的 action chunk：

```python
# 执行 act_horizon 步动作后再更新策略
for step in range(act_horizon):
    action = action_chunk[:, step, :]
    next_obs, reward, done, info = env.step(action)
    cumulative_reward += gamma^step * reward
    if done:
        break

# SMDP Bellman equation
# Q(s, a_chunk) = R_cumulative + γ^τ * V(s_{t+τ})
```

### 3. 可学习探索噪声

使用 `ExploreNoiseNet` 预测每个维度的探索噪声标准差：

```python
class ExploreNoiseNet:
    """预测探索噪声 scale"""
    def forward(self, time_emb, obs_emb) -> noise_std  # [B, act_dim]

class NoisyVelocityUNet1D:
    """包装基础 velocity network，添加可学习探索"""
    def forward(self, x, t, d, obs, sample_noise=True):
        velocity = base_net(x, t, d, obs)
        noise_std = explore_noise_net(t_emb, obs_emb)
        if sample_noise:
            noise = randn_like(x) * noise_std
        return velocity, noise, noise_std
```

## 关键组件

### ReinFlowAgent

```python
agent = ReinFlowAgent(
    velocity_net=ShortCutVelocityUNet1D(...),  # 基础网络（与 AW-ShortCut Flow 对齐）
    obs_dim=256,
    act_dim=7,
    pred_horizon=16,
    act_horizon=8,                    # 完整执行的 chunk 长度
    num_inference_steps=8,            # 固定推理步数
    
    # PPO 参数
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    
    # Critic Warmup
    critic_warmup_steps=5000,         # 先训练 critic 的步数
    
    # 噪声调度
    min_noise_std=0.01,
    max_noise_std=0.3,
    noise_decay_type="linear",        # constant/linear/exponential
    noise_decay_steps=500000,
)
```

### RolloutBuffer

存储 SMDP 级别的 transition：

```python
buffer = RolloutBuffer(
    buffer_size=rollout_steps * num_envs,
    obs_dim=obs_dim,
    pred_horizon=pred_horizon,
    act_dim=act_dim,
    gamma=0.99,
    gae_lambda=0.95,
)

# 添加 chunk 级别的 transition
buffer.add(
    obs=obs_features,           # 观测
    actions=action_chunk,       # 完整 action chunk
    reward=cumulative_reward,   # SMDP 累积奖励
    value=V(s_0),              # Critic 预测
    log_prob=log_pi(a|s),      # 策略 log prob
    done=chunk_done,            # chunk 内是否终止
)

# 计算 GAE
buffer.compute_returns_and_advantages(last_value, last_done)
```

## 训练流程

### 1. 加载预训练检查点

```python
# 从 AW-ShortCut Flow 加载 velocity network
agent.load_from_aw_shortcut_flow(
    checkpoint_path="runs/aw_shortcut_flow-PickCube-v1/best_model.pt",
    device="cuda",
)

# 可选：冻结 visual encoder
if freeze_visual_encoder:
    for param in visual_encoder.parameters():
        param.requires_grad = False
```

### 2. Critic Warmup 阶段

```python
# 前 critic_warmup_steps 只训练 value network
if agent.is_in_critic_warmup():
    loss = value_coef * F.mse_loss(V(s), returns)
    # 不更新 policy
```

**Warmup 的意义：**
- 新初始化的 value network 预测不准确
- 直接用错误的 advantage 更新 policy 会导致不稳定
- 先让 critic 学习到合理的 value estimate

### 3. 正常 PPO 更新

```python
# Critic warmup 结束后，正常 PPO 更新
loss_dict = agent.compute_ppo_loss(
    obs_cond=batch["obs"],
    actions=batch["actions"],
    old_log_probs=batch["log_probs"],
    advantages=batch["advantages"],
    returns=batch["returns"],
)

# loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

### 4. 噪声调度

```python
# 每步更新噪声边界
agent.update_noise_schedule(step=global_step)

# 支持三种衰减方式
if noise_decay_type == "constant":
    max_std = initial_max_std
elif noise_decay_type == "linear":
    max_std = initial_max_std * (1 - progress) + min_std * progress
elif noise_decay_type == "exponential":
    max_std = min_std + (initial_max_std - min_std) * exp(-3 * progress)
```

## 使用示例

### 基础用法

```bash
python train_online_finetune.py \
    --env_id PickCube-v1 \
    --pretrained_path runs/aw_shortcut_flow-PickCube-v1-seed0/best_model.pt \
    --num_inference_steps 8 \
    --critic_warmup_steps 5000 \
    --total_timesteps 1000000
```

### 完整参数

```bash
python train_online_finetune.py \
    --env_id PickCube-v1 \
    --pretrained_path runs/aw_shortcut_flow-PickCube-v1-seed0/best_model.pt \
    --num_envs 16 \
    --total_timesteps 1000000 \
    --rollout_steps 128 \
    --ppo_epochs 4 \
    --minibatch_size 64 \
    --lr 3e-5 \
    --lr_critic 1e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_ratio 0.2 \
    --entropy_coef 0.01 \
    --value_coef 0.5 \
    --num_inference_steps 8 \
    --critic_warmup_steps 5000 \
    --min_noise_std 0.01 \
    --max_noise_std 0.3 \
    --noise_decay_type linear \
    --noise_decay_steps 500000 \
    --freeze_visual_encoder \
    --track \
    --wandb_project_name ManiSkill_ReinFlow
```

---

## 参数扫描指南

### Ablation 设计原则

为了系统地理解各参数的影响，采用**控制变量**方法：

1. **确定 baseline 配置**：基于离线阶段最佳配置
2. **每次只改变一个参数类别**：便于归因
3. **避免重复实验**：与 baseline 相同的配置不重复运行
4. **选择有意义的参数范围**：基于先验知识和初步实验

### Baseline Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| critic_warmup_steps | 5000 | 足够让 critic 收敛 |
| noise_decay_type | linear | 稳定衰减 |
| noise_decay_steps | 500000 | 半程衰减 |
| min_noise_std | 0.01 | 最小探索 |
| max_noise_std | 0.3 | 初始探索 |
| clip_ratio | 0.2 | PPO 标准值 |
| entropy_coef | 0.01 | 适度探索 |
| value_coef | 0.5 | 平衡策略与价值 |
| lr | 3e-5 | 较小学习率保持稳定 |
| lr_critic | 1e-4 | Critic 可以学快一些 |
| num_inference_steps | 8 | 推理效率与质量平衡 |
| rollout_steps | 128 | 足够的样本 |
| ppo_epochs | 4 | PPO 标准值 |

### Ablation Categories

共 **22 个实验**（1 baseline + 21 ablations）

#### 1. Critic Warmup Ablation (3 variants)
**目的**：探索 warmup 步数对训练稳定性的影响

| Config | warmup_steps | vs Baseline | 假设 |
|--------|-------------|-------------|------|
| baseline | 5000 | - | 默认配置 |
| `warmup:0` | 0 | −5000 | 无 warmup，可能不稳定 |
| `warmup:2k` | 2000 | −3000 | 较短 warmup |
| `warmup:10k` | 10000 | +5000 | 较长 warmup |

#### 2. Noise Schedule Ablation (5 variants)
**目的**：探索探索噪声衰减策略

| Config | decay_type | max_noise | decay_steps | vs Baseline |
|--------|-----------|-----------|-------------|-------------|
| baseline | linear | 0.3 | 500k | - |
| `noise:constant` | constant | 0.3 | - | 不衰减 |
| `noise:exp` | exponential | 0.3 | 500k | 指数衰减 |
| `noise:fast` | linear | 0.3 | 200k | 快速衰减 |
| `noise:low_init` | linear | 0.15 | 500k | 初始噪声减半 |
| `noise:high_init` | linear | 0.5 | 500k | 初始噪声增加 |

#### 3. PPO Hyperparameters Ablation (5 variants)
**目的**：调整 PPO 核心参数

| Config | clip_ratio | entropy_coef | value_coef | vs Baseline |
|--------|-----------|--------------|------------|-------------|
| baseline | 0.2 | 0.01 | 0.5 | - |
| `ppo:clip_tight` | 0.1 | 0.01 | 0.5 | 更保守 |
| `ppo:clip_loose` | 0.3 | 0.01 | 0.5 | 更激进 |
| `ppo:high_entropy` | 0.2 | 0.05 | 0.5 | 更多探索 |
| `ppo:low_entropy` | 0.2 | 0.001 | 0.5 | 更少探索 |
| `ppo:high_value` | 0.2 | 0.01 | 1.0 | 更重视 value |

#### 4. Learning Rate Ablation (3 variants)
**目的**：探索学习率配置

| Config | lr_policy | lr_critic | vs Baseline |
|--------|----------|-----------|-------------|
| baseline | 3e-5 | 1e-4 | - |
| `lr:low` | 1e-5 | 3e-5 | 学习率降低 3x |
| `lr:high` | 1e-4 | 3e-4 | 学习率提高 3x |
| `lr:equal` | 5e-5 | 5e-5 | 相同学习率 |

#### 5. Inference Steps Ablation (2 variants)
**目的**：探索推理步数的影响

| Config | num_inference_steps | vs Baseline |
|--------|-------------------|-------------|
| baseline | 8 | - |
| `infer:4steps` | 4 | 更快但可能质量下降 |
| `infer:16steps` | 16 | 更慢但可能质量提升 |

#### 6. Rollout Configuration Ablation (3 variants)
**目的**：探索数据收集配置

| Config | rollout_steps | ppo_epochs | minibatch_size | vs Baseline |
|--------|--------------|------------|----------------|-------------|
| baseline | 128 | 4 | 64 | - |
| `rollout:short` | 64 | 4 | 32 | 更小 batch |
| `rollout:long` | 256 | 4 | 128 | 更大 batch |
| `rollout:more_epochs` | 128 | 8 | 64 | 更多优化次数 |

### 运行 Sweep

```bash
# 完整 sweep（22 个实验）
./sweep_reinflow_parallel.sh --gpus "0 1 2 3 4 5 6 7" --env PickCube-v1

# 仅测试 warmup category
./sweep_reinflow_parallel.sh --gpus "0 1" --configs "baseline:default warmup:0 warmup:2k warmup:10k"

# Dry run 检查命令
./sweep_reinflow_parallel.sh --dry-run

# 指定预训练检查点
./sweep_reinflow_parallel.sh --pretrained-path /path/to/checkpoint.pt

# 多 seed 实验
./sweep_reinflow_parallel.sh --seeds "0 1 2"
```

### 监控进度

```bash
# 默认日志目录
./monitor_reinflow.sh

# 指定日志目录
./monitor_reinflow.sh --log-dir /tmp/reinflow_grid --interval 10
```

### 结果分析建议

1. **收敛速度对比**：比较不同配置达到相同性能所需的 timesteps
2. **最终性能对比**：比较 1M timesteps 后的 success rate
3. **稳定性对比**：比较训练曲线的 variance
4. **效率对比**：比较每个 epoch 的计算时间

### 预期结论

| Category | Expected Best | Reasoning |
|----------|--------------|-----------|
| warmup | warmup:5k (baseline) | 太短不稳定，太长浪费时间 |
| noise | noise:linear or noise:exp | 需要逐渐减少探索 |
| ppo | baseline 或 clip_tight | 过于激进可能不稳定 |
| lr | baseline | 太大可能发散，太小收敛慢 |
| infer | infer:8steps (baseline) | 平衡质量与效率 |
| rollout | baseline 或 rollout:long | 更多数据通常更好 |

---

## 与其他算法的对比

| 特性 | DPPO | ReinFlow |
|-----|------|----------|
| Base 架构 | DDPM | Flow Matching |
| Denoising 步数 | 100 (partial: 5) | 8 (fixed) |
| Fine-tune 范围 | 最后 K 步 | 全部步 |
| 探索机制 | Learnable log_std | ExploreNoiseNet |
| Action chunk | 支持 | 支持 (SMDP) |

## 注意事项

1. **Pretrained checkpoint 格式**：需要与 `ShortCutVelocityUNet1D` 兼容
2. **Visual encoder 冻结**：推荐冻结以保持视觉特征稳定
3. **Critic warmup**：对于新环境或困难任务，建议使用较长的 warmup
4. **噪声衰减**：过快衰减可能导致探索不足，过慢可能导致训练不稳定
5. **内存管理**：`return_chains=False` 可以节省 GPU 内存，但 log_prob 精度略低

## 参考文献

- [ReinFlow](https://github.com/ReinFlow/ReinFlow) - Official implementation
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - Foundation work
- [DPPO](https://diffusion-ppo.github.io/) - Diffusion PPO
- [ShortCut Flow](shortcut_flow.py) - Stage 1 pretraining
- [AW-ShortCut Flow](aw_shortcut_flow.py) - Stage 2 offline RL

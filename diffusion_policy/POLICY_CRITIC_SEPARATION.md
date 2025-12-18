# Policy-Critic Data Separation 功能说明

## 背景

在 RLPD 在线微调阶段，模仿学习环节会引入失败的在线样本。虽然 AWAC 使用 Q-weighted BC，失败样本权重较低，但仍会对 policy 训练产生负面影响。

## 解决方案

实现 **Policy-Critic 数据分离**：
- **Critic**: 使用所有数据（包括失败样本），学习"什么是坏的"
- **Policy**: 只使用高质量数据（demos + 高 advantage 的在线样本）

## 新增功能

### 1. 数据源标记 (`is_demo`)

`OnlineReplayBufferRaw.sample_mixed()` 现在返回 `is_demo` 字段：
- `True`: 来自离线 demo 数据
- `False`: 来自在线探索数据

### 2. AWSCAgent 数据过滤

新增参数：
- `filter_policy_data: bool = False` - 是否启用数据过滤
- `advantage_threshold: float = 0.0` - 在线样本的 advantage 阈值

过滤逻辑：
1. 保留所有 demo 样本 (`is_demo=True`)
2. 仅保留 `advantage > threshold` 的在线样本
3. 如果过滤后无有效样本，回退到仅使用 demo

### 3. SuccessReplayBuffer

存储成功的在线轨迹（暂时仅保存，不参与训练）：
- 仅存储 `success=True` 的 episode
- 可用于未来的课程学习或自我提升

## 使用方法

### 启用数据过滤

```bash
python train_rlpd_online.py \
    --algorithm awsc \
    --env_id PickCube-v1 \
    --pretrained_path runs/awsc-PickCube-v1-offline/best.pt \
    --load_critic \
    --demo_path demos/PickCube-v1.h5 \
    --filter_policy_data \
    --advantage_threshold 0.0
```

### 启用 Success Buffer

```bash
python train_rlpd_online.py \
    --algorithm awsc \
    ... \
    --use_success_buffer \
    --success_buffer_capacity 100000
```

## 新增监控指标

训练时会记录以下指标：

| 指标 | 说明 |
|------|------|
| `policy_batch_size` | 过滤后用于 policy 训练的样本数 |
| `n_demo_samples` | batch 中 demo 样本数 |
| `n_online_kept` | 通过 advantage 过滤的在线样本数 |
| `n_online_filtered` | 被过滤掉的在线样本数 |
| `success_buffer_size` | Success Buffer 当前大小 |
| `total_successes_stored` | 累计存储的成功 episode 数 |

## 代码修改清单

1. [replay_buffer.py](diffusion_policy/rlpd/replay_buffer.py)
   - `sample()` 返回 `is_demo=False`
   - `_sample_offline()` 返回 `is_demo=True`
   - `sample_mixed()` 合并 `is_demo` 字段
   - 新增 `SuccessReplayBuffer` 类

2. [awsc_agent.py](diffusion_policy/rlpd/awsc_agent.py)
   - 新增 `filter_policy_data`, `advantage_threshold` 参数
   - `compute_actor_loss()` 添加数据过滤逻辑
   - 新增 metrics: `policy_batch_size`, `n_demo_samples`, `n_online_kept`, `n_online_filtered`

3. [train_rlpd_online.py](train_rlpd_online.py)
   - 新增 CLI 参数: `--filter_policy_data`, `--advantage_threshold`, `--use_success_buffer`
   - 创建 `SuccessReplayBuffer`
   - 传递 `is_demo` 到 `compute_actor_loss()`

4. [__init__.py](diffusion_policy/rlpd/__init__.py)
   - 导出 `SuccessReplayBuffer`

## 测试

```bash
cd /home/amax/rlft/diffusion_policy
python tests/test_policy_critic_separation.py
```

## 参考

- RLPD (Ball et al., ICML 2023) - 离线数据混合
- AWAC (Nair et al., NeurIPS 2020) - Advantage Weighted Actor-Critic

# ManiSkill3 并行数据采样问题与解决方案

## 问题描述

在使用 ManiSkill3 的 GPU 并行环境进行数据采集时，发现除了第一批并行环境（等于 `num_envs` 数量）能正常采集外，后续的采集速度异常快且成功率极低（从预期的 70-80% 降至 7-20%）。

### 症状表现

1. **前 N 个 episode 正常采集**（N = 并行环境数），速度较慢
2. **之后的 episode 采集极快**，几乎瞬间完成
3. **成功率骤降**：从单环境测试的 84% 降至并行采集的 7-20%
4. **Transition 数量异常少**：2000 episodes 只有 5000-11000 transitions（预期应为 50000-100000）

## 根本原因

### 1. 未使用官方 VectorEnv Wrapper

直接使用 `gym.make()` 创建的 ManiSkill3 并行环境**不会自动重置**。当 episode 结束（`done=True`）时：

```python
# ❌ 错误理解
next_obs  # 以为是新 episode 的初始观测

# ✅ 实际情况
next_obs  # 实际上是当前 episode 的最终观测（cube 可能在空中）
```

### 2. Expert 策略状态混乱

由于 `next_obs` 包含的是旧 episode 的最终状态（cube 已被抬起），Expert 的 `initial_cube_z` 被错误设置为空中的高度（如 0.24）而非桌面高度（0.02），导致后续计算的抓取位置完全错误。

### 3. Episode 快速"完成"

因为 Expert 策略行为异常，环境很快达到 truncation 条件（max_episode_steps），表现为采集速度极快但无实际有效 transitions。

## 解决方案

### 使用 `ManiSkillVectorEnv` Wrapper

ManiSkill3 官方提供了 `ManiSkillVectorEnv` wrapper 来正确处理并行环境的自动重置：

```python
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# 创建基础环境
env = gym.make("PickCube-v1", num_envs=100, obs_mode="state", ...)

# ✅ 使用官方 wrapper 包装
vec_env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
```

### Wrapper 的关键行为

当 `auto_reset=True` 时：

| 返回值 | 含义 |
|--------|------|
| `next_obs` | **新 episode 的初始观测**（环境已自动重置） |
| `info["final_observation"]` | **旧 episode 的最终观测** |
| `info["final_info"]` | **旧 episode 的最终 info**（包含 success 等） |

### 修复后的采集逻辑

```python
next_obs, rewards, terminated, truncated, info = vec_env.step(actions)
dones = terminated | truncated

for i in range(num_envs):
    if dones[i]:
        # 存储时使用 final_observation（旧 episode 的最终状态）
        final_obs = info["final_observation"][i]
        data["next_obs"].append(final_obs)
        
        # 检查成功使用 final_info
        success = info["final_info"]["success"][i]
        
        # 重置 expert（next_obs[i] 已经是新 episode 的初始观测）
        env_experts[i].reset()
    else:
        # 非结束状态正常存储
        data["next_obs"].append(next_obs[i])
```

## 修复效果对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 成功率 | 7-20% | **70.2%** |
| 2000 ep Transitions | 5,000-11,000 | **99,363** |
| 平均 ep 长度 | ~3-5 steps | **~50 steps** |
| 采集行为 | 异常快速 | 正常稳定 |

## 官方文档参考

- [RL Setup - Gym Vectorized Environment API](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html#gym-vectorized-environment-api)
- [GPU Simulation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/gpu_simulation.html)

## 关键代码位置

- 数据采集脚本：`scripts/generate_maniskill_dataset.py`
- `collect_episodes_parallel()` 函数
- `ManiSkillScriptedExpert` 类

## 注意事项

1. **设备一致性**：确保 actions tensor 与环境在同一 GPU 上
   ```python
   actions_tensor = actions_tensor.to(vec_env.device)
   ```

2. **单 GPU 运行**：建议使用 `CUDA_VISIBLE_DEVICES=0` 避免多 GPU 设备不匹配问题

3. **PickCube 任务特性**：这是一个 "pick and hold" 任务，目标位置在空中，不需要放下 cube

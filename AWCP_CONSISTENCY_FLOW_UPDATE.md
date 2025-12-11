# AWCP Consistency Flow 参数更新

## 更新背景
根据 `sweep_consistency_flow_parallel.sh` 的 13 个配置 sweep 结果，推荐使用 **flow_endpoint** 风格的配置作为 AWCP 的默认设置。

## 主要更新

### 1. 时间范围参数
**变更**：从受限范围改为完整范围
- `t_min`: 0.05 → **0.0**（完整 t 范围）
- `t_max`: 0.95 → **1.0**（完整 t 范围）

**原因**：完整时间范围提供更好的时间多样性和覆盖，避免只在中间时间步学习。

### 2. Delta 参数
**变更**：从随机范围改为小的固定值
- `delta_min`: 0.02 → **0.01**（小固定 delta）
- `delta_max`: 0.15 → **0.01**（小固定 delta）

**原因**：小的固定 delta 确保：
- 稳定的 teacher 目标（不会有大的 teacher 误差）
- 更精准的一致性约束
- 避免动态范围导致的不稳定性

### 3. Teacher 和 Student 设计

#### Teacher 网络（使用 EMA）
- **起始点**：从 `t_cons` 开始（不是 `t_plus`）
- **集成步数**：`teacher_steps = 2`
- **目标**：将 x(t_cons) 集成到 x(1)（最终状态）

#### Student 网络（可训练）
- **起始点**：从 `t_plus` 开始
- **目标**：也将 x(t_plus) 集成到 x(1)
- **约束**：两个网络应该在 x(1) 处达到一致

### 4. 一致性损失空间
**变更**：从 velocity-space 改为 endpoint-space
- **前**：计算速度目标 `v = x(1) - x(0)` 的 MSE
- **后**：直接计算最终状态 `x(1)` 的 MSE

**损失公式**：
```
Loss_endpoint = ||x_student(1) - x_teacher(1)||^2
```

**原因**：
- Endpoint 一致性更直接，更容易优化
- 避免 velocity-space 中的梯度分散
- 更快的收敛和更稳定的训练

### 5. Q-加权机制（保留）
AWCP 的核心 AWAC-风格权重机制保持不变：
```python
weights = exp(β * advantage)  # advantage = Q(s,a) - baseline
```
- 高 Q 值的样本获得更高权重
- 避免直接 Q 最大化导致的分布偏移
- Flow BC 和一致性损失都按权重加权

## 配置对比

| 参数 | 旧值 (CPQL-style) | 新值 (flow_endpoint) | 原因 |
|------|------------------|-------------------|------|
| t_min | 0.05 | 0.0 | 完整时间覆盖 |
| t_max | 0.95 | 1.0 | 完整时间覆盖 |
| delta_min | 0.02 | 0.01 | 小固定值 |
| delta_max | 0.15 | 0.01 | 小固定值 |
| teacher_from | t_plus | t_cons | sweep 最优 |
| student_at | t_cons | t_plus | sweep 最优 |
| loss_space | velocity | **endpoint** | sweep 最优 |
| teacher_steps | 2 | 2 | 推荐 |

## 明确避免的设计

根据 sweep 结果，**明确不推荐**以下组合：

1. ❌ **Velocity-space 一致性损失**
   - 性能不如 endpoint-space
   - 梯度更分散，收敛慢

2. ❌ **动态大范围 delta**
   - 导致 teacher 目标不稳定
   - 容易出现训练抖动

3. ❌ **Student/Teacher 都在 t_cons 对齐**
   - 不如 teacher(t_cons) + student(t_plus) 配置
   - 时间多样性不足

## 代码变更

在 `awcp.py` 中的变更：

### 参数初始化
```python
# __init__ 中的 consistency 参数
self.t_min = 0.0          # Full t range
self.t_max = 1.0          # Full t range
self.delta_min = 0.01     # Small fixed delta
self.delta_max = 0.01     # Small fixed delta
self.teacher_steps = 2    # Multi-step teacher

self.cons_teacher_from = "t_cons"      # Teacher starts from t_cons
self.cons_student_point = "t_plus"     # Student from t_plus
self.cons_loss_space = "endpoint"      # Endpoint consistency loss
```

### Consistency Loss 计算
```python
# _compute_policy_loss 中的主要变更

# 1. 固定 delta（不是随机范围）
delta_t = torch.full_like(t_cons, self.delta_min)
t_plus = torch.clamp(t_cons + delta_t, max=1.0)

# 2. Teacher 从 t_cons 开始集成
x_teacher = x_t_cons.clone()  # NOT x_t_plus
current_t = t_cons.clone()

# 3. 集成到 x(1)
for _ in range(self.teacher_steps):
    v_teacher = self.velocity_net_ema(x_teacher, current_t, global_cond=obs_cond)
    x_teacher = x_teacher + v_teacher * dt_expand
    current_t = current_t + dt_teacher

# 4. Student 也集成到 x(1)（从 t_plus 开始）
x_student = x_t_plus.clone()
# ... 类似集成过程 ...

# 5. Endpoint 一致性损失（不是速度损失）
consistency_loss = MSE(x_student(1), x_teacher(1))
```

## 性能期望

使用 `flow_endpoint` 配置的 AWCP 应该表现出：
- ✅ 更稳定的训练（更小的损失波动）
- ✅ 更快的收敛速度
- ✅ 更好的最终任务性能
- ✅ 更可靠的泛化

## 参考

- **Sweep 配置**：`diffusion_policy/sweep_consistency_flow_parallel.sh`
- **推荐配置**：`flow_endpoint` (lines 138-149 in sweep script)
- **相关论文**：Consistency Flow Matching, AWAC

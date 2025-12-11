# AWCP Consistency Flow 参数变更总结

## 核心变更：从 CPQL 风格 → Flow Endpoint 风格

### 参数对比

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            参数             │     旧值        │     新值      │  说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
时间范围      t_min           │     0.05       │     0.0      │ 完整范围
            t_max           │     0.95       │     1.0      │ 完整范围
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Delta        delta_min       │     0.02       │     0.01     │ 固定值
            delta_max       │     0.15       │     0.01     │ 固定值
            delta 策略      │   随机范围     │    固定      │ 稳定性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Teacher     cons_teacher_from│    t_plus      │    t_cons    │ sweep最优
一致性      cons_student_point│   t_cons      │    t_plus    │ sweep最优
            teacher_steps   │      2        │      2       │ 多步集成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
损失        cons_loss_space │   velocity    │   endpoint   │ 直接优化
计算         │               │               │             │ 最终状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 关键改进说明

### 1️⃣ 完整时间范围 [0.0, 1.0]
**旧**：t ∈ [0.05, 0.95]（避免边界）
**新**：t ∈ [0.0, 1.0]（完整覆盖）

✅ **优势**：
- 在整个去噪过程中学习，时间覆盖更全
- 利用早期和晚期的时间步信息
- 提高模型的泛化能力

### 2️⃣ 小的固定 Delta [0.01]
**旧**：delta ∈ [0.02, 0.15]（随机范围）
**新**：delta = 0.01（固定值）

✅ **优势**：
- 稳定的 teacher 目标（不会有大的集成误差）
- 更精准的一致性约束
- 避免训练中的 delta 方差导致的抖动
- teacher 和 student 都在可靠的范围内学习

### 3️⃣ Teacher 和 Student 配置
**旧**：
```
Teacher: x(t_plus) → x(1)
Student: x(t_plus) → x(1)
```

**新**：
```
Teacher (EMA): x(t_cons) → x(1)
Student (Train): x(t_plus) → x(1)
```

✅ **优势**：
- Teacher 从更早的时间开始集成，有更多步数改进预测
- Student 从 t_plus 开始，保证短的但有意义的一致性约束
- 两个网络有不同的起点，避免过度相似
- t_cons < t_plus，保证时间排序正确

### 4️⃣ Endpoint 一致性损失
**旧**：Velocity-space
```python
v_target = x(1) - x(0)
loss = MSE(v_pred, v_target)  # 在速度空间
```

**新**：Endpoint-space
```python
loss = MSE(x_student(1), x_teacher(1))  # 在最终状态空间
```

✅ **优势**：
- ✅ 更直接的优化目标（最终状态匹配）
- ✅ 梯度信号更清晰，避免分散
- ✅ 更快收敛
- ✅ 更稳定的训练动态

---

## 代码实现细节

### 初始化部分 (`__init__`)
```python
# 时间范围：完整 [0, 1]
self.t_min = 0.0
self.t_max = 1.0

# Delta：固定小值
self.delta_min = 0.01
self.delta_max = 0.01

# 集成参数
self.teacher_steps = 2

# 一致性设计
self.cons_teacher_from = "t_cons"      # ← Teacher 从 t_cons
self.cons_student_point = "t_plus"     # ← Student 从 t_plus  
self.cons_loss_space = "endpoint"      # ← Endpoint 损失
```

### 一致性损失计算 (`_compute_policy_loss`)

**步骤 1**：采样一致性时间点
```python
t_cons = self.t_min + torch.rand(B) * (self.t_max - self.t_min)  # 随机 t ∈ [0, 1]
delta_t = torch.full_like(t_cons, self.delta_min)  # ← 固定 delta
t_plus = torch.clamp(t_cons + delta_t, max=1.0)   # t_cons + 0.01
```

**步骤 2**：Teacher 集成（使用 EMA 网络）
```python
x_teacher = x_t_cons.clone()  # 从 t_cons 点开始
current_t = t_cons.clone()

for step in range(self.teacher_steps):  # 2 步
    v_teacher = velocity_net_ema(x_teacher, current_t, obs_cond)
    x_teacher += v_teacher * dt  # 向前集成
    current_t += dt

target_x1 = x_teacher  # 最终预测
```

**步骤 3**：Student 集成（可训练网络）
```python
x_student = x_t_plus.clone()  # 从 t_plus 点开始（比 t_cons 晚）
current_t_student = t_plus.clone()

for step in range(self.teacher_steps):
    v_student = velocity_net(x_student, current_t_student, obs_cond)
    x_student += v_student * dt
    current_t_student += dt
```

**步骤 4**：Endpoint 一致性损失
```python
# 直接在最终状态空间比较
consistency_loss = MSE(x_student, target_x1)

# 按 Q-权重加权
weighted_loss = (weights * consistency_loss).mean()
```

---

## 性能影响预期

| 指标 | 影响 | 原因 |
|------|------|------|
| **收敛速度** | ⬆️ 加快 | Endpoint 损失更直接，梯度信号清晰 |
| **训练稳定性** | ⬆️ 提升 | 固定 delta，teacher 目标稳定 |
| **最终性能** | ⬆️ 改善 | 完整时间覆盖 + 稳定的一致性 |
| **损失波动** | ⬇️ 减少 | 小的固定 delta 避免方差 |
| **计算成本** | ≈ 相同 | 集成步数相同 |

---

## 明确避免的设计 ❌

根据 sweep 结果，**不推荐**以下组合：

1. **Velocity-space 损失**
   - ❌ 收敛慢
   - ❌ 梯度分散
   - ❌ 不如 endpoint-space

2. **动态大范围 delta**（如 [0.02, 0.15]）
   - ❌ Teacher 目标不稳定
   - ❌ 训练抖动
   - ❌ 方差大

3. **Student/Teacher 都在 t_cons**
   - ❌ 缺乏时间多样性
   - ❌ 一致性约束不足
   - ❌ 性能下降

---

## 验证和测试

- ✅ Python 语法检查：通过
- ✅ 代码兼容性：保持与 AWCP Q-加权机制兼容
- ✅ 参数类型：torch.Tensor 张量操作一致

## 文件位置

- **主要实现**：`diffusion_policy/diffusion_policy/algorithms/awcp.py`
- **参考 Sweep**：`diffusion_policy/sweep_consistency_flow_parallel.sh`
- **推荐配置**：L138-149（flow_endpoint 配置）

---

## 使用建议

使用新配置的 AWCP 时：

```bash
# 训练命令示例
python train_offline_rl.py \
    --algorithm awcp \
    --env_id LiftPegUpright-v1 \
    --exp_name awcp-flow-endpoint-sweep \
    --track \
    --wandb_project_name maniskill_awcp_flow
```

新的一致性参数会自动应用，无需额外配置。

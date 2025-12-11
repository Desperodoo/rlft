# 🎯 AWCP Consistency Flow 快速参考

## 更新内容一览

### ✅ 已应用的变更

在 `diffusion_policy/algorithms/awcp.py` 中：

```python
# 新的 consistency 参数配置（基于 flow_endpoint 推荐）

# 时间范围：完整 [0, 1]
t_min = 0.0   ← 从 0.05 改为 0.0（完整时间覆盖）
t_max = 1.0   ← 从 0.95 改为 1.0

# Delta：小的固定值
delta_min = 0.01  ← 从 0.02 改为 0.01（固定值）
delta_max = 0.01  ← 从 0.15 改为 0.01（固定值）

# Teacher/Student 配置
cons_teacher_from = "t_cons"   ← 新增：Teacher 从 t_cons 开始
cons_student_point = "t_plus"  ← 新增：Student 从 t_plus 开始
cons_loss_space = "endpoint"   ← 新增：Endpoint 一致性损失（非 velocity）

# 集成参数
teacher_steps = 2  ← 保持不变
```

---

## 🔄 一致性损失变更

### 之前（CPQL 风格）
```
t_cons ─────── (随机 delta ∈ [0.02, 0.15]) ──── t_plus
                                      ↓
                          v_cons_pred = velocity_net(x_t_plus, t_plus)
                          v_cons_target = x(1) - x(0)
                          loss = MSE(v_pred, v_target)  [velocity-space]
```

### 现在（Flow Endpoint 风格）✨
```
t_cons ──────────── (固定 delta = 0.01) ─────── t_plus
  ↓                                              ↓
[Teacher EMA]                            [Student (trainable)]
 集成 → x(1)                              集成 → x(1)
  │                                         │
  └─────────────────────────────────────────┘
              loss = MSE(x_s, x_t)
           [endpoint-space 一致性]
```

---

## 📊 关键改进

| 改进项 | 效果 |
|------|------|
| **完整时间范围** | 整个去噪过程都学到信息 |
| **固定小 delta** | Teacher 目标稳定，训练不抖动 |
| **不同起点** | teacher(t_cons) vs student(t_plus) |
| **Endpoint 损失** | 更直接的优化目标 |

---

## ❌ 明确避免

- ❌ Velocity-space 一致性损失
- ❌ 动态大范围 delta（如 [0.02, 0.15]）
- ❌ Student/Teacher 都在 t_cons

---

## 📝 实现要点

```python
# 固定 delta（关键差异）
delta_t = torch.full_like(t_cons, self.delta_min)  # NOT random!
t_plus = torch.clamp(t_cons + delta_t, max=1.0)

# Teacher 从 t_cons 开始（关键差异）
x_teacher = x_t_cons.clone()  # NOT x_t_plus
# ... 集成到 x(1) ...

# Student 从 t_plus 开始
x_student = x_t_plus.clone()
# ... 集成到 x(1) ...

# Endpoint 一致性（关键差异）
consistency_loss = MSE(x_student, x_teacher)  # NOT velocity MSE
```

---

## 🚀 使用方式

无需更改！使用新的 AWCP 时自动应用这些参数：

```bash
python train_offline_rl.py --algorithm awcp ...
```

---

## 📚 参考文献

- Sweep 脚本：`diffusion_policy/sweep_consistency_flow_parallel.sh`
- 推荐配置：`flow_endpoint` (lines 138-149)
- 13 个配置对比结果推荐 `flow_endpoint` 作为最优

---

## 🔍 验证

✅ Python 语法检查：通过
✅ 张量操作：正确
✅ 兼容性：与 AWCP Q-加权机制完全兼容

---

**生成时间**：2025-12-10
**更新文件**：`diffusion_policy/algorithms/awcp.py`
**状态**：✅ 生效中

# Toy Diffusion RL

A minimal, clean PyTorch codebase for comparing diffusion/flow-based policies with various RL algorithms on simple continuous-control tasks.

## 🎯 Overview

This project implements **toy versions** of state-of-the-art generative model + RL algorithms:

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Diffusion Policy** | Offline/BC | DDPM-based policy for behavior cloning |
| **Flow Matching Policy** | Offline/BC | ODE-based generative policy (vanilla, reflected, consistency) |
| **Diffusion Double Q** | Offline RL | Diffusion actor + Double Q critics |
| **CPQL** | Offline RL | Consistency Policy Q-Learning |
| **DPPO** | Online RL | Diffusion Policy Policy Optimization |
| **ReinFlow** | Online RL | Flow Matching + Online RL fine-tuning |

## 📁 Project Structure

```
toy_diffusion_rl/
├── envs/                           # Toy environments
│   ├── point_mass_2d.py           # 2D point mass navigation
│   └── pendulum_continuous_wrapper.py
├── common/                         # Shared components
│   ├── networks.py                # MLP, noise predictors, Q networks
│   ├── replay_buffer.py           # Experience replay
│   └── utils.py                   # Diffusion helpers, utilities
├── algorithms/
│   ├── diffusion_policy/          # DDPM-based behavior cloning
│   ├── flow_matching/             # Flow matching variants
│   │   ├── fm_policy.py          # Vanilla flow matching
│   │   ├── reflected_flow.py     # Boundary-reflected flow
│   │   └── consistency_flow.py   # Consistency-style training
│   ├── diffusion_double_q/        # Diffusion-QL + Double Q
│   ├── cpql/                      # Consistency Policy Q-Learning
│   ├── dppo/                      # Diffusion PPO
│   └── reinflow/                  # Flow + Online RL
├── configs/                        # YAML configurations
├── train.py                        # Training script
└── eval.py                         # Evaluation script
```

## 🚀 Quick Start

### Installation

```bash
cd toy_diffusion_rl
pip install -r requirements.txt
```

### Training

```bash
# Train Diffusion Policy on Point Mass 2D
python train.py --algorithm diffusion_policy --env point_mass_2d --seed 42

# Train with config file
python train.py --config configs/diffusion_policy_pendulum.yaml

# Train Flow Matching Policy
python train.py --algorithm flow_matching --env point_mass_2d

# Train online RL algorithms
python train.py --algorithm dppo --env point_mass_2d
python train.py --algorithm reinflow --env point_mass_2d
```

### Evaluation

```bash
# Evaluate trained model
python eval.py --checkpoint results/experiment/best_model.pt \
               --algorithm diffusion_policy --env point_mass_2d \
               --num_episodes 100 --save_plots
```

## 🎮 Environments

### 2D Point Mass (Recommended for Testing)
- **State**: (x, y, vx, vy) - position and velocity
- **Action**: (ax, ay) - acceleration in [-1, 1]
- **Reward**: -distance_to_goal - action_penalty + goal_bonus
- **Goal**: Navigate to origin (0, 0)

### Pendulum
- Wrapped Gymnasium Pendulum-v1
- Normalized action space [-1, 1]
- Energy-based expert controller

## 📚 Algorithms

### 1. Diffusion Policy
Behavior cloning using DDPM. Learns to denoise actions conditioned on states.

```python
from algorithms import DiffusionPolicyAgent

agent = DiffusionPolicyAgent(
    state_dim=4, action_dim=2,
    num_diffusion_steps=100,
    device="cuda"
)
action = agent.sample_action(state)
```

### 2. Flow Matching Policy
ODE-based generative model. Three variants:
- **Vanilla**: Standard conditional flow matching
- **Reflected**: Boundary reflection for bounded actions
- **Consistency**: Single-step generation via consistency training

```python
from algorithms import FlowMatchingPolicy, ConsistencyFlowPolicy

# Vanilla
agent = FlowMatchingPolicy(state_dim=4, action_dim=2)

# Consistency (fast inference)
agent = ConsistencyFlowPolicy(state_dim=4, action_dim=2)
```

### 3. Diffusion Double Q Learning
Offline RL combining diffusion actor with double Q-critics.

```python
from algorithms import DiffusionDoubleQAgent

agent = DiffusionDoubleQAgent(
    state_dim=4, action_dim=2,
    alpha=1.0,  # Q-value weight
    device="cuda"
)
```

### 4. CPQL
Consistency Policy Q-Learning - fast single-step action generation with Q-learning.

```python
from algorithms import CPQLAgent

agent = CPQLAgent(state_dim=4, action_dim=2)
```

### 5. DPPO
Diffusion Policy Policy Optimization - online RL with diffusion policy.

```python
from algorithms import DPPOAgent

agent = DPPOAgent(
    state_dim=4, action_dim=2,
    num_diffusion_steps=5  # Keep small for online RL
)

# Training loop
buffer = agent.collect_rollout(env, rollout_steps=2048)
metrics = agent.update(buffer)
```

### 6. ReinFlow
Online RL fine-tuning for flow matching policies.

```python
from algorithms import ReinFlowAgent

agent = ReinFlowAgent(state_dim=4, action_dim=2)

# Pretrain offline
agent.offline_pretrain(expert_data, num_steps=5000)

# Online fine-tuning
buffer = agent.collect_rollout(env, rollout_steps=2048)
metrics = agent.online_update(buffer)
```

## ⚙️ Configuration

Example config (`configs/diffusion_policy_pendulum.yaml`):

```yaml
algorithm: "diffusion_policy"

env:
  name: "point_mass_2d"
  max_episode_steps: 200

training:
  seed: 42
  total_steps: 20000
  batch_size: 256

diffusion:
  num_diffusion_steps: 100
  noise_schedule: "linear"

network:
  hidden_dims: [256, 256, 256]

optimizer:
  learning_rate: 1.0e-4
```

## 📖 References

### Papers & Repositories

| Algorithm | Paper | Code |
|-----------|-------|------|
| Diffusion Policy | [Chi et al., 2023](https://diffusion-policy.cs.columbia.edu/) | [GitHub](https://github.com/real-stanford/diffusion_policy) |
| Flow Matching | [Lipman et al., 2022](https://arxiv.org/abs/2210.02747) | [HRI-EU](https://github.com/HRI-EU/flow_matching) |
| Diffusion-QL | [Wang et al., 2022](https://arxiv.org/abs/2208.06193) | [GitHub](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) |
| CPQL | [Chen et al., 2024](https://arxiv.org/abs/2404.07503) | [GitHub](https://github.com/cccedric/cpql) |
| DPPO | [Ren et al., 2024](https://diffusion-ppo.github.io/) | [GitHub](https://github.com/irom-princeton/dppo) |
| ReinFlow | [Lu et al., 2024](https://reinflow.github.io/) | [GitHub](https://github.com/ReinFlow/ReinFlow) |

### Additional References
- Rectified Flow: [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch)
- ManiFlow: [allenai/maniflow](https://github.com/allenai/maniflow)
- IDQL: [philippe-eecs/IDQL](https://github.com/philippe-eecs/IDQL)

## 🔧 Development

```bash
# Run tests
pytest tests/

# Format code
black toy_diffusion_rl/
isort toy_diffusion_rl/
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests.

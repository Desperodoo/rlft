# Toy Diffusion RL

A minimal, clean PyTorch codebase for comparing diffusion/flow-based policies with various RL algorithms on simple continuous-control tasks.

## 🎯 Overview

This project implements **toy versions** of state-of-the-art generative model + RL algorithms:

| Algorithm | Type | Description | EMD (Ring) |
|-----------|------|-------------|------------|
| **Diffusion Policy** | Offline/BC | DDPM-based policy for behavior cloning | 0.0728 ✅ |
| **Flow Matching Policy** | Offline/BC | ODE-based generative policy | 0.1193 ✅ |
| **Consistency Flow** | Offline/BC | Fast few-step generation via consistency training | 0.1949 ✅ |
| **Reflected Flow** | Offline/BC | Boundary-reflected flow for bounded actions | 0.1678 ✅ |
| **Diffusion Double Q** | Offline RL | Diffusion actor + Double Q critics | 0.1487 ✅ |
| **CPQL** | Offline RL | Consistency-Flow Q-Learning (reimplemented) | 0.2134 ✅ |
| **DPPO** | Online RL | Diffusion Policy Policy Optimization (pretrain) | 0.1168 ✅ |
| **ReinFlow** | Online RL | Flow Matching + Online RL fine-tuning (pretrain) | 0.1249 ✅ |

> **Note**: EMD (Earth Mover's Distance) scores shown are from validation on 8-mode ring distribution after 40k training steps. Lower is better. All algorithms successfully learn multimodal distributions in offline training phase.

## 📁 Project Structure

```
toy_diffusion_rl/
├── envs/                           # Toy environments
│   ├── multimodal_particle.py     # Multimodal distribution learning task
│   ├── point_mass_2d.py           # 2D point mass navigation
│   └── pendulum_continuous_wrapper.py
├── common/                         # Shared components
│   ├── networks.py                # MLP, noise predictors, Q networks
│   ├── replay_buffer.py           # Experience replay
│   └── utils.py                   # Diffusion helpers, utilities
├── algorithms/
│   ├── diffusion_policy/          # DDPM-based behavior cloning ✅
│   ├── flow_matching/             # Flow matching variants
│   │   ├── fm_policy.py          # Vanilla flow matching ✅
│   │   ├── reflected_flow.py     # Boundary-reflected flow ✅
│   │   ├── consistency_flow.py   # Consistency-style training ✅
│   │   └── base_flow.py          # Base class for flow models
│   ├── diffusion_double_q/        # Diffusion-QL + Double Q ✅
│   ├── cpql/                      # Consistency-Flow Q-Learning ✅
│   ├── dppo/                      # Diffusion PPO (pretrain) ✅
│   └── reinflow/                  # Flow + Online RL (pretrain) ✅
├── scripts/
│   └── validate_all_algorithms.py # Validation script with progress tracking
├── configs/                        # YAML configurations
├── train.py                        # Training script
└── eval.py                         # Evaluation script
```

**Legend**: ✅ = Validated and working

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

# Train online RL algorithms (pretrain phase)
python train.py --algorithm dppo --env point_mass_2d
python train.py --algorithm reinflow --env point_mass_2d

# Validate all algorithms on multimodal distribution
python toy_diffusion_rl/scripts/validate_all_algorithms.py \
    --distribution ring \
    --num_steps 40000 \
    --num_checkpoints 6
```

### Evaluation

```bash
# Evaluate trained model
python eval.py --checkpoint results/experiment/best_model.pt \
               --algorithm diffusion_policy --env point_mass_2d \
               --num_episodes 100 --save_plots
```

## 🧪 Validation Results

All algorithms have been validated on an 8-mode ring distribution (multimodal toy environment) with 5000 expert samples over 40,000 training steps. The Earth Mover's Distance (EMD) measures how well each algorithm learns the multimodal distribution:

| Algorithm | EMD (↓) | Training Status | Notes |
|-----------|---------|----------------|-------|
| Diffusion Policy | 0.0728 | ✅ Converged | Best performance |
| Flow Matching | 0.1193 | ✅ Converged | Fast convergence |
| DPPO | 0.1168 | ✅ Converged | BC pretrain phase |
| ReinFlow | 0.1249 | ✅ Converged | BC pretrain phase |
| Diffusion QL | 0.1487 | ✅ Converged | Stable Q-learning |
| Reflected Flow | 0.1678 | ✅ Converged | Satisfies constraints |
| Consistency Flow | 0.1949 | ✅ Converged | 5-step inference |
| CPQL | 0.2134 | ✅ Converged | Flow + Q-learning |

**Visualization**: Training progress and final distributions are saved to `toy_diffusion_rl/validation/ring_8_mode_validation_<timestamp>/`

**Key Findings**:
- All algorithms successfully learn the multimodal distribution (offline training phase)
- Diffusion Policy achieves best distribution matching (EMD = 0.0728)
- Flow Matching and DPPO show excellent balance of speed and accuracy
- CPQL (redesigned with Consistency Flow + Q-learning) now converges properly

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

### Fetch Pick-and-Place (Gymnasium Robotics)
- **Environment**: FetchPickAndPlace-v3 from gymnasium-robotics
- **Observation**: Robot + object state (25D) + goal (3D)
- **Action**: End-effector delta pose + gripper (4D)
- **Task**: Pick up object and place at goal position
- **Supports**: state, image, and state_image observation modes

### ManiSkill3 PickCube (GPU-Accelerated)
- **Environment**: PickCube-v1 from ManiSkill3
- **Observation**: Robot + object + goal state (varies) + RGB-D camera
- **Action**: End-effector delta pose + gripper (7D, pd_ee_delta_pose)
- **Task**: Pick up cube and place at goal position
- **Features**:
  - GPU-accelerated parallel simulation (VecEnv with num_envs > 1)
  - Supports state, image, and state_image observation modes
  - Headless rendering for SSH/server environments
  - Compatible with online RL fine-tuning (DPPO/ReinFlow)

#### ManiSkill3 Installation

ManiSkill3 requires a separate conda environment due to dependency conflicts:

```bash
# Create ManiSkill3 environment
cd toy_diffusion_rl
bash setup_ms3_env.sh

# Activate environment
conda activate rlft_ms3

# Validate installation
python scripts/validate_maniskill.py --quick
```

#### ManiSkill3 Usage

```python
from envs import make_maniskill_env, check_maniskill_available

# Check if ManiSkill3 is available
if check_maniskill_available():
    # Single environment (NumPy mode)
    env = make_maniskill_env(
        task="PickCube-v1",
        obs_mode="state_image",
        num_envs=1,
        use_numpy=True,
    )
    
    # VecEnv for parallel training (GPU, PyTorch tensors)
    env = make_maniskill_env(
        task="PickCube-v1",
        obs_mode="state",
        num_envs=16,
        use_numpy=False,  # Returns PyTorch tensors
    )
```

#### Generate ManiSkill3 Dataset

```bash
# Single environment
python scripts/generate_maniskill_dataset.py \
    --obs_mode state_image \
    --num_episodes 200 \
    --output_path data/maniskill_pickcube_state_image.h5

# Parallel collection (faster, requires GPU)
python scripts/generate_maniskill_dataset.py \
    --obs_mode state \
    --num_episodes 200 \
    --num_envs 16 \
    --output_path data/maniskill_pickcube_state.h5
```

## 📚 Algorithms

### 1. Diffusion Policy (EMD: 0.0728)
Behavior cloning using DDPM. Learns to denoise actions conditioned on states.

**Status**: ✅ Working - Best performer on multimodal distributions

```python
from algorithms import DiffusionPolicyAgent

agent = DiffusionPolicyAgent(
    state_dim=4, action_dim=2,
    num_diffusion_steps=100,
    device="cuda"
)
action = agent.sample_action(state)
```

### 2. Flow Matching Policy (EMD: 0.1193)
ODE-based generative model. Three variants:
- **Vanilla** (EMD: 0.1193): Standard conditional flow matching
- **Reflected** (EMD: 0.1678): Boundary reflection for bounded actions
- **Consistency** (EMD: 0.1949): Few-step generation via consistency training (5 steps)

**Status**: ✅ All variants working

```python
from algorithms import FlowMatchingPolicy, ConsistencyFlowPolicy

# Vanilla
agent = FlowMatchingPolicy(state_dim=4, action_dim=2)

# Consistency (fast inference)
agent = ConsistencyFlowPolicy(
    state_dim=4, action_dim=2,
    num_inference_steps=5  # Faster than vanilla
)
```

### 3. Diffusion Double Q Learning (EMD: 0.1487)
Offline RL combining diffusion actor with double Q-critics.

**Status**: ✅ Working

```python
from algorithms import DiffusionDoubleQAgent

agent = DiffusionDoubleQAgent(
    state_dim=4, action_dim=2,
    alpha=1.0,  # Q-value weight
    device="cuda"
)
```

### 4. CPQL (EMD: 0.2134)
Consistency-Flow Q-Learning - fast few-step action generation with Q-learning.

**Status**: ✅ Working (reimplemented using Consistency Flow + Q-learning)

**Note**: Our implementation uses Consistency Flow Matching as the base instead of Karras-style Consistency Models, making it simpler and more stable.

```python
from algorithms import CPQLAgent

agent = CPQLAgent(
    state_dim=4, action_dim=2,
    num_inference_steps=5
)
```

### 5. DPPO (EMD: 0.1168)
Diffusion Policy Policy Optimization - online RL with diffusion policy.

**Status**: ✅ Pretrain phase working (online RL not yet implemented)

```python
from algorithms import DPPOAgent

agent = DPPOAgent(
    state_dim=4, action_dim=2,
    num_diffusion_steps=10  # Keep small for efficiency
)

# Pretrain with BC
metrics = agent.pretrain_bc(expert_data, num_steps=5000)

# Training loop (for future online RL)
# buffer = agent.collect_rollout(env, rollout_steps=2048)
# metrics = agent.update(buffer)
```

### 6. ReinFlow (EMD: 0.1249)
Online RL fine-tuning for flow matching policies.

**Status**: ✅ Pretrain phase working (online RL not yet implemented)

```python
from algorithms import ReinFlowAgent

agent = ReinFlowAgent(
    state_dim=4, action_dim=2,
    num_flow_steps=10
)

# Pretrain offline
metrics = agent.offline_pretrain(expert_data, num_steps=5000)

# Online fine-tuning (for future implementation)
# buffer = agent.collect_rollout(env, rollout_steps=2048)
# metrics = agent.online_update(buffer)
```

## 🧪 Validation Results

All algorithms have been validated on an 8-mode ring distribution task (40k training steps):

| Algorithm | EMD Score | Status | Notes |
|-----------|-----------|--------|-------|
| Diffusion Policy | 0.0728 | ✅ Excellent | Best overall performance |
| Flow Matching | 0.1193 | ✅ Good | Stable ODE-based generation |
| DPPO | 0.1168 | ✅ Good | Pretrain phase only |
| ReinFlow | 0.1249 | ✅ Good | Pretrain phase only |
| Diffusion QL | 0.1487 | ✅ Good | Q-learning working |
| Reflected Flow | 0.1678 | ✅ Good | Handles boundaries well |
| Consistency Flow | 0.1949 | ✅ Good | 5-step inference |
| CPQL | 0.2134 | ✅ Good | Consistency Flow + Q-learning |

**Test Setup**: 
- Distribution: 8-mode ring (radius ~3.0)
- Training steps: 40,000
- Batch size: 256
- Dataset size: 5,000 expert samples
- Metric: Earth Mover's Distance (lower is better)

See validation results visualization in `results/progress_ring_*.png`

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

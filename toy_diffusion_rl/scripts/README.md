# Scripts for ManiSkill3 PickCube Task

This directory contains scripts for working with the ManiSkill3 PickCube-v1 robotics task.

## Prerequisites

Activate the ManiSkill3 environment before running any script:
```bash
conda activate rlft_ms3
```

## Scripts

### 1. `generate_maniskill_dataset.py`

Generate offline demonstration dataset using expert policy.

```bash
# Single environment collection (slower but simpler)
python scripts/generate_maniskill_dataset.py \
    --obs_mode state_image \
    --num_episodes 200 \
    --output_path data/maniskill_pickcube_200.h5

# Parallel collection using GPU VecEnv (recommended)
CUDA_VISIBLE_DEVICES=0 python scripts/generate_maniskill_dataset.py \
    --obs_mode state_image \
    --num_episodes 1000 \
    --num_envs 100 \
    --output_path data/maniskill_pickcube_1k.h5
```

**Arguments:**
- `--obs_mode`: `state`, `image`, or `state_image` (default: `state_image`)
- `--num_episodes`: Number of episodes to collect
- `--num_envs`: Number of parallel environments (>1 uses GPU)
- `--max_episode_steps`: Steps per episode (default: 50)
- `--image_size`: Image resolution (default: 128)
- `--output_path`: Output HDF5 file path

### 2. `filter_dataset.py`

Filter dataset by episode quality (cumulative reward).

```bash
# Filter to keep only high-quality episodes
python scripts/filter_dataset.py \
    --input_path data/maniskill_pickcube_1k.h5 \
    --output_path data/maniskill_pickcube_filtered.h5 \
    --min_reward 100

# Analyze dataset without filtering
python scripts/filter_dataset.py \
    --input_path data/maniskill_pickcube_1k.h5 \
    --analyze_only
```

**Arguments:**
- `--min_reward`: Minimum cumulative episode reward to keep
- `--max_length`: Maximum episode length to keep
- `--remove_rotation`: Remove zero rotation dimensions (7D → 4D actions)
- `--analyze_only`: Only analyze, don't create filtered dataset

### 3. `validate_maniskill_offline.py`

Train and evaluate all diffusion/flow algorithms on ManiSkill3.

```bash
# Full offline training validation
python scripts/validate_maniskill_offline.py \
    --dataset_path data/maniskill_pickcube_1k.h5 \
    --obs_mode state_image \
    --num_steps 50000 \
    --eval_interval 5000 \
    --record_video

# Quick test with fewer steps
python scripts/validate_maniskill_offline.py \
    --dataset_path data/maniskill_pickcube_200.h5 \
    --num_steps 5000 \
    --eval_interval 1000
```

**Algorithms tested:**
1. Diffusion Policy (BC)
2. Flow Matching Policy (BC)
3. Consistency Flow Policy
4. Reflected Flow Policy
5. Diffusion QL (Offline RL)
6. CPQL (Offline RL)
7. DPPO (pretrain only)
8. ReinFlow (pretrain only)

**Arguments:**
- `--dataset_path`: Path to HDF5 dataset
- `--obs_mode`: `state` or `state_image` (default: `state_image`)
- `--num_steps`: Total training steps (default: 50000)
- `--eval_interval`: Steps between evaluations (default: 5000)
- `--eval_episodes`: Episodes per evaluation (default: 40)
- `--record_video`: Enable video recording of evaluations
- `--video_dir`: Directory to save videos

### 4. `validate_maniskill_online.py`

Online fine-tuning for DPPO and ReinFlow using PPO on ManiSkill3.

```bash
# Fine-tune from pretrained checkpoint
CUDA_VISIBLE_DEVICES=1 python scripts/validate_maniskill_online.py \
    --checkpoint_dir ./results/maniskill_checkpoints_YYYYMMDD_HHMMSS \
    --obs_mode state_image \
    --num_iters 100 \
    --rollout_steps 2048 \
    --num_envs 20 \
    --record_video

# Train from scratch using dataset for normalizer
CUDA_VISIBLE_DEVICES=1 python scripts/validate_maniskill_online.py \
    --dataset_path data/maniskill_pickcube_1k.h5 \
    --algorithm DPPO \
    --num_iters 100 \
    --rollout_steps 2048

# Fine-tune specific algorithm only
CUDA_VISIBLE_DEVICES=1 python scripts/validate_maniskill_online.py \
    --checkpoint_dir ./results/maniskill_checkpoints_XXX \
    --algorithm ReinFlow \
    --num_iters 50
```

**Algorithms:**
- **DPPO**: Diffusion Policy with PPO (partial chain fine-tuning)
- **ReinFlow**: Flow Matching with PPO (learnable exploration noise)

**Arguments:**
- `--checkpoint_dir`: Directory containing pretrained checkpoints (from `validate_maniskill_offline.py`)
- `--dataset_path`: Path to dataset for creating normalizer (if no checkpoint)
- `--algorithm`: `DPPO`, `ReinFlow`, or `all` (default: `all`)
- `--obs_mode`: `state` or `state_image` (default: `state_image`)
- `--num_envs`: Parallel training environments (default: 20)
- `--num_eval_envs`: Parallel evaluation environments (default: 20)
- `--num_iters`: Number of PPO update iterations (default: 100)
- `--rollout_steps`: Steps per rollout (default: 2048)
- `--eval_interval`: Iterations between evaluations (default: 10)
- `--eval_episodes`: Episodes per evaluation (default: 20)
- `--save_interval`: Iterations between checkpoint saves (default: 50)
- `--record_video`: Enable video recording of evaluations
- `--video_dir`: Directory to save videos

**Output:**
- Checkpoints saved to `./results/online_finetuning_YYYYMMDD_HHMMSS/`
- Training progress plot (`training_progress.png`)
- Results JSON with metrics per iteration
- Videos (if `--record_video` enabled)

## Dataset Format

Generated HDF5 files contain:
- `obs`: State observations (N, state_dim)
- `images`: RGB images (N, H, W, 3) - uint8
- `actions`: Actions (N, 7) - [dx, dy, dz, droll, dpitch, dyaw, gripper]
- `rewards`: Step rewards (N,)
- `dones`: Episode done flags (N,)
- `episode_ids`: Episode indices (N,)
- `timesteps`: Step indices within episodes (N,)

## Environment API

The environment uses a simplified wrapper around ManiSkill3:

```python
from envs.maniskill_env import make_maniskill_env, ManiSkillExpertPolicy

# Single environment
env = make_maniskill_env(obs_mode="state_image", num_envs=1, seed=42)
obs, info = env.reset()
print(obs["state"].shape, obs["rgb"].shape)

# Get expert action
expert = ManiSkillExpertPolicy()
expert.reset()
raw_obs = info["raw_obs"]  # Contains structured obs for expert
action = expert.get_action(raw_obs, info)

# VecEnv (GPU parallel)
vec_env = make_maniskill_env(obs_mode="state", num_envs=16, seed=42)
obs, info = vec_env.reset()
print(obs.shape)  # (16, state_dim) tensor
```

## PickCube-v1 Task Details

**Success Conditions:**
- Cube position within 0.025m of goal
- Robot is static (joint velocities < 0.2)

**Action Space:** `pd_ee_delta_pose` (7D)
- Position delta: [dx, dy, dz]
- Rotation delta: [droll, dpitch, dyaw]
- Gripper: [-1 close, +1 open]

**Observation Modes:**
- `state`: 42D state vector (qpos, qvel, tcp_pose, obj_pose, goal_pos, etc.)
- `state_image`: State + 128x128 RGB image
- `image`: 128x128 RGB image only

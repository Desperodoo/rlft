# Pick-and-Place Multi-Step Manipulation Task

You are extending an existing PyTorch codebase called **toy_diffusion_rl** that already implements:

- Diffusion Policy
- Flow Matching / Reflected Flow / Consistency Flow
- Diffusion Double Q
- CPQL
- DPPO (BC pretrain only)
- ReinFlow (BC pretrain only)

and validates them on simple particle / ring distributions and 2D point mass.

Now I want to add **a multi-step, manipulation-style task** and fully validate **online fine-tuning** for **DPPO** and **ReinFlow** under both **state** and **image** observations.

Please carefully follow this specification and modify/extend the repo accordingly.

## High-Level Goal

Add a new toy **robotic manipulation**-style environment and dataset, and wire it into DPPO and ReinFlow so that I can:

1. Do **offline pretraining** (BC / flow / diffusion-style) on a fixed dataset.
2. Then do **online fine-tuning** in the environment using:
   - DPPO (Policy Gradient with diffusion policy)
   - ReinFlow (Policy Gradient with flow-matching policy)

### Requirements for the new task

1. **Manipulation-related**: pick-and-place style or push-style task.
2. **Simple to configure**: no heavy engines like Isaac / SAPIEN. Using Gymnasium-Robotics + MuJoCo is OK.
3. **Ready-to-use offline dataset**:
   - Provide a script that generates and saves a reusable offline dataset file for this environment.
   - Offline training for DPPO/ReinFlow should directly load this dataset.
4. **Both vector state and image state**:
   - Implement both a **state-based** and an **image-based** version of the environment / dataset.
   - The code should allow me to easily switch between them.

Your implementation should be clean, modular, and consistent with the existing toy_diffusion_rl structure.

## Reference Environment / Datasets

Use an existing, simple robotic manipulation environment as the base:

- **Gymnasium-Robotics Fetch Pick And Place**:
  - Docs: https://robotics.farama.org/envs/fetch/pick_and_place/ (FetchPickAndPlace)
  - This is a 7-DoF manipulator + gripper doing pick-and-place. It is a classic example of a multi-step manipulation task.

We will use **gymnasium-robotics** with **MuJoCo** as the primary backend for the Fetch environment.

## Step 1 – New Multi-Step Manipulation Environment(s)

Add a new submodule under `envs/`:

- `envs/pick_and_place.py`

Inside this file implement the following:

### 1.1 Factory Function

A high-level factory function:

```python
def make_pick_and_place_env(
    backend: str = "fetch",       # "fetch"
    obs_mode: str = "state",      # "state" or "image"
    seed: int = 0,
    max_episode_steps: int = 50,
) -> gym.Env:
    ...
```

### 1.2 Fetch-based Environment Wrapper (Preferred Backend)

If `backend == "fetch"`:

- Use `gymnasium.make("FetchPickAndPlace-v2", ...)` or the current robotics name (check robotics docs).
- Wrap it so that:

#### For `obs_mode == "state"`:

- Return a 1D numpy array concatenating:
  - `obs_dict["observation"]`
  - `obs_dict["achieved_goal"]`
  - `obs_dict["desired_goal"]`

#### For `obs_mode == "image"`:

- Use `env.render()` / `env.unwrapped.mujoco_renderer.render(mode="rgb_array")` to get an RGB image.
- Option A: Observation = image only.
- Option B (better): Observation = dict with "state" and "image".
- Provide helper wrappers to map this dict into either:
  - Just image (for vision-only policy)
  - Or (state, image) pair (for multimodal policies)

#### General:

- Ensure `max_episode_steps` is enforced (TimeLimit wrapper).
## Step 2 – Offline Dataset Generation (State + Image)

Create a script: `scripts/generate_pick_and_place_dataset.py`

### Script Arguments (via argparse)

- `--backend` ("fetch")
- `--obs_mode` ("state" or "image" or "state_image")
- `--num_episodes`
- `--max_episode_steps`
- `--output_path` (e.g., `data/pick_and_place_fetch_state.h5`)

### Implementation Details

1. Use `make_pick_and_place_env` to create the environment.

2. Implement a simple scripted expert policy for data collection (no need for deep RL here):

   - **For Fetch**:
     - Use the positions of gripper, block, and target.
     - Phased heuristic: Move gripper above block; close gripper; move with block toward goal; release when near goal.

3. For each episode, record transitions:

   - `state`: vector observation (even if `obs_mode="image"`, still record the internal state for analysis).
   - `image`: if `obs_mode` includes images, record image as uint8 (H, W, C).
   - `action`
   - `next_state`
   - `next_image` (if applicable)
   - `reward`
   - `done`
   - `episode_id` and `timestep`

4. Save as a compressed HDF5 or NPZ file, for example with keys:

   - `"obs"` / `"next_obs"` (state vectors)
   - `"images"` / `"next_images"` (optional)
   - `"actions"`, `"rewards"`, `"dones"`, `"episode_ids"`

### Example Usage

```bash
python scripts/generate_pick_and_place_dataset.py \
    --backend fetch \
    --obs_mode state_image \
    --num_episodes 200 \
    --max_episode_steps 50 \
    --output_path data/fetch_pick_and_place_state_image.h5
```

## Step 3 – Dataset Loader Utilities

In `common/`, add a new file: `common/dataset_loader.py`

### Implementation

Implement helper functions/classes:

#### PickAndPlaceOfflineDataset

```python
class PickAndPlaceOfflineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        obs_mode: str = "state",    # "state", "image", or "state_image"
        transform: Optional[Callable] = None,
    ): ...
```

The class should:

1. Load the HDF5 file.

2. Provide items based on `obs_mode`:

   - **For state-only**:
     ```python
     {"obs": ..., "action": ..., "next_obs": ..., "reward": ..., "done": ...}
     ```

   - **For image-only**:
     ```python
     {"image": ..., "action": ..., ...}
     ```

   - **For state+image**:
     ```python
     {"obs": ..., "image": ..., "action": ..., ...}
     ```

3. Optionally provide a `make_dataloader(...)` helper that creates a PyTorch DataLoader with shuffling, batch_size, etc.

4. Make sure this is consistent with how your existing offline algorithms (Diffusion Policy, Flow Matching, etc.) expect batches.

## Step 3.5 – Design Unified Vision Encoder Interface

To efficiently support both state and image observations across all algorithms (Diffusion Policy, Flow Matching, Diffusion-QL, CPQL, DPPO, ReinFlow), we need a **unified and pluggable vision encoder architecture**.

### Vision Encoder Families

Implement two complementary encoder families in `common/vision_encoders.py`:

#### 1. Lightweight CNN Encoder (Diffusion Policy Style)

A compact, trainable-from-scratch CNN encoder:

```python
class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for image observations (train from scratch).
    
    Architecture: 3-4 conv layers → flatten → MLP head
    Output: Fixed-dimension embedding
    """
    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # (H, W, C)
        output_dim: int = 128,
        channels: List[int] = [32, 64, 64],
        kernel_sizes: List[int] = [8, 4, 3],
        strides: List[int] = [4, 2, 1],
    ):
        ...
```

**Key features**:
- Train from scratch on pick-and-place task
- Fast inference
- Suitable for online RL fine-tuning (no risk of encoder collapse)

#### 2. Pre-trained Vision Transformer Encoder (Frozen or Semi-Frozen)

Use a pre-trained DINOv3-ViT-S/16 backbone:

```python
class DINOv3Encoder(nn.Module):
    """Pre-trained DINOv3-ViT-S/16 encoder for image observations.
    
    Can be frozen entirely or allow semi-supervised fine-tuning.
    Uses DINO self-supervised features directly.
    """
    def __init__(
        self,
        output_dim: int = 384,  # ViT-S output is 384-dim
        freeze: bool = True,    # Freeze backbone
        fine_tune_layers: int = 0,  # Number of last layers to unfreeze (0 = fully frozen)
    ):
        ...
```

**Key features**:
- Leverage pre-trained semantic features
- Option to freeze or partially fine-tune
- Handles diverse visual inputs robustly

### Unified Encoder Interface

Create a factory function for flexible encoder selection:

```python
def make_vision_encoder(
    encoder_type: str,        # "cnn" or "dinov3"
    image_shape: Tuple[int, int, int],
    output_dim: int = 128,
    freeze: bool = True,      # For DINOv3
    **kwargs
) -> nn.Module:
    """Create a vision encoder based on config.
    
    Args:
        encoder_type: "cnn" or "dinov3"
        image_shape: (H, W, C) of input images
        output_dim: Output embedding dimension
        freeze: Whether to freeze (for DINOv3)
    
    Returns:
        Encoder module with input (images) → output (embeddings)
    """
    if encoder_type == "cnn":
        return CNNEncoder(image_shape=image_shape, output_dim=output_dim, **kwargs)
    elif encoder_type == "dinov3":
        return DINOv3Encoder(output_dim=output_dim, freeze=freeze, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
```

### Encoder Integration with Policy Networks

For any algorithm handling state+image, modify network classes to accept an optional vision encoder:

```python
class DiffusionNoisePredictor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        vision_encoder: Optional[nn.Module] = None,  # Optional image encoder
        vision_encoder_output_dim: int = 128,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        
        # Compute total input dimension
        total_input_dim = state_dim
        if vision_encoder is not None:
            total_input_dim += vision_encoder_output_dim
        
        # Policy MLP
        self.mlp = MLP(total_input_dim, action_dim, hidden_dims)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor, 
                image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            state: State vector (B, state_dim)
            action: Action (B, action_dim)
            t: Time embedding (B,)
            image: Optional image observations (B, C, H, W)
        
        Returns:
            Predicted noise (B, action_dim)
        """
        if image is not None and self.vision_encoder is not None:
            # Encode image
            image_features = self.vision_encoder(image)  # (B, vision_encoder_output_dim)
            # Concatenate with state
            state_action = torch.cat([state, image_features], dim=1)
        else:
            state_action = state
        
        # Rest of the policy forward pass...
        return self.mlp(state_action, action, t)
```

### Config-Based Encoder Selection

In YAML configs, specify which encoder to use:

```yaml
vision_encoder:
  type: "cnn"              # or "dinov3"
  output_dim: 128
  freeze: false            # For DINOv3: whether to freeze backbone
  fine_tune_layers: 2      # For DINOv3: number of layers to unfreeze
  
  # CNN-specific
  cnn:
    channels: [32, 64, 64]
    kernel_sizes: [8, 4, 3]
    strides: [4, 2, 1]
```

### Implementation Checklist

- [ ] Create `common/vision_encoders.py` with `CNNEncoder` and `DINOv3Encoder`
- [ ] Add `make_vision_encoder()` factory function
- [ ] Update all network classes (`DiffusionNoisePredictor`, `FlowVelocityPredictor`, etc.) to optionally accept a `vision_encoder`
- [ ] Update `create_agent()` in `train.py` to instantiate vision encoder from config
- [ ] Ensure all algorithms can handle state-only, image-only, and state+image modes



The DPPO agent already has full online RL fine-tuning implemented in `algorithms/dppo/agent.py`.

Now we only need to **adapt it for the new pick-and-place task** with support for both state and image observations.

**Reference**:
- DPPO agent: `algorithms/dppo/agent.py` (already has `collect_rollout` and `update` methods)
- Paper / website: https://diffusion-ppo.github.io/

### Adaptation Tasks

#### 1. Support Image-based Observations

Extend the agent's encoder to handle:

- State-only input (existing MLP encoder)
- Image input (add CNN encoder)
- (state + image) combined input (add fusion layer)

Modify `DiffusionNoisePredictor` in `common/networks.py` or create a wrapper.

#### 2. Load Pick-and-Place Dataset for Offline Pretraining

- Use `PickAndPlaceOfflineDataset` (from Step 3) to pretrain the diffusion policy
- Keep existing BC-style pretraining logic

#### 3. Test Online Fine-Tuning on New Environment

- Ensure `collect_rollout` and `update` methods work with `make_pick_and_place_env`
- Verify GAE advantage computation and PPO loss work correctly
- Test partial chain fine-tuning (last K denoising steps trainable)

## Step 5 – Adapt ReinFlow for Pick-and-Place Task

The ReinFlow agent already has online RL fine-tuning implemented in `algorithms/reinflow/agent.py`.

Now we only need to **adapt it for the new pick-and-place task** with support for both state and image observations.

**Reference**:
- ReinFlow agent: `algorithms/reinflow/agent.py` (already has `collect_rollout` and `online_update` methods)
- Website: https://reinflow.github.io/

### Adaptation Tasks

#### 1. Support Image-based Observations

Extend the agent's encoder to handle:

- State-only input (existing MLP encoder)
- Image input (add CNN encoder)
- (state + image) combined input (add fusion layer)

Modify the `FlowVelocityPredictor` and `NoisyFlowMLP` in `common/networks.py` or wrapper class.

#### 2. Load Pick-and-Place Dataset for Offline Pretraining

- Use `PickAndPlaceOfflineDataset` (from Step 3) to pretrain the flow-matching policy
- Keep existing flow-matching loss (momentum matching or teacher forcing)

#### 3. Test Online Fine-Tuning on New Environment

- Ensure `collect_rollout` and `online_update` methods work with `make_pick_and_place_env`
- Verify that the `NoisyFlowMLP` exploration noise network adapts properly
- Test actor-critic learning with value function baseline
- Verify REINFORCE/PPO-style updates of flow parameters

## Step 6 – Add Configs and Update Training Scripts

### Add New Config Files in `configs/`

Create config files for pick-and-place task:

- `configs/dppo_pick_and_place_state.yaml`
- `configs/dppo_pick_and_place_image.yaml`
- `configs/reinflow_pick_and_place_state.yaml`
- `configs/reinflow_pick_and_place_image.yaml`

### Config Specification

Each config should specify:

```yaml
env:
  name: "pick_and_place"
  backend: "fetch"
  obs_mode: "state" or "image" or "state_image"
  max_episode_steps: 50

training:
  total_iterations: ...  # Number of PPO iterations
  rollout_steps: ...     # Steps per rollout
  eval_interval: ...

pretrain:
  enabled: true
  offline_dataset: "data/fetch_pick_and_place_state_image.h5"
  num_steps: ...         # Number of offline pretraining steps

diffusion:  # for DPPO
  num_diffusion_steps: 5
  noise_schedule: "linear"

flow:  # for ReinFlow
  num_flow_steps: 10

ppo:
  clip_ratio: 0.2
  ppo_epochs: 10
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5

network:
  hidden_dims: [256, 256]
  # For image-based: add CNN architecture
  # cnn_channels: [32, 64, 64]
  # cnn_kernel_sizes: [8, 4, 3]
  # cnn_strides: [4, 2, 1]

optimizer:
  learning_rate: 3.0e-4
```

### Update `train.py` and `eval.py`

#### In `train.py`:

1. Register `"pick_and_place"` environment in `create_agent()` 
2. Add `--obs_mode` CLI argument
3. Load offline dataset from config if `pretrain.offline_dataset` is specified
4. Verify DPPO/ReinFlow agents work with new environment

#### In `eval.py`:

1. Add support for evaluating on pick-and-place environment
2. Add `--obs_mode` argument
3. Add `--save_video` flag to render episodes

## Step 7 – Minimal Docs / README Update
In the main `README.md`, add a new section: **"Multi-step Manipulation Task: Pick-and-Place"**

### Content

1. **Briefly describe the env**

2. **Show how to generate dataset**:

```bash
python scripts/generate_pick_and_place_dataset.py \
    --backend fetch \
    --obs_mode state_image \
    --num_episodes 200 \
    --max_episode_steps 50 \
    --output_path data/fetch_pick_and_place_state_image.h5
```

3. **Show example training commands** for DPPO and ReinFlow (state vs image).

4. **Code style**: Make sure all new code is consistent with the existing coding style in `toy_diffusion_rl`, with clear comments and docstrings.

---

## Notes

如有其他改进或优化需求，欢迎继续反馈。
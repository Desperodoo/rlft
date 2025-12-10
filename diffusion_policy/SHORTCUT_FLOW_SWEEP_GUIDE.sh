#!/bin/bash
#
# ShortCut Flow Design Sweep - Quick Start Guide
#
# This script shows how to run the ShortCut Flow parameter sweep with monitoring

# 1. Start the sweep (using GPUs 2-9)
# ./sweep_shortcut_flow_parallel.sh --gpus "2 3 4 5 6 7 8 9"

# 2. In another terminal, monitor the sweep
# ./monitor_shortcut_flow.sh

# Optional parameters for sweep:
#   --gpus "2 3 4"              - Specify GPU IDs (default: 2 3 4 5 6 7 8 9)
#   --dry-run                   - Show commands without running
#   --env ENV_ID                - Environment ID (default: LiftPegUpright-v1)
#   --iters N                   - Total iterations (default: 30000)
#   --num-demos N               - Number of demos (default: 1000)
#   --wandb-project PROJECT     - W&B project name
#   --control-mode MODE         - Control mode
#   --sim-backend BACKEND       - Simulation backend

# Optional parameters for monitor:
#   --log-dir DIR               - Log directory (default: /tmp/shortcut_flow_grid)
#   --interval N                - Refresh interval in seconds (default: 5)

# === Ablation Categories ===
#
# 1. BASELINE (default config)
#    - power2 step size sampling
#    - uniform time sampling
#    - velocity target mode
#    - 2-step teacher
#    - EMA teacher
#    - adaptive inference
#
# 2. STEP SIZE SAMPLING (3 variants)
#    - uniform: Uniform sampling in [min, max]
#    - fixed_small: Fixed small step (0.05)
#    - fixed_large: Fixed large step (0.25)
#
# 3. TIME SAMPLING (2 variants)
#    - truncated: Sample directly in valid [t_min, 1-2d]
#    - restricted: Limit to [0.1, 0.9] range
#
# 4. TARGET COMPUTATION (1 variant)
#    - endpoint: Use endpoint x1 instead of velocity
#
# 5. TEACHER NETWORK (3 variants)
#    - 1step: Single-step teacher rollout
#    - 3step: Three-step teacher rollout
#    - online: Use online network instead of EMA
#
# 6. INFERENCE MODE (3 variants)
#    - uniform: Uniform dt throughout inference
#    - 4steps: Fixed 4 steps (fewer steps)
#    - 16steps: Fixed 16 steps (more steps)
#
# 7. LOSS WEIGHTS (3 variants)
#    - flow_only: Only flow loss (consistency_weight=0)
#    - shortcut_heavy: Emphasize shortcut loss (2:1 ratio)
#    - flow_heavy: Emphasize flow loss (1:2 ratio)
#
# 8. CONSISTENCY FRACTION (3 variants)
#    - k10: Only 10% of batch for consistency
#    - k50: 50% of batch for consistency
#    - k100: 100% of batch for consistency

echo "ShortCut Flow Sweep - Quick Start"

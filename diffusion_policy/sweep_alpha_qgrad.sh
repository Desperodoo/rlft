#!/bin/bash
#
# Grid search for Alpha and Q-gradient chain length (Diffusion Double Q)
#
# This script performs a 2D grid search over:
# - Alpha: Regularization strength for Q-loss
# - Q-grad modes: Single step, partial chain, or full chain gradient
#
# Usage: ./sweep_alpha_qgrad.sh [--dry-run] [--gpu GPU_ID] [--env ENV_ID]

set -e

# Default configurations
GPU_ID=0
DRY_RUN=false
TOTAL_ITERS=30000
EVAL_FREQ=2000
LOG_FREQ=100
NUM_EVAL_EPISODES=100
NUM_EVAL_ENVS=100
NUM_DEMOS=1000
SIM_BACKEND="physx_cuda"
WANDB_PROJECT="maniskill_alpha_qgrad_grid"
MAX_EPISODE_STEPS=100
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --env)
            ENV_ID="$2"
            shift 2
            ;;
        --iters)
            TOTAL_ITERS="$2"
            shift 2
            ;;
        --num-demos)
            NUM_DEMOS="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --control-mode)
            CONTROL_MODE="$2"
            shift 2
            ;;
        --sim-backend)
            SIM_BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Sparse grid for alpha and q-grad modes
# Alpha values - coarse grid to avoid too many runs
ALPHAS=(
    0.01
    0.05
    0.2
    1.0
)

# Q-gradient modes - representative points
# single_step (fast), last_few variants (moderate), whole_grad (slow but accurate)
Q_GRAD_CONFIGS=(
    "single_step:0"       # Fast approximation
    "last_few:5"          # Moderate (default)
    "whole_grad:0"        # Full chain
)

# Seeds for multiple runs
SEEDS=(0)

# Demo path based on environment
DEMO_PATH="$HOME/.maniskill/demos/${ENV_ID}/rl/trajectory.rgb.${CONTROL_MODE}.physx_cuda.h5"

echo "=========================================="
echo "Alpha + Q-gradient Grid Search"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Algorithm: diffusion_double_q"
echo "Environment: $ENV_ID"
echo "Control Mode: $CONTROL_MODE"
echo "Sim Backend: $SIM_BACKEND"
echo "Demo Path: $DEMO_PATH"
echo "Alpha values: ${ALPHAS[*]}"
echo "Q-grad configs: ${Q_GRAD_CONFIGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total iterations: $TOTAL_ITERS"
echo "Eval frequency: $EVAL_FREQ"
echo "Num demos: $NUM_DEMOS"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

# Change to the diffusion_policy directory
cd /home/wjz/rlft/diffusion_policy

# Check if demo file exists
if [ ! -f "$DEMO_PATH" ]; then
    echo "Error: Demo file not found: $DEMO_PATH"
    echo "Please check the environment ID and demo path."
    exit 1
fi

# Calculate total experiments
TOTAL_EXPS=$((${#ALPHAS[@]} * ${#Q_GRAD_CONFIGS[@]} * ${#SEEDS[@]}))
EXP_IDX=0

for alpha in "${ALPHAS[@]}"; do
    for config in "${Q_GRAD_CONFIGS[@]}"; do
        # Parse config (format: "mode:steps")
        Q_GRAD_MODE="${config%%:*}"
        Q_GRAD_STEPS="${config##*:}"
        
        for seed in "${SEEDS[@]}"; do
            EXP_IDX=$((EXP_IDX + 1))
            
            # Generate experiment name
            if [ "$Q_GRAD_MODE" = "single_step" ]; then
                q_grad_suffix="q_single"
            elif [ "$Q_GRAD_MODE" = "whole_grad" ]; then
                q_grad_suffix="q_whole"
            else
                q_grad_suffix="q_last${Q_GRAD_STEPS}"
            fi
            
            exp_name="ddql-${ENV_ID}-alpha${alpha}-${q_grad_suffix}-seed${seed}"
            
            echo "[$EXP_IDX/$TOTAL_EXPS] Running: $exp_name"
            
            # Build command
            CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python train_offline_rl.py \
                --algorithm diffusion_double_q \
                --env_id $ENV_ID \
                --obs_mode rgb \
                --demo_path $DEMO_PATH \
                --control-mode $CONTROL_MODE \
                --sim-backend $SIM_BACKEND \
                --num-demos $NUM_DEMOS \
                --seed $seed \
                --total_iters $TOTAL_ITERS \
                --eval_freq $EVAL_FREQ \
                --log_freq $LOG_FREQ \
                --num_eval_episodes $NUM_EVAL_EPISODES \
                --num_eval_envs $NUM_EVAL_ENVS \
                --exp_name $exp_name \
                --track \
                --wandb_project_name $WANDB_PROJECT \
                --max_episode_steps $MAX_EPISODE_STEPS \
                --alpha $alpha \
                --q_grad_mode $Q_GRAD_MODE"
            
            # Add q_grad_steps for last_few mode
            if [ "$Q_GRAD_MODE" = "last_few" ]; then
                CMD="$CMD --q_grad_steps $Q_GRAD_STEPS"
            fi
            
            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] Would execute:"
                echo "  $CMD"
                echo ""
            else
                echo "  Starting..."
                # Run and capture output
                if eval "$CMD"; then
                    echo "  ✓ $exp_name completed successfully"
                else
                    echo "  ✗ $exp_name failed"
                    # Continue with other experiments even if one fails
                fi
            fi
            
            echo ""
        done
    done
done

echo "=========================================="
echo "Grid search completed!"
echo "=========================================="

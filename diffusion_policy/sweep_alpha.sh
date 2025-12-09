#!/bin/bash
# Alpha parameter sweep for CPQL and Diffusion Double Q algorithms
# Usage: ./sweep_alpha.sh [--dry-run] [--gpu GPU_ID] [--env ENV_ID]

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
WANDB_PROJECT="maniskill_alpha_sweep"
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

# Algorithms to sweep
ALGORITHMS=(
    "cpql"
    "diffusion_double_q"
)

# Alpha values to sweep
ALPHAS=(
    0.005
    0.01
    0.02
    0.05
    0.1
    0.2
    0.5
    1.0
)

# Demo path based on environment
DEMO_PATH="$HOME/.maniskill/demos/${ENV_ID}/rl/trajectory.rgb.${CONTROL_MODE}.physx_cuda.h5"

# Seeds for multiple runs
SEEDS=(0)

echo "=========================================="
echo "Alpha Parameter Sweep"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Environment: $ENV_ID"
echo "Control Mode: $CONTROL_MODE"
echo "Sim Backend: $SIM_BACKEND"
echo "Demo Path: $DEMO_PATH"
echo "Alpha values: ${ALPHAS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total iterations: $TOTAL_ITERS"
echo "Eval frequency: $EVAL_FREQ"
echo "Num demos: $NUM_DEMOS"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

# Change to the diffusion_policy directory
cd /home/amax/rlft/diffusion_policy

# Check if demo file exists
if [ ! -f "$DEMO_PATH" ]; then
    echo "Error: Demo file not found: $DEMO_PATH"
    echo "Please check the environment ID and demo path."
    exit 1
fi

# Calculate total experiments
TOTAL_EXPS=$((${#ALGORITHMS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]}))
EXP_IDX=0

for algo in "${ALGORITHMS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP_IDX=$((EXP_IDX + 1))
            
            exp_name="${algo}-${ENV_ID}-alpha${alpha}-seed${seed}"
            
            echo "[$EXP_IDX/$TOTAL_EXPS] Running: $exp_name"
            
            CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python train_offline_rl.py \
                --algorithm $algo \
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
                --alpha $alpha"
            
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
echo "Alpha sweep completed!"
echo "=========================================="

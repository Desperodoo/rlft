#!/bin/bash
# Batch experiments for Offline RL algorithms on ManiSkill3
# Usage: ./run_experiments.sh [--dry-run] [--gpu GPU_ID] [--algorithms "algo1 algo2 ..."]

set -e

# Default configurations
GPU_ID=0
DRY_RUN=false
TOTAL_ITERS=30000
EVAL_FREQ=5000
LOG_FREQ=1000
NUM_EVAL_EPISODES=100
NUM_EVAL_ENVS=10
NUM_DEMOS=950
SIM_BACKEND="physx_cpu"
WANDB_PROJECT="maniskill_offline_rl"
MAX_EPISODE_STEPS=100

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
        --algorithms)
            SELECTED_ALGORITHMS="$2"
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
        --max-episode-steps)
            MAX_EPISODE_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Algorithms to run
if [ -z "$SELECTED_ALGORITHMS" ]; then
    ALGORITHMS=(
        "diffusion_policy"
        "flow_matching"
        "reflected_flow"
        "consistency_flow"
        "shortcut_flow"
        "diffusion_double_q"
        "cpql"
    )
else
    read -ra ALGORITHMS <<< "$SELECTED_ALGORITHMS"
fi

# Environments and their demo paths (state_dict+rgb format)
declare -A ENVS
ENVS["PickCube-v1"]="$HOME/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
# Add more environments here:
# ENVS["PegInsertionSide-v1"]="$HOME/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
# ENVS["StackCube-v1"]="$HOME/.maniskill/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"

# Seeds for multiple runs
SEEDS=(0)

echo "=========================================="
echo "Offline RL Batch Experiments"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Environments: ${!ENVS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total iterations: $TOTAL_ITERS"
echo "Eval frequency: $EVAL_FREQ"
echo "Num demos: $NUM_DEMOS"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

# Change to the diffusion_policy directory
cd /home/amax/rlft/diffusion_policy

# Calculate total experiments
TOTAL_EXPS=$((${#ALGORITHMS[@]} * ${#ENVS[@]} * ${#SEEDS[@]}))
EXP_IDX=0

for env in "${!ENVS[@]}"; do
    demo_path="${ENVS[$env]}"
    
    # Check if demo file exists
    if [ ! -f "$demo_path" ]; then
        echo "Warning: Demo file not found for $env: $demo_path"
        continue
    fi
    
    for algo in "${ALGORITHMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP_IDX=$((EXP_IDX + 1))
            
            exp_name="${algo}-${env}-seed${seed}"
            
            echo "[$EXP_IDX/$TOTAL_EXPS] Running: $exp_name"
            
            CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python train_offline_rl.py \
                --algorithm $algo \
                --env_id $env \
                --obs_mode rgb \
                --demo_path $demo_path \
                --control-mode pd_ee_delta_pos \
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
                --max_episode_steps $MAX_EPISODE_STEPS"
            
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
echo "Batch experiments completed!"
echo "=========================================="

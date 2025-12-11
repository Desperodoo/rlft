#!/bin/bash
#
# Parallel Grid search for AWCP beta parameter
# Auto-allocates tasks to available GPUs
#
# Usage: ./sweep_awcp_beta_parallel.sh [--gpus "3 4 5 6 7 8"] [--dry-run] [--env ENV_ID]

set -e

# Default configurations
GPUS=(2 3 4 5 6 7 8 9)
DRY_RUN=false
TOTAL_ITERS=30000
EVAL_FREQ=2000
LOG_FREQ=100
NUM_EVAL_EPISODES=100
NUM_EVAL_ENVS=50
NUM_DEMOS=1000
SIM_BACKEND="physx_cuda"
WANDB_PROJECT="maniskill_awcp_finetune_beta_grid"
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
        --gpus)
            IFS=' ' read -ra GPUS <<< "$2"
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

# Beta grid: from very low (almost no weighting) to high (very selective)
# beta=0 means uniform weighting (equivalent to pure BC)
# beta>0 means high-advantage samples get higher weight
# Recommended range: 0.1 to 10.0
BETAS=(10.0 20.0 50.0 100.0)
WEIGHT_CLIPS=(50.0 100.0 200.0 400.0)  # Can also sweep this if needed
SEEDS=(0)

DEMO_PATH="$HOME/.maniskill/demos/${ENV_ID}/rl/trajectory.rgb.${CONTROL_MODE}.physx_cuda.h5"
LOG_DIR="/tmp/awcp_finetune_beta_grid_search"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "AWCP Beta Parameter Sweep"
echo "=========================================="
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Beta values: ${BETAS[*]}"
echo "Total experiments: $((${#BETAS[@]} * ${#WEIGHT_CLIPS[@]} * ${#SEEDS[@]}))"
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

cd /home/wjz/rlft/diffusion_policy

# Check demo file
if [ ! -f "$DEMO_PATH" ]; then
    echo "Error: Demo file not found: $DEMO_PATH"
    exit 1
fi

# Generate all task combinations
declare -a TASKS
TASK_IDX=0
for beta in "${BETAS[@]}"; do
    for weight_clip in "${WEIGHT_CLIPS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            TASKS[$TASK_IDX]="$beta|$weight_clip|$seed"
            ((TASK_IDX+=1))
        done
    done
done

TOTAL_TASKS=${#TASKS[@]}
COMPLETED=0
FAILED=0

# Function to run a single task
run_task() {
    local gpu=$1
    local task=$2
    local task_num=$3
    
    IFS='|' read -r beta weight_clip seed <<< "$task"
    
    # Generate experiment name
    exp_name="awcp-${ENV_ID}-beta${beta}-wclip${weight_clip}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"
    
    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"
    
    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_offline_rl.py \
        --algorithm awcp \
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
        --beta $beta \
        --weight_clip $weight_clip"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[GPU $gpu] [DRY RUN] $exp_name"
        echo "$CMD" > "$log_file"
        return 0
    else
        if eval "$CMD" >> "$log_file" 2>&1; then
            echo "[GPU $gpu] ✓ Completed: $exp_name"
            return 0
        else
            echo "[GPU $gpu] ✗ Failed: $exp_name (see $log_file)"
            return 1
        fi
    fi
}

# Parallel task dispatch
declare -A gpu_queue
for gpu in "${GPUS[@]}"; do
    gpu_queue[$gpu]=""
done

next_task=0
active_pids=()
gpu_mapping=()

# Function to assign tasks to available GPUs
dispatch_tasks() {
    for gpu in "${GPUS[@]}"; do
        # Check if process is still running
        if [ -n "${gpu_queue[$gpu]}" ]; then
            local pid=${gpu_queue[$gpu]}
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished
                if wait "$pid" 2>/dev/null; then
                    ((COMPLETED+=1))
                else
                    ((FAILED+=1))
                fi
                gpu_queue[$gpu]=""
            fi
        fi
        
        # Assign new task if GPU is free
        if [ -z "${gpu_queue[$gpu]}" ] && [ $next_task -lt $TOTAL_TASKS ]; then
            run_task "$gpu" "${TASKS[$next_task]}" $((next_task + 1)) &
            gpu_queue[$gpu]=$!
            ((next_task+=1))
        fi
    done
}

# Main dispatch loop
while [ $next_task -lt $TOTAL_TASKS ] || [ $COMPLETED -lt $TOTAL_TASKS ]; do
    dispatch_tasks
    sleep 2
done

# Wait for remaining processes
for gpu in "${GPUS[@]}"; do
    if [ -n "${gpu_queue[$gpu]}" ]; then
        wait "${gpu_queue[$gpu]}" || ((FAILED+=1))
        ((COMPLETED+=1))
    fi
done

echo ""
echo "=========================================="
echo "AWCP Beta sweep completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="

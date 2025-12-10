#!/bin/bash
#
# Parallel Grid search for Alpha and Q-gradient chain length (cpql)
# Auto-allocates tasks to available GPUs
#
# Usage: ./sweep_alpha_qgrad_parallel.sh [--gpus "3 4 5 6 7 8"] [--dry-run] [--env ENV_ID]

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
WANDB_PROJECT="maniskill_cpql_qlnorm_alpha_qgrad_grid"
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

# Sparse grid for alpha and q-grad modes
ALPHAS=(0.01 0.05 0.2 1.0)
Q_GRAD_CONFIGS=(
    "single_step:0"
    "last_few:5"
    # "whole_grad:0"
)
SEEDS=(0)

DEMO_PATH="$HOME/.maniskill/demos/${ENV_ID}/rl/trajectory.rgb.${CONTROL_MODE}.physx_cuda.h5"
LOG_DIR="/tmp/cpql_qlnorm_grid_search"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Parallel Grid Search (Alpha + Q-gradient)"
echo "=========================================="
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Total experiments: $((${#ALPHAS[@]} * ${#Q_GRAD_CONFIGS[@]} * ${#SEEDS[@]}))"
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
for alpha in "${ALPHAS[@]}"; do
    for config in "${Q_GRAD_CONFIGS[@]}"; do
        Q_GRAD_MODE="${config%%:*}"
        Q_GRAD_STEPS="${config##*:}"
        for seed in "${SEEDS[@]}"; do
            TASKS[$TASK_IDX]="$alpha|$Q_GRAD_MODE|$Q_GRAD_STEPS|$seed"
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
    
    IFS='|' read -r alpha q_grad_mode q_grad_steps seed <<< "$task"
    
    # Generate experiment name
    if [ "$q_grad_mode" = "single_step" ]; then
        q_grad_suffix="q_single"
    elif [ "$q_grad_mode" = "whole_grad" ]; then
        q_grad_suffix="q_whole"
    else
        q_grad_suffix="q_last${q_grad_steps}"
    fi
    
    exp_name="cpql-${ENV_ID}-alpha${alpha}-${q_grad_suffix}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"
    
    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"
    
    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_offline_rl.py \
        --algorithm cpql \
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
        --q_grad_mode $q_grad_mode"
    
    if [ "$q_grad_mode" = "last_few" ]; then
        CMD="$CMD --q_grad_steps $q_grad_steps"
    fi
    
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
echo "Grid search completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="

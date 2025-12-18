#!/bin/bash
#
# Parallel hyperparameter sweep for RLPD Online training
# Auto-allocates tasks to available GPUs
#
# Based on official RLPD paper and ManiSkill RLPD implementation:
# - Paper: "Efficient Online Reinforcement Learning with Offline Data" (Ball et al., ICML 2023)
# - Code: https://github.com/ikostrikov/rlpd
# - ManiSkill: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/rlpd
#
# Key hyperparameter configurations:
# 1. Walltime Efficient: num_envs=50, num_qs=2, utd_ratio=4 (fast but more samples)
# 2. Sample Efficient: num_envs=8, num_qs=10, utd_ratio=16-20 (slower but fewer samples)
#
# Critical RLPD settings (from paper):
# - discount (gamma): 0.9 for short episodes (ManiSkill), 0.99 for long episodes
# - backup_entropy: False (critical for sparse rewards)
# - num_qs: 10, num_min_qs: 2 (REDQ-style ensemble with subsample)
# - critic_layer_norm: True (improves stability with offline data)
# - online_ratio: 0.5 (50% online + 50% offline data mixing)
#
# Usage:
#   ./sweep_rlpd_online_parallel.sh [--gpus "0 1 2 3"] [--dry-run] [--env ENV_ID] [--mode walltime|sample]
#
# Examples:
#   # Walltime efficient sweep on GPUs 0,1
#   ./sweep_rlpd_online_parallel.sh --gpus "0 1" --mode walltime --env LiftPegUpright-v1
#
#   # Sample efficient sweep (uses fewer envs)
#   ./sweep_rlpd_online_parallel.sh --gpus "0 1" --mode sample --env LiftPegUpright-v1

set -e

# ============== Default Configuration ==============
# These defaults are based on the RLPD paper and ManiSkill experiments
GPUS=(0 1)
DRY_RUN=false
MODE="walltime"  # "walltime" or "sample"

# Environment settings
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
SIM_BACKEND="physx_cuda"
MAX_EPISODE_STEPS=100
OBS_MODE="rgb"

# Training settings (adjusted per mode below)
TOTAL_TIMESTEPS=500000
EVAL_FREQ=10000
LOG_FREQ=1000
SAVE_FREQ=50000
NUM_EVAL_EPISODES=50
NUM_EVAL_ENVS=10
NUM_SEED_STEPS=5000

# Algorithm to sweep
ALGORITHM="sac"  # "sac" or "awsc"

# Pretrained path for AWSC (optional)
PRETRAINED_PATH=""

# Wandb settings
WANDB_PROJECT="maniskill_rlpd_online"
TRACK=true

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
        --mode)
            MODE="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --total-timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --pretrained-path)
            PRETRAINED_PATH="$2"
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
        --no-track)
            TRACK=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============== Mode-specific Parameter Grids ==============
# Based on RLPD paper and ManiSkill RLPD configs

if [ "$MODE" = "walltime" ]; then
    # Walltime Efficient: More parallel envs, fewer Q-networks, lower UTD
    # Goal: Fast wall-clock time, may use more samples
    NUM_ENVS_OPTIONS=(50)
    UTD_RATIOS=(4 8)
    NUM_QS_OPTIONS=(2 10)
    NUM_MIN_QS_OPTIONS=(2)
    GAMMAS=(0.9)
    LR_OPTIONS=(3e-4)
    BATCH_SIZES=(256)
    BACKUP_ENTROPY_OPTIONS=(False)
    
elif [ "$MODE" = "sample" ]; then
    # Sample Efficient: Fewer envs, more Q-networks, higher UTD
    # Goal: Best sample efficiency, may be slower
    NUM_ENVS_OPTIONS=(8 16)
    UTD_RATIOS=(16 20)
    NUM_QS_OPTIONS=(10)
    NUM_MIN_QS_OPTIONS=(2)
    GAMMAS=(0.9)
    LR_OPTIONS=(3e-4)
    BATCH_SIZES=(256)
    BACKUP_ENTROPY_OPTIONS=(False)
    
else
    echo "Unknown mode: $MODE. Use 'walltime' or 'sample'"
    exit 1
fi

# Common sweep dimensions
SEEDS=(0 1 2)

# Demo settings (only if demo_path is available)
DEMO_PATH="$HOME/.maniskill/demos/${ENV_ID}/rl/trajectory.rgb.${CONTROL_MODE}.${SIM_BACKEND}.h5"
ONLINE_RATIOS=(0.5)  # RLPD default: 50% online, 50% offline

# Log directory
LOG_DIR="/tmp/rlpd_online_sweep_${MODE}"
mkdir -p "$LOG_DIR"

# ============== Display Configuration ==============
echo "=========================================="
echo "RLPD Online Hyperparameter Sweep"
echo "=========================================="
echo "Mode: $MODE"
echo "Algorithm: $ALGORITHM"
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Control mode: $CONTROL_MODE"
echo "Obs mode: $OBS_MODE"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo ""
echo "Sweep dimensions:"
echo "  NUM_ENVS: ${NUM_ENVS_OPTIONS[*]}"
echo "  UTD_RATIO: ${UTD_RATIOS[*]}"
echo "  NUM_QS: ${NUM_QS_OPTIONS[*]}"
echo "  GAMMA: ${GAMMAS[*]}"
echo "  SEEDS: ${SEEDS[*]}"
echo ""

# Check demo file
if [ -f "$DEMO_PATH" ]; then
    echo "Demo file found: $DEMO_PATH"
    USE_DEMOS=true
else
    echo "Demo file not found: $DEMO_PATH"
    echo "Running without offline demos (pure online RL)"
    USE_DEMOS=false
fi

cd /home/amax/rlft/diffusion_policy

# ============== Generate Task Combinations ==============
declare -a TASKS
TASK_IDX=0

for num_envs in "${NUM_ENVS_OPTIONS[@]}"; do
    for utd_ratio in "${UTD_RATIOS[@]}"; do
        for num_qs in "${NUM_QS_OPTIONS[@]}"; do
            for num_min_qs in "${NUM_MIN_QS_OPTIONS[@]}"; do
                for gamma in "${GAMMAS[@]}"; do
                    for lr in "${LR_OPTIONS[@]}"; do
                        for batch_size in "${BATCH_SIZES[@]}"; do
                            for backup_entropy in "${BACKUP_ENTROPY_OPTIONS[@]}"; do
                                for online_ratio in "${ONLINE_RATIOS[@]}"; do
                                    for seed in "${SEEDS[@]}"; do
                                        TASKS[$TASK_IDX]="$num_envs|$utd_ratio|$num_qs|$num_min_qs|$gamma|$lr|$batch_size|$backup_entropy|$online_ratio|$seed"
                                        ((TASK_IDX+=1))
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "Total experiments: $TOTAL_TASKS"
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

COMPLETED=0
FAILED=0

# ============== Task Runner ==============
run_task() {
    local gpu=$1
    local task=$2
    local task_num=$3
    
    IFS='|' read -r num_envs utd_ratio num_qs num_min_qs gamma lr batch_size backup_entropy online_ratio seed <<< "$task"
    
    # Generate experiment name
    exp_name="${ALGORITHM}-online-${ENV_ID}-${MODE}-nenv${num_envs}-utd${utd_ratio}-nq${num_qs}-g${gamma}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"
    
    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"
    
    # Build command
    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_rlpd_online.py \
        --algorithm $ALGORITHM \
        --env_id $ENV_ID \
        --obs_mode $OBS_MODE \
        --control_mode $CONTROL_MODE \
        --sim_backend $SIM_BACKEND \
        --max_episode_steps $MAX_EPISODE_STEPS \
        --num_envs $num_envs \
        --num_eval_envs $NUM_EVAL_ENVS \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --total_timesteps $TOTAL_TIMESTEPS \
        --num_seed_steps $NUM_SEED_STEPS \
        --utd_ratio $utd_ratio \
        --batch_size $batch_size \
        --gamma $gamma \
        --lr_actor $lr \
        --lr_critic $lr \
        --num_qs $num_qs \
        --num_min_qs $num_min_qs \
        --online_ratio $online_ratio \
        --seed $seed \
        --eval_freq $EVAL_FREQ \
        --log_freq $LOG_FREQ \
        --save_freq $SAVE_FREQ \
        --exp_name $exp_name"
    
    # Add backup_entropy flag
    if [ "$backup_entropy" = "True" ]; then
        CMD="$CMD --backup_entropy"
    fi
    
    # Add demo path if available
    if [ "$USE_DEMOS" = true ]; then
        CMD="$CMD --demo_path $DEMO_PATH"
    fi
    
    # Add pretrained path for AWSC
    if [ "$ALGORITHM" = "awsc" ] && [ -n "$PRETRAINED_PATH" ]; then
        CMD="$CMD --pretrained_path $PRETRAINED_PATH"
    fi
    
    # Add tracking
    if [ "$TRACK" = true ]; then
        CMD="$CMD --track --wandb_project_name $WANDB_PROJECT"
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

# ============== Parallel Dispatch ==============
declare -A gpu_queue
for gpu in "${GPUS[@]}"; do
    gpu_queue[$gpu]=""
done

next_task=0

dispatch_tasks() {
    for gpu in "${GPUS[@]}"; do
        # Check if process is still running
        if [ -n "${gpu_queue[$gpu]}" ]; then
            local pid=${gpu_queue[$gpu]}
            if ! kill -0 "$pid" 2>/dev/null; then
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
while [ $next_task -lt $TOTAL_TASKS ] || [ $COMPLETED -lt $((TOTAL_TASKS - FAILED)) ]; do
    dispatch_tasks
    sleep 5
done

# Wait for remaining processes
for gpu in "${GPUS[@]}"; do
    if [ -n "${gpu_queue[$gpu]}" ]; then
        wait "${gpu_queue[$gpu]}" 2>/dev/null || ((FAILED+=1))
        ((COMPLETED+=1))
    fi
done

echo ""
echo "=========================================="
echo "RLPD Online sweep completed!"
echo "Mode: $MODE"
echo "Completed: $COMPLETED / $TOTAL_TASKS"
echo "Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="

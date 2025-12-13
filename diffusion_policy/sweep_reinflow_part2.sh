#!/bin/bash
#
# ReinFlow Sweep Part 2: Training Dynamics + Critic Stability
# Focus: LR, inference steps, rollout/epochs, GAE, critic stability
#
# Usage: ./sweep_reinflow_part2.sh [--gpus "0 1 2 3"] [--dry-run] [--env ENV_ID]
#

set -e

# Default configurations
GPUS=(2 3 4 5 6 7 8 9)
DRY_RUN=false
TOTAL_UPDATES=10000
EVAL_FREQ=100
LOG_FREQ=1
NUM_EVAL_EPISODES=50
NUM_EVAL_ENVS=10
NUM_ENVS=50
SIM_BACKEND="physx_cuda"
WANDB_PROJECT="maniskill_reinflow_sweep"
MAX_EPISODE_STEPS=64
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
OBS_MODE="rgb"

PRETRAINED_PATH_PATTERN="/home/wjz/rlft/diffusion_policy/runs/awsc-{ENV_ID}-seed0/checkpoints/best_eval_success_once.pt"

# ============================================================================
# Part 2 Configurations (15 experiments)
# ============================================================================
CONFIGS=(
    # === LEARNING RATE ABLATION (4) ===
    "lr:high"
    "lr:low"
    
    # === PPO EPOCHS ABLATION (3) ===
    "epochs:one"
    
    # === CRITIC STABILITY ABLATION (5) ===
    "critic:no_target_vnet"
    "critic:fast_target_update"
    "critic:no_return_norm"
    "critic:high_value_clip"
    "critic:high_reward_scale"
)

SEEDS=(0)
LOG_DIR="/tmp/reinflow_sweep_part2"

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
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --seeds)
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PRETRAINED_PATH="${PRETRAINED_PATH_PATTERN//\{ENV_ID\}/$ENV_ID}"
mkdir -p "$LOG_DIR"
TOTAL_TASKS=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

echo "=========================================="
echo "ReinFlow Sweep Part 2: Training Dynamics"
echo "=========================================="
echo "GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Seeds: ${SEEDS[*]}"
echo "Total experiments: $TOTAL_TASKS"
echo "Log directory: $LOG_DIR"
echo "Pretrained: $PRETRAINED_PATH"
echo "=========================================="

cd /home/wjz/rlft/diffusion_policy

declare -a TASKS
idx=0
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASKS[$idx]="$cfg|$seed"
        ((idx+=1))
    done
done

COMPLETED=0
FAILED=0

run_task() {
    local gpu=$1
    local task=$2
    local task_num=$3
    IFS='|' read -r cfg seed <<< "$task"
    IFS=':' read -r category variant <<< "$cfg"

    # ========== Default values ==========
    # Critic warmup
    critic_warmup_steps=50
    
    # Noise settings
    noise_decay_type="linear"
    min_noise_std=0.01
    max_noise_std=0.15
    
    # PPO hyperparameters
    clip_ratio=0.1
    entropy_coef=0.00
    value_coef=0.5
    target_kl=0.02
    kl_early_stop=true
    
    # Flow/inference settings
    num_inference_steps=8
    
    # Rollout/training settings
    rollout_steps=64
    ppo_epochs=10
    minibatch_size=5120
    
    # GAE/discount settings
    gamma=0.99
    gae_lambda=0.95
    
    # Gradient/optimization settings
    max_grad_norm=10.0
    lr=3e-5
    lr_critic=3e-5
    
    # Critic stability settings
    reward_scale=0.5
    value_target_tau=0.005
    use_target_value_net=true
    value_target_clip=100.0
    normalize_returns=true
    normalize_rewards=true
    
    # Visual encoder
    freeze_visual_encoder=true
    
    profile="${category}-${variant}"

    # ========== Apply configuration variants ==========
    case "$category" in
        lr)
            case "$variant" in
                high) lr=3e-4 ; lr_critic=3e-4 ;;
                low) lr=1e-6 ; lr_critic=1e-6 ;;
            esac
            ;;
        epochs)
            case "$variant" in
                one) ppo_epochs=1 ;;
            esac
            ;;
        critic)
            case "$variant" in
                no_target_vnet) use_target_value_net=false ;;
                fast_target_update) value_target_tau=0.05 ;;
                no_return_norm) normalize_returns=false ;;
                high_value_clip) value_target_clip=500.0 ;;
                high_reward_scale) reward_scale=1.0 ;;
            esac
            ;;
    esac

    exp_name="reinflow-${ENV_ID}-${profile}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"
    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"

    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_online_finetune.py \
        --env_id $ENV_ID --obs_mode $OBS_MODE --control_mode $CONTROL_MODE \
        --sim_backend $SIM_BACKEND --pretrained_path $PRETRAINED_PATH \
        --seed $seed --total_updates $TOTAL_UPDATES --eval_freq $EVAL_FREQ \
        --log_freq $LOG_FREQ --num_eval_episodes $NUM_EVAL_EPISODES \
        --num_eval_envs $NUM_EVAL_ENVS --num_envs $NUM_ENVS \
        --max_episode_steps $MAX_EPISODE_STEPS --exp_name $exp_name \
        --track --wandb_project_name $WANDB_PROJECT \
        --critic_warmup_steps $critic_warmup_steps \
        --noise_decay_type $noise_decay_type \
        --min_noise_std $min_noise_std --max_noise_std $max_noise_std \
        --clip_ratio $clip_ratio --entropy_coef $entropy_coef --value_coef $value_coef \
        --num_inference_steps $num_inference_steps \
        --rollout_steps $rollout_steps --ppo_epochs $ppo_epochs \
        --minibatch_size $minibatch_size --gamma $gamma --gae_lambda $gae_lambda \
        --max_grad_norm $max_grad_norm --lr $lr --lr_critic $lr_critic \
        --reward_scale $reward_scale \
        --value_target_tau $value_target_tau --value_target_clip $value_target_clip \
        --target_kl $target_kl"
    
    # Boolean flags
    if [ "$freeze_visual_encoder" = true ]; then CMD+=" --freeze_visual_encoder"; fi
    if [ "$normalize_rewards" = true ]; then CMD+=" --normalize_rewards"; fi
    if [ "$use_target_value_net" = true ]; then CMD+=" --use_target_value_net"; fi
    if [ "$normalize_returns" = true ]; then CMD+=" --normalize_returns"; fi
    if [ "$kl_early_stop" = true ]; then CMD+=" --kl_early_stop"; fi

    if [ "$DRY_RUN" = true ]; then
        echo "[GPU $gpu] [DRY RUN] $exp_name"
        echo "$CMD" > "$log_file"
        return 0
    fi

    if eval "$CMD" >> "$log_file" 2>&1; then
        echo "[GPU $gpu] ✓ Completed: $exp_name"
        return 0
    else
        echo "[GPU $gpu] ✗ Failed: $exp_name (see $log_file)"
        return 1
    fi
}

# GPU task queue management
declare -A gpu_queue
for gpu in "${GPUS[@]}"; do gpu_queue[$gpu]=""; done
next_task=0

assign_tasks() {
    for gpu in "${GPUS[@]}"; do
        if [ -n "${gpu_queue[$gpu]}" ]; then
            local pid=${gpu_queue[$gpu]}
            if ! kill -0 "$pid" 2>/dev/null; then
                if wait "$pid" 2>/dev/null; then ((COMPLETED+=1)); else ((FAILED+=1)); fi
                gpu_queue[$gpu]=""
            fi
        fi
        if [ -z "${gpu_queue[$gpu]}" ] && [ $next_task -lt $TOTAL_TASKS ]; then
            run_task "$gpu" "${TASKS[$next_task]}" $((next_task + 1)) &
            gpu_queue[$gpu]=$!
            ((next_task+=1))
        fi
    done
}

while [ $next_task -lt $TOTAL_TASKS ] || [ $COMPLETED -lt $TOTAL_TASKS ]; do
    assign_tasks
    sleep 2
done

for gpu in "${GPUS[@]}"; do
    if [ -n "${gpu_queue[$gpu]}" ]; then
        if wait "${gpu_queue[$gpu]}"; then ((COMPLETED+=1)); else ((FAILED+=1)); fi
    fi
done

echo ""
echo "=========================================="
echo "Part 2 completed! Completed: $COMPLETED, Failed: $FAILED"
echo "=========================================="

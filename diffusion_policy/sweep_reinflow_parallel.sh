#!/bin/bash
#
# Parallel grid search for ReinFlow online RL fine-tuning
# Organized by ablation categories for systematic comparison
#
# Usage: ./sweep_reinflow_parallel.sh [--gpus "0 1 2 3"] [--dry-run] [--env ENV_ID]
#
# =============================================================================
# ABLATION STUDY ORGANIZATION
# =============================================================================
#
# 1. BASELINE: Default configuration with reasonable hyperparameters
#
# 2. CRITIC WARMUP ABLATION: How long to warm up the critic before policy updates
#    - warmup_0: No warmup (risky)
#    - warmup_2k: Short warmup (2000 steps)
#    - warmup_5k: Default warmup (5000 steps)
#    - warmup_10k: Long warmup (10000 steps)
#
# 3. NOISE SCHEDULE ABLATION: How exploration noise decays over training
#    - noise_constant: Constant noise (no decay)
#    - noise_linear: Linear decay (default)
#    - noise_exp: Exponential decay
#    - noise_fast: Fast linear decay (200k steps)
#    - noise_low_init: Lower initial noise (0.15)
#    - noise_high_init: Higher initial noise (0.5)
#
# 4. PPO HYPERPARAMETERS ABLATION: Core PPO settings
#    - ppo_default: Standard PPO settings
#    - ppo_clip_tight: Tighter clipping (0.1)
#    - ppo_clip_loose: Looser clipping (0.3)
#    - ppo_high_entropy: Higher entropy bonus
#    - ppo_low_entropy: Lower entropy bonus
#    - ppo_high_value: Higher value coefficient
#
# 5. LEARNING RATE ABLATION: Learning rate configuration
#    - lr_default: Policy 3e-5, Critic 1e-4
#    - lr_low: Lower learning rates
#    - lr_high: Higher learning rates
#    - lr_equal: Equal rates for policy and critic
#
# 6. INFERENCE STEPS ABLATION: Number of denoising steps at inference
#    - infer_4steps: 4 inference steps
#    - infer_8steps: 8 inference steps (default)
#    - infer_16steps: 16 inference steps
#
# 7. ROLLOUT CONFIG ABLATION: Data collection and training configuration
#    - rollout_default: 128 steps, 4 epochs
#    - rollout_short: 64 steps, 4 epochs
#    - rollout_long: 256 steps, 4 epochs
#    - rollout_more_epochs: 128 steps, 8 epochs
#
# =============================================================================

set -e

# Default configurations
GPUS=(2 3 4 5 6 7 8 9)
DRY_RUN=false
TOTAL_UPDATES=10000           # Aligned with train_online_finetune.py default
EVAL_FREQ=100                   # In number of updates (aligned with train_online_finetune.py)
LOG_FREQ=1                     # In number of updates (aligned with train_online_finetune.py)
NUM_EVAL_EPISODES=20           # Aligned with train_online_finetune.py default
NUM_EVAL_ENVS=5               # Aligned with train_online_finetune.py default
NUM_ENVS=25                  # Aligned with train_online_finetune.py default
SIM_BACKEND="physx_cuda"
WANDB_PROJECT="maniskill_reinflow_grid"
MAX_EPISODE_STEPS=100
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
OBS_MODE="rgb"

# Pretrained checkpoint path pattern (will be formatted with ENV_ID)
# Note: Use best_eval_success_once.pt which includes visual_encoder
PRETRAINED_PATH_PATTERN="/home/wjz/rlft/diffusion_policy/runs/awsc-{ENV_ID}-seed0/checkpoints/best_eval_success_once.pt"

# Config variants organized by ablation category
# Format: "category:variant"
# Note: baseline uses default config, so we don't need separate configs 
#       that are identical to baseline (e.g., warmup:5k, noise:linear, ppo:default, lr:default)
CONFIGS=(
    # === BASELINE ===
    "baseline:default"
    
    # === CRITIC WARMUP ABLATION ===
    "warmup:0"
    "warmup:2k"
    "warmup:5k"
    
    # === NOISE SCHEDULE ABLATION ===
    "noise:constant"
    # "noise:linear"  # Same as baseline
    "noise:exp"
    "noise:fast"
    "noise:low_init"
    "noise:high_init"
    
    # === PPO HYPERPARAMETERS ABLATION ===
    # "ppo:default"  # Same as baseline
    "ppo:clip_tight"
    "ppo:clip_loose"
    "ppo:high_entropy"
    "ppo:low_entropy"
    "ppo:high_value"
    
    # === LEARNING RATE ABLATION ===
    # "lr:default"  # Same as baseline
    "lr:low"
    "lr:high"
    "lr:equal"
    
    # === INFERENCE STEPS ABLATION ===
    "infer:4steps"
    # "infer:8steps"  # Same as baseline
    "infer:16steps"
    
    # === ROLLOUT CONFIG ABLATION ===
    # "rollout:default"  # Same as baseline
    "rollout:short"
    "rollout:long"
    "rollout:more_epochs"
    
    # === NEW: CRITIC STABILITY ABLATION (borrowed from AWCP) ===
    "critic:no_target_vnet"
    "critic:high_reward_scale"
    "critic:low_reward_scale"
    "critic:no_return_norm"
    "critic:high_value_clip"
    "critic:fast_target_update"
)

SEEDS=(0)
LOG_DIR="/tmp/reinflow_grid"

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
        --timesteps)
            TOTAL_TIMESTEPS="$2"
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
        --configs)
            IFS=' ' read -ra CONFIGS <<< "$2"
            shift 2
            ;;
        --pretrained-path)
            PRETRAINED_PATH_PATTERN="$2"
            shift 2
            ;;
        --seeds)
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Format pretrained path with environment ID
PRETRAINED_PATH="${PRETRAINED_PATH_PATTERN//\{ENV_ID\}/$ENV_ID}"

mkdir -p "$LOG_DIR"

TOTAL_TASKS=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

echo "=========================================="
echo "ReinFlow Online RL Sweep"
echo "=========================================="
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Pretrained: $PRETRAINED_PATH"
echo "Total experiments: $TOTAL_TASKS"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo ""
echo "Ablation Categories:"
echo "  - baseline: Default configuration"
echo "  - warmup: Critic warmup steps"
echo "  - noise: Noise schedule configuration"
echo "  - ppo: PPO hyperparameters"
echo "  - lr: Learning rate settings"
echo "  - infer: Inference steps"
echo "  - rollout: Rollout configuration"
echo "=========================================="
echo ""

cd /home/wjz/rlft/diffusion_policy

# Check pretrained checkpoint
if [ "$DRY_RUN" = false ] && [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Warning: Pretrained checkpoint not found: $PRETRAINED_PATH"
    echo "Make sure the checkpoint exists before running experiments."
    echo ""
fi

# Build task list
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

    # =========================================================================
    # DEFAULT VALUES (baseline configuration)
    # =========================================================================
    # Critic warmup
    critic_warmup_steps=10
    
    # Noise schedule
    noise_decay_type="linear"
    noise_decay_steps=500000
    min_noise_std=0.01
    max_noise_std=0.3
    
    # PPO hyperparameters
    clip_ratio=0.2
    entropy_coef=0.00
    value_coef=0.5
    
    # Learning rate
    lr=3e-5
    lr_critic=1e-4
    
    # Inference
    num_inference_steps=8
    
    # Rollout config (aligned with train_online_finetune.py defaults)
    rollout_steps=128
    ppo_epochs=1                    # Aligned with train_online_finetune.py default
    minibatch_size=5120             # Aligned with train_online_finetune.py default
    
    # Other defaults
    gamma=0.99
    gae_lambda=0.95
    max_grad_norm=0.5
    freeze_visual_encoder=true
    normalize_rewards=true
    
    # === NEW: Critic stability defaults (borrowed from AWCP) ===
    reward_scale=0.1
    value_target_tau=0.005
    use_target_value_net=true
    value_target_clip=100.0
    normalize_returns=true
    
    # Profile name for logging
    profile="${category}-${variant}"

    # =========================================================================
    # APPLY CONFIGURATION BASED ON CATEGORY AND VARIANT
    # =========================================================================
    case "$category" in
        baseline)
            # Use all defaults
            profile="baseline"
            ;;
        
        warmup)
            case "$variant" in
                0)
                    critic_warmup_steps=0
                    ;;
                2k)
                    critic_warmup_steps=20
                    ;;
                5k)
                    critic_warmup_steps=50
                    ;;
                *)
                    echo "Unknown warmup variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        noise)
            case "$variant" in
                constant)
                    noise_decay_type="constant"
                    ;;
                linear)
                    noise_decay_type="linear"
                    noise_decay_steps=500000
                    ;;
                exp)
                    noise_decay_type="exponential"
                    noise_decay_steps=500000
                    ;;
                fast)
                    noise_decay_type="linear"
                    noise_decay_steps=200000
                    ;;
                low_init)
                    max_noise_std=0.15
                    ;;
                high_init)
                    max_noise_std=0.5
                    ;;
                *)
                    echo "Unknown noise variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        ppo)
            case "$variant" in
                default)
                    # Use defaults
                    ;;
                clip_tight)
                    clip_ratio=0.1
                    ;;
                clip_loose)
                    clip_ratio=0.3
                    ;;
                high_entropy)
                    entropy_coef=0.05
                    ;;
                low_entropy)
                    entropy_coef=0.001
                    ;;
                high_value)
                    value_coef=1.0
                    ;;
                *)
                    echo "Unknown ppo variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        lr)
            case "$variant" in
                default)
                    lr=3e-5
                    lr_critic=1e-4
                    ;;
                low)
                    lr=1e-5
                    lr_critic=3e-5
                    ;;
                high)
                    lr=1e-4
                    lr_critic=3e-4
                    ;;
                equal)
                    lr=5e-5
                    lr_critic=5e-5
                    ;;
                *)
                    echo "Unknown lr variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        infer)
            case "$variant" in
                4steps)
                    num_inference_steps=4
                    ;;
                8steps)
                    num_inference_steps=8
                    ;;
                16steps)
                    num_inference_steps=16
                    ;;
                *)
                    echo "Unknown infer variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        rollout)
            case "$variant" in
                default)
                    rollout_steps=256
                    minibatch_size=5120
                    ;;
                short)
                    rollout_steps=128
                    minibatch_size=2048
                    ;;
                long)
                    rollout_steps=512
                    minibatch_size=10240
                    ;;
                more_epochs)
                    rollout_steps=256
                    ppo_epochs=10
                    minibatch_size=5120
                    ;;
                *)
                    echo "Unknown rollout variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        # === NEW: CRITIC STABILITY ABLATION ===
        critic)
            case "$variant" in
                no_target_vnet)
                    use_target_value_net=false
                    ;;
                high_reward_scale)
                    reward_scale=0.5
                    ;;
                low_reward_scale)
                    reward_scale=0.01
                    ;;
                no_return_norm)
                    normalize_returns=false
                    ;;
                high_value_clip)
                    value_target_clip=500.0
                    ;;
                fast_target_update)
                    value_target_tau=0.05
                    ;;
                *)
                    echo "Unknown critic variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        *)
            echo "Unknown category: $category"
            return 1
            ;;
    esac

    exp_name="reinflow-${ENV_ID}-${profile}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"

    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"

    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_online_finetune.py \
        --env_id $ENV_ID \
        --obs_mode $OBS_MODE \
        --control_mode $CONTROL_MODE \
        --sim_backend $SIM_BACKEND \
        --pretrained_path $PRETRAINED_PATH \
        --seed $seed \
        --total_updates $TOTAL_UPDATES \
        --eval_freq $EVAL_FREQ \
        --log_freq $LOG_FREQ \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --num_eval_envs $NUM_EVAL_ENVS \
        --num_envs $NUM_ENVS \
        --max_episode_steps $MAX_EPISODE_STEPS \
        --exp_name $exp_name \
        --track \
        --wandb_project_name $WANDB_PROJECT \
        --critic_warmup_steps $critic_warmup_steps \
        --noise_decay_type $noise_decay_type \
        --noise_decay_steps $noise_decay_steps \
        --min_noise_std $min_noise_std \
        --max_noise_std $max_noise_std \
        --clip_ratio $clip_ratio \
        --entropy_coef $entropy_coef \
        --value_coef $value_coef \
        --lr $lr \
        --lr_critic $lr_critic \
        --num_inference_steps $num_inference_steps \
        --rollout_steps $rollout_steps \
        --ppo_epochs $ppo_epochs \
        --minibatch_size $minibatch_size \
        --gamma $gamma \
        --gae_lambda $gae_lambda \
        --max_grad_norm $max_grad_norm \
        --reward_scale $reward_scale \
        --value_target_tau $value_target_tau \
        --value_target_clip $value_target_clip"
    
    # Append boolean flags
    if [ "$freeze_visual_encoder" = true ]; then CMD+=" --freeze_visual_encoder"; fi
    if [ "$normalize_rewards" = true ]; then CMD+=" --normalize_rewards"; fi
    if [ "$use_target_value_net" = true ]; then CMD+=" --use_target_value_net"; fi
    if [ "$normalize_returns" = true ]; then CMD+=" --normalize_returns"; fi

    if [ "$DRY_RUN" = true ]; then
        echo "[GPU $gpu] [DRY RUN] $exp_name"
        echo "$CMD" > "$log_file"
        return 0
    fi

    if eval "$CMD" >> "$log_file" 2>&1; then
        echo "[GPU $gpu] ✓ Completed: $exp_name"
        echo "completed" >> "$log_file"
        return 0
    else
        echo "[GPU $gpu] ✗ Failed: $exp_name (see $log_file)"
        echo "failed" >> "$log_file"
        return 1
    fi
}

# Parallel dispatcher
declare -A gpu_queue
for gpu in "${GPUS[@]}"; do
    gpu_queue[$gpu]=""
done

next_task=0
active_pids=()

assign_tasks() {
    for gpu in "${GPUS[@]}"; do
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
        if wait "${gpu_queue[$gpu]}"; then
            ((COMPLETED+=1))
        else
            ((FAILED+=1))
        fi
    fi
done

echo ""
echo "=========================================="
echo "ReinFlow sweep completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="

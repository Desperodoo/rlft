#!/bin/bash
#
# Parallel grid search for Consistency Flow design variants
# Mirrors GPU auto-allocation style of sweep_alpha_qgrad_parallel.sh
#
# Usage: ./sweep_consistency_flow_parallel.sh [--gpus "2 3 4"] [--dry-run] [--env ENV_ID]

set -e

# Default configurations (aligned with sweep_alpha_qgrad_parallel.sh)
GPUS=(2 3 4 5 6 7 8 9)
DRY_RUN=false
TOTAL_ITERS=30000
EVAL_FREQ=2000
LOG_FREQ=100
NUM_EVAL_EPISODES=100
NUM_EVAL_ENVS=50
NUM_DEMOS=1000
SIM_BACKEND="physx_cuda"
WANDB_PROJECT="maniskill_consistency_grid"
MAX_EPISODE_STEPS=100
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
OBS_MODE="rgb"
BC_WEIGHT=1.0
CONSISTENCY_WEIGHT=1.0
# Config variants to sweep (add/remove as needed)
CONFIGS=(
    "cpql_style"            # restricted t, random delta, teacher from t+, student t_cons, velocity loss
    "cpql_full_t"           # full t range, random delta, velocity loss
    "cpql_fixed_delta"      # restricted t, fixed delta, velocity loss
    "cpql_endpoint"         # restricted t, random delta, endpoint loss
    "cpql_student_tplus"    # restricted t, random delta, student at t_plus, velocity loss
    "flow_endpoint"         # full t, fixed delta, teacher from t_cons, student t_plus, endpoint loss
    "flow_velocity"         # full t, fixed delta, teacher from t_cons, student t_plus, velocity loss
    "flow_teacher_tplus"    # full t, fixed delta, teacher from t_plus, endpoint loss
    "flow_student_tcons"    # full t, fixed delta, student at t_cons, endpoint loss
    "flow_small_delta"      # full t, tiny fixed delta, endpoint loss
    "flow_random_delta"     # full t, random delta in [0.01,0.1], endpoint loss
    "flow_teacher3"         # full t, fixed delta, teacher 3 steps, endpoint loss
    "flow_resample_t"       # full t, fixed delta, resample t for consistency (no flow reuse)
)
SEEDS=(0)
LOG_DIR="/tmp/consistency_flow_grid"

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

DEMO_PATH="$HOME/.maniskill/demos/${ENV_ID}/rl/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5"
mkdir -p "$LOG_DIR"

TOTAL_TASKS=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

echo "=========================================="
echo "Consistency Flow Design Sweep"
echo "=========================================="
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Demo: $DEMO_PATH"
echo "Total experiments: $TOTAL_TASKS"
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

    # Default knobs
    cons_use_flow_t=false
    cons_full_t_range=false
    cons_t_min=0.05
    cons_t_max=0.95
    cons_t_upper=0.95
    cons_delta_mode="random"
    cons_delta_min=0.02
    cons_delta_max=0.15
    cons_delta_fixed=0.01
    cons_delta_dynamic_max=false
    cons_delta_cap=0.99
    cons_teacher_steps=2
    cons_teacher_from="t_plus"
    cons_student_point="t_plus"
    cons_loss_space="velocity"

    case "$cfg" in
        cpql_style)
            cons_use_flow_t=false
            cons_full_t_range=false
            cons_t_min=0.05
            cons_t_max=0.95
            cons_t_upper=0.99
            cons_delta_mode="random"
            cons_delta_min=0.02
            cons_delta_max=0.99
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=true
            cons_delta_cap=0.99
            cons_teacher_steps=1
            cons_teacher_from="t_plus"
            cons_student_point="t_cons"
            cons_loss_space="velocity"
            profile="cpql"
            ;;
        cpql_full_t)
            cons_use_flow_t=false
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=0.99
            cons_delta_mode="random"
            cons_delta_min=0.02
            cons_delta_max=0.99
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=true
            cons_delta_cap=0.99
            cons_teacher_steps=1
            cons_teacher_from="t_plus"
            cons_student_point="t_cons"
            cons_loss_space="velocity"
            profile="cpql-full"
            ;;
        cpql_fixed_delta)
            cons_use_flow_t=false
            cons_full_t_range=false
            cons_t_min=0.05
            cons_t_max=0.95
            cons_t_upper=0.99
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=0.99
            cons_teacher_steps=1
            cons_teacher_from="t_plus"
            cons_student_point="t_cons"
            cons_loss_space="velocity"
            profile="cpql-fixed"
            ;;
        cpql_endpoint)
            cons_use_flow_t=false
            cons_full_t_range=false
            cons_t_min=0.05
            cons_t_max=0.95
            cons_t_upper=0.99
            cons_delta_mode="random"
            cons_delta_min=0.02
            cons_delta_max=0.99
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=true
            cons_delta_cap=0.99
            cons_teacher_steps=1
            cons_teacher_from="t_plus"
            cons_student_point="t_cons"
            cons_loss_space="endpoint"
            profile="cpql-endpoint"
            ;;
        cpql_student_tplus)
            cons_use_flow_t=false
            cons_full_t_range=false
            cons_t_min=0.05
            cons_t_max=0.95
            cons_t_upper=0.99
            cons_delta_mode="random"
            cons_delta_min=0.02
            cons_delta_max=0.99
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=true
            cons_delta_cap=0.99
            cons_teacher_steps=1
            cons_teacher_from="t_plus"
            cons_student_point="t_plus"
            cons_loss_space="velocity"
            profile="cpql-student-tplus"
            ;;
        flow_endpoint)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_cons"
            cons_student_point="t_plus"
            cons_loss_space="endpoint"
            profile="flow-endpoint"
            ;;
        consistency_style)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_cons"
            cons_student_point="t_plus"
            cons_loss_space="endpoint"
            profile="flow"
            ;;
        flow_velocity)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_cons"
            cons_student_point="t_plus"
            cons_loss_space="velocity"
            profile="flow-vel"
            ;;
        flow_teacher_tplus)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_plus"
            cons_student_point="t_plus"
            cons_loss_space="endpoint"
            profile="flow-teacher-tplus"
            ;;
        flow_student_tcons)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_cons"
            cons_student_point="t_cons"
            cons_loss_space="endpoint"
            profile="flow-student-tcons"
            ;;
        flow_small_delta)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.005
            cons_delta_max=0.005
            cons_delta_fixed=0.005
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_cons"
            cons_student_point="t_plus"
            cons_loss_space="endpoint"
            profile="flow-small-delta"
            ;;
        flow_random_delta)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="random"
            cons_delta_min=0.01
            cons_delta_max=0.10
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_cons"
            cons_student_point="t_plus"
            cons_loss_space="endpoint"
            profile="flow-rand-delta"
            ;;
        flow_teacher3)
            cons_use_flow_t=true
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=3
            cons_teacher_from="t_cons"
            cons_student_point="t_plus"
            cons_loss_space="endpoint"
            profile="flow-teacher3"
            ;;
        flow_resample_t)
            cons_use_flow_t=false
            cons_full_t_range=true
            cons_t_min=0.0
            cons_t_max=1.0
            cons_t_upper=1.0
            cons_delta_mode="fixed"
            cons_delta_min=0.01
            cons_delta_max=0.01
            cons_delta_fixed=0.01
            cons_delta_dynamic_max=false
            cons_delta_cap=1.0
            cons_teacher_steps=2
            cons_teacher_from="t_cons"
            cons_student_point="t_plus"
            cons_loss_space="endpoint"
            profile="flow-resample-t"
            ;;
        *)
            echo "Unknown config: $cfg"
            return 1
            ;;
    esac

    exp_name="cf-${ENV_ID}-${profile}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"

    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"

    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_offline_rl.py \
        --algorithm consistency_flow \
        --env_id $ENV_ID \
        --obs_mode $OBS_MODE \
        --demo_path $DEMO_PATH \
        --control_mode $CONTROL_MODE \
        --sim_backend $SIM_BACKEND \
        --num_demos $NUM_DEMOS \
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
        --bc_weight $BC_WEIGHT \
        --consistency_weight $CONSISTENCY_WEIGHT \
        --cons_t_min $cons_t_min \
        --cons_t_max $cons_t_max \
        --cons_t_upper $cons_t_upper \
        --cons_delta_mode $cons_delta_mode \
        --cons_delta_min $cons_delta_min \
        --cons_delta_max $cons_delta_max \
        --cons_delta_fixed $cons_delta_fixed \
        --cons_delta_cap $cons_delta_cap \
        --cons_teacher_steps $cons_teacher_steps \
        --cons_teacher_from $cons_teacher_from \
        --cons_student_point $cons_student_point \
        --cons_loss_space $cons_loss_space"
    
    # Append boolean flags only when true
    if [ "$cons_use_flow_t" = true ]; then CMD+=" --cons_use_flow_t"; fi
    if [ "$cons_full_t_range" = true ]; then CMD+=" --cons_full_t_range"; fi
    if [ "$cons_delta_dynamic_max" = true ]; then CMD+=" --cons_delta_dynamic_max"; fi

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
echo "Consistency flow sweep completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="
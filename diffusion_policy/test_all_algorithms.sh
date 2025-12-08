#!/bin/bash
# Quick test all 7 offline RL algorithms

set -e  # Exit on error

DEMO_PATH="$HOME/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
COMMON_ARGS="--env_id PickCube-v1 --obs_mode rgb --demo_path $DEMO_PATH --control-mode pd_ee_delta_pos --sim-backend physx_cpu --num-demos 10 --max_episode_steps 100 --total_iters 20 --log_freq 5 --eval_freq 10 --num_eval_episodes 2 --num_eval_envs 2"

# Algorithms to test
ALGORITHMS=(
    "diffusion_policy"
    "flow_matching"
    "reflected_flow"
    "consistency_flow"
    "shortcut_flow"
    "diffusion_double_q"
    "cpql"
)

echo "=== Quick Test All 7 Offline RL Algorithms ==="
echo ""

cd /home/amax/rlft/diffusion_policy

PASSED=0
FAILED=0

for algo in "${ALGORITHMS[@]}"; do
    echo "=========================================="
    echo "Testing: $algo"
    echo "=========================================="
    
    if CUDA_VISIBLE_DEVICES=0 python train_offline_rl.py --algorithm $algo $COMMON_ARGS 2>&1 | tail -20; then
        echo "✓ $algo PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "✗ $algo FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=========================================="
echo "Summary: $PASSED PASSED, $FAILED FAILED"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi

#!/bin/bash
#
# Monitor RLPD Online sweep tasks
# Shows real-time progress, status, and recent training metrics
#
# Usage: ./monitor_rlpd_online.sh [--log-dir /tmp/rlpd_online_sweep_walltime] [--interval 5]

LOG_DIR="/tmp/rlpd_online_sweep_walltime"
REFRESH=5  # seconds

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --interval)
            REFRESH="$2"
            shift 2
            ;;
        --mode)
            # Auto-set log dir based on mode
            LOG_DIR="/tmp/rlpd_online_sweep_$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory not found: $LOG_DIR"
    echo "Available directories:"
    ls -d /tmp/rlpd_online_sweep_* 2>/dev/null || echo "  (none)"
    exit 1
fi

clear
while true; do
    echo "=========================================="
    echo "RLPD Online Sweep Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log directory: $LOG_DIR"
    echo "=========================================="
    echo ""
    
    # Count status (safe when no logs)
    total=$(find "$LOG_DIR" -maxdepth 1 -name "*.log" 2>/dev/null | wc -l)
    if [ "$total" -eq 0 ]; then
        echo "No logs yet. Waiting for runs to start..."
        echo ""
        echo "Press Ctrl+C to exit, refreshing every 10s..."
        sleep 10
        clear
        continue
    fi
    
    # Count completed and failed
    completed=$(grep -l "Training complete\|Best success rate\|completed successfully" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    failed=$(grep -l "Error\|Traceback\|CUDA out of memory\|RuntimeError" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    running=$((total - completed - failed))
    
    echo "Status Summary:"
    echo "  Total: $total | Running: $running | Completed: $completed | Failed: $failed"
    if [ "$total" -gt 0 ]; then
        progress=$(( (completed + failed) * 100 / total ))
        echo "  Progress: ${progress}%"
    fi
    echo ""
    
    # Show running experiments with recent progress
    echo "Active Experiments (last 10 lines, latest metrics):"
    echo "------------------------------------------"
    
    running_logs=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | while read f; do
        # Check if still active (no completion marker)
        if ! grep -q "Training complete\|Best success rate\|Error\|Traceback" "$f" 2>/dev/null; then
            echo "$f"
        fi
    done | head -6)
    
    if [ -n "$running_logs" ]; then
        echo "$running_logs" | while read f; do
            if [ -n "$f" ]; then
                exp_name=$(basename "$f" .log)
                mod_time=$(date -r "$f" '+%H:%M:%S')
                echo ""
                echo "[$mod_time] $exp_name"
                # Flatten carriage returns, strip ANSI, extract key metrics
                tr '\r' '\n' < "$f" | sed $'s/\x1b\[[0-9;]*[A-Za-z]//g' | \
                    grep -E "Step|success|eval|actor_loss|critic_loss|temperature|Training" | \
                    tail -3 | cut -c1-120 | sed 's/^/    /'
            fi
        done
    else
        echo "  (no active experiments)"
    fi
    
    echo ""
    echo "------------------------------------------"
    
    # Show best results so far
    echo ""
    echo "Best Results So Far:"
    echo "------------------------------------------"
    
    # Extract best success rates from completed logs
    best_results=""
    for f in "$LOG_DIR"/*.log; do
        if [ -f "$f" ]; then
            exp_name=$(basename "$f" .log)
            # Look for "Best success rate" or similar
            best_rate=$(grep -oE "Best success rate: [0-9.]+%" "$f" 2>/dev/null | tail -1)
            if [ -n "$best_rate" ]; then
                rate_value=$(echo "$best_rate" | grep -oE "[0-9.]+" | head -1)
                best_results="$best_results\n$rate_value $exp_name: $best_rate"
            fi
        fi
    done
    
    if [ -n "$best_results" ]; then
        echo -e "$best_results" | sort -rn | head -10 | while read line; do
            if [ -n "$line" ]; then
                echo "  $line" | cut -d' ' -f2-
            fi
        done
    else
        echo "  (no results yet)"
    fi
    
    echo ""
    echo "------------------------------------------"
    
    # Show failed experiments
    if [ "$failed" -gt 0 ]; then
        echo ""
        echo "Failed Experiments (last error):"
        echo "------------------------------------------"
        grep -l "Error\|Traceback\|CUDA out of memory" "$LOG_DIR"/*.log 2>/dev/null | head -3 | while read f; do
            exp_name=$(basename "$f" .log)
            echo "  ✗ $exp_name"
            tr '\r' '\n' < "$f" | sed $'s/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "Error|Exception" | tail -1 | cut -c1-100 | sed 's/^/      /'
        done
        echo ""
    fi
    
    # GPU utilization (if nvidia-smi available)
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Utilization:"
        echo "------------------------------------------"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
            while read line; do
                echo "  $line"
            done
    fi
    
    echo ""
    echo "Press Ctrl+C to exit, refreshing every ${REFRESH}s..."
    sleep "$REFRESH"
    clear
done

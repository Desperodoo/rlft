#!/bin/bash
#
# Monitor ReinFlow online RL sweep
# Shows real-time progress and status organized by ablation category
#
# Usage: ./monitor_reinflow.sh [--log-dir /tmp/reinflow_grid] [--interval 5]

LOG_DIR="/tmp/reinflow_grid"
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
        *)
            shift
            ;;
    esac
done

if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory not found: $LOG_DIR"
    exit 1
fi

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
while true; do
    echo "=========================================="
    echo -e "${BLUE}ReinFlow Sweep Monitor${NC} - $(date '+%Y-%m-%d %H:%M:%S')"
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
    
    completed=$(grep -l "completed$" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    failed=$(grep -l "failed$\|Error\|Traceback" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    running=$((total - completed - failed))
    
    echo -e "Total: ${BLUE}$total${NC} | Completed: ${GREEN}$completed${NC} | Running: ${YELLOW}$running${NC} | Failed: ${RED}$failed${NC}"
    if [ "$total" -gt 0 ]; then
        progress=$(( (completed + failed) * 100 / total ))
        echo "Progress: $progress%"
    fi
    echo ""
    
    # Show status by category
    echo "Status by Ablation Category:"
    echo "----------------------------"
    
    for category in baseline warmup noise ppo lr infer rollout; do
        cat_logs=$(find "$LOG_DIR" -maxdepth 1 -name "*-${category}-*.log" -o -name "*-${category}.log" 2>/dev/null | wc -l)
        if [ "$cat_logs" -gt 0 ]; then
            cat_completed=$(grep -l "completed$" "$LOG_DIR"/*-${category}*.log 2>/dev/null | wc -l)
            cat_failed=$(grep -l "failed$\|Error\|Traceback" "$LOG_DIR"/*-${category}*.log 2>/dev/null | wc -l)
            cat_running=$((cat_logs - cat_completed - cat_failed))
            printf "  %-10s: %d total | ${GREEN}%d done${NC} | ${YELLOW}%d run${NC} | ${RED}%d fail${NC}\n" \
                "$category" "$cat_logs" "$cat_completed" "$cat_running" "$cat_failed"
        fi
    done
    echo ""
    
    # Show running experiments with current metrics
    echo "Running Experiments:"
    echo "-------------------"
    for f in "$LOG_DIR"/*.log; do
        [ -f "$f" ] || continue
        if ! grep -q "completed$\|failed$" "$f" 2>/dev/null; then
            exp_name=$(basename "$f" .log)
            mod_time=$(date -r "$f" '+%H:%M:%S')
            
            # Try to extract latest metrics from log
            latest_return=$(grep -oP 'return[=:]\s*[\d.]+' "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+' || echo "N/A")
            latest_step=$(grep -oP 'step[=:]\s*\d+|timestep[=:]\s*\d+' "$f" 2>/dev/null | tail -1 | grep -oP '\d+' || echo "N/A")
            
            echo -e "  ${YELLOW}●${NC} [$mod_time] $exp_name"
            if [ "$latest_step" != "N/A" ] || [ "$latest_return" != "N/A" ]; then
                echo "       step: $latest_step | return: $latest_return"
            fi
        fi
    done
    echo ""
    
    # Show failed experiments
    failed_files=$(grep -l "failed$\|Error\|Traceback" "$LOG_DIR"/*.log 2>/dev/null)
    if [ -n "$failed_files" ]; then
        echo -e "${RED}Failed Experiments:${NC}"
        echo "-------------------"
        for f in $failed_files; do
            exp_name=$(basename "$f" .log)
            # Get last error line
            error_line=$(grep -E "Error|Exception|Traceback" "$f" 2>/dev/null | tail -1 | cut -c1-80)
            echo -e "  ${RED}✗${NC} $exp_name"
            if [ -n "$error_line" ]; then
                echo "       $error_line..."
            fi
        done
        echo ""
    fi
    
    # Show recent activity
    echo "Recent Activity (latest 8 logs, last 2 lines):"
    echo "----------------------------------------------"
    ls -t "$LOG_DIR"/*.log 2>/dev/null | head -8 | while read f; do
        exp_name=$(basename "$f" .log)
        mod_time=$(date -r "$f" '+%H:%M:%S')
        
        # Status indicator
        if grep -q "completed$" "$f" 2>/dev/null; then
            status="${GREEN}✓${NC}"
        elif grep -q "failed$\|Error\|Traceback" "$f" 2>/dev/null; then
            status="${RED}✗${NC}"
        else
            status="${YELLOW}●${NC}"
        fi
        
        echo -e "- [$mod_time] $status $exp_name"
        # Flatten carriage returns, strip ANSI, show last 2 lines, truncate long lines
        tr '\r' '\n' < "$f" | sed $'s/\x1b\[[0-9;]*[A-Za-z]//g' | tail -2 | cut -c1-120 | sed 's/^/    /'
    done
    
    echo ""
    echo "Press Ctrl+C to exit, refreshing every ${REFRESH}s..."
    sleep "$REFRESH"
    clear
done

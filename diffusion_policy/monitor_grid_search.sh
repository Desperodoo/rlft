#!/bin/bash
#
# Monitor parallel grid search tasks
# Shows real-time progress and status
#
# Usage: ./monitor_grid_search.sh [--log-dir /tmp/ddql_grid_search]

LOG_DIR="/tmp/ddql_grid_search"

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
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

clear
while true; do
    echo "=========================================="
    echo "Grid Search Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Count status
    total=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    completed=$(grep -l "completed successfully\|DRY RUN" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    failed=$(grep -l "failed\|Error" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    running=$((total - completed - failed))
    
    echo "Total: $total | Completed: $completed | Running: $running | Failed: $failed"
    echo "Progress: $(( (completed + failed) * 100 / total ))%"
    echo ""
    
    # Show recent logs
    echo "Recent activity:"
    ls -t "$LOG_DIR"/*.log 2>/dev/null | head -3 | while read f; do
        exp_name=$(basename "$f" .log)
        status=$(tail -1 "$f" 2>/dev/null)
        echo "  $exp_name: $(tail -1 "$f" | head -c 60)"
    done
    
    echo ""
    echo "Press Ctrl+C to exit, refreshing every 10s..."
    sleep 10
    clear
done

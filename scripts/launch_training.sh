#!/bin/bash
# Keural Vision Encoder — Training Watchdog Launcher
#
# Launches training and auto-restarts it within 5 minutes if it crashes.
# Keeps running until max restarts reached or training completes normally.
#
# Usage:
#   bash scripts/launch_training.sh [--max-steps N] [--batch-size N]
#
# Monitor:
#   tail -f logs/train_run1.log
#   tail -f logs/watchdog.log

PYTHON="/home/mkd-s002/.local/bin/python3.10"
WORK_DIR="/home/mkd-s002/vision-encoder/poc"
CONFIG="configs/keural_tiny_poc.yaml"
OUTPUT_DIR="outputs/run"
LOG_FILE="logs/train_run1.log"
WATCHDOG_LOG="logs/watchdog.log"
PID_FILE="logs/train.pid"

RESTART_DELAY=300     # 5 minutes between restarts
MAX_RESTARTS=10       # stop watchdog after this many consecutive crashes

# Pass through any extra args (--max-steps, --batch-size, etc.)
EXTRA_ARGS="$@"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$WORK_DIR/logs"
cd "$WORK_DIR" || exit 1

# Prevent multiple watchdog instances
LOCK_FILE="logs/watchdog.lock"
if [ -f "$LOCK_FILE" ] && kill -0 $(cat "$LOCK_FILE") 2>/dev/null; then
    echo "[watchdog] Already running (PID $(cat $LOCK_FILE)). Exiting."
    exit 1
fi
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

log() {
    echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$WATCHDOG_LOG"
}

run_training() {
    log "Starting training (restart #${restart_count})"
    $PYTHON -u -m training.train \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        $EXTRA_ARGS \
        >> "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$PID_FILE"
    log "Training PID: $TRAIN_PID"
    wait $TRAIN_PID
    return $?
}

restart_count=0

log "=========================================="
log "Keural Training Watchdog Started"
log "Config: $CONFIG"
log "Extra args: $EXTRA_ARGS"
log "Max restarts: $MAX_RESTARTS"
log "Restart delay: ${RESTART_DELAY}s"
log "=========================================="

while true; do
    run_training
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "Training completed successfully (exit code 0). Watchdog exiting."
        rm -f "$PID_FILE"
        exit 0
    fi

    restart_count=$((restart_count + 1))

    # Read error reason if available
    ERROR_REASON="unknown"
    if [ -f "$OUTPUT_DIR/error.log" ]; then
        ERROR_REASON=$(grep "^Reason:" "$OUTPUT_DIR/error.log" | head -1)
    fi

    log "Training stopped with exit code $EXIT_CODE. $ERROR_REASON"
    log "Restart #${restart_count} of ${MAX_RESTARTS}"

    if [ $restart_count -ge $MAX_RESTARTS ]; then
        log "Max restarts ($MAX_RESTARTS) reached. Watchdog giving up."
        log "Check $OUTPUT_DIR/error.log for the last error."
        rm -f "$PID_FILE"
        exit 1
    fi

    log "Waiting ${RESTART_DELAY}s before restart..."
    for i in $(seq $RESTART_DELAY -30 1); do
        sleep 30
        log "Restarting in ${i}s..."
    done
    sleep $((RESTART_DELAY % 30))

    log "Restarting now."
    # Append a separator to the log so restarts are clearly visible
    echo "" >> "$LOG_FILE"
    echo "========== RESTART #${restart_count} — $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG_FILE"
done

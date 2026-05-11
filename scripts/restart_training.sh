#!/bin/bash
# Safe training restart — kills existing watchdog and trainer before starting fresh.
# Usage: bash scripts/restart_training.sh

WORK_DIR="/home/mkd-s002/vision-encoder/poc"
cd "$WORK_DIR" || exit 1

LOCK_FILE="logs/watchdog.lock"
PID_FILE="logs/train.pid"

# Kill existing watchdog
if [ -f "$LOCK_FILE" ]; then
    OLD_WATCHDOG=$(cat "$LOCK_FILE")
    if kill -0 "$OLD_WATCHDOG" 2>/dev/null; then
        echo "Killing watchdog PID $OLD_WATCHDOG..."
        kill "$OLD_WATCHDOG"
        sleep 2
    fi
    rm -f "$LOCK_FILE"
fi

# Kill existing training process
if [ -f "$PID_FILE" ]; then
    OLD_TRAIN=$(cat "$PID_FILE")
    if kill -0 "$OLD_TRAIN" 2>/dev/null; then
        echo "Killing training PID $OLD_TRAIN..."
        kill "$OLD_TRAIN"
        sleep 3
    fi
    rm -f "$PID_FILE"
fi

# Confirm nothing is left
REMAINING=$(pgrep -u mkd-s002 -f "training.train" | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "Warning: $REMAINING training process(es) still alive. Force killing..."
    pkill -u mkd-s002 -f "training.train"
    sleep 2
fi

echo "Starting fresh watchdog..."
nohup bash scripts/launch_training.sh > /dev/null 2>&1 &
echo "Watchdog PID: $!"

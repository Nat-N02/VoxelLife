#!/usr/bin/env bash

set -euo pipefail

# =========================
# CONFIG
# =========================
REFINE_LIST="refinement_jobs.csv"
EXE="./voxel"
PARAMFILE="params.txt"
METRICS_DIR="metrics"
SEEDS=(1 2 3 4)

MAX_JOBS=8

mkdir -p "$METRICS_DIR"

runId=0
running=0

# =========================
# PARALLEL SLOT CONTROL
# =========================
wait_for_slot() {
    while [ "$running" -ge "$MAX_JOBS" ]; do
        wait -n
        running=$((running - 1))
    done
}

# =========================
# MAIN LOOP
# =========================
# Skip CSV header, read rows
tail -n +2 "$REFINE_LIST" | while IFS=, read -r r f w; do
    for s in "${SEEDS[@]}"; do
        runId=$((runId + 1))
        wait_for_slot

        metricsFile="$METRICS_DIR/metrics_${runId}_r${r}_f${f}_w${w}_s${s}.csv"

        echo "QUEUE radius=$r repair_frac=$f W_decay=$w seed=$s"
        echo "  -> $metricsFile"

        (
            "$EXE" \
              --params "$PARAMFILE" \
              --set "sent_tail_radius=$r" \
              --set "repair_tail_frac=$f" \
              --set "W_decay=$w" \
              --seed "$s" \
              --metrics "$metricsFile"
        ) &

        running=$((running + 1))
    done
done

# =========================
# WAIT FOR ALL
# =========================
wait
echo "All jobs completed."

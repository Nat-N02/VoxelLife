#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
EXE="./voxel"
SEEDS=(1 2 3 4)

MAX_JOBS=8

# -------------------------
# RUN PLAN (SEQUENTIAL)
# name : params_file : metrics_dir : refinement_csv
# -------------------------
RUN_MODES=(
  "PHASE:params_phase.txt:metrics_phase:refine_phase.csv"
  "BASIN:params_basin.txt:metrics_basin:refine_basin.csv"
  "BIG:params_big.txt:metrics_big:refine_big.csv"
)

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
# MAIN LOOP (SEQUENTIAL MODES)
# =========================
for MODE in "${RUN_MODES[@]}"; do
    IFS=":" read -r MODE_NAME PARAMFILE METRICS_DIR REFINE_LIST <<< "$MODE"

    echo "========================================"
    echo "STARTING MODE: $MODE_NAME"
    echo "  Params:     $PARAMFILE"
    echo "  Output:     $METRICS_DIR"
    echo "  Refinement: $REFINE_LIST"
    echo "========================================"

    DONE_FILE="completed_runs_${MODE_NAME}.txt"

    mkdir -p "$METRICS_DIR"
    touch "$DONE_FILE"

    runId=0
    running=0

    {
        read  # skip header
        while IFS=, read -r r f w; do
            for s in "${SEEDS[@]}"; do

                key="${r},${f},${w},${s}"

                # ---- Crash recovery ----
                if grep -q "^$key$" "$DONE_FILE"; then
                    continue
                fi

                runId=$((runId + 1))
                wait_for_slot

                metricsFile="$METRICS_DIR/metrics_${runId}_r${r}_f${f}_w${w}_s${s}.csv"

                echo "[$MODE_NAME] QUEUE r=$r f=$f w=$w seed=$s"
                echo "  -> $metricsFile"

                (
                    "$EXE" \
                      --params "$PARAMFILE" \
                      --set "sent_tail_radius=$r" \
                      --set "repair_tail_frac=$f" \
                      --set "W_decay=$w" \
                      --seed "$s" \
                      --metrics "$metricsFile"

                    # ---- Mark done only if successful ----
                    echo "$key" >> "$DONE_FILE"
                ) &

                running=$((running + 1))
            done
        done
    } < "$REFINE_LIST"

    # ---- Wait for this MODE to finish completely ----
    wait
    echo "MODE COMPLETE: $MODE_NAME"
    echo
done

echo "========================================"
echo "ALL MODES COMPLETED"
echo "========================================"

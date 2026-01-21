import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================
# CONFIG
# =========================
METRICS_DIR = "metrics"
POLL_INTERVAL = 2.0   # seconds between scans

# --- Phase thresholds (TUNE THESE) ---
THRESH = {
    "dead": {
        "SPI_max": 0.01,
        "sent_q95_max": 0.05
    },
    "pid": {
        "SPI_min": 0.05,
        "BSI_max": 2.5
    },
    "crystal": {
        "BSI_min": 3.0
    },
    "high_damage": {
        "D_q95_min": 10.0
    }
}

# =========================
# PHASE CLASSIFIER
# =========================
def classify(row):
    SPI = row["SPI"]
    BSI = row["BSI"]
    D95 = row["D_q95"]
    S95 = row["sent_q95"]

    if BSI >= THRESH["crystal"]["BSI_min"]:
        return "Crystal"
    if D95 >= THRESH["high_damage"]["D_q95_min"]:
        return "High Damage"
    if SPI <= THRESH["dead"]["SPI_max"] and S95 <= THRESH["dead"]["sent_q95_max"]:
        return "Dead"
    if SPI >= THRESH["pid"]["SPI_min"] and BSI <= THRESH["pid"]["BSI_max"]:
        return "PID"

    return "Other"

# =========================
# LIVE STATE
# =========================
seen_files = set()
phase_counts = defaultdict(int)
timeline = []

# =========================
# PLOT SETUP
# =========================
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))

def redraw():
    ax.clear()

    phases = list(phase_counts.keys())
    counts = [phase_counts[p] for p in phases]

    ax.bar(phases, counts)
    ax.set_title("Live Phase Population")
    ax.set_ylabel("Runs observed")
    ax.set_xlabel("Phase")
    ax.set_ylim(0, max(1, max(counts) + 5))

    total = sum(counts)
    ax.text(0.95, 0.95, f"Total runs: {total}",
            transform=ax.transAxes,
            ha="right", va="top")

    fig.canvas.draw()
    fig.canvas.flush_events()

# =========================
# MAIN LOOP
# =========================
print("Watching:", METRICS_DIR)
print("Press Ctrl+C to stop")

while True:
    try:
        files = [
            f for f in os.listdir(METRICS_DIR)
            if f.endswith(".csv")
        ]

        new_files = [f for f in files if f not in seen_files]

        for fname in new_files:
            path = os.path.join(METRICS_DIR, fname)

            try:
                df = pd.read_csv(path)
                if len(df) == 0:
                    continue

                row = df.iloc[0]
                phase = classify(row)

                phase_counts[phase] += 1
                timeline.append((time.time(), phase))
                seen_files.add(fname)

                print(f"{fname:40s} → {phase}")

            except Exception as e:
                print("Error reading", fname, e)

        if new_files:
            redraw()

        time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")
        break

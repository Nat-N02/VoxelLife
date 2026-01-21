import pandas as pd
import numpy as np

SPI_THRESH = 0.05

# Zoom parameters
F_STEPS = [0.02, 0.01, 0.005]
W_REL_STEP = 0.1
W_MIN_STEP = 1e-5

# Load data
df = pd.read_csv("all_metrics.csv")

R_vals = sorted(df["sent_tail_radius"].unique())
F_vals = sorted(df["repair_tail_frac"].unique())
W_vals = sorted(df["W_decay"].unique())

def snap(val, grid):
    """Snap to nearest valid grid value"""
    return min(grid, key=lambda x: abs(x - val))

# Find hot points
hot = df[df["SPI"] >= SPI_THRESH][
    ["sent_tail_radius", "repair_tail_frac", "W_decay"]
].drop_duplicates()

refine_set = set()

for _, row in hot.iterrows():
    r, f, w = row

    # --- Radius: discrete topology ---
    R = [rv for rv in [r-1, r, r+1] if rv in R_vals]

    # --- Repair fraction: fine control surface ---
    F = {f}
    for df_step in F_STEPS:
        F.add(snap(f + df_step, F_vals))
        F.add(snap(f - df_step, F_vals))

    # --- W_decay: entropy scale ---
    dw = max(w * W_REL_STEP, W_MIN_STEP)
    W = {
        w,
        snap(w + dw, W_vals),
        snap(w - dw, W_vals),
        snap(w + 0.5 * dw, W_vals),
        snap(w - 0.5 * dw, W_vals),
    }

    # --- Combine ---
    for rr in R:
        for ff in F:
            for ww in W:
                refine_set.add((rr, ff, ww))

# Save
out = pd.DataFrame(
    sorted(refine_set),
    columns=["sent_tail_radius", "repair_tail_frac", "W_decay"]
)

out.to_csv("refinement_jobs.csv", index=False)
print(f"Refinement set size: {len(out)}")

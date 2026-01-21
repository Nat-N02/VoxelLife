import pandas as pd
import numpy as np

SPI_THRESH = 0.05

# Bounds (safety rails)
F_MIN, F_MAX = 0.0, 1.0
W_MIN, W_MAX = 0.0, 0.01

# Zoom behavior
F_STEPS = [0.02, 0.01, 0.005]
W_REL = [0.5, 0.25, 0.1]   # multiplicative zoom

# Load
df = pd.read_csv("all_metrics.csv")

R_vals = sorted(df["sent_tail_radius"].unique())

# Hot points
hot = df[df["SPI"] >= SPI_THRESH][
    ["sent_tail_radius", "repair_tail_frac", "W_decay"]
].drop_duplicates()

refine_set = set()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

for _, row in hot.iterrows():
    r, f, w = row

    # --- Radius (topology only) ---
    R = [rr for rr in (r-1, r, r+1) if rr >= 1]

    # --- Repair fraction (linear zoom) ---
    F = {f}
    for step in F_STEPS:
        F.add(clamp(f + step, F_MIN, F_MAX))
        F.add(clamp(f - step, F_MIN, F_MAX))

    # --- W_decay (log-style zoom) ---
    W = {w}
    for rel in W_REL:
        delta = max(w * rel, 1e-6)
        W.add(clamp(w + delta, W_MIN, W_MAX))
        W.add(clamp(w - delta, W_MIN, W_MAX))

    # --- Combine ---
    for rr in R:
        for ff in F:
            for ww in W:
                refine_set.add((
                    int(rr),
                    round(float(ff), 6),
                    round(float(ww), 8)
                ))

# Save
out = pd.DataFrame(
    sorted(refine_set),
    columns=["sent_tail_radius", "repair_tail_frac", "W_decay"]
)

out.to_csv("refinement_jobs.csv", index=False)
print(f"Refinement set size: {len(out)}")
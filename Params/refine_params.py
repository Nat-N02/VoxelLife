import pandas as pd
import numpy as np

# -------------------------
# CONFIG
# -------------------------
RADIUS = 10

TAU_MIN = 0.6
TAU_MAX = 0.8
TAU_STEP = 0.001

SIGS = np.logspace(0, 1, num=31)
EXP_MIN = -2
EXP_MAX = 0

OUTFILE = "refinement_jobs.csv"

# -------------------------
# Generate τ values
# -------------------------
taus = np.round(
    np.arange(TAU_MIN, TAU_MAX + 1e-9, TAU_STEP),
    6
)

# -------------------------
# Generate W_decay values (± log grid)
# -------------------------
w_vals = set()

for n in range(EXP_MIN, EXP_MAX + 1):
    for s in SIGS:
        base = s * (10 ** n)
        w_vals.add(round(float(base), 8))

# Optional: keep zero explicitly if you want a neutral column
w_vals.add(0.0)

# Sort for nice CSV structure
w_vals = sorted(w_vals)

# -------------------------
# Build sweep
# -------------------------
rows = []
for tau in taus:
    for w in w_vals:
        rows.append((RADIUS, tau, w))

df = pd.DataFrame(
    rows,
    columns=["sent_tail_radius", "repair_tail_frac", "W_decay"]
)

# -------------------------
# Save
# -------------------------
df.to_csv(OUTFILE, index=False)

print("Sweep generated.")
print(f"  Radius: {RADIUS}")
print(f"  τ values: {len(taus)}")
print(f"  W_decay values: {len(w_vals)}")
print(f"  Total jobs: {len(df)}")
print(f"  Output: {OUTFILE}")

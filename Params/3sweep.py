import pandas as pd
import numpy as np
from itertools import product

# ==========================
# CORE PARAM SPACE
# ==========================

# τ range (repair_tail_frac)
TAU_MIN = 0.6
TAU_MAX = 0.9

# W_decay ladder (signed, log-spaced)
W_VALUES = [
    -1.0,
    -0.1,
    -0.01,
    -0.001,
    -0.0001,
     0.0,
     0.0001,
     0.001,
     0.01,
     0.1,
     1.0
]

# ==========================
# SWEEP PROFILES
# ==========================

RUNS = {
    # Coarse global phase map
    "refine_phase.csv": {
        "radius": 10,
        "tau_steps": 0.05
    },

    # Dense ridge / basin probe
    "refine_basin.csv": {
        "radius": 10,
        "tau_steps": 0.01
    },

    # Scaling + invariance test
    "refine_big.csv": {
        "radius": 15,
        "tau_steps": 0.02
    }
}

# ==========================
# GENERATOR
# ==========================

def generate_tau_values(step):
    vals = np.arange(TAU_MIN, TAU_MAX + 1e-9, step)
    return [round(float(v), 6) for v in vals]

def generate_jobs(radius, tau_step):
    tau_vals = generate_tau_values(tau_step)

    jobs = []
    for r, f, w in product([radius], tau_vals, W_VALUES):
        jobs.append((
            int(r),
            round(float(f), 6),
            round(float(w), 8)
        ))

    return pd.DataFrame(
        jobs,
        columns=["sent_tail_radius", "repair_tail_frac", "W_decay"]
    )

# ==========================
# WRITE FILES
# ==========================

total = 0

for filename, cfg in RUNS.items():
    df = generate_jobs(
        radius=cfg["radius"],
        tau_step=cfg["tau_steps"]
    )

    df.to_csv(filename, index=False)
    print(f"{filename}: {len(df)} jobs")
    total += len(df)

print(f"\nTotal jobs generated: {total}")

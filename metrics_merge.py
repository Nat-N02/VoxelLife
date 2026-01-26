import os
import glob

METRICS_DIR = "metrics_claw"
OUT_FILE = "all_metrics.csv"

files = sorted(glob.glob(os.path.join(METRICS_DIR, "*.csv")))

if not files:
    raise RuntimeError("No CSV files found in metrics/")

header_written = False
rows_written = 0

with open(OUT_FILE, "w", newline="") as out:
    for path in files:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # First line is header
                if i == 0:
                    if not header_written:
                        out.write(line + "\n")
                        header_written = True
                else:
                    out.write(line + "\n")
                    rows_written += 1

print(f"Merged {len(files)} files")
print(f"Wrote {rows_written} data rows to {OUT_FILE}")

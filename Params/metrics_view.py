import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider

# =========================
# CONFIG
# =========================
CSV_PATH = "all_metrics.csv"

AXIS_PARAMS = [
    "sent_tail_radius",
    "repair_tail_frac",
    "W_decay"
]

METRICS = [
    "BSI",
    "RFC",
    "RLI",
    "SPI",
    "FTP",
    "D_q50",
    "D_q95",
    "sent_q95"
]
VAR_MODES = ["mean", "abs_diff", "std"]

# =========================
# LOAD & PREP
# =========================
df = pd.read_csv(CSV_PATH)

DROP_COLS = ["seed", "nx", "ny", "nz"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

raw = df.copy()

# Still keep a mean version for normal mode
agg = raw.groupby(AXIS_PARAMS, as_index=False).mean(numeric_only=True)

# =========================
# PLOT CORE
# =========================
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(right=0.65)

current = {
    "x": AXIS_PARAMS[0],
    "y": AXIS_PARAMS[1],
    "metric": METRICS[0],
    "mode": "mean"
}

current["slice_param"] = None
current["slice_value"] = None

im = None

def draw():
    global im, slice_slider

    ax.clear()

    x = current["x"]
    y = current["y"]
    m = current["metric"]

    slice_param = get_slice_param()
    current["slice_param"] = slice_param

    values = np.sort(agg[slice_param].unique())

    # Initialize slice value if needed
    if current["slice_value"] not in values:
        current["slice_value"] = values[len(values) // 2]

    # Filter raw data by slice
    df_slice_raw = raw[raw[slice_param] == current["slice_value"]]

    if current["mode"] == "mean":
        g = df_slice_raw.groupby([x, y])[m].mean().reset_index()

    elif current["mode"] == "abs_diff":
        g = (
            df_slice_raw
            .groupby([x, y])[m]
            .apply(lambda v: (v.quantile(0.75) - v.quantile(0.25)) if len(v) >= 2 else np.nan)
            .reset_index(name=m)
        )

    elif current["mode"] == "std":
        g = df_slice_raw.groupby([x, y])[m].std().reset_index()

    pivot = g.pivot(index=y, columns=x, values=m)

    x_vals = pivot.columns.values.astype(float)
    y_vals = pivot.index.values.astype(float)
    Z = pivot.values

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            x_vals.min(), x_vals.max(),
            y_vals.min(), y_vals.max()
        ],
        interpolation="nearest"
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{m} ({current['mode']}) | {slice_param} = {current['slice_value']}")

    # --- Colorbar ---
    if not hasattr(draw, "cbar"):
        draw.cbar = plt.colorbar(im, ax=ax, label=m)
    else:
        draw.cbar.update_normal(im)
        draw.cbar.set_label(m)

    # --- Slider ---
    ax_slice.clear()

    slice_slider = Slider(
        ax_slice,
        slice_param,
        0,
        len(values) - 1,
        valinit=list(values).index(current["slice_value"]),
        valstep=1
    )

    def on_slide(val):
        idx = int(val)
        current["slice_value"] = values[idx]
        draw()

    slice_slider.on_changed(on_slide)

    fig.canvas.draw_idle()

# =========================
# UI
# =========================
# =========================
# UI (RIGHT SIDE)
# =========================

ax_slice = plt.axes([0.70, 0.88, 0.25, 0.04])

ax_x = plt.axes([0.70, 0.60, 0.25, 0.20])
ax_y = plt.axes([0.70, 0.35, 0.25, 0.20])
ax_m = plt.axes([0.70, 0.10, 0.25, 0.20])
ax_mode = plt.axes([0.70, 0.02, 0.25, 0.07])

radio_x = RadioButtons(ax_x, AXIS_PARAMS)
radio_y = RadioButtons(ax_y, AXIS_PARAMS)
radio_m = RadioButtons(ax_m, METRICS)
radio_mode = RadioButtons(ax_mode, VAR_MODES)

ax_slice.set_title("Slice Parameter")

ax_x.set_title("X Axis")
ax_y.set_title("Y Axis")
ax_m.set_title("Metric")
ax_mode.set_title("Display Mode")


def update_mode(label):
    current["mode"] = label
    draw()

radio_mode.on_clicked(update_mode)

ax_slice = plt.axes([0.05, 0.90, 0.25, 0.04])
slice_slider = None

def get_slice_param():
    for p in AXIS_PARAMS:
        if p != current["x"] and p != current["y"]:
            return p
    return None

ax_x.set_title("X Axis")
ax_y.set_title("Y Axis")
ax_m.set_title("Metric")

def update_x(label):
    if label != current["y"]:
        current["x"] = label
        draw()

def update_y(label):
    if label != current["x"]:
        current["y"] = label
        draw()

def update_m(label):
    current["metric"] = label
    draw()

radio_x.on_clicked(update_x)
radio_y.on_clicked(update_y)
radio_m.on_clicked(update_m)

# =========================
# START
# =========================
draw()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

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


def make_log_lattice(n_min=-5, n_max=0, mantissas=(1, 2, 3, 5, 7)):
    vals = []
    for n in range(n_min, n_max + 1):
        for m in mantissas:
            v = m * (10 ** n)
            vals.append(v)
            vals.append(-v)
    return np.array(sorted(vals), dtype=float)

# W_decay display controls
W_MAX_ABS = 1.0      # cutoff: keep only -1 <= W_decay <= 1
W_STRIDE = 2        # keep every Nth lattice value (2 = half density, 3 = third, etc.)

_full_lattice = make_log_lattice()

# Cut to range
W_LATTICE = _full_lattice[np.abs(_full_lattice) <= W_MAX_ABS]

W_INDEX = {v: i for i, v in enumerate(W_LATTICE)}


def snap_to_lattice_index(values, lattice, tol=1e-12):
    """Snap floats to nearest lattice value, returning integer lattice indices (or NaN if not within tol)."""
    values = np.asarray(values, dtype=float)
    idxs = np.full(values.shape, np.nan, dtype=float)
    for k, v in enumerate(values):
        i = int(np.argmin(np.abs(lattice - v)))
        if abs(lattice[i] - v) <= tol:
            idxs[k] = i
    return idxs


def categorical_edges(vals):
    """
    Return edges for categorical axis positions [0..n-1] so each cell is centered on an integer.
    """
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return np.array([0.0, 1.0])
    # vals are expected to be sorted integer positions (or float equivalents)
    return np.concatenate(([vals[0] - 0.5], vals + 0.5))


# =========================
# LOAD & PREP
# =========================
df = pd.read_csv(CSV_PATH)

DROP_COLS = ["seed", "nx", "ny", "nz"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

raw = df.copy()

# Prepare a snapped categorical index for W_decay for ALL rows up front.
# This prevents float-near-equality from exploding the pivot index.
snapped_idx = snap_to_lattice_index(raw["W_decay"].values, W_LATTICE, tol=1e-12)
mask = ~np.isnan(snapped_idx)
raw = raw[mask].copy()
raw["W_idx"] = snapped_idx[mask].astype(int)
raw["W_decay"] = W_LATTICE[raw["W_idx"].values]

# Mean version for slice slider options
agg = raw.groupby(AXIS_PARAMS, as_index=False).mean(numeric_only=True)

# =========================
# PLOT CORE
# =========================
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(right=0.65)

current = {
    "x": "repair_tail_frac",
    "y": "W_decay",
    "metric": METRICS[0],
    "mode": "mean",
    "slice_param": None,
    "slice_value": None,
}

im = None
slice_slider = None


def get_slice_param():
    for p in AXIS_PARAMS:
        if p != current["x"] and p != current["y"]:
            return p
    return None


def value_col_for_axis(axis_name: str) -> str:
    """Use W_idx for grouping if axis is W_decay, otherwise use the raw column."""
    return "W_idx" if axis_name == "W_decay" else axis_name


def set_w_decay_ticks(a):
    """Pretty tick labels for the W_decay axis when it's categorical positions."""
    a.set_yticks(np.arange(0, len(W_LATTICE), 2))

    def fmt(_pos, _):
        # _pos might not be integer due to pan/zoom; only label near integers
        i = int(round(_pos))
        if 0 <= i < len(W_LATTICE) and abs(_pos - i) < 1e-6:
            v = W_LATTICE[i]
            return f"{v:.0e}"
        return ""

    a.yaxis.set_major_formatter(FuncFormatter(fmt))


def draw():
    global im, slice_slider

    ax.clear()

    x = current["x"]
    y = current["y"]
    m = current["metric"]
    mode = current["mode"]

    slice_param = get_slice_param()
    current["slice_param"] = slice_param

    values = np.sort(agg[slice_param].unique())
    if values.size == 0:
        ax.set_title("No data after snapping/filtering.")
        fig.canvas.draw_idle()
        return

    if current["slice_value"] not in values:
        current["slice_value"] = values[len(values) // 2]

    df_slice = raw[raw[slice_param] == current["slice_value"]].copy()

    gx = value_col_for_axis(x)
    gy = value_col_for_axis(y)

    if mode == "mean":
        g = df_slice.groupby([gx, gy])[m].mean().reset_index()
    elif mode == "abs_diff":
        g = (
            df_slice.groupby([gx, gy])[m]
            .apply(lambda v: (v.quantile(0.75) - v.quantile(0.25)) if len(v) >= 2 else np.nan)
            .reset_index(name=m)
        )
    elif mode == "std":
        g = df_slice.groupby([gx, gy])[m].std().reset_index()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Map categorical indices back to physical values if needed (for labels/title only)
    if x == "W_decay":
        g["W_decay_x"] = W_LATTICE[g["W_idx"].values]
    if y == "W_decay":
        g["W_decay_y"] = W_LATTICE[g["W_idx"].values]

    g = g.sort_values([gy, gx])
    pivot = g.pivot(index=gy, columns=gx, values=m)

    # Build axis coordinates
    x_vals = pivot.columns.values.astype(float)
    y_vals = pivot.index.values.astype(float)
    Z = pivot.values

    # Guard against all-NaN grids
    finite_Z = Z[np.isfinite(Z)]
    if finite_Z.size == 0:
        ax.set_title(f"{m} ({mode}) | {slice_param} = {current['slice_value']} | (no finite values)")
        fig.canvas.draw_idle()
        return

    # For W_decay we treat axis as categorical positions (indices), not numeric log space.
    if x == "W_decay":
        # categorical positions: x_vals are W_idx
        X = categorical_edges(x_vals)
        ax.set_xlim(X[0], X[-1])
        ax.set_xlabel("W_decay")
        # optional: pretty ticks if W_decay on x (rare)
        ax.set_xticks(np.arange(len(W_LATTICE)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda pos, _: f"{W_LATTICE[int(round(pos))]:.0e}"
                                                   if abs(pos - round(pos)) < 1e-6 and 0 <= int(round(pos)) < len(W_LATTICE)
                                                   else ""))
        ax.tick_params(axis="x", labelrotation=45)
    else:
        # numeric axis edges
        x_vals_sorted = np.sort(x_vals)
        dx = np.diff(x_vals_sorted)
        if dx.size == 0:
            X = np.array([x_vals_sorted[0] - 0.5, x_vals_sorted[0] + 0.5])
        else:
            X = np.concatenate(([x_vals_sorted[0] - dx[0] / 2], x_vals_sorted[:-1] + dx / 2, [x_vals_sorted[-1] + dx[-1] / 2]))
        ax.set_xlabel(x)

    if y == "W_decay":
        Y = categorical_edges(y_vals)
        ax.set_ylim(Y[0], Y[-1])
        ax.set_ylabel("W_decay")
        set_w_decay_ticks(ax)
    else:
        y_vals_sorted = np.sort(y_vals)
        dy = np.diff(y_vals_sorted)
        if dy.size == 0:
            Y = np.array([y_vals_sorted[0] - 0.5, y_vals_sorted[0] + 0.5])
        else:
            Y = np.concatenate(([y_vals_sorted[0] - dy[0] / 2], y_vals_sorted[:-1] + dy / 2, [y_vals_sorted[-1] + dy[-1] / 2]))
        ax.set_ylabel(y)

    # Ensure X and Y are increasing
    if x != "W_decay":
        # X built from sorted values; OK
        pass
    if y != "W_decay":
        # Y built from sorted values; OK
        pass

    # pcolormesh expects:
    #   len(X) == Z.shape[1] + 1
    #   len(Y) == Z.shape[0] + 1
    # So we must align edges with pivot ordering.
    # Rebuild edges in pivot order, not sorted order:
    if x == "W_decay":
        X = categorical_edges(x_vals)  # pivot order (already ascending if W_idx ascending)
    else:
        # pivot columns order may be ascending but keep it explicit
        xv = x_vals
        dx = np.diff(xv)
        if dx.size == 0:
            X = np.array([xv[0] - 0.5, xv[0] + 0.5])
        else:
            X = np.concatenate(([xv[0] - dx[0] / 2], xv[:-1] + dx / 2, [xv[-1] + dx[-1] / 2]))

    if y == "W_decay":
        Y = categorical_edges(y_vals)
    else:
        yv = y_vals
        dy = np.diff(yv)
        if dy.size == 0:
            Y = np.array([yv[0] - 0.5, yv[0] + 0.5])
        else:
            Y = np.concatenate(([yv[0] - dy[0] / 2], yv[:-1] + dy / 2, [yv[-1] + dy[-1] / 2]))

    # Color scaling
    use_lognorm = np.nanmin(finite_Z) > 0
    norm = LogNorm(vmin=np.nanmin(finite_Z), vmax=np.nanmax(finite_Z)) if use_lognorm else None

    im = ax.pcolormesh(
        X,
        Y,
        Z,
        shading="auto",
        norm=norm
    )

    ax.set_title(f"{m} ({mode}) | {slice_param} = {current['slice_value']}")

    # --- Colorbar ---
    if not hasattr(draw, "cbar"):
        draw.cbar = plt.colorbar(im, ax=ax, label=m)
    else:
        draw.cbar.update_normal(im)
        draw.cbar.set_label(m)

    # --- Slider ---
    ax_slice.clear()
    ax_slice.set_title("Slice Parameter")

    slice_slider = Slider(
        ax_slice,
        slice_param,
        0,
        len(values) - 1,
        valinit=int(np.where(values == current["slice_value"])[0][0]),
        valstep=1
    )

    def on_slide(val):
        idx = int(val)
        current["slice_value"] = values[idx]
        draw()

    slice_slider.on_changed(on_slide)

    fig.canvas.draw_idle()


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

ax_x.set_title("X Axis")
ax_y.set_title("Y Axis")
ax_m.set_title("Metric")
ax_mode.set_title("Display Mode")


def update_mode(label):
    current["mode"] = label
    draw()


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


radio_mode.on_clicked(update_mode)
radio_x.on_clicked(update_x)
radio_y.on_clicked(update_y)
radio_m.on_clicked(update_m)

# =========================
# START
# =========================
draw()
plt.show()

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# -------------------------
# PID DEFINITION (single source of truth)
# -------------------------
PID_X = 100    # center x (array coordinates)
PID_Y = 317    # center y
INSET_RADIUS = 16

def pid_bounds():
    x0 = PID_X - INSET_RADIUS
    y0 = PID_Y - INSET_RADIUS
    w = 2 * INSET_RADIUS
    h = 2 * INSET_RADIUS
    return x0, y0, w, h

def dump_name(idx):
    return f"dump/dump_t{idx:08d}.bin"


def load_dump(fname):
    with open(fname, "rb") as f:
        nx = np.fromfile(f, dtype="<i4", count=1)[0]
        ny = np.fromfile(f, dtype="<i4", count=1)[0]
        nz = np.fromfile(f, dtype="<i4", count=1)[0]
        tick = np.fromfile(f, dtype="<u8", count=1)[0]

        nvox = nx * ny * nz

        E = np.fromfile(f, dtype="<f4", count=nvox)
        D = np.fromfile(f, dtype="<f4", count=nvox)
        P = np.fromfile(f, dtype="<f4", count=nvox)
        R = np.fromfile(f, dtype="<f4", count=nvox)

    shape = (nz, ny, nx)  # z, y, x
    return {
        "tick": tick,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "E": E.reshape(shape),
        "D": D.reshape(shape),
        "P": P.reshape(shape),
        "R_boost": R.reshape(shape),
    }

# -------------------------
# CONFIG (paper defaults)
# -------------------------
FIELDS = ["E", "D", "R_boost", "P"]
PID_X, PID_Y = 100, 317
INSET_RADIUS = 16

def crop(arr, cx, cy, r):
    return arr[
        max(0, cy - r):cy + r,
        max(0, cx - r):cx + r
    ]

FIELD_LABELS = {
    "E": "Energy (E)",
    "D": "Damage (D)",
    "R_boost": "Repair Boost (R)",
    "P": "Precursor (P)"
}

COLORMAPS = {
    "E": "viridis",
    "D": "cividis",
    "R_boost": "viridis",
    "P": "viridis"
}

# Fixed scales = visual honesty
CLIMS = {
    "E": (0.0, 1.0),
    "D": (0.0, 20.0),
    "R_boost": (0.0, 1.0),
    "P": (0.0, 1.0),
}

OUTPUT_DIR = "figures"
DPI = 300

def draw_pid_box(ax):
    x0, y0, w, h = pid_bounds()
    rect = Rectangle(
        (x0, y0),
        w,
        h,
        linewidth=0.8,
        edgecolor="red",
        facecolor="none",
        alpha=0.9
    )
    ax.add_patch(rect)


def extract_slice(data, field, axis, idx):
    arr = data[field]
    if axis == 0:
        return arr[idx, :, :]
    elif axis == 1:
        return arr[:, idx, :]
    else:
        return arr[:, :, idx]


# -------------------------
# Main render function
# -------------------------
def render_figure(
    dump_idx,
    axis=0,
    slice_idx=1,
    outname=None,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_dump(dump_name(dump_idx))

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

    for ax, field in zip(axs.flat, FIELDS):
        img = extract_slice(data, field, axis, slice_idx)

        im = ax.imshow(
            img,
            origin="lower",
            cmap=COLORMAPS[field],
            vmin=CLIMS[field][0],
            vmax=CLIMS[field][1],
            interpolation="nearest"
        )

        ax.set_title(FIELD_LABELS[field], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        inset = inset_axes(ax, width="30%", height="30%", loc="upper right")

        zoom = crop(img, PID_X, PID_Y, INSET_RADIUS)

        inset.imshow(
            zoom,
            origin="lower",
            cmap=COLORMAPS[field],
            vmin=CLIMS[field][0],
            vmax=CLIMS[field][1],
            interpolation="nearest"
        )

        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("PID", fontsize=7)
        
        draw_pid_box(ax)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Voxel Repair Dynamics | tick={data['tick']} | "
        f"{['z','y','x'][axis]}={slice_idx}",
        fontsize=12
    )

    if outname is None:
        outname = f"fig_vrd_t{dump_idx:08d}.pdf"

    outpath = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(outpath, dpi=DPI)
    plt.close(fig)

    print(f"[Figure saved] {outpath}")

render_figure(40001)
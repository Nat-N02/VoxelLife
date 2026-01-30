import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from matplotlib.widgets import Slider, RadioButtons, TextBox, Button, CheckButtons

# =========================
# FILE IO
# =========================

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

# =========================
# VISUALIZER CORE
# =========================

def visualize_time_slices(start_idx=1, step=1, export_args=None):
    state = {
        "file_idx": start_idx,
        "step": step,
        "data": None,
        "field": "E",
        "field2": "R_boost",
        "axis": 0,
        "slice": 0,
        "lock_scale": True,
        "playing": False,
        "delay_ms": 200,
        "sxs": False,
    }

    def load_current():
        fname = dump_name(state["file_idx"])
        state["data"] = load_dump(fname)
        dims = [state["data"]["nz"], state["data"]["ny"], state["data"]["nx"]]
        state["slice"] = min(state["slice"], dims[state["axis"]] - 1)

    load_current()

    fields = ["E", "D", "P", "R_boost"]
    axis_names = ["z", "y", "x"]

    # Macroblock safe: 704x608 px
    fig, axes = plt.subplots(1, 2, figsize=(14.08, 6.08), dpi=100)
    ax1, ax2 = axes

    plt.subplots_adjust(left=0.3, bottom=0.3, wspace=0.25)

    def get_slice(field):
        arr = state["data"][field]
        s = state["slice"]
        if state["axis"] == 0:
            return arr[s, :, :]
        elif state["axis"] == 1:
            return arr[:, s, :]
        else:
            return arr[:, :, s]

    img1 = ax1.imshow(get_slice(state["field"]), origin="lower", aspect="auto")
    img2 = ax2.imshow(get_slice(state["field2"]), origin="lower", aspect="auto")

    cbar1 = plt.colorbar(img1, ax=ax1)
    cbar2 = plt.colorbar(img2, ax=ax2)

    def autoscale(img, field):
        if state["lock_scale"] and field in ("E", "D"):
            img.set_clim(0.0, 10.0)
        elif state["lock_scale"] and field == "R_boost":
            img.set_clim(0.0, 1.0)
        else:
            img.autoscale()

    def refresh():
        img1.set_data(get_slice(state["field"]))
        img2.set_data(get_slice(state["field2"]))

        autoscale(img1, state["field"])
        autoscale(img2, state["field2"])

        ax1.set_title(f"{state['field']}")
        ax2.set_title(f"{state['field2']}")

        fig.suptitle(
            f"{axis_names[state['axis']]}={state['slice']} | "
            f"tick={state['data']['tick']} | file={state['file_idx']:08d}"
        )

        fig.canvas.draw_idle()

    refresh()

    # =========================
    # EXPORT MODE
    # =========================

    if export_args is not None:
        out = Path(export_args["out"])
        out.parent.mkdir(parents=True, exist_ok=True)

        state["field"] = export_args["field"]
        state["field2"] = export_args["field2"]
        state["axis"] = axis_names.index(export_args["axis"])
        state["slice"] = export_args["slice"]
        state["lock_scale"] = True

        writer = imageio.get_writer(out, fps=export_args["fps"], codec="libx264")

        print("Exporting video to:", out)

        for idx in range(export_args["start"], export_args["end"] + 1, step):
            try:
                state["file_idx"] = idx
                load_current()
                refresh()

                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape((h, w, 4))

                frame = buf[:, :, :3]
                writer.append_data(frame)

            except FileNotFoundError:
                print("Missing file:", dump_name(idx))
                break

        writer.close()
        print("Done.")
        plt.close(fig)
        return

    # =========================
    # UI CONTROLS
    # =========================

    ax_slice = plt.axes([0.3, 0.18, 0.6, 0.03])
    slice_slider = Slider(ax_slice, "Slice", 0, state["data"]["nz"] - 1, valinit=0, valstep=1)

    def on_slice(val):
        state["slice"] = int(val)
        refresh()

    slice_slider.on_changed(on_slice)

    ax_field = plt.axes([0.05, 0.55, 0.2, 0.25])
    radio_field = RadioButtons(ax_field, fields)

    def on_field(label):
        state["field"] = label
        refresh()

    radio_field.on_clicked(on_field)

    ax_field2 = plt.axes([0.05, 0.30, 0.2, 0.25])
    radio_field2 = RadioButtons(ax_field2, fields)

    def on_field2(label):
        state["field2"] = label
        refresh()

    radio_field2.on_clicked(on_field2)

    ax_axis = plt.axes([0.05, 0.05, 0.2, 0.2])
    radio_axis = RadioButtons(ax_axis, axis_names)

    def on_axis(label):
        state["axis"] = axis_names.index(label)
        dims = [state["data"]["nz"], state["data"]["ny"], state["data"]["nx"]]
        slice_slider.valmax = dims[state["axis"]] - 1
        slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
        slice_slider.set_val(min(state["slice"], slice_slider.valmax))
        refresh()

    radio_axis.on_clicked(on_axis)

    ax_scale = plt.axes([0.3, 0.05, 0.2, 0.08])
    check_scale = CheckButtons(ax_scale, ["Lock scale"], [True])

    def toggle_scale(label):
        state["lock_scale"] = not state["lock_scale"]
        refresh()

    check_scale.on_clicked(toggle_scale)

    plt.show()

# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=int, help="file step size")
    parser.add_argument("--export", nargs=2, metavar=("START", "END"), type=int)
    parser.add_argument("--field", default="D", choices=["E", "D", "P", "R_boost"])
    parser.add_argument("--field2", default="R_boost", choices=["E", "D", "P", "R_boost"])
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"])
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", default="vrd_export.mp4")
    parser.add_argument("--sxs", action="store_true")

    args = parser.parse_args()

    export_cfg = None
    if args.export:
        export_cfg = {
            "start": args.export[0],
            "end": args.export[1],
            "field": args.field,
            "field2": args.field2,
            "axis": args.axis,
            "slice": args.slice,
            "fps": args.fps,
            "out": args.out,
        }

    visualize_time_slices(
        start_idx=args.export[0] if args.export else 1,
        step=args.step,
        export_args=export_cfg
    )

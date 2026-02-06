import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from matplotlib.widgets import Slider, RadioButtons, TextBox, Button, CheckButtons
from collections import defaultdict

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

def pid_mask_from_D(D, q=0.20):
    thr = np.quantile(D, q)
    return D < thr, thr
# =========================
# DERIVED STRUCTURES
# =========================

def pid_mask_from_D(D, q=0.20):
    thr = np.quantile(D, q)
    return D < thr, thr

def label_components(mask):
    structure = ndi.generate_binary_structure(rank=mask.ndim, connectivity=1)
    labels, nlab = ndi.label(mask, structure=structure)
    return labels, nlab

def boundary_shell(mask, iters=2):
    dil = ndi.binary_dilation(mask, iterations=iters)
    ero = ndi.binary_erosion(mask, iterations=iters)
    return dil & (~ero)

def hoarder_score(D, P, q=0.20):
    mask, _ = pid_mask_from_D(D, q)
    shell = boundary_shell(mask, 2)
    if mask.sum() == 0 or shell.sum() == 0:
        return 0.0
    return float(P[mask].mean() / (P[shell].mean() + 1e-6))

def fire_score(curr, prev):
    if prev is None:
        return np.zeros_like(curr)
    return np.abs(curr - prev)

def label_components(mask, connectivity=1):
    # connectivity=1 => 6-neigh in 3D
    if not HAVE_SCIPY:
        raise RuntimeError("Install scipy for component labeling: pip install scipy")
    structure = ndi.generate_binary_structure(rank=mask.ndim, connectivity=connectivity)
    labels, nlab = ndi.label(mask, structure=structure)
    return labels, nlab

def component_stats(labels, nlab, fields):
    # fields: dict of name->3D array (D,P,R,E)
    stats = []
    for lab in range(1, nlab+1):
        vox = (labels == lab)
        n = int(vox.sum())
        if n == 0:
            continue
        # centroid in z,y,x coords
        coords = np.argwhere(vox)
        cz, cy, cx = coords.mean(axis=0)
        s = {
            "label": lab,
            "n": n,
            "cz": float(cz),
            "cy": float(cy),
            "cx": float(cx),
        }
        for k, arr in fields.items():
            s[f"mean_{k}"] = float(arr[vox].mean())
        stats.append(s)
    return stats

def boundary_shell(mask, iters=1):
    # shell = dilated(mask) - eroded(mask)
    if not HAVE_SCIPY:
        raise RuntimeError("Install scipy for morphology: pip install scipy")
    dil = ndi.binary_dilation(mask, iterations=iters)
    ero = ndi.binary_erosion(mask, iterations=iters)
    shell = dil & (~ero)
    return shell

def hoarder_score(D, P, q=0.20, shell_iters=2):
    mask, thr = pid_mask_from_D(D, q=q)
    shell = boundary_shell(mask, iters=shell_iters)
    Pin = P[mask].mean() if mask.any() else 0.0
    Psh = P[shell].mean() if shell.any() else (Pin + 1e-6)
    return float(Pin / (Psh + 1e-6)), thr

def fire_score(currP, prevP=None, currR=None, prevR=None, mode="P"):
    if prevP is None and prevR is None:
        return None
    if mode == "P":
        return np.abs(currP - prevP)
    else:
        return np.abs(currR - prevR)

def match_components_by_overlap(prev_labels, curr_labels, prev_nlab, curr_nlab):
    # Greedy overlap matching: for each curr label, find prev label with max overlap
    mapping = {}
    if prev_labels is None:
        return mapping

    # compute overlap counts via contingency
    # (works fine at these sizes)
    for c in range(1, curr_nlab+1):
        vox = (curr_labels == c)
        prev_ids, counts = np.unique(prev_labels[vox], return_counts=True)
        # ignore 0 (background)
        best = 0
        bestcnt = 0
        for pid, cnt in zip(prev_ids, counts):
            if pid == 0:
                continue
            if cnt > bestcnt:
                bestcnt = int(cnt)
                best = int(pid)
        if best != 0:
            mapping[c] = best
    return mapping

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
        C = np.fromfile(f, dtype="<f4", count=nvox)

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
        "C": C.reshape(shape),
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
        "axis": 0,
        "slice": 0,
        "lock_scale": True,
        "playing": False,
        "delay_ms": 200,
        "pid_q": 0.20,
        "derived": "None",
        "prev_frame": None,
        "events": [],
        "event_idx": 0,
    }

    def load_current():
        fname = dump_name(state["file_idx"])
        state["prev_frame"] = state["data"]
        state["data"] = load_dump(fname)
        dims = [state["data"]["nz"], state["data"]["ny"], state["data"]["nx"]]
        state["slice"] = min(state["slice"], dims[state["axis"]] - 1)

    load_current()

    fields = ["E", "D", "P", "R_boost", "C", "PID_mask", "Components", "Fire", "Hoard"]
    axis_names = ["z", "y", "x"]

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(left=0.3, bottom=0.3)

    def get_slice():
        data = state["data"]
        field = state["field"]

        D = data["D"]
        P = data["P"]
        R = data["R_boost"]
        C = data["C"]

        if field == "PID_mask":
            mask, _ = pid_mask_from_D(D, state["pid_q"])
            arr = mask.astype(float)

        elif field == "Components":
            mask, _ = pid_mask_from_D(D, state["pid_q"])
            labels, _ = label_components(mask)
            arr = labels.astype(float)

        elif field == "Fire":
            prev = state["prev_frame"]
            if prev is None:
                arr = np.zeros_like(D)
            else:
                arr = fire_score(P, prev["P"])

        elif field == "Hoard":
            mask, _ = pid_mask_from_D(D, state["pid_q"])
            shell = boundary_shell(mask)
            arr = np.zeros_like(D)
            arr[mask] = P[mask]
            arr[shell] = P[shell] * 2.0

        else:
            arr = data[field]

        s = state["slice"]
        if state["axis"] == 0:
            return arr[s, :, :]
        elif state["axis"] == 1:
            return arr[:, s, :]
        else:
            return arr[:, :, s]


    img = ax.imshow(get_slice(), origin="lower", aspect="auto")
    cbar = plt.colorbar(img, ax=ax)

    def refresh():
        img.set_data(get_slice())

        if state["lock_scale"] and state["field"] in ("E", "D"):
            img.set_clim(0.0, 10.0)
        elif state["lock_scale"] and state["field"] == "R_boost":
            img.set_clim(0.0, 1.0)
        else:
            img.autoscale()

        ax.set_title(
            f"{state['field']} | {axis_names[state['axis']]}={state['slice']} "
            f"| tick={state['data']['tick']} | file={state['file_idx']:08d}"
        )
        fig.canvas.draw_idle()

    def scan_events(start, end):
        events = []
        prev_labels = None
        prev_n = 0

        for idx in range(start, end + 1, state["step"]):
            try:
                f = load_dump(dump_name(idx))
            except:
                break

            D = f["D"]
            P = f["P"]

            mask, _ = pid_mask_from_D(D, state["pid_q"])
            labels, nlab = label_components(mask)

            hoard = hoarder_score(D, P, state["pid_q"])

            if prev_labels is not None:
                delta = abs(nlab - prev_n)
                if delta > 0:
                    events.append((idx, "SPLIT/MERGE", delta))
                if hoard > 1.5:
                    events.append((idx, "HOARD_SPIKE", hoard))

            prev_labels = labels
            prev_n = nlab

        state["events"] = sorted(events, key=lambda x: -x[2])
        state["event_idx"] = 0
        print("Top events:")
        for e in state["events"][:10]:
            print(e)

    refresh()

    # =========================
    # EXPORT MODE
    # =========================

    if export_args is not None:
        out = Path(export_args["out"])
        out.parent.mkdir(parents=True, exist_ok=True)

        state["field"] = export_args["field"]
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

                # Drop alpha channel (RGBA → RGB)
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

    ax_pidq = plt.axes([0.3, 0.13, 0.6, 0.03])
    pid_slider = Slider(ax_pidq, "PID q", 0.05, 0.45, valinit=state["pid_q"], valstep=0.01)

    def on_pidq(val):
        state["pid_q"] = val
        refresh()

    pid_slider.on_changed(on_pidq)

    ax_field = plt.axes([0.05, 0.55, 0.2, 0.25])
    radio_field = RadioButtons(ax_field, fields)

    def on_field(label):
        state["field"] = label
        refresh()

    radio_field.on_clicked(on_field)

    ax_axis = plt.axes([0.05, 0.3, 0.2, 0.2])
    radio_axis = RadioButtons(ax_axis, axis_names)

    def on_axis(label):
        state["axis"] = axis_names.index(label)
        dims = [state["data"]["nz"], state["data"]["ny"], state["data"]["nx"]]
        slice_slider.valmax = dims[state["axis"]] - 1
        slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
        slice_slider.set_val(min(state["slice"], slice_slider.valmax))
        refresh()

    radio_axis.on_clicked(on_axis)

    ax_scale = plt.axes([0.05, 0.15, 0.2, 0.08])
    check_scale = CheckButtons(ax_scale, ["Lock scale (E,D)"], [True])

    def toggle_scale(label):
        state["lock_scale"] = not state["lock_scale"]
        refresh()

    check_scale.on_clicked(toggle_scale)

    ax_prev = plt.axes([0.3, 0.08, 0.1, 0.05])
    ax_next = plt.axes([0.42, 0.08, 0.1, 0.05])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")

    Trading = lambda d: lambda _: step_time(d)

    def step_time(direction):
        state["file_idx"] += direction * state["step"]
        load_current()
        on_axis(axis_names[state["axis"]])

    btn_prev.on_clicked(Trading(-1))
    btn_next.on_clicked(Trading(+1))

    ax_step = plt.axes([0.6, 0.08, 0.15, 0.05])
    step_box = TextBox(ax_step, "Δfile", initial=str(step))

    def set_step(text):
        try:
            state["step"] = max(1, int(text))
        except ValueError:
            pass

    step_box.on_submit(set_step)

    timer = fig.canvas.new_timer(interval=state["delay_ms"])

    def autoplay_step():
        if not state["playing"]:
            return
        try:
            state["file_idx"] += state["step"]
            load_current()
            on_axis(axis_names[state["axis"]])
        except FileNotFoundError:
            state["playing"] = False
            play_btn.label.set_text("Play")

    timer.add_callback(autoplay_step)

    ax_play = plt.axes([0.54, 0.08, 0.08, 0.05])
    play_btn = Button(ax_play, "Play")
    
    ax_scan = plt.axes([0.05, 0.05, 0.2, 0.05])
    scan_btn = Button(ax_scan, "Scan Events")

    def do_scan(_):
        scan_events(state["file_idx"], state["file_idx"] + 1000)

    scan_btn.on_clicked(do_scan)

    ax_jump = plt.axes([0.27, 0.05, 0.15, 0.05])
    jump_btn = Button(ax_jump, "Next Event")

    def jump_event(_):
        if not state["events"]:
            return
        tick, label, score = state["events"][state["event_idx"]]
        state["event_idx"] = (state["event_idx"] + 1) % len(state["events"])
        state["file_idx"] = tick
        load_current()
        refresh()

    jump_btn.on_clicked(jump_event)

    def toggle_play(_):
        state["playing"] = not state["playing"]
        play_btn.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            timer.start()
        else:
            timer.stop()

    play_btn.on_clicked(toggle_play)

    ax_speed = plt.axes([0.78, 0.08, 0.15, 0.05])
    speed_box = TextBox(ax_speed, "ms/frame", initial=str(state["delay_ms"]))

    def set_speed(text):
        try:
            state["delay_ms"] = max(10, int(text))
            timer.interval = state["delay_ms"]
        except ValueError:
            pass

    speed_box.on_submit(set_speed)

    plt.show()

# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=int, help="file step size")
    parser.add_argument("--export", nargs=2, metavar=("START", "END"), type=int)
    parser.add_argument("--field", default="D", choices=["E", "D", "P", "R_boost"])
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"])
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", default="vrd_export.mp4")

    args = parser.parse_args()

    export_cfg = None
    if args.export:
        export_cfg = {
            "start": args.export[0],
            "end": args.export[1],
            "field": args.field,
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

import struct
import numpy as np
import dearpygui.dearpygui as dpg

MAGIC = 0x564F5843
VERSION = 1
VIEW_W = 640
VIEW_H = 640

# ======================
# Viridis Colormap (your table)
# ======================
# View transform
view_x = 0.0   # top-left voxel coordinate
view_y = 0.0
view_scale = 1.0   # screen pixels per voxel

VIRIDIS = np.array([
    [0.267, 0.005, 0.329],
    [0.268, 0.010, 0.335],
    [0.269, 0.014, 0.341],
    [0.271, 0.019, 0.347],
    [0.272, 0.023, 0.353],
    [0.273, 0.028, 0.358],
    [0.274, 0.032, 0.364],
    [0.276, 0.037, 0.370],
    [0.277, 0.042, 0.376],
    [0.278, 0.046, 0.381],
    [0.280, 0.051, 0.387],
    [0.281, 0.056, 0.392],
    [0.282, 0.061, 0.398],
    [0.283, 0.065, 0.403],
    [0.285, 0.070, 0.409],
    [0.286, 0.075, 0.414],
    [0.287, 0.080, 0.420],
    [0.289, 0.084, 0.425],
    [0.290, 0.089, 0.431],
    [0.291, 0.094, 0.436],
    [0.292, 0.099, 0.442],
    [0.294, 0.103, 0.447],
    [0.295, 0.108, 0.453],
    [0.296, 0.113, 0.458],
    [0.298, 0.118, 0.464],
    [0.299, 0.122, 0.469],
    [0.300, 0.127, 0.475],
    [0.301, 0.132, 0.480],
    [0.303, 0.137, 0.486],
    [0.304, 0.141, 0.491],
    [0.305, 0.146, 0.497],
    [0.307, 0.151, 0.502],
    [0.308, 0.156, 0.507],
    [0.309, 0.160, 0.513],
    [0.311, 0.165, 0.518],
    [0.312, 0.170, 0.524],
    [0.313, 0.175, 0.529],
    [0.314, 0.180, 0.535],
    [0.316, 0.184, 0.540],
    [0.317, 0.189, 0.546],
    [0.318, 0.194, 0.551],
    [0.320, 0.199, 0.557],
    [0.321, 0.204, 0.562],
    [0.322, 0.208, 0.567],
    [0.324, 0.213, 0.573],
    [0.325, 0.218, 0.578],
    [0.326, 0.223, 0.584],
    [0.328, 0.228, 0.589],
    [0.329, 0.233, 0.594],
    [0.330, 0.237, 0.600],
    [0.332, 0.242, 0.605],
    [0.333, 0.247, 0.611],
    [0.334, 0.252, 0.616],
    [0.336, 0.257, 0.621],
    [0.337, 0.262, 0.627],
    [0.338, 0.267, 0.632],
    [0.340, 0.271, 0.638],
    [0.341, 0.276, 0.643],
    [0.342, 0.281, 0.648],
    [0.344, 0.286, 0.654],
    [0.345, 0.291, 0.659],
    [0.346, 0.296, 0.664],
    [0.348, 0.301, 0.670],
    [0.349, 0.306, 0.675],
    [0.350, 0.311, 0.680],
    [0.352, 0.316, 0.686],
    [0.353, 0.321, 0.691],
    [0.354, 0.326, 0.696],
    [0.356, 0.331, 0.702],
    [0.357, 0.336, 0.707],
    [0.358, 0.341, 0.712],
    [0.360, 0.346, 0.717],
    [0.361, 0.351, 0.723],
    [0.362, 0.356, 0.728],
    [0.364, 0.361, 0.733],
    [0.365, 0.366, 0.738],
    [0.366, 0.371, 0.744],
    [0.368, 0.376, 0.749],
    [0.369, 0.381, 0.754],
    [0.370, 0.386, 0.759],
    [0.372, 0.391, 0.765],
    [0.373, 0.396, 0.770],
    [0.374, 0.401, 0.775],
    [0.376, 0.406, 0.780],
    [0.377, 0.411, 0.786],
    [0.378, 0.416, 0.791],
    [0.380, 0.421, 0.796],
    [0.381, 0.426, 0.801],
    [0.382, 0.431, 0.807],
    [0.384, 0.436, 0.812],
    [0.385, 0.441, 0.817],
    [0.386, 0.446, 0.822],
    [0.388, 0.451, 0.828],
    [0.389, 0.456, 0.833],
    [0.390, 0.461, 0.838],
    [0.392, 0.466, 0.843],
    [0.393, 0.471, 0.849],
    [0.394, 0.476, 0.854],
    [0.396, 0.481, 0.859],
    [0.397, 0.486, 0.864],
    [0.398, 0.491, 0.870],
    [0.400, 0.496, 0.875],
    [0.401, 0.501, 0.880],
    [0.402, 0.506, 0.885],
    [0.404, 0.511, 0.891],
    [0.405, 0.516, 0.896],
    [0.406, 0.521, 0.901],
    [0.408, 0.526, 0.906],
    [0.409, 0.531, 0.912],
    [0.410, 0.536, 0.917],
    [0.412, 0.541, 0.922],
    [0.413, 0.546, 0.927],
    [0.414, 0.551, 0.933],
    [0.416, 0.556, 0.938],
    [0.417, 0.561, 0.943],
    [0.418, 0.566, 0.948],
    [0.420, 0.571, 0.954],
    [0.421, 0.576, 0.959],
    [0.422, 0.581, 0.964],
    [0.424, 0.586, 0.969],
    [0.425, 0.591, 0.975],
    [0.426, 0.596, 0.980],
    [0.428, 0.601, 0.985],
], dtype=np.float32)

# ======================
# Checkpoint I/O
# ======================

def read_vec(f, dtype):
    n = struct.unpack("<Q", f.read(8))[0]
    need = np.dtype(dtype).itemsize * n
    buf = f.read(need)
    if len(buf) != need:
        raise RuntimeError(f"Truncated checkpoint vector: expected {need} bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype=dtype).copy()

def write_vec(f, arr):
    f.write(struct.pack("<Q", arr.size))
    f.write(arr.tobytes())

class Checkpoint:
    def __init__(self, path):
        with open(path, "rb") as f:
            magic, ver = struct.unpack("<II", f.read(8))
            if magic != MAGIC or ver != VERSION:
                raise RuntimeError("Bad checkpoint (magic/version)")

            self.nx, self.ny, self.nz = struct.unpack("<iii", f.read(12))
            self.tick, self.seed = struct.unpack("<QQ", f.read(16))
            self.nvox = self.nx * self.ny * self.nz

            self.E = read_vec(f, np.float32)
            self.D = read_vec(f, np.float32)
            self.W = read_vec(f, np.float32)
            self.P = read_vec(f, np.float32)
            self.repair_elig = read_vec(f, np.float32)
            self.R_boost = read_vec(f, np.float32)

            self.have_prev = struct.unpack("<B", f.read(1))[0]
            self.Dm_prev = struct.unpack("<d", f.read(8))[0]
            self.D_prev = read_vec(f, np.float32)

            # sanity checks (catch mismatched dumps)
            if self.E.size != self.nvox: raise RuntimeError("E size mismatch")
            if self.D.size != self.nvox: raise RuntimeError("D size mismatch")
            if self.P.size != self.nvox: raise RuntimeError("P size mismatch")
            if self.R_boost.size != self.nvox: raise RuntimeError("R_boost size mismatch")
            if self.W.size != self.nvox * 6: raise RuntimeError("W size mismatch (expected nvox*6)")

    def save(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<II", MAGIC, VERSION))
            f.write(struct.pack("<iii", self.nx, self.ny, self.nz))
            f.write(struct.pack("<QQ", self.tick, self.seed))

            write_vec(f, self.E)
            write_vec(f, self.D)
            write_vec(f, self.W)
            write_vec(f, self.P)
            write_vec(f, self.repair_elig)
            write_vec(f, self.R_boost)

            f.write(struct.pack("<B", self.have_prev))
            f.write(struct.pack("<d", self.Dm_prev))
            write_vec(f, self.D_prev)

# ======================
# GUI STATE
# ======================

ckpt = None
z_slice = 0
current_field = "D"
brush_value = 0.0
brush_radius = 2

FIELDS = {
    "D": lambda c: c.D,
    "E": lambda c: c.E,
    "R_boost": lambda c: c.R_boost,
    "P": lambda c: c.P,
}

# texture bookkeeping
TEXTURE_TAG = "slice_tex_0"
texture_counter = 0

def new_texture_tag():
    global texture_counter
    texture_counter += 1
    return f"slice_tex_{texture_counter}"

def apply_viridis(norm_img):
    n = len(VIRIDIS) - 1
    idx = np.clip((norm_img * n).astype(np.int32), 0, n)
    return VIRIDIS[idx]

# ======================
# Rendering
# ======================

def render_slice():
    global ckpt, z_slice
    if ckpt is None:
        return

    field = FIELDS[current_field](ckpt)
    img = np.zeros((VIEW_H, VIEW_W), dtype=np.float32)

    stride = ckpt.nx * ckpt.ny
    zoff = z_slice * stride

    for sy in range(VIEW_H):
        for sx in range(VIEW_W):
            vx = int(view_x + sx / view_scale)
            vy = int(view_y + (VIEW_H - 1 - sy) / view_scale)

            if 0 <= vx < ckpt.nx and 0 <= vy < ckpt.ny:
                i = zoff + vy * ckpt.nx + vx
                img[sy, sx] = field[i]

    lo = float(np.min(img))
    hi = float(np.max(img))
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img.fill(0.0)

    rgb = apply_viridis(img)
    alpha = np.ones((VIEW_H, VIEW_W, 1), dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=2)

    dpg.set_value(TEXTURE_TAG, rgba.ravel().tolist())


# ======================
# Editing
# ======================

def paint_at(x, y):
    global ckpt, z_slice, brush_radius, brush_value, current_field
    if ckpt is None:
        return

    field = FIELDS[current_field](ckpt)

    # paint into the 1D array directly
    for dy in range(-brush_radius, brush_radius + 1):
        for dx in range(-brush_radius, brush_radius + 1):
            if dx*dx + dy*dy > brush_radius*brush_radius:
                continue
            px = x + dx
            py = y + dy
            if 0 <= px < ckpt.nx and 0 <= py < ckpt.ny:
                i = (z_slice * ckpt.nx * ckpt.ny) + (py * ckpt.nx + px)
                field[i] = brush_value

    render_slice()

# ======================
# Callbacks
# ======================

_last_hover = None
DEBUG = True

def dbg(*args):
    if DEBUG:
        print(*args)

def _paint_from_local(lx, ly):
    if ckpt is None:
        dbg("PAINT: no ckpt loaded")
        return

    w, h = dpg.get_item_rect_size("slice_image")
    dbg(f"PAINT: widget size = ({w:.2f}, {h:.2f})")

    if w <= 0 or h <= 0:
        dbg("PAINT: invalid widget size")
        return

    # Clamp
    lx0, ly0 = lx, ly
    lx = max(0, min(lx, w - 1))
    ly = max(0, min(ly, h - 1))

    u = lx / w
    v = ly / h

    x = int(u * ckpt.nx)
    y = int(v * ckpt.ny)

    x = max(0, min(x, ckpt.nx - 1))
    y = max(0, min(y, ckpt.ny - 1))

    dbg(f"PAINT: local=({lx0:.2f},{ly0:.2f}) clamped=({lx:.2f},{ly:.2f}) "
        f"uv=({u:.4f},{v:.4f}) voxel=({x},{y})")

    paint_at(x, y)

def mouse_paint():
    if ckpt is None:
        return

    mx, my = dpg.get_mouse_pos(local=False)
    rmin = dpg.get_item_rect_min("slice_image")
    rmax = dpg.get_item_rect_max("slice_image")

    if not (rmin[0] <= mx <= rmax[0] and
            rmin[1] <= my <= rmax[1]):
        return

    lx = mx - rmin[0]
    ly = my - rmin[1]

    vx = int(view_x + lx / view_scale)
    vy = int(view_y + (VIEW_H - 1 - ly) / view_scale)

    if 0 <= vx < ckpt.nx and 0 <= vy < ckpt.ny:
        paint_at(vx, vy)

def mouse_wheel(sender, app_data):
    global view_scale
    delta = app_data

    old = view_scale
    if delta > 0:
        view_scale *= 1.25
    else:
        view_scale /= 1.25

    view_scale = np.clip(view_scale, 0.25, 32.0)
    render_slice()

def on_img_hover(sender, app_data):
    global _last_hover

    if not ckpt:
        return

    dbg("HOVER app_data =", app_data)

    if isinstance(app_data, (tuple, list)) and len(app_data) >= 2:
        lx, ly = float(app_data[0]), float(app_data[1])
        _last_hover = (lx, ly)

        mx, my = dpg.get_mouse_pos(local=False)
        rmin = dpg.get_item_rect_min("slice_image")
        rmax = dpg.get_item_rect_max("slice_image")

        dbg(f"HOVER: global=({mx:.1f},{my:.1f}) "
            f"rect_min={rmin} rect_max={rmax} local=({lx:.2f},{ly:.2f})")

        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            dbg("HOVER: painting (LMB down)")
            _paint_from_local(lx, ly)


def on_img_click(sender, app_data):
    dbg("CLICK app_data =", app_data)

    if not ckpt:
        dbg("CLICK: no ckpt")
        return

    if _last_hover is None:
        dbg("CLICK: no hover history")
        return

    if app_data != dpg.mvMouseButton_Left:
        dbg("CLICK: not left button")
        return

    lx, ly = _last_hover
    dbg(f"CLICK: using last hover ({lx:.2f},{ly:.2f})")

    _paint_from_local(lx, ly)

last_pan = None

def mouse_pan(sender, app_data):
    global view_x, view_y, last_pan

    if not dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
        last_pan = None
        return

    mx, my = dpg.get_mouse_pos(local=False)

    if last_pan is None:
        last_pan = (mx, my)
        return

    dx = mx - last_pan[0]
    dy = my - last_pan[1]

    view_x -= dx / view_scale
    view_y += dy / view_scale

    view_x = np.clip(view_x, 0, ckpt.nx - 1)
    view_y = np.clip(view_y, 0, ckpt.ny - 1)

    last_pan = (mx, my)
    render_slice()

def load_file(sender, app_data):
    global ckpt, z_slice, TEXTURE_TAG, view_x, view_y, view_scale

    ckpt = Checkpoint(app_data["file_path_name"])
    z_slice = 0
    view_x = 0.0
    view_y = 0.0
    view_scale = 1.0

    # delete old texture item (NOT the registry)
    if dpg.does_item_exist(TEXTURE_TAG):
        dpg.delete_item(TEXTURE_TAG)

    # new unique tag
    TEXTURE_TAG = new_texture_tag()

    # create texture at FIXED viewport resolution
    dpg.add_dynamic_texture(
        VIEW_W,
        VIEW_H,
        [0.0, 0.0, 0.0, 1.0] * (VIEW_W * VIEW_H),
        tag=TEXTURE_TAG,
        parent="texreg"
    )

    # IMPORTANT: rebind image to the NEW texture tag
    dpg.configure_item("slice_image", texture_tag=TEXTURE_TAG)

    # z slider
    dpg.configure_item("z_slider", max_value=max(0, ckpt.nz - 1))
    dpg.set_value("z_slider", 0)

    view_x = 0
    view_y = 0
    view_scale = min(
        VIEW_W / ckpt.nx,
        VIEW_H / ckpt.ny
    )

    render_slice()

def save_file():
    if ckpt is None:
        return
    ckpt.save("edited_ckpt.bin")
    print("Saved edited_ckpt.bin")

def set_field(sender, app_data):
    global current_field
    current_field = app_data
    render_slice()

def set_z(sender, app_data):
    global z_slice
    z_slice = int(app_data)
    render_slice()

def set_brush_value(sender, app_data):
    global brush_value
    brush_value = float(app_data)

def set_brush_radius(sender, app_data):
    global brush_radius
    brush_radius = int(app_data)

# ======================
# GUI
# ======================

dpg.create_context()
dpg.create_viewport(title="VRD Checkpoint Editor", width=900, height=700)

# one stable registry forever
dpg.add_texture_registry(tag="texreg")

INIT_W = 640
INIT_H = 640

dpg.add_dynamic_texture(
    INIT_W,
    INIT_H,
    [0.0, 0.0, 0.0, 1.0] * (INIT_W * INIT_H),
    tag=TEXTURE_TAG,
    parent="texreg"
)

with dpg.window(label="Controls", width=250, height=700):
    dpg.add_button(label="Load Checkpoint", callback=lambda: dpg.show_item("file_dialog"))
    dpg.add_button(label="Save As edited_ckpt.bin", callback=save_file)

    dpg.add_separator()
    dpg.add_text("Field")
    dpg.add_radio_button(items=list(FIELDS.keys()), default_value="D", callback=set_field)

    dpg.add_separator()
    dpg.add_text("Z Slice")
    dpg.add_slider_int(tag="z_slider", min_value=0, max_value=0, callback=set_z)

    dpg.add_separator()
    dpg.add_text("Brush Value")
    dpg.add_slider_float(min_value=-10.0, max_value=50.0, default_value=0.0, callback=set_brush_value)

    dpg.add_text("Brush Radius")
    dpg.add_slider_int(min_value=1, max_value=20, default_value=2, callback=set_brush_radius)

with dpg.window(label="Slice View", pos=(260, 10), width=700, height=700):
    dpg.add_image(TEXTURE_TAG, width=640, height=640, tag="slice_image")

with dpg.handler_registry():
    dpg.add_mouse_click_handler(
        button=dpg.mvMouseButton_Left,
        callback=lambda s, a: mouse_paint()
    )
    dpg.add_mouse_drag_handler(
        button=dpg.mvMouseButton_Left,
        callback=lambda s, a: mouse_paint()
    )

with dpg.file_dialog(tag="file_dialog", show=False, callback=load_file):
    dpg.add_file_extension(".bin")
with dpg.handler_registry():
    dpg.add_mouse_wheel_handler(callback=mouse_wheel)
    dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=mouse_pan)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

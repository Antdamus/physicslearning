# plane_wave_2d_2modes_reim.py
# 2D/3D visualization of e^{i(kx x + ky y)} with:
#  - Field toggle: Re or Im (cos or sin)
#  - 2D colormap (stripes) OR 3D surface
#  - Build (reveal) along constant-phase fronts (⊥ k)
#  - Wavevector arrow k and constant-phase fronts/planes
#
# Controls:
#  - Sliders: kx, ky, speed
#  - Buttons: Play, Pause, Reset build, Mode (2D/3D), Field (Re/Im)
#
# Requirements: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------
# Domain/grid
# -------------------
L = 2*np.pi
N = 150
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")

# -------------------
# State
# -------------------
FPS = 60
DT  = 1.0 / FPS
state = dict(
    kx=2.0,
    ky=1.5,
    speed=1.0,      # build rate
    u=0.0,          # reveal progress [0,1]
    playing=True,
    mode="2D",      # "2D" or "3D"
    field="Re"      # "Re" or "Im"
)

# -------------------
# Core fields
# -------------------
def compute_phase(kx, ky):
    return kx*X + ky*Y

def phase_to_unit(phi):
    pmin, pmax = phi.min(), phi.max()
    return (phi - pmin) / max(pmax - pmin, 1e-12), pmin, pmax

def build_mask(u, tnorm, tau=0.02):
    # Smooth logistic front: 0 -> hidden, 1 -> revealed
    return 1.0 / (1.0 + np.exp((tnorm - u)/tau))

def field_from_phase(phi, which):
    return np.cos(phi) if which == "Re" else np.sin(phi)

def refresh_fields():
    global PHI, TNORM, PHI_MIN, PHI_MAX, ZFULL
    PHI = compute_phase(state["kx"], state["ky"])
    TNORM, PHI_MIN, PHI_MAX = phase_to_unit(PHI)
    ZFULL = field_from_phase(PHI, state["field"])

refresh_fields()

# -------------------
# Figure layout
# -------------------
plt.close("all")
fig = plt.figure(figsize=(12, 6))
gs  = fig.add_gridspec(3, 7, height_ratios=[6, 0.9, 0.9],
                       width_ratios=[1,1,1,1,1,1,1], hspace=0.6, wspace=0.6)

ax_main = fig.add_subplot(gs[0, 0:4])      # 2D or 3D view goes here
ax_side = fig.add_subplot(gs[0, 4:7])      # k-panel
ax_side.set_aspect("equal", adjustable="box")
ax_side.axhline(0, linewidth=1); ax_side.axvline(0, linewidth=1)
ax_side.set_xlim(-4, 4); ax_side.set_ylim(-4, 4)
ax_side.set_title("k-vector & directions")

# Controls
ax_kx   = fig.add_subplot(gs[1, 0:2]); slider_kx = Slider(ax=ax_kx, label="k_x", valmin=-6.0, valmax=6.0, valinit=state["kx"])
ax_ky   = fig.add_subplot(gs[1, 2:4]); slider_ky = Slider(ax=ax_ky, label="k_y", valmin=-6.0, valmax=6.0, valinit=state["ky"])
ax_spd  = fig.add_subplot(gs[1, 4:7]); slider_sp = Slider(ax=ax_spd, label="speed", valmin=0.0, valmax=4.0, valinit=state["speed"])

ax_play  = fig.add_subplot(gs[2, 0]); btn_play  = Button(ax_play,  "Play")
ax_pause = fig.add_subplot(gs[2, 1]); btn_pause = Button(ax_pause, "Pause")
ax_reset = fig.add_subplot(gs[2, 2]); btn_reset = Button(ax_reset, "Reset build")
ax_mode  = fig.add_subplot(gs[2, 3]); btn_mode  = Button(ax_mode,  "Mode: 2D/3D")
ax_field = fig.add_subplot(gs[2, 4]); btn_field = Button(ax_field, "Field: Re/Im")

# -------------------
# k panel artists
# -------------------
k_arrow_side, = ax_side.plot([], [], linewidth=2, color='k')
u_line_side,  = ax_side.plot([], [], linestyle="--", linewidth=1, label="‖ k (propagation)", color='tab:blue')
n_line_side,  = ax_side.plot([], [], linestyle=":",  linewidth=1, label="⊥ k (fronts)",     color='tab:orange')
ax_side.legend(loc="lower right", fontsize=8)

def update_k_panel():
    kx, ky = state["kx"], state["ky"]
    k_norm = np.hypot(kx, ky)
    k_arrow_side.set_data([0, kx], [0, ky])
    if k_norm < 1e-9:
        u = np.array([1.0, 0.0])
        n = np.array([0.0, 1.0])
    else:
        u = np.array([kx, ky]) / k_norm
        n = np.array([-ky, kx]) / k_norm
    u_line_side.set_data([-2*u[0], 2*u[0]], [-2*u[1], 2*u[1]])
    n_line_side.set_data([-2*n[0], 2*n[0]], [-2*n[1], 2*n[1]])

# -------------------
# Constant-phase levels (φ = multiples of 2π)
# -------------------
def phase_levels(phi_min, phi_max, spacing=2*np.pi, max_lines=9):
    n0 = int(np.ceil(phi_min/spacing))
    n1 = int(np.floor(phi_max/spacing))
    levels = (np.arange(n0, n1+1) * spacing).astype(float)
    if len(levels) > max_lines and max_lines > 0:
        step = int(np.ceil(len(levels)/max_lines))
        levels = levels[::step]
    return levels

# -------------------
# 2D mode artists
# -------------------
img2d = None
front_lines_2d = []
k_quiver2d = None
lambda_text2d = None

def setup_2d():
    global img2d, front_lines_2d, k_quiver2d, lambda_text2d
    ax_main.clear()
    title_txt = f"2D: {state['field']}" + "{e^{i(k·r)}} — building along constant phase"
    ax_main.set_title(title_txt)
    ax_main.set_xlabel("x"); ax_main.set_ylabel("y")
    img2d = ax_main.imshow(np.zeros_like(ZFULL), origin="lower",
                           extent=[x.min(), x.max(), y.min(), y.max()],
                           cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
    front_lines_2d = []
    k_quiver2d = None
    lambda_text2d = ax_main.text(0.02, 0.98, "", transform=ax_main.transAxes,
                                 va="top", color="w",
                                 bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.3"))

def render_2d():
    global front_lines_2d, k_quiver2d
    # update title (Re/Im)
    title_txt = f"2D: {state['field']}" + "{e^{i(k·r)}} — building along constant phase"
    ax_main.set_title(title_txt)

    # reveal mask and image
    M = build_mask(state["u"], TNORM, tau=0.02)
    img2d.set_data(ZFULL * M)

    # Remove previous quiver arrow
    if k_quiver2d is not None:
        k_quiver2d.remove()
        k_quiver2d = None

    # k arrow on 2D
    kx, ky = state["kx"], state["ky"]
    k_norm = np.hypot(kx, ky)
    if k_norm < 1e-9:
        u = np.array([1.0, 0.0]); n = np.array([0.0, 1.0])
    else:
        u = np.array([kx, ky]) / k_norm
        n = np.array([-ky, kx]) / k_norm
    k_quiver2d = ax_main.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy',
                                scale=1.5, width=0.005, color='w')

    # Remove old front lines
    for ln in front_lines_2d:
        ln.remove()
    front_lines_2d.clear()

    # Draw constant-phase fronts (lines ⟂ k)
    if k_norm >= 1e-9:
        levels = phase_levels(PHI_MIN, PHI_MAX)
        Lseg = L
        for c in levels:
            r0 = (c / k_norm) * u
            p1 = r0 - (Lseg/2) * n
            p2 = r0 + (Lseg/2) * n
            ln, = ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], color='w', alpha=0.5, linewidth=1)
            front_lines_2d.append(ln)

    # wavelength label
    lam_text = "λ = ∞" if k_norm < 1e-9 else f"λ = {2*np.pi/k_norm:.3f}"
    # Find or create text; here we assume the last added text is lambda label:
    # Safer: if no texts, add one; else update first text.
    if not ax_main.texts:
        ax_main.text(0.02, 0.98, lam_text, transform=ax_main.transAxes,
                     va="top", color="w",
                     bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.3"))
    else:
        ax_main.texts[0].set_text(lam_text)

# -------------------
# 3D mode artists
# -------------------
ax3d = None
surf3d = None
front_planes_3d = []
k_line3d = None
lambda_text3d = None

def setup_3d():
    global ax3d, surf3d, front_planes_3d, k_line3d, lambda_text3d
    ax_main.clear()
    ax3d = fig.add_subplot(gs[0, 0:4], projection='3d')
    title_txt = "3D: z = " + f"{state['field']}" + "{e^{i(k·r)}} — building along constant phase"
    ax3d.set_title(title_txt)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_xlim(x.min(), x.max()); ax3d.set_ylim(y.min(), y.max()); ax3d.set_zlim(-1.1, 1.1)
    Z0 = np.full_like(ZFULL, np.nan)
    surf3d = ax3d.plot_surface(X, Y, Z0, linewidth=0, antialiased=True, cmap="viridis")
    front_planes_3d = []
    k_line3d = None
    lambda_text3d = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes)

def render_3d():
    global surf3d, front_planes_3d, k_line3d, lambda_text3d
    # title (Re/Im)
    title_txt = "3D: z = " + f"{state['field']}" + "{e^{i(k·r)}} — building along constant phase"
    ax3d.set_title(title_txt)

    # update surface with reveal mask
    M = build_mask(state["u"], TNORM, tau=0.02)
    Z = ZFULL * M
    if surf3d is not None:
        try: surf3d.remove()
        except Exception: pass
    surf3d = ax3d.plot_surface(X, Y, Z, linewidth=0, antialiased=True, cmap="viridis")

    # Draw k arrow on z=0 plane
    if k_line3d is not None:
        try: k_line3d.remove()
        except Exception: pass
        k_line3d = None
    kx, ky = state["kx"], state["ky"]
    k_line3d = ax3d.plot([0, kx], [0, ky], [0, 0], linewidth=3, color='k')[0]

    # Remove old planes
    for poly in front_planes_3d:
        try: poly.remove()
        except Exception: pass
    front_planes_3d.clear()

    # Draw constant-phase planes (vertical sheets ⟂ k)
    k_norm = np.hypot(kx, ky)
    if k_norm >= 1e-9:
        u = np.array([kx, ky]) / k_norm
        n = np.array([-ky, kx]) / k_norm
        levels = phase_levels(PHI_MIN, PHI_MAX)
        Lseg = L
        zmin, zmax = -1.1, 1.1
        for c in levels:
            r0 = (c / k_norm) * u
            p1 = r0 - (Lseg/2)*n
            p2 = r0 + (Lseg/2)*n
            verts = [[(p1[0], p1[1], zmin),
                      (p2[0], p2[1], zmin),
                      (p2[0], p2[1], zmax),
                      (p1[0], p1[1], zmax)]]
            poly = Poly3DCollection(verts, alpha=0.15, facecolor='w', edgecolor='none')
            ax3d.add_collection3d(poly)
            front_planes_3d.append(poly)

    # wavelength label in 3D
    lam_text = "λ = ∞" if k_norm < 1e-9 else f"λ = {2*np.pi/k_norm:.3f}"
    if lambda_text3d is None:
        lambda_text3d = ax3d.text2D(0.02, 0.96, lam_text, transform=ax3d.transAxes)
    else:
        lambda_text3d.set_text(lam_text)

# -------------------
# Toggle & rendering
# -------------------
def switch_mode():
    if state["mode"] == "2D":
        setup_2d()
    else:
        setup_3d()

def render():
    if state["mode"] == "2D":
        render_2d()
    else:
        render_3d()
    update_k_panel()

# -------------------
# Callbacks
# -------------------
def on_kx(val):
    state["kx"] = float(val)
    refresh_fields()
def on_ky(val):
    state["ky"] = float(val)
    refresh_fields()
def on_sp(val):
    state["speed"] = float(val)
def on_play(evt):  state["playing"] = True
def on_pause(evt): state["playing"] = False
def on_reset(evt): state["u"] = 0.0

def on_mode(evt):
    state["mode"] = "3D" if state["mode"] == "2D" else "2D"
    switch_mode()

def on_field(evt):
    state["field"] = "Im" if state["field"] == "Re" else "Re"
    refresh_fields()
    switch_mode()  # rebuild titles/axes for new field

slider_kx.on_changed(on_kx)
slider_ky.on_changed(on_ky)
slider_sp.on_changed(on_sp)
btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)
btn_mode.on_clicked(on_mode)
btn_field.on_clicked(on_field)

# -------------------
# Animation
# -------------------
def init():
    switch_mode()
    update_k_panel()
    render()
    return []

def update(frame):
    if state["playing"]:
        state["u"] = (state["u"] + 0.003 * state["speed"]) % 1.0
    render()
    return []

ani = FuncAnimation(fig, update, init_func=init,
                    frames=10**9, interval=1000/FPS, blit=False, repeat=True)

plt.tight_layout()
plt.show()

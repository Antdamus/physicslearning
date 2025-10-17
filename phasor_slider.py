# phasor_sliders.py
# Two synchronized graphs + live sliders for k, amplitude R, phase phi, and speed.
# Dependencies: numpy, matplotlib
# Run: python phasor_sliders.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# -------- Parameters you can tweak --------
FPS       = 60          # visual smoothness
DURATION  = 9999        # run "forever" until closed
T_WINDOW  = 6.0         # seconds shown in the waveform panel
INIT_K    = 2.0         # initial frequency
INIT_R    = 1.0         # initial radius
INIT_PHI  = 0.0         # initial phase (rad)
INIT_SPD  = 1.0         # initial speed multiplier (playback rate)
RADIUS_CIRCLE = 1.0     # reference unit circle radius (left panel)

# -------- Time base --------
N_FRAMES = int(FPS * DURATION)
dt       = 1.0 / FPS

# -------- Figure + layout --------
plt.close("all")
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(3, 4, height_ratios=[5, 0.5, 0.5], width_ratios=[1, 1, 1, 1])

ax_plane   = fig.add_subplot(gs[0, 0:2])   # complex plane (left)
ax_wave    = fig.add_subplot(gs[0, 2:4])   # waveform (right)

# Slider axes (bottom rows)
ax_k    = fig.add_subplot(gs[1, 0])
ax_R    = fig.add_subplot(gs[1, 1])
ax_phi  = fig.add_subplot(gs[1, 2])
ax_spd  = fig.add_subplot(gs[1, 3])

ax_play = fig.add_subplot(gs[2, 0])
ax_stop = fig.add_subplot(gs[2, 1])
ax_rst  = fig.add_subplot(gs[2, 2])

# -------- Complex plane init --------
ax_plane.set_aspect("equal", adjustable="box")
lim = 1.35 * max(RADIUS_CIRCLE, INIT_R)
ax_plane.set_xlim(-lim, lim)
ax_plane.set_ylim(-lim, lim)
ax_plane.axhline(0, linewidth=1)
ax_plane.axvline(0, linewidth=1)

theta = np.linspace(0, 2*np.pi, 400)
circle_line, = ax_plane.plot(RADIUS_CIRCLE*np.cos(theta), RADIUS_CIRCLE*np.sin(theta), linewidth=1)

phasor_line, = ax_plane.plot([], [], linewidth=2)      # arrow
tip_point,   = ax_plane.plot([], [], "o")              # tip
proj_x_line, = ax_plane.plot([], [], linestyle="--", linewidth=1)
proj_y_line, = ax_plane.plot([], [], linestyle="--", linewidth=1)
plane_text   = ax_plane.text(-lim+0.05, lim-0.1, "", fontsize=10, va="top")

# -------- Waveform init --------
t_plot = np.linspace(0.0, T_WINDOW, int(FPS*T_WINDOW))
wave_line,  = ax_wave.plot(t_plot, np.zeros_like(t_plot), linewidth=2, label="Re[R e^{i(kt+φ)}]")
mag_line,   = ax_wave.plot(t_plot, np.zeros_like(t_plot), linestyle="--", label="|R e^{i(kt+φ)}|")
marker,     = ax_wave.plot([0], [0], "o")
ax_wave.axhline(0, linewidth=1)
ax_wave.set_xlim(0, T_WINDOW)
ax_wave.set_ylim(-2.0, 2.0)
ax_wave.set_xlabel("t (s)")
ax_wave.set_title("Real projection vs time")
ax_wave.legend(loc="upper right")

# -------- Sliders --------
slider_k   = Slider(ax=ax_k,   label="k (frequency)", valmin=-10.0, valmax=10.0, valinit=INIT_K)
slider_R   = Slider(ax=ax_R,   label="R (radius)",    valmin=0.0,  valmax=2.0,  valinit=INIT_R)
slider_phi = Slider(ax=ax_phi, label="φ (phase rad)", valmin=-np.pi, valmax=np.pi, valinit=INIT_PHI)
slider_spd = Slider(ax=ax_spd, label="speed",         valmin=0.0,  valmax=4.0,   valinit=INIT_SPD)

# -------- Buttons --------
btn_play = Button(ax_play, "Play")
btn_stop = Button(ax_stop, "Pause")
btn_rst  = Button(ax_rst,  "Reset")

# -------- State --------
state = {
    "t": 0.0,
    "playing": True,
}

def update_axes_limits_for_R(R):
    # Keep complex plane nicely framed if R changes a lot
    L = 1.35 * max(RADIUS_CIRCLE, R)
    ax_plane.set_xlim(-L, L)
    ax_plane.set_ylim(-L, L)
    plane_text.set_position((-L+0.05, L-0.1))

def update_frame(frame):
    # Read slider values live
    k   = float(slider_k.val)
    R   = float(slider_R.val)
    phi = float(slider_phi.val)
    spd = float(slider_spd.val)

    # Advance time if playing
    if state["playing"]:
        state["t"] += dt * spd

    t = state["t"]

    # --- Complex plane phasor ---
    angle = k * t + phi
    x = R * np.cos(angle)
    y = R * np.sin(angle)

    phasor_line.set_data([0, x], [0, y])
    tip_point.set_data([x], [y])
    proj_x_line.set_data([0, x], [0, 0])
    proj_y_line.set_data([x, x], [0, y])
    plane_text.set_text(f"t={t:5.2f}s\nangle={angle:6.2f} rad\nRe={x:5.2f}  Im={y:5.2f}")
    update_axes_limits_for_R(R)

    # --- Waveform (real part and magnitude) over time window ---
    # Align the window to end at current t for a "live" feel
    t0 = max(0.0, t - T_WINDOW)
    t_view = t0 + t_plot

    real_wave = R * np.cos(k * t_view + phi)
    wave_line.set_data(t_plot, real_wave)
    mag_line.set_data(t_plot, np.full_like(t_plot, R))

    # Move the marker to current time end
    current_real = R * np.cos(k * t + phi)
    marker.set_data([t - t0], [current_real])

    # Auto-scale y subtly to fit R range (clamped)
    y_lim = max(1.5, R*1.2)
    ax_wave.set_ylim(-y_lim, y_lim)

    return (phasor_line, tip_point, proj_x_line, proj_y_line,
            wave_line, mag_line, marker, circle_line, plane_text)

def on_play(event):
    state["playing"] = True

def on_stop(event):
    state["playing"] = False

def on_reset(event):
    state["t"] = 0.0

btn_play.on_clicked(on_play)
btn_stop.on_clicked(on_stop)
btn_rst.on_clicked(on_reset)

ani = FuncAnimation(fig, update_frame, frames=N_FRAMES, interval=1000/FPS, blit=False)
plt.tight_layout()
plt.show()

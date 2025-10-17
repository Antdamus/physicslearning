# forward_fourier_educational.py
# Geometric, windowed forward Fourier transform (single k) with didactic overlays.
# Shows: f(x), Re[f(x) e^{-ikx}] (the true integrand), window w(x),
# instantaneous phasor f(x) e^{-ikx}, and the running accumulation ∫ f w e^{-ikx} dx.
#
# Key ideas:
# - Top: f(x) does NOT depend on k, but the integrand f(x)e^{-ikx} DOES (so we plot both).
# - Windowing (w) stabilizes the accumulation (reduces spectral leakage).
# - Bottom-right vector sum approaches F(k) (over the finite window) as x sweeps the domain.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons

# ------------------------------
# Parameters
# ------------------------------
FPS        = 60            # visual smoothness
X_WINDOW   = 6*np.pi       # x-range length
N_SAMPLES  = 1600          # resolution in x
INIT_K     = 1.0           # starting detector frequency
INIT_SPEED = 1.0           # playback rate
DT         = 1.0 / FPS

x_vals = np.linspace(0.0, X_WINDOW, N_SAMPLES)
dx     = x_vals[1] - x_vals[0]

# ------------------------------
# Signals library f(x)
# (complex-valued OK; real signals are a special case)
# ------------------------------
def f_sum_of_cos(x):
    amps   = [1.0, 0.7, 0.5]
    omegas = [1.0, 2.0, 3.0]    # rad/unit
    phases = [np.pi/6, np.pi/3, -np.pi/4]
    out = np.zeros_like(x, dtype=complex)
    for A, w, ph in zip(amps, omegas, phases):
        out += A * np.exp(1j*(w*x + ph))  # A*cos(wx+ph) + iA*sin(...)
    return out

def f_gaussian(x):
    mu  = 0.5 * X_WINDOW
    sig = 0.12 * X_WINDOW
    g = np.exp(-0.5*((x - mu)/sig)**2)
    return g.astype(complex)

def f_square(x):
    # Period ~ 2π: sign(sin(x))
    return np.sign(np.sin(x)) + 0j

def f_chirp(x):
    a = 0.12           # chirp rate
    return np.exp(1j*(a * x**2))

SIGNALS = {
    "sum of cos": f_sum_of_cos,
    "gaussian":   f_gaussian,
    "square":     f_square,
    "chirp":      f_chirp,
}

# ------------------------------
# Windows w(x)
# ------------------------------
def window_rect(x):
    return np.ones_like(x)

def window_hann(x):
    # Hann over full span [0, X_WINDOW]
    n = len(x)
    return 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/ (n-1))

def window_gauss(x, width_frac=0.3):
    # Gaussian centered in the middle; width_frac controls sigma as fraction of span
    mu = 0.5 * X_WINDOW
    sigma = width_frac * X_WINDOW
    return np.exp(-0.5*((x - mu)/sigma)**2)

WINDOWS = {
    "rect":   ("Rectangular", window_rect),
    "hann":   ("Hann",        window_hann),
    "gauss":  ("Gaussian",    window_gauss),
}

# ------------------------------
# Build UI
# ------------------------------
plt.close("all")
fig = plt.figure(figsize=(12, 7.5))
gs  = fig.add_gridspec(4, 4, height_ratios=[4.2, 0.9, 0.9, 0.9], hspace=0.75)

ax_top     = fig.add_subplot(gs[0, 0:4])   # f(x), integrand, window
ax_inst    = fig.add_subplot(gs[1, 0:2])   # instantaneous phasor
ax_accum   = fig.add_subplot(gs[1, 2:4])   # accumulation path

# Controls
ax_k       = fig.add_subplot(gs[2, 0])
ax_speed   = fig.add_subplot(gs[2, 1])
ax_play    = fig.add_subplot(gs[2, 2])
ax_pause   = fig.add_subplot(gs[2, 3])

rax_sig    = plt.axes([0.80, 0.66, 0.17, 0.22])   # signal chooser
rax_win    = plt.axes([0.80, 0.44, 0.17, 0.20])   # window chooser
ax_gw      = plt.axes([0.80, 0.36, 0.17, 0.04])   # gaussian width slider

radio_sig = RadioButtons(rax_sig, list(SIGNALS.keys()), active=0)
radio_sig.ax.set_title("Signal f(x)")
radio_win = RadioButtons(rax_win, ["rect", "hann", "gauss"], active=1)
radio_win.ax.set_title("Window w(x)")
slider_gw = Slider(ax=ax_gw, label="gauss σ frac", valmin=0.05, valmax=0.6, valinit=0.3)

slider_k     = Slider(ax=ax_k,     label="k (freq)", valmin=-8.0, valmax=8.0,  valinit=INIT_K)
slider_speed = Slider(ax=ax_speed, label="speed",    valmin=0.0,  valmax=4.0,  valinit=INIT_SPEED)
btn_play     = Button(ax_play,  "Play")
btn_pause    = Button(ax_pause, "Pause")

# ------------------------------
# State + helpers
# ------------------------------
state = {
    "playing": True,
    "t": 0.0,
    "f_name": list(SIGNALS.keys())[0],
    "w_name": "hann",
    "gauss_width": slider_gw.val,
}

def compute_signal(name):
    return SIGNALS[name](x_vals)

def compute_window(wname, width_frac):
    if wname == "gauss":
        return WINDOWS[wname][1](x_vals, width_frac)
    else:
        return WINDOWS[wname][1](x_vals)

f_vals = compute_signal(state["f_name"])
w_vals = compute_window(state["w_name"], state["gauss_width"])

# ------------------------------
# Top panel setup
# ------------------------------
line_f,      = ax_top.plot(x_vals, f_vals.real, linewidth=2, label="Re[f(x)]")
line_int,    = ax_top.plot(x_vals, np.zeros_like(x_vals), linestyle="--", label="Re[f(x) e^{-ikx}]")
line_win,    = ax_top.plot(x_vals, w_vals * 0.9*np.max([1, np.max(np.abs(f_vals.real))+1e-9]),
                           linewidth=1, label="window w(x)")
marker_top,  = ax_top.plot([x_vals[0]], [f_vals.real[0]], "o")

ax_top.axhline(0, linewidth=1)
ax_top.set_xlim(x_vals[0], x_vals[-1])
ymax = 1.4 * max(1.0, np.max(np.abs(f_vals.real)))
ax_top.set_ylim(-ymax, ymax)
ax_top.set_title("Signal f(x), integrand Re[f(x) e^{-ikx}], and window w(x)")
ax_top.set_xlabel("x")
ax_top.legend(loc="upper right")

# ------------------------------
# Instantaneous phasor panel
# ------------------------------
inst_line, = ax_inst.plot([], [], linewidth=2)
inst_tip,  = ax_inst.plot([], [], "o")
ax_inst.axhline(0, linewidth=1)
ax_inst.axvline(0, linewidth=1)
ax_inst.set_aspect("equal", adjustable="box")
ax_inst.set_xlim(-2.2, 2.2)
ax_inst.set_ylim(-2.2, 2.2)
ax_inst.set_title("Instantaneous phasor  f(x)·w(x)·e^{-ikx}")
ax_inst.set_xlabel("Real")
ax_inst.set_ylabel("Imag")

# ------------------------------
# Accumulation panel
# ------------------------------
acc_line, = ax_accum.plot([], [], linewidth=2)
acc_tip,  = ax_accum.plot([], [], "o")
ax_accum.axhline(0, linewidth=1)
ax_accum.axvline(0, linewidth=1)
ax_accum.set_aspect("equal", adjustable="box")
ax_accum.set_xlim(-3.0, 3.0)
ax_accum.set_ylim(-3.0, 3.0)
acc_text = ax_accum.text(0.02, 0.98, "", transform=ax_accum.transAxes, va="top")
ax_accum.set_title("Accumulated projection  S(x)=∫ f(t)w(t)e^{-ikt} dt  (running)")
ax_accum.set_xlabel("Real")
ax_accum.set_ylabel("Imag")

# ------------------------------
# Update helpers
# ------------------------------
def update_signal(sig_name):
    global f_vals
    f_vals = compute_signal(sig_name)
    line_f.set_data(x_vals, f_vals.real)
    # rescale y if needed
    ymax = 1.4 * max(1.0, np.max(np.abs(f_vals.real)))
    ax_top.set_ylim(-ymax, ymax)

def update_window(win_name, gw):
    global w_vals
    w_vals = compute_window(win_name, gw)
    # show window as a thin guide scaled to the visible range
    scale = 0.9 * max(1.0, np.max(np.abs(f_vals.real)))
    line_win.set_data(x_vals, w_vals * scale)

def running_integral(idx_end, k):
    z = f_vals[:idx_end+1] * w_vals[:idx_end+1] * np.exp(-1j * k * x_vals[:idx_end+1])
    return np.cumsum(z) * dx

# ------------------------------
# Callbacks
# ------------------------------
def on_sig(label):
    state["f_name"] = label
    update_signal(label)
radio_sig.on_clicked(on_sig)

def on_win(label):
    state["w_name"] = label
    update_window(state["w_name"], state["gauss_width"])
radio_win.on_clicked(on_win)

def on_gw(val):
    state["gauss_width"] = val
    if state["w_name"] == "gauss":
        update_window("gauss", state["gauss_width"])
slider_gw.on_changed(on_gw)

def on_play(evt):
    state["playing"] = True
btn_play.on_clicked(on_play)

def on_pause(evt):
    state["playing"] = False
btn_pause.on_clicked(on_pause)

# ------------------------------
# Animation loop
# ------------------------------
def update(frame):
    if state["playing"]:
        state["t"] += DT * float(slider_speed.val)

    # Map time → a scanning position x_now over [0, X_WINDOW] (wrap)
    x_now = (state["t"] / (6.0)) * X_WINDOW      # 6.0 controls scan speed feel
    x_now = x_now % X_WINDOW
    idx = int(np.clip(np.searchsorted(x_vals, x_now), 0, N_SAMPLES-1))

    k = float(slider_k.val)

    # ---- Top panel: integrand changes with k ----
    integrand_re = np.real(f_vals * np.exp(-1j * k * x_vals))
    line_int.set_data(x_vals, integrand_re)
    marker_top.set_data([x_vals[idx]], [f_vals.real[idx]])

    # ---- Instantaneous phasor (with window) ----
    inst = f_vals[idx] * w_vals[idx] * np.exp(-1j * k * x_vals[idx])
    inst_line.set_data([0, inst.real], [0, inst.imag])
    inst_tip.set_data([inst.real], [inst.imag])

    # ---- Accumulation up to current index ----
    acc = running_integral(idx, k)
    acc_line.set_data(acc.real, acc.imag)
    acc_tip.set_data([acc[-1].real], [acc[-1].imag])
    acc_text.set_text(f"Partial F(k) at x={x_vals[idx]:.2f}\n≈ {acc[-1].real:.3f} + i{acc[-1].imag:.3f}\n|·|={np.abs(acc[-1]):.3f}")

    return (line_int, marker_top, inst_line, inst_tip, acc_line, acc_tip, acc_text)

ani = FuncAnimation(fig, update, frames=int(FPS*9999), interval=1000/FPS, blit=False, repeat=True)
plt.tight_layout()
plt.show()

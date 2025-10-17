# phasor_lab.py
# Animate rotating phasors e^{i k x} on the complex plane for multiple k.
# Run: python phasor_lab.py
# Optional: tweak K_VALUES, FPS, and DURATION.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---- Settings ----
K_VALUES   = [-3, -1, 1, 2, 20]   # frequencies to show (speed ∝ |k|)
FPS        = 1000                   # frames per second
DURATION   = 6.0                  # seconds of animation
RADIUS     = 1.0                  # phasor radius (|e^{ikx}| = 1)
SHOW_GUIDES = True                # draw projections and axes

# Time base: x plays the role of time parameter
N_FRAMES = int(FPS * DURATION)
x_vals   = np.linspace(0, DURATION, N_FRAMES, endpoint=False)

# ---- Figure layout ----
n = len(K_VALUES)
cols = min(3, n)
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
if n == 1:
    axes = np.array([[axes]])

axes = axes.reshape(rows, cols)

# Prepare per-axes artists we’ll update each frame
artists = []
for idx, k in enumerate(K_VALUES):
    r, c = divmod(idx, cols)
    ax = axes[r, c]
    ax.set_aspect("equal", adjustable="box")
    lim = 1.25 * RADIUS
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if SHOW_GUIDES:
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 400)
    (circle_line,) = ax.plot(np.cos(theta)*RADIUS, np.sin(theta)*RADIUS, linewidth=1)

    # Phasor line and tip
    (phasor_line,) = ax.plot([], [], linewidth=2)        # from (0,0) to tip
    (tip_point,)   = ax.plot([], [], marker="o")         # tip marker

    # Projections
    if SHOW_GUIDES:
        (proj_x_line,) = ax.plot([], [], linestyle="--", linewidth=1)
        (proj_y_line,) = ax.plot([], [], linestyle="--", linewidth=1)
    else:
        proj_x_line = proj_y_line = None

    title = ax.set_title(f"k = {k}")
    text  = ax.text(-lim + 0.05, lim - 0.1, "", fontsize=10, va="top")

    artists.append({
        "ax": ax,
        "k": k,
        "phasor_line": phasor_line,
        "tip_point": tip_point,
        "proj_x_line": proj_x_line,
        "proj_y_line": proj_y_line,
        "text": text,
    })

# Hide any extra axes (if grid has more cells than K_VALUES)
for idx in range(len(K_VALUES), rows*cols):
    r, c = divmod(idx, cols)
    axes[r, c].axis("off")

fig.tight_layout()

def update(frame):
    x = x_vals[frame]
    for a in artists:
        k = a["k"]
        angle = k * x   # angle = k * x (speed scales with k)
        tip_x = RADIUS * np.cos(angle)
        tip_y = RADIUS * np.sin(angle)

        # Update phasor line and tip
        a["phasor_line"].set_data([0, tip_x], [0, tip_y])
        a["tip_point"].set_data([tip_x], [tip_y])

        # Update projections
        if SHOW_GUIDES:
            a["proj_x_line"].set_data([0, tip_x], [0, 0])
            a["proj_y_line"].set_data([tip_x, tip_x], [0, tip_y])

        # Update info text
        a["text"].set_text(f"angle = {angle:.2f} rad\nRe = {tip_x:.2f}, Im = {tip_y:.2f}")
    return []

ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS, blit=False, repeat=True)

# ---- To save (optional): uncomment one of these lines ----
# ani.save("phasor_lab.gif", writer="pillow", fps=FPS)
# ani.save("phasor_lab.mp4", writer="ffmpeg", fps=FPS)

plt.show()

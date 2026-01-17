from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def save_interception_gif(log: Dict[str, Any], out_path: Path, *, R_capture: float, stride: int = 2, fps: int = 20) -> None:
    """Save a clean GIF animation (no ffmpeg required).

    Fixes the common matplotlib error: Line2D.set_data expects sequences.
    """
    p_m = np.asarray(log["p_m"], dtype=float)
    p_t = np.asarray(log["p_t"], dtype=float)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Downsample for file size
    idx = np.arange(0, len(p_m), int(stride))
    p_m = p_m[idx]
    p_t = p_t[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Bounds
    all_xy = np.vstack([p_m, p_t])
    xmin, ymin = np.min(all_xy, axis=0) - 10
    xmax, ymax = np.max(all_xy, axis=0) + 10
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    missile_line, = ax.plot([], [], lw=2, label="missile")
    target_line, = ax.plot([], [], lw=2, label="target")
    missile_pt, = ax.plot([], [], marker="o", markersize=7)
    target_pt, = ax.plot([], [], marker="x", markersize=7)

    capture = plt.Circle((0, 0), float(R_capture), fill=False, linestyle="--")
    ax.add_patch(capture)

    ax.legend(loc="best")

    def init():
        missile_line.set_data([], [])
        target_line.set_data([], [])
        missile_pt.set_data([], [])
        target_pt.set_data([], [])
        capture.center = (p_t[0, 0], p_t[0, 1])
        return missile_line, target_line, missile_pt, target_pt, capture

    def update(k: int):
        # Trails
        missile_line.set_data(p_m[: k + 1, 0], p_m[: k + 1, 1])
        target_line.set_data(p_t[: k + 1, 0], p_t[: k + 1, 1])

        # Points: MUST be sequences
        missile_pt.set_data([p_m[k, 0]], [p_m[k, 1]])
        target_pt.set_data([p_t[k, 0]], [p_t[k, 1]])

        capture.center = (p_t[k, 0], p_t[k, 1])
        return missile_line, target_line, missile_pt, target_pt, capture

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(p_m),
        init_func=init,
        interval=int(1000 / max(1, fps)),
        blit=False,
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer, dpi=150)
    plt.close(fig)

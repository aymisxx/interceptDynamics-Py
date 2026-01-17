from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(log: Dict[str, Any], out_path: Path, title: str) -> None:
    p_m = np.asarray(log["p_m"], dtype=float)
    p_t = np.asarray(log["p_t"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p_m[:, 0], p_m[:, 1], label="missile")
    ax.plot(p_t[:, 0], p_t[:, 1], label="target")
    ax.scatter([p_m[0, 0]], [p_m[0, 1]], marker="o", s=50, label="missile start")
    ax.scatter([p_t[0, 0]], [p_t[0, 1]], marker="x", s=60, label="target start")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distance(log: Dict[str, Any], out_path: Path, title: str) -> None:
    ts = np.asarray(log["ts"], dtype=float)
    d = np.asarray(log["d"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, d)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance ||p_t - p_m|| (m)")
    ax.grid(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_controls(log: Dict[str, Any], cfg: Dict[str, Any], out_path: Path, title: str) -> None:
    ts = np.asarray(log["ts"], dtype=float)
    u = np.asarray(log["u"], dtype=float)
    a_max = float(cfg["a_max"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, u[:, 0], label="u_x")
    ax.plot(ts, u[:, 1], label="u_y")
    ax.axhline(+a_max, linestyle="--")
    ax.axhline(-a_max, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("accel command (m/s^2)")
    ax.grid(True)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

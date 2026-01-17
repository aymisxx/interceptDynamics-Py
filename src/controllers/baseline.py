from __future__ import annotations

from typing import Dict, Any

import numpy as np

from ..utils import clip_box, clip_slew, rel_state


def controller_pd(t: float, p_m: np.ndarray, v_m: np.ndarray, p_t: np.ndarray, v_t: np.ndarray, u_prev: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """PD on relative state (baseline). Returns saturated + slew-limited acceleration."""
    gains = cfg.get("baseline", {})
    kp = float(gains.get("kp_pos", 0.8))
    kd = float(gains.get("kd_vel", 1.6))

    x = rel_state(p_m, v_m, p_t, v_t)
    r = x[0:2]
    vrel = x[2:4]

    u_raw = kp * r + kd * vrel

    # Apply constraints
    u = clip_box(u_raw, cfg["a_max"])
    u = clip_slew(u, u_prev, cfg["du_max"])
    u = clip_box(u, cfg["a_max"])
    return u

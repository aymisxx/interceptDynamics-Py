from __future__ import annotations

import numpy as np


def rel_state(p_m: np.ndarray, v_m: np.ndarray, p_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
    """Relative state x = [r_x, r_y, v_x, v_y]^T where r = p_t - p_m and v = v_t - v_m."""
    p_m = np.asarray(p_m, dtype=float).reshape(2,)
    v_m = np.asarray(v_m, dtype=float).reshape(2,)
    p_t = np.asarray(p_t, dtype=float).reshape(2,)
    v_t = np.asarray(v_t, dtype=float).reshape(2,)
    r = p_t - p_m
    v = v_t - v_m
    return np.hstack([r, v])


def rel_distance(p_m: np.ndarray, p_t: np.ndarray) -> float:
    """Euclidean distance ||p_t - p_m||."""
    p_m = np.asarray(p_m, dtype=float).reshape(2,)
    p_t = np.asarray(p_t, dtype=float).reshape(2,)
    return float(np.linalg.norm(p_t - p_m))


def intercepted(p_m: np.ndarray, p_t: np.ndarray, R_capture: float) -> bool:
    """Capture condition."""
    return rel_distance(p_m, p_t) <= float(R_capture)


def clip_box(u: np.ndarray, u_max: float) -> np.ndarray:
    """Per-axis box saturation."""
    u = np.asarray(u, dtype=float).reshape(2,)
    return np.clip(u, -float(u_max), float(u_max))


def clip_slew(u: np.ndarray, u_prev: np.ndarray, du_max: float) -> np.ndarray:
    """Per-axis slew-rate limiting."""
    u = np.asarray(u, dtype=float).reshape(2,)
    u_prev = np.asarray(u_prev, dtype=float).reshape(2,)
    du = u - u_prev
    du = np.clip(du, -float(du_max), float(du_max))
    return u_prev + du

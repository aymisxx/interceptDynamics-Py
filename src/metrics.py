from __future__ import annotations

from typing import Dict, Any

import numpy as np


def control_energy(u: np.ndarray, dt: float) -> float:
    u = np.asarray(u, dtype=float)
    return float(np.sum(np.sum(u * u, axis=1)) * float(dt))


def accel_saturation_pct(u: np.ndarray, a_max: float, tol: float = 1e-6) -> float:
    u = np.asarray(u, dtype=float)
    a_max = float(a_max)
    sat = np.any(np.abs(u) >= (a_max - tol), axis=1)
    return float(100.0 * np.mean(sat))


def slew_activity_pct(u: np.ndarray, du_max: float, tol: float = 1e-6) -> float:
    u = np.asarray(u, dtype=float)
    du_max = float(du_max)
    du = np.diff(u, axis=0)
    if len(du) == 0:
        return 0.0
    active = np.any(np.abs(du) >= (du_max - tol), axis=1)
    return float(100.0 * np.mean(active))


def compute_metrics(log: Dict[str, Any], cfg: Dict[str, Any], controller_name: str) -> Dict[str, Any]:
    dt = float(cfg["dt"])
    d = np.asarray(log["d"], dtype=float)
    u = np.asarray(log["u"], dtype=float)

    # If intercepted, use time-to-intercept from log; else None
    t_int = log.get("t_intercept", None)

    return {
        "controller": controller_name,
        "intercepted": bool(log.get("intercepted", False)),
        "time_to_intercept_s": float(t_int) if t_int is not None else None,
        "min_distance_m": float(np.min(d)),
        "control_energy_int_u2_dt": control_energy(u, dt),
        "accel_saturation_pct": accel_saturation_pct(u, cfg["a_max"]),
        "slew_activity_pct": slew_activity_pct(u, cfg["du_max"]),
    }

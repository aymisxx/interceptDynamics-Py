from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Scenario:
    name: str
    p_m0: np.ndarray
    v_m0: np.ndarray
    p_t0: np.ndarray
    v_t0: np.ndarray
    a_t_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def scenario_straight(name: str = "straight_target") -> Scenario:
    """Target moves straight with zero acceleration."""
    return Scenario(
        name=name,
        p_m0=np.array([0.0, 0.0]),
        v_m0=np.array([0.0, 0.0]),
        p_t0=np.array([200.0, 50.0]),
        v_t0=np.array([-8.0, 0.0]),
        a_t_fn=lambda t, p_t, v_t: np.array([0.0, 0.0]),
    )


def scenario_turning(name: str = "turning_target", a_lat: float = 1.5) -> Scenario:
    """Simple turning: lateral accel perpendicular to current velocity."""

    def a_t(t: float, p_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        v = np.asarray(v_t, dtype=float).reshape(2,)
        speed = float(np.linalg.norm(v)) + 1e-9
        v_hat = v / speed
        perp = np.array([-v_hat[1], v_hat[0]])
        return float(a_lat) * perp

    return Scenario(
        name=name,
        p_m0=np.array([0.0, 0.0]),
        v_m0=np.array([0.0, 0.0]),
        p_t0=np.array([200.0, 0.0]),
        v_t0=np.array([-8.0, 2.0]),
        a_t_fn=a_t,
    )

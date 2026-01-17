from __future__ import annotations

from typing import Callable, Dict, Any

import numpy as np

from .dynamics import step_missile, step_target
from .utils import rel_distance, intercepted


Controller = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]


def run_episode(controller: Controller, scenario, cfg: Dict[str, Any], *, integration: str = "rk4") -> Dict[str, Any]:
    """Run one simulation.

    controller signature:
      u = controller(t, p_m, v_m, p_t, v_t, u_prev, cfg)
    """
    dt = float(cfg["dt"])
    t_max = float(cfg["t_max"])
    steps = int(np.ceil(t_max / dt))

    p_m = np.asarray(scenario.p_m0, dtype=float).reshape(2,)
    v_m = np.asarray(scenario.v_m0, dtype=float).reshape(2,)
    p_t = np.asarray(scenario.p_t0, dtype=float).reshape(2,)
    v_t = np.asarray(scenario.v_t0, dtype=float).reshape(2,)

    u_prev = np.zeros(2)

    # Logs
    ts = np.zeros(steps)
    p_m_log = np.zeros((steps, 2))
    v_m_log = np.zeros((steps, 2))
    p_t_log = np.zeros((steps, 2))
    v_t_log = np.zeros((steps, 2))
    u_log = np.zeros((steps, 2))
    a_t_log = np.zeros((steps, 2))
    d_log = np.zeros(steps)

    hit = False
    t_hit = None

    for k in range(steps):
        t = k * dt
        ts[k] = t

        a_t = np.asarray(scenario.a_t_fn(t, p_t, v_t), dtype=float).reshape(2,)
        u = np.asarray(controller(t, p_m, v_m, p_t, v_t, u_prev, cfg), dtype=float).reshape(2,)

        # Step
        p_m, v_m = step_missile(p_m, v_m, u, dt, method=integration)
        p_t, v_t = step_target(p_t, v_t, a_t, dt, method=integration)

        # Log
        p_m_log[k] = p_m
        v_m_log[k] = v_m
        p_t_log[k] = p_t
        v_t_log[k] = v_t
        u_log[k] = u
        a_t_log[k] = a_t
        d_log[k] = rel_distance(p_m, p_t)

        u_prev = u

        if not hit and intercepted(p_m, p_t, cfg["R_capture"]):
            hit = True
            t_hit = t
            # keep sim to end for consistent array sizes; metrics can stop early

    return {
        "scenario": scenario.name,
        "ts": ts,
        "p_m": p_m_log,
        "v_m": v_m_log,
        "p_t": p_t_log,
        "v_t": v_t_log,
        "u": u_log,
        "a_t": a_t_log,
        "d": d_log,
        "intercepted": bool(hit),
        "t_intercept": float(t_hit) if t_hit is not None else None,
    }

from __future__ import annotations

import numpy as np


def f_point_mass(state: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """Continuous-time point-mass dynamics in 2D.

    state = [p_x, p_y, v_x, v_y]
    accel = [a_x, a_y]
    returns d/dt(state)
    """
    state = np.asarray(state, dtype=float).reshape(4,)
    accel = np.asarray(accel, dtype=float).reshape(2,)
    v = state[2:4]
    dp = v
    dv = accel
    return np.hstack([dp, dv])


def euler_step(state: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
    return state + float(dt) * f_point_mass(state, accel)


def rk4_step(state: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
    dt = float(dt)
    k1 = f_point_mass(state, accel)
    k2 = f_point_mass(state + 0.5 * dt * k1, accel)
    k3 = f_point_mass(state + 0.5 * dt * k2, accel)
    k4 = f_point_mass(state + dt * k3, accel)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def step_agent(p: np.ndarray, v: np.ndarray, accel: np.ndarray, dt: float, method: str = "rk4"):
    """Step an agent forward: returns (p_next, v_next)."""
    p = np.asarray(p, dtype=float).reshape(2,)
    v = np.asarray(v, dtype=float).reshape(2,)
    accel = np.asarray(accel, dtype=float).reshape(2,)

    state = np.hstack([p, v])
    m = method.lower()
    if m == "euler":
        state_next = euler_step(state, accel, dt)
    elif m == "rk4":
        state_next = rk4_step(state, accel, dt)
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return state_next[0:2], state_next[2:4]


def step_missile(p_m, v_m, u, dt, method: str = "rk4"):
    return step_agent(p_m, v_m, u, dt, method)


def step_target(p_t, v_t, a_t, dt, method: str = "rk4"):
    return step_agent(p_t, v_t, a_t, dt, method)


def get_ABG(dt: float):
    """Closed-form discrete relative dynamics matrices for:

    r_{k+1} = r_k + dt v_k + 0.5 dt^2 (a_tk - u_k)
    v_{k+1} = v_k + dt (a_tk - u_k)

    x = [r_x, r_y, v_x, v_y], u = [u_x, u_y], a_t = [a_tx, a_ty]
    """
    dt = float(dt)
    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))

    A = np.block([
        [I2, dt * I2],
        [Z2, I2],
    ])

    B = np.block([
        [-0.5 * dt ** 2 * I2],
        [-dt * I2],
    ])

    G = np.block([
        [0.5 * dt ** 2 * I2],
        [dt * I2],
    ])

    return A, B, G

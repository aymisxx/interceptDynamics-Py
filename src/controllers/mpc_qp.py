from __future__ import annotations

from typing import Callable, Dict, Any

import numpy as np

try:
    import cvxpy as cp
except Exception as e:  # pragma: no cover
    cp = None

from ..dynamics import get_ABG
from ..utils import clip_box, clip_slew, rel_state


ATPredFn = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def make_mpc_controller(a_t_pred_fn: ATPredFn) -> Callable:
    """Return a controller(t, p_m, v_m, p_t, v_t, u_prev, cfg) that solves a QP MPC.

    a_t_pred_fn: provides the target acceleration at time t based on (p_t, v_t).
    For simplicity, we predict constant acceleration across the horizon using the current estimate.
    """

    def controller_mpc(t: float, p_m: np.ndarray, v_m: np.ndarray, p_t: np.ndarray, v_t: np.ndarray, u_prev: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        if cp is None:
            raise ImportError("cvxpy is required for MPC. Install via requirements.txt")

        dt = float(cfg["dt"])
        N = int(cfg["N_mpc"])
        a_max = float(cfg["a_max"])
        du_max = float(cfg["du_max"])

        w = cfg.get("mpc_weights", {})
        w_r = float(w.get("w_r", 10.0))
        w_v = float(w.get("w_v", 1.0))
        w_u = float(w.get("w_u", 0.05))
        w_du = float(w.get("w_du", 0.5))

        x0 = rel_state(p_m, v_m, p_t, v_t)  # [r; v]
        A, B, G = get_ABG(dt)

        a_t_now = np.asarray(a_t_pred_fn(t, p_t, v_t), dtype=float).reshape(2,)
        a_seq = np.tile(a_t_now.reshape(2, 1), (1, N))

        # Decision variables
        X = cp.Variable((4, N + 1))
        U = cp.Variable((2, N))

        constraints = [X[:, 0] == x0]

        # Input constraints
        constraints += [U <= a_max, U >= -a_max]

        # Slew constraints with u_prev
        u_prev = np.asarray(u_prev, dtype=float).reshape(2,)
        constraints += [U[:, 0] - u_prev <= du_max, U[:, 0] - u_prev >= -du_max]
        for k in range(1, N):
            constraints += [U[:, k] - U[:, k - 1] <= du_max, U[:, k] - U[:, k - 1] >= -du_max]

        # Dynamics
        for k in range(N):
            constraints += [X[:, k + 1] == A @ X[:, k] + B @ U[:, k] + G @ a_seq[:, k]]

        # Cost
        cost = 0
        for k in range(N):
            r_k = X[0:2, k]
            v_k = X[2:4, k]
            cost += w_r * cp.sum_squares(r_k)
            cost += w_v * cp.sum_squares(v_k)
            cost += w_u * cp.sum_squares(U[:, k])
            if k == 0:
                cost += w_du * cp.sum_squares(U[:, k] - u_prev)
            else:
                cost += w_du * cp.sum_squares(U[:, k] - U[:, k - 1])

        prob = cp.Problem(cp.Minimize(cost), constraints)
        # OSQP is a great default for QPs
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if U.value is None:
            # Fallback: behave like a mild PD if solver fails
            # (still respects constraints)
            u_fallback = 0.4 * x0[0:2] + 0.8 * x0[2:4]
            u = clip_box(u_fallback, a_max)
        else:
            u = np.asarray(U.value[:, 0], dtype=float).reshape(2,)

        # Apply same hard constraints after solve (safety)
        u = clip_slew(u, u_prev, du_max)
        u = clip_box(u, a_max)
        return u

    return controller_mpc

from __future__ import annotations

"""Reproduce the baseline vs MPC comparison (plots + metrics + GIF).

Run:
  python scripts/run_all.py

Outputs go to results/plots, results/logs, results/animations.
"""

from pathlib import Path
import json

import sys

import numpy as np
import pandas as pd

# Allow running as a plain script: `python scripts/run_all.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import make_paths, default_config, save_config
from src.scenarios import scenario_turning
from src.sim import run_episode
from src.controllers.baseline import controller_pd
from src.controllers.mpc_qp import make_mpc_controller
from src.metrics import compute_metrics
from src.plotting import plot_trajectories, plot_distance, plot_controls
from src.animation import save_interception_gif


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = make_paths(root)
    cfg = default_config(paths)

    # Stressed config that made the baseline struggle in notebook
    cfg_stressed = dict(cfg)
    cfg_stressed["a_max"] = 15.0
    cfg_stressed["du_max"] = 5.0

    # Tuned weights from sweep (best-first)
    cfg_stressed["mpc_weights"] = {"w_r": 10.0, "w_v": 1.0, "w_u": 0.05, "w_du": 0.2}

    scenario = scenario_turning(name="turning_target_hard", a_lat=1.5)

    # Baseline
    log_pd = run_episode(controller_pd, scenario, cfg_stressed)

    # MPC (closure uses the scenario target accel model)
    mpc_controller = make_mpc_controller(scenario.a_t_fn)
    log_mpc = run_episode(mpc_controller, scenario, cfg_stressed)

    # Plots
    plot_trajectories(log_pd, paths.plots_dir / "baseline_pd_traj_turning_stressed.png", "Trajectories — PD baseline (stressed turning target)")
    plot_distance(log_pd, paths.plots_dir / "baseline_pd_distance_turning_stressed.png", "Distance vs time — PD baseline (stressed turning target)")
    plot_controls(log_pd, cfg_stressed, paths.plots_dir / "baseline_pd_controls_turning_stressed.png", "Controls — PD baseline (stressed turning target)")

    plot_trajectories(log_mpc, paths.plots_dir / "mpc_qp_traj_turning_stressed.png", "Trajectories — MPC QP (stressed turning target)")
    plot_distance(log_mpc, paths.plots_dir / "mpc_qp_distance_turning_stressed.png", "Distance vs time — MPC QP (stressed turning target)")
    plot_controls(log_mpc, cfg_stressed, paths.plots_dir / "mpc_qp_controls_turning_stressed.png", "Controls — MPC QP (stressed turning target)")

    # Metrics
    rows = [
        compute_metrics(log_pd, cfg_stressed, "PD_baseline_stressed"),
        compute_metrics(log_mpc, cfg_stressed, "MPC_QP_stressed"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(paths.logs_dir / "metrics_comparison.csv", index=False)

    # Save logs (compact)
    np.savez_compressed(paths.logs_dir / "log_pd_stressed.npz", **{k: v for k, v in log_pd.items() if k not in ["scenario"]})
    np.savez_compressed(paths.logs_dir / "log_mpc_stressed.npz", **{k: v for k, v in log_mpc.items() if k not in ["scenario"]})

    # Config snapshot
    save_config(cfg_stressed, paths.logs_dir / "config_final.json")

    # Animation (GIF only, by design)
    save_interception_gif(log_mpc, paths.anims_dir / "mpc_qp_stressed.gif", R_capture=cfg_stressed["R_capture"], stride=2, fps=20)

    print("DONE")
    print(df)


if __name__ == "__main__":
    main()

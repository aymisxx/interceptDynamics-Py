from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class Paths:
    root: Path
    results_dir: Path
    plots_dir: Path
    anims_dir: Path
    logs_dir: Path


def make_paths(root: Path | str = ".") -> Paths:
    root = Path(root)
    results_dir = root / "results"
    plots_dir = results_dir / "plots"
    anims_dir = results_dir / "animations"
    logs_dir = results_dir / "logs"
    for d in [results_dir, plots_dir, anims_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, results_dir=results_dir, plots_dir=plots_dir, anims_dir=anims_dir, logs_dir=logs_dir)


def default_config(paths: Paths, seed: int = 7) -> Dict[str, Any]:
    """Single source of truth config (mirrors the notebook)."""
    cfg: Dict[str, Any] = {
        # Time
        "dt": 0.05,
        "t_max": 25.0,
        "N_mpc": 25,
        # Interception
        "R_capture": 5.0,
        # Missile limits (box constraints)
        "a_max": 30.0,
        "du_max": 10.0,
        # Baseline gains
        "baseline": {"kp_pos": 0.8, "kd_vel": 1.6},
        # MPC weights
        "mpc_weights": {"w_r": 10.0, "w_v": 1.0, "w_u": 0.05, "w_du": 0.5},
        # Housekeeping
        "seed": int(seed),
        "results_dir": str(paths.results_dir),
        "plots_dir": str(paths.plots_dir),
        "anims_dir": str(paths.anims_dir),
        "logs_dir": str(paths.logs_dir),
    }
    return cfg


def save_config(cfg: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

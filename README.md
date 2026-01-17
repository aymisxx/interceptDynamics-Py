# **interceptDynamics-Py**  

**2D pursuit-evasion interception with constrained MPC (QP) vs classical PD guidance**

A clean, reproducible **planar interception simulator** built to study **non-RL optimal control** for pursuit-evasion problems.  
The project compares a **classical PD baseline** against a **constraint-aware MPC (QP)** controller under increasingly stressed target maneuvers, with full metrics, plots, and portable animations.

This repo is intentionally **engineering-first**: transparent assumptions, interpretable controllers, and deterministic results.

---

## What this project does

- Models a **2D point-mass missile-target interception problem**
- Implements a **classical PD guidance baseline**
- Implements a **constrained MPC controller (QP, CVXPY/OSQP)**
- Enforces:
  - acceleration bounds
  - slew-rate bounds
- Evaluates controllers on **straight, turning, and stressed turning targets**
- Produces:
  - trajectory plots
  - distance-to-target plots
  - control activity plots
  - quantitative metrics
  - **GIF animations** (no FFmpeg dependency)

---

## Repository structure

```text
interceptDynamics-Py/
│
├─ src/
│  ├─ dynamics.py
│  ├─ scenarios.py
│  ├─ sim.py
│  ├─ controllers/
│  │  ├─ baseline.py
│  │  └─ mpc_qp.py
│  ├─ metrics.py
│  ├─ plotting.py
│  └─ animation.py
│
├─ scripts/
│  └─ run_all.py
│
├─ notebooks/
│  ├─ interceptDynamics-experiment.zip
│  └─ interceptDynamics-experiment.pdf
│
├─ results/
│  ├─ plots/
│  ├─ animations/
│  └─ logs/
│
└─ README.md
```

## Quickstart

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the full experiment
```bash
python scripts/run_all.py
```

All outputs are saved automatically to `results/`.

---

## Mathematical modeling

### Kinematics

Both missile and target are modeled as **planar point masses** with second-order dynamics:

$$
\dot{p}_m = v_m,\qquad \dot{v}_m = u
$$

$$
\dot{p}_t = v_t,\qquad \dot{v}_t = a_t(t)
$$

where:
- $p_m, p_t \in \mathbb{R}^2$ are positions
- $v_m, v_t \in \mathbb{R}^2$ are velocities
- $u \in \mathbb{R}^2$ is the missile acceleration command
- $a_t(t)$ is a scenario-defined target acceleration

### Relative-state formulation

For guidance and control, the system is expressed in **relative coordinates**:

$$
r = p_t - p_m,\qquad v_{rel} = v_t - v_m
$$

with dynamics:

$$
\dot{r} = v_{rel}
$$
$$
\dot{v}_{rel} = a_t(t) - u
$$

This formulation directly exposes the interception geometry and simplifies controller design.

### Discretization

The system is simulated in discrete time with step size $dt$:

$$
x_{k+1} = A x_k + B u_k + d_k
$$

where

$$
x = \begin{bmatrix}
r \\
v_{rel}
\end{bmatrix},
$$

and $d_k$ captures known target acceleration effects.

Forward Euler discretization is used for clarity and reproducibility.

### Capture condition

Interception is defined using a geometric proximity criterion:

$$
\|r_k\| = \|p_t - p_m\| \le R_{capture}
$$

No terminal explosion, fuse logic, or post-intercept dynamics are modeled.

## Controllers

### PD baseline

A classical proportional-derivative controller on the relative state:

$$
u_k = K_p r_k + K_d v_{rel,k}
$$

followed by actuator saturation and slew-rate limiting.  
This controller is reactive, interpretable, and serves as a reference baseline.

### MPC (QP)

At each timestep, the controller solves a finite-horizon quadratic program.

**Objective**

### Objective

$$
\min_{\{u_i\}_{i=0}^{N-1}}
\sum_{i=0}^{N-1}
\left(
w_r \|r_i\|^2
+ w_v \|v_{rel,i}\|^2
+ w_u \|u_i\|^2
+ w_{\Delta u} \|u_i - u_{i-1}\|^2
\right)
$$

**Subject to**

$$
x_{i+1} = A x_i + B u_i + d_i
$$

$$
\|u_i\|_\infty \le a_{\max},
\qquad
\|u_i - u_{i-1}\|_\infty \le du_{\max}
$$

Only the first control input $u_0^*$ is applied before re-solving at the next timestep (receding-horizon control).

---

## Results summary (stressed turning target)

- Both controllers successfully intercept the target
- **MPC achieves interception significantly earlier**
- MPC trades higher control saturation and energy for speed
- PD baseline is smoother but slower and more reactive

Metrics, plots, and logs are saved in `results/` for full transparency.

## About the **multiple touches** & **trajectory coincide** near interception

In some runs, the distance-to-target curve may cross the capture threshold more than once before the trajectories visually converge.

This is expected because:
- point-mass modeling
- discrete-time simulation
- proximity-based capture
- no terminal kill modeling

This behavior reflects a modeling boundary, not a guidance failure.

## Reproducibility

- All parameters are centralized in a config block
- Deterministic seeds are used where applicable
- Logs, metrics, and configs are saved alongside plots
- A cold `Run All` reproduces the same results

## Known limitations (by design)

- 2D kinematics only
- Scenario-defined target acceleration (perfect information)
- No sensor noise or state estimation
- No terminal kill / blast modeling
- Discrete capture detection

---

## Acknowledgements

AI-based development tools were used to assist with debugging and documentation drafting.

---

# Author
### **Ayushman M.**

>LinkedIn: https://www.linkedin.com/in/aymisxx

>GitHub: https://github.com/aymisxx

---
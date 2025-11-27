Here is a cleaned-up, precise `Plan.md` you can drop into the repo.

---

# Plan: DDPM-Inspired ODE Solvers for Flow Matching Inference

**Goal.** Implement DDPM/DPM-Solver–inspired inference methods (non-uniform time grids + higher-order ODE solvers + adaptive steps) on top of Meta’s `flow_matching` library, first for the CIFAR-10 example in `examples/image`. Documentation and report come **after** the code and experiments.

---

## 0. Current Status (Phase 0 – Completed)

You already have:

* A working CIFAR-10 flow-matching checkpoint (`checkpoint-1799.pth`).
* `examples/image/analysis.py` that:

  * Reproduces midpoint ODE sampling (same as `eval_loop.py`).
  * Computes cosine similarity and angular difference between consecutive velocity evaluations.
  * Aggregates statistics over multiple batches (e.g., 30 × 32 = 960 samples).
  * Saves:

    * `angular_analysis.png`
    * `summary.txt`
    * `analysis_data.csv`

Example result for `step_size = 0.2` (5 steps, 10 evaluations):

* Global angular difference: mean ≈ 5.6°, std ≈ 3.1°.
* Early evaluations have larger angles (higher curvature); mid-range is smoother.

**Phase-0 objective is met:** we have a quantitative baseline of the velocity field smoothness under the current midpoint solver with a uniform time grid.

The only optional extension later is to add a simple FID evaluation script for image quality comparison.

---

## 1. Phase 1 – Non-uniform Time Grids (DDPM-like Schedules)

**Goal.** Introduce non-uniform time grids for inference, inspired by DDPM/DPM-Solver schedules, while keeping the solver itself fixed (midpoint / existing ODE solver).

### 1.1 Implement time schedule utilities

Create a new module, e.g.:

* `examples/image/training/time_schedules.py`

Implement several schedules that return a tensor of time points `t[0..K]` in `[0, 1]`:

```python
import torch

def uniform_schedule(num_steps: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, num_steps + 1)

def front_dense_schedule(num_steps: int, gamma: float = 3.0) -> torch.Tensor:
    # Concentrate steps near t = 0
    k = torch.linspace(0.0, 1.0, num_steps + 1)
    return k**gamma

def back_dense_schedule(num_steps: int, gamma: float = 3.0) -> torch.Tensor:
    # Concentrate steps near t = 1
    k = torch.linspace(0.0, 1.0, num_steps + 1)
    return 1.0 - (1.0 - k)**gamma

def curvature_based_schedule(num_steps: int, csv_path: str) -> torch.Tensor:
    # Load per-evaluation angular statistics from analysis_data.csv
    # Use average angle as a “curvature weight”
    # Build a cumulative distribution over t and invert it to get a non-uniform grid.
    ...
```

The curvature-based schedule should:

1. Read the mean angular difference per evaluation index.
2. Map evaluation indices to the underlying uniform times used during analysis.
3. Use the angle magnitude as a weight (higher angle → finer grid).
4. Normalize weights and compute a cumulative distribution.
5. Interpolate to obtain `num_steps + 1` time points.

### 1.2 Integrate schedules into image inference

Modify `examples/image/training/eval_loop.py` (or the equivalent sampling script) to:

* Accept new CLI arguments:

```bash
--time_schedule {uniform,front_dense,back_dense,curvature}
--num_steps N
--curvature_csv path/to/analysis_data.csv    # only for curvature mode
```

* Construct the time grid by calling the corresponding schedule instead of using a fixed `step_size` loop.
* Pass this `time_grid` into the ODE solver (midpoint) in ascending order.

### 1.3 Experiments for Phase 1

For a fixed checkpoint:

* Compare, at the same `num_steps` (e.g., 5, 10, 20):

  * `uniform_schedule`
  * `front_dense_schedule`
  * `curvature_based_schedule`
* For each configuration:

  * Run `analysis.py` (or a lighter variant) to verify how the angular differences change.
  * Optionally run a FID script (once you have it) to measure image quality.

**Success criterion for Phase 1.**
At least one non-uniform schedule achieves better image quality (lower FID or visibly better samples) than the uniform schedule at the same number of ODE steps.

---

## 2. Phase 2 – Higher-Order ODE Solvers (RK3/RK4)

**Goal.** Add higher-order ODE methods (e.g., RK3/RK4) to the continuous solver and combine them with non-uniform time grids from Phase 1.

### 2.1 Extend `flow_matching/solver/ode_solver.py`

In `flow_matching/solver/ode_solver.py`:

* Add support for multiple methods inside `ODESolver` (or introduce a subclass), e.g.:

```python
class ODESolver(Solver):
    def __init__(self, model, method: str = "midpoint", ...):
        self.method = method
        ...

    def _step(self, x, t0, t1):
        dt = t1 - t0
        if self.method == "euler":
            k1 = f(x, t0)
            return x + dt * k1
        elif self.method == "midpoint":
            ...
        elif self.method == "rk3":
            ...
        elif self.method == "rk4":
            ...
```

* Implement RK3/RK4 using standard Butcher tableaux, **allowing non-uniform `dt`** between adjacent time points.

### 2.2 Wire solver selection into the CIFAR example

In `examples/image/training/eval_loop.py`:

* Add CLI option:

```bash
--ode_method {midpoint,rk3,rk4}
```

* Propagate this argument when constructing the `ODESolver` instance.

### 2.3 Experiments for Phase 2

For each solver (`midpoint`, `rk3`, `rk4`) and each schedule (best one from Phase 1 + uniform):

* Run inference with `num_steps ∈ {5, 8, 10}`.
* Measure:

  * Visual quality / FID.
  * Runtime and total number of model evaluations (NFE).
* Use `analysis.py` on a subset (e.g., 960 samples) to ensure angular behaviour is stable and not exploding.

**Success criterion for Phase 2.**
There exists a configuration `RK3/RK4 + non-uniform schedule` that matches or exceeds the baseline 20–50-step midpoint + uniform schedule, with significantly fewer steps (e.g., 8–10).

---

## 3. Phase 3 – FM-Specific DPM-Style Solver (DPMFM-2)

**Goal.** Design a DPM-Solver–style multi-step method tailored to flow matching, using a suitable time reparameterization and velocity-based updates.

### 3.1 Identify the path and reparameterization

* Inspect `examples/image/training/edm_time_discretization.py` to confirm which EDM/VP-style path is used (e.g., a mapping between data and Gaussian with a specific noise schedule).
* Derive a monotone reparameterization `s = g(t)` (e.g., based on log-SNR or an EDM-style scaling) and its inverse `t = g⁻¹(s)`.

The aim is to rewrite the FM ODE in the form:

[
\frac{dx}{ds} = \tilde{f}(x, s)
]

where `s` is more “linear” with respect to the difficulty of integration, enabling evenly spaced steps in `s`.

### 3.2 Implement a DPMFM-style solver

In `flow_matching/solver/ode_solver.py` or a new file:

* Implement `DPMSolverFMSolver`, with interface similar to `ODESolver`, but:

  * Accept a fixed number of steps `K`.
  * Construct an evenly spaced grid in `s`, then map to `t`.
  * Use a **two-step or three-step** update analogous to DPM-Solver-2:

    * Use `v(x_t, t)` evaluated at a small set of `t` points per step and combine them with analytical coefficients derived under simplifying assumptions (e.g., locally linear drift).

### 3.3 2D toy verification

Use the 2D examples:

* `examples/2d_flow_matching.ipynb` (or similar):

  1. Use a dense RK4 solution (with many small steps) as the “ground truth” trajectory.
  2. Run `DPMSolverFM` with a small number of steps (e.g., 8).
  3. Measure trajectory error:

     * L2 distance between paths.
     * Final distribution error (e.g., KDE or sample plots).

### 3.4 CIFAR-10 experiments

* Add `--ode_method dpmfm2` to `eval_loop.py`.
* Run the same protocol as Phase 2:

  * Compare `dpmfm2` vs `rk3/rk4` vs `midpoint` for several step counts.
  * Measure NFE, runtime, and FID.

**Success criterion for Phase 3.**
`DPMSolverFM` achieves similar or better FID than RK3/RK4 at the same NFE, and toy experiments show clearly reduced trajectory error compared to midpoint/Euler.

---

## 4. Phase 4 – Adaptive Step Size (DPM-Solver-v3 Spirit)

**Goal.** Make the FM solver adaptive: allocate more steps where the velocity field is rapidly changing, and fewer steps where it is smooth, targeting a fixed NFE budget.

### 4.1 Design a local error estimator

Within the solver:

* Use one of the following heuristics:

  * Difference of velocities:
    [
    \text{err} \approx |v(x_t, t) - v(x_{t-\Delta t}, t-\Delta t)|
    ]
  * Embedded step comparison:

    * Compute one large step of size `Δt`.
    * Compute two half-steps of size `Δt/2`.
    * Use the difference between the two end states as an error estimate.

### 4.2 Implement `AdaptiveDPMSolverFM`

* Add a new solver class with parameters:

  * `max_nfe` (maximum allowed model evaluations).
  * `err_tol` (target local error tolerance).

* Algorithm outline:

  1. Start at `t = 0` (or `t = 1`, depending on convention).
  2. Propose a step size `Δt` (initially coarse).
  3. Estimate local error using the method above.
  4. If error > tolerance: shrink `Δt` and retry.
  5. If error < tolerance: accept the step and consider increasing `Δt`.
  6. Stop when reaching the final time or when `max_nfe` is exhausted.

* Log the actual sequence of `t` values for later analysis.

### 4.3 Experiments for Phase 4

For a fixed `max_nfe ∈ {8, 12, 16}`:

* Compare:

  * Fixed-grid `dpmfm2`.
  * Adaptive `AdaptiveDPMSolverFM` with the same or lower **realized** NFE.
* Metrics:

  * Realized average NFE per sample.
  * FID and runtime.
  * Distribution of step sizes `Δt` over time (to verify that the solver indeed refines around high-curvature regions, as indicated by Phase-0 analysis).

**Success criterion for Phase 4.**
Adaptive solver matches or improves upon fixed-grid DPMFM at the same or lower NFE, and the step-size distribution is consistent with the measured velocity curvature.

---

## 5. Final Cleanup and Documentation (After Code)

Only after the above phases are implemented and tested:

1. Refactor shared utilities in `examples/image/training`:

   * Move schedule and solver wrappers into dedicated modules.
   * Remove duplicated code between `eval_loop.py`, `analysis.py`, and any FID scripts.
2. Write a concise technical README / report draft:

   * Problem statement and motivation.
   * Description of non-uniform schedules, high-order solvers, DPMFM, and adaptive stepping.
   * Experimental tables and plots (FID vs NFE, velocity angular statistics, etc.).
3. If targeting a course project or paper submission, expand the draft into a full report.

---

If you want, I can next:

* Write the initial `time_schedules.py` implementation (uniform + front-dense + back-dense + a simple curvature-based version), and
* Show the minimal diff needed in `eval_loop.py` to switch from `step_size` to `num_steps + schedule`.

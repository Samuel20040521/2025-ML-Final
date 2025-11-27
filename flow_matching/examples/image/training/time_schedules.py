# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""
Pure time-grid schedule functions for non-uniform ODE discretization.

These functions ONLY generate time grids for the ODE solver.
They do NOT modify the velocity field, scheduler, or any α(t)/σ(t) coefficients.

This is strictly a discretization experiment - same ODE, different time grids.
"""

import torch
from typing import Optional

# Minimum number of steps required for meaningful discretization
MIN_NUM_STEPS = 1


def _enforce_boundaries(time_grid: torch.Tensor) -> torch.Tensor:
    """
    Enforce exact t=0 and t=1 boundaries to avoid floating-point drift.
    
    Many flow-matching schedulers assume:
        t == 0.0 → noise distribution
        t == 1.0 → data distribution
    
    Floating-point operations may produce values like 0.99999994039 instead of 1.0,
    which can cause reference comparisons or solvers to behave incorrectly.
    
    Args:
        time_grid: Time grid tensor of shape (num_steps + 1,).
    
    Returns:
        Time grid with exact 0.0 and 1.0 at boundaries.
    """
    time_grid = time_grid.clone()
    time_grid[0] = 0.0
    time_grid[-1] = 1.0
    return time_grid


def _validate_num_steps(num_steps: int, schedule_name: str = "schedule") -> None:
    """
    Validate that num_steps is at least MIN_NUM_STEPS.
    
    Args:
        num_steps: Number of ODE integration steps.
        schedule_name: Name of the schedule for error messages.
    
    Raises:
        ValueError: If num_steps < MIN_NUM_STEPS.
    """
    if num_steps < MIN_NUM_STEPS:
        raise ValueError(
            f"{schedule_name} requires num_steps >= {MIN_NUM_STEPS}, got {num_steps}. "
            f"With num_steps=1, the time grid is always [0, 1] regardless of schedule."
        )


def uniform_schedule(num_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate a uniform time grid from 0 to 1.
    
    Args:
        num_steps: Number of ODE integration steps (must be >= 1).
        device: Target device for the tensor.
    
    Returns:
        Tensor of shape (num_steps + 1,) with uniformly spaced time points.
    
    Example:
        >>> uniform_schedule(5)
        tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000])
    
    Raises:
        ValueError: If num_steps < 1.
    """
    _validate_num_steps(num_steps, "uniform_schedule")
    time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    return _enforce_boundaries(time_grid)


def front_dense_schedule(
    num_steps: int,
    gamma: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a front-dense (early-concentrated) time grid.
    
    Time points are concentrated near t=0, where the velocity field
    typically has higher curvature (transitions from noise to signal).
    
    Formula: t_k = (k / num_steps)^gamma
    
    Args:
        num_steps: Number of ODE integration steps (must be >= 1).
        gamma: Power exponent (gamma > 1 concentrates points near t=0).
        device: Target device for the tensor.
    
    Returns:
        Tensor of shape (num_steps + 1,) with front-dense time points.
    
    Example:
        >>> front_dense_schedule(5, gamma=3.0)
        tensor([0.0000, 0.0080, 0.0640, 0.2160, 0.5120, 1.0000])
    
    Raises:
        ValueError: If num_steps < 1.
    """
    _validate_num_steps(num_steps, "front_dense_schedule")
    k = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    time_grid = k.pow(gamma)
    return _enforce_boundaries(time_grid)


def back_dense_schedule(
    num_steps: int,
    gamma: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a back-dense (late-concentrated) time grid.
    
    Time points are concentrated near t=1, useful if the velocity field
    has higher curvature near the end of the trajectory.
    
    Formula: t_k = 1 - (1 - k / num_steps)^gamma
    
    Args:
        num_steps: Number of ODE integration steps (must be >= 1).
        gamma: Power exponent (gamma > 1 concentrates points near t=1).
        device: Target device for the tensor.
    
    Returns:
        Tensor of shape (num_steps + 1,) with back-dense time points.
    
    Example:
        >>> back_dense_schedule(5, gamma=3.0)
        tensor([0.0000, 0.4880, 0.7840, 0.9360, 0.9920, 1.0000])
    
    Raises:
        ValueError: If num_steps < 1.
    """
    _validate_num_steps(num_steps, "back_dense_schedule")
    k = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    time_grid = 1.0 - (1.0 - k).pow(gamma)
    return _enforce_boundaries(time_grid)


def cosine_schedule(
    num_steps: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a cosine-based time grid.
    
    Inspired by DDPM cosine schedules, this places more points
    in the middle of the trajectory.
    
    Formula: t_k = (1 - cos(π * k / num_steps)) / 2
    
    Args:
        num_steps: Number of ODE integration steps (must be >= 1).
        device: Target device for the tensor.
    
    Returns:
        Tensor of shape (num_steps + 1,) with cosine-distributed time points.
    
    Raises:
        ValueError: If num_steps < 1.
    """
    _validate_num_steps(num_steps, "cosine_schedule")
    k = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    time_grid = (1.0 - torch.cos(torch.pi * k)) / 2.0
    return _enforce_boundaries(time_grid)


def sigmoid_schedule(
    num_steps: int,
    start: float = -3.0,
    end: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a sigmoid-based time grid.
    
    Places more points in the middle of the trajectory with smooth transitions.
    
    Args:
        num_steps: Number of ODE integration steps (must be >= 1).
        start: Start value for sigmoid input range.
        end: End value for sigmoid input range.
        device: Target device for the tensor.
    
    Returns:
        Tensor of shape (num_steps + 1,) with sigmoid-distributed time points.
    
    Raises:
        ValueError: If num_steps < 1.
    """
    _validate_num_steps(num_steps, "sigmoid_schedule")
    k = torch.linspace(start, end, num_steps + 1, device=device)
    sigmoid_k = torch.sigmoid(k)
    # Normalize to [0, 1]
    time_grid = (sigmoid_k - sigmoid_k[0]) / (sigmoid_k[-1] - sigmoid_k[0])
    return _enforce_boundaries(time_grid)


def quadratic_schedule(
    num_steps: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a quadratic (front-dense) time grid with gamma=2.
    
    Convenience wrapper for front_dense_schedule with gamma=2.0.
    
    Args:
        num_steps: Number of ODE integration steps (must be >= 1).
        device: Target device for the tensor.
    
    Returns:
        Tensor of shape (num_steps + 1,) with quadratic time points.
    
    Raises:
        ValueError: If num_steps < 1.
    """
    # Validation is done inside front_dense_schedule
    return front_dense_schedule(num_steps, gamma=2.0, device=device)


def get_time_schedule(
    schedule_name: str,
    num_steps: int,
    gamma: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Factory function to get a time schedule by name.
    
    Args:
        schedule_name: One of 'uniform', 'front_dense', 'back_dense', 
                       'cosine', 'sigmoid', 'quadratic'.
        num_steps: Number of ODE integration steps.
        gamma: Power exponent for front_dense/back_dense schedules.
        device: Target device for the tensor.
    
    Returns:
        Tensor of shape (num_steps + 1,) with the requested time grid.
    
    Raises:
        ValueError: If schedule_name is not recognized.
    """
    schedule_name = schedule_name.lower()
    
    if schedule_name == "uniform":
        return uniform_schedule(num_steps, device=device)
    elif schedule_name == "front_dense":
        return front_dense_schedule(num_steps, gamma=gamma, device=device)
    elif schedule_name == "back_dense":
        return back_dense_schedule(num_steps, gamma=gamma, device=device)
    elif schedule_name == "cosine":
        return cosine_schedule(num_steps, device=device)
    elif schedule_name == "sigmoid":
        return sigmoid_schedule(num_steps, device=device)
    elif schedule_name == "quadratic":
        return quadratic_schedule(num_steps, device=device)
    else:
        raise ValueError(
            f"Unknown schedule '{schedule_name}'. "
            f"Available: uniform, front_dense, back_dense, cosine, sigmoid, quadratic"
        )


# List of available schedule names for CLI help
AVAILABLE_SCHEDULES = [
    "uniform",
    "front_dense", 
    "back_dense",
    "cosine",
    "sigmoid",
    "quadratic",
]

"""
Evaluator for the 2D two-group neutron diffusion equation analytical solution.

Evaluates a candidate Python program that defines phi1(x,y) and phi2(x,y)
by computing PDE residuals (interior points) and BC residuals (boundary points).

Interface:
    evaluate(path_user_py, task_name, timeout) -> dict
"""
import importlib.util
import os
import sys
import time
import traceback
from typing import Dict

import numpy as np


# Physical parameters
D1 = 1.0
D2 = 0.5
SIGMA_R = 0.02
SIGMA_A2 = 0.1
NU = 2.5
SIGMA_F1 = 0.005
SIGMA_F2 = 0.1
SIGMA_12 = 0.015

# PDE test points (interior)
PDE_TEST_POINTS = [
    (0.0, 0.0),
    (0.2, 0.2),
    (-0.2, -0.3),
    (0.4, -0.4),
]

# BC test points: 5 uniformly spaced on each boundary
BC_LEFT_POINTS = [(-0.5, y) for y in np.linspace(-0.5, 0.5, 5)]
BC_RIGHT_POINTS = [(0.5, y) for y in np.linspace(-0.5, 0.5, 5)]
BC_TOP_POINTS = [(x, 0.5) for x in np.linspace(-0.5, 0.5, 5)]
BC_BOTTOM_POINTS = [(x, -0.5) for x in np.linspace(-0.5, 0.5, 5)]

# Step size for numerical differentiation
H = 1e-6


def _load_module(path_user_py: str):
    """Load user module from file path."""
    spec = importlib.util.spec_from_file_location("user_solution", path_user_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_solution_func(module):
    """Extract the solution function from user module.

    Looks for: solution(x, y), phi(x, y), or (phi1(x,y), phi2(x,y)).
    """
    if hasattr(module, "solution"):
        return module.solution
    elif hasattr(module, "phi"):
        return module.phi
    elif hasattr(module, "phi1") and hasattr(module, "phi2"):
        def combined(x, y):
            return module.phi1(x, y), module.phi2(x, y)
        return combined
    else:
        raise AttributeError(
            "User module must define one of: solution(x,y), phi(x,y), "
            "or both phi1(x,y) and phi2(x,y)"
        )


def _numerical_laplacian(func, x, y, h=H):
    """Compute Laplacian via central finite differences.

    nabla^2 f = (f(x+h,y) + f(x-h,y) + f(x,y+h) + f(x,y-h) - 4*f(x,y)) / h^2
    """
    f_center = func(x, y)
    f_xp = func(x + h, y)
    f_xm = func(x - h, y)
    f_yp = func(x, y + h)
    f_ym = func(x, y - h)
    return (f_xp + f_xm + f_yp + f_ym - 4.0 * f_center) / (h * h)


def _numerical_dfdx(func, x, y, h=H):
    """Compute df/dx via central differences."""
    return (func(x + h, y) - func(x - h, y)) / (2.0 * h)


def _numerical_dfdy(func, x, y, h=H):
    """Compute df/dy via central differences."""
    return (func(x, y + h) - func(x, y - h)) / (2.0 * h)


def _compute_pde_residuals(sol_func):
    """Compute PDE residuals at interior test points.

    Fast group:  Res1 = -D1 * nabla^2(phi1) + Sigma_r * phi1
                        - nu * Sigma_f1 * phi1 - nu * Sigma_f2 * phi2
    Thermal:     Res2 = -D2 * nabla^2(phi2) + Sigma_a2 * phi2
                        - Sigma_12 * phi1
    """
    residuals = []
    for x, y in PDE_TEST_POINTS:
        phi1_val, phi2_val = sol_func(x, y)
        phi1_val = float(np.asarray(phi1_val).ravel()[0])
        phi2_val = float(np.asarray(phi2_val).ravel()[0])

        def phi1_func(xx, yy):
            v1, _ = sol_func(xx, yy)
            return float(np.asarray(v1).ravel()[0])

        def phi2_func(xx, yy):
            _, v2 = sol_func(xx, yy)
            return float(np.asarray(v2).ravel()[0])

        lap_phi1 = _numerical_laplacian(phi1_func, x, y)
        lap_phi2 = _numerical_laplacian(phi2_func, x, y)

        res1 = (-D1 * lap_phi1 + SIGMA_R * phi1_val
                - NU * SIGMA_F1 * phi1_val - NU * SIGMA_F2 * phi2_val)
        res2 = (-D2 * lap_phi2 + SIGMA_A2 * phi2_val
                - SIGMA_12 * phi1_val)

        residuals.append(res1)
        residuals.append(res2)

    return np.array(residuals)


def _compute_bc_residuals(sol_func):
    """Compute boundary condition residuals.

    Left  (x=-0.5): -D_i * dphi_i/dx - y = 0
    Right (x= 0.5): -D_i * dphi_i/dx = 0
    Top   (y= 0.5): -D_i * dphi_i/dy = 0
    Bottom(y=-0.5): -D_i * dphi_i/dy = 0
    """
    residuals = []

    def phi1_func(xx, yy):
        v1, _ = sol_func(xx, yy)
        return float(np.asarray(v1).ravel()[0])

    def phi2_func(xx, yy):
        _, v2 = sol_func(xx, yy)
        return float(np.asarray(v2).ravel()[0])

    # Left boundary: -D * dphi/dx = y
    for x, y in BC_LEFT_POINTS:
        dphi1_dx = _numerical_dfdx(phi1_func, x, y)
        dphi2_dx = _numerical_dfdx(phi2_func, x, y)
        residuals.append(-D1 * dphi1_dx - y)
        residuals.append(-D2 * dphi2_dx - y)

    # Right boundary: -D * dphi/dx = 0
    for x, y in BC_RIGHT_POINTS:
        dphi1_dx = _numerical_dfdx(phi1_func, x, y)
        dphi2_dx = _numerical_dfdx(phi2_func, x, y)
        residuals.append(-D1 * dphi1_dx)
        residuals.append(-D2 * dphi2_dx)

    # Top boundary: -D * dphi/dy = 0
    for x, y in BC_TOP_POINTS:
        dphi1_dy = _numerical_dfdy(phi1_func, x, y)
        dphi2_dy = _numerical_dfdy(phi2_func, x, y)
        residuals.append(-D1 * dphi1_dy)
        residuals.append(-D2 * dphi2_dy)

    # Bottom boundary: -D * dphi/dy = 0
    for x, y in BC_BOTTOM_POINTS:
        dphi1_dy = _numerical_dfdy(phi1_func, x, y)
        dphi2_dy = _numerical_dfdy(phi2_func, x, y)
        residuals.append(-D1 * dphi1_dy)
        residuals.append(-D2 * dphi2_dy)

    return np.array(residuals)


def evaluate(path_user_py: str, task_name: str = "default",
             timeout: int = 3600) -> Dict:
    """Evaluate a candidate solution for the 2D two-group neutron diffusion.

    Args:
        path_user_py: Path to the candidate Python file.
        task_name: Task identifier (unused, kept for interface compatibility).
        timeout: Maximum evaluation time in seconds.

    Returns:
        dict with keys: validity, combined_score, cost_time, error_info.
    """
    start_time = time.time()

    try:
        # Load user module
        module = _load_module(path_user_py)
        sol_func = _get_solution_func(module)

        # Quick sanity check: can we evaluate at origin?
        test_result = sol_func(0.0, 0.0)
        if not (isinstance(test_result, (tuple, list)) and len(test_result) == 2):
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "cost_time": time.time() - start_time,
                "error_info": "solution must return a tuple (phi1, phi2)",
            }

        # Check for NaN/Inf in basic evaluation
        p1, p2 = test_result
        p1_val = float(np.asarray(p1).ravel()[0])
        p2_val = float(np.asarray(p2).ravel()[0])
        if not (np.isfinite(p1_val) and np.isfinite(p2_val)):
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "cost_time": time.time() - start_time,
                "error_info": "solution returns NaN or Inf at origin",
            }

        # Compute residuals
        pde_residuals = _compute_pde_residuals(sol_func)
        bc_residuals = _compute_bc_residuals(sol_func)

        # Check for NaN in residuals
        if not (np.all(np.isfinite(pde_residuals))
                and np.all(np.isfinite(bc_residuals))):
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "cost_time": time.time() - start_time,
                "error_info": "residual computation produced NaN/Inf",
            }

        # Compute norms
        pde_norm = np.sqrt(np.mean(pde_residuals ** 2))
        bc_norm = np.sqrt(np.mean(bc_residuals ** 2))

        # Score: higher is better, perfect solution -> 1.0
        combined_score = 1.0 / (1.0 + pde_norm + bc_norm)

        cost_time = time.time() - start_time

        return {
            "validity": 1.0,
            "combined_score": float(combined_score),
            "cost_time": float(cost_time),
            "error_info": "",
        }

    except Exception as e:
        return {
            "validity": 0.0,
            "combined_score": 0.0,
            "cost_time": time.time() - start_time,
            "error_info": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <path_to_solution.py>")
        sys.exit(1)

    result = evaluate(sys.argv[1])
    print(f"validity:       {result['validity']}")
    print(f"combined_score: {result['combined_score']}")
    print(f"cost_time:      {result['cost_time']:.4f}s")
    print(f"error_info:     {result['error_info']}")

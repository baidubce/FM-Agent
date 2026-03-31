"""Truncated Analytical baseline for two-group neutron diffusion.

Uses only the first K=5 odd Fourier modes (n=1,3,5,7,9) from the
analytical solution, compared to N_modes=30 used by the full solution.
"""
import numpy as np
import json
import time
import os
import sys

# Add parent paths to import init.py's helper functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "famou", "task1"))
from init import (  # noqa: E402
    D1, D2, A11, A12, A21, A22,
    _solve_eigenvalues, _basis_func, _basis_deriv, _eigenvector_ratio,
)


def phi_truncated(x, y, K: int = 5):
    """Compute solution with only K odd Fourier modes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    phi1 = np.zeros_like(x)
    phi2 = np.zeros_like(x)

    for k in range(K):
        n = 2 * k + 1
        kappa_n = n * np.pi
        cn = -4.0 / (kappa_n**2)
        beta1, beta2, B11, B22 = _solve_eigenvalues(n)
        r1 = _eigenvector_ratio(beta1, B11)
        r2 = _eigenvector_ratio(beta2, B11)
        f1p = _basis_deriv(beta1, -0.5)
        f2p = _basis_deriv(beta2, -0.5)

        mat = np.array([
            [-D1 * f1p, -D1 * f2p],
            [-D2 * r1 * f1p, -D2 * r2 * f2p],
        ], dtype=complex)
        rhs_vec = np.array([cn, cn], dtype=complex)

        try:
            coeffs = np.linalg.solve(mat, rhs_vec)
        except np.linalg.LinAlgError:
            continue

        C_A = coeffs[0].real
        C_B = coeffs[1].real
        f1_x = _basis_func(beta1, x)
        f2_x = _basis_func(beta2, x)
        Y_n = np.cos(kappa_n * (y + 0.5))

        phi1 += (C_A * f1_x + C_B * f2_x) * Y_n
        phi2 += (C_A * r1.real * f1_x + C_B * r2.real * f2_x) * Y_n

    return phi1, phi2


def evaluate_truncated():
    """Evaluate truncated analytical solution."""
    t0 = time.time()

    def p1(xi, yi):
        v1, _ = phi_truncated(xi, yi, K=5)
        return float(v1)

    def p2(xi, yi):
        _, v2 = phi_truncated(xi, yi, K=5)
        return float(v2)

    elapsed = time.time() - t0

    # PDE residuals
    pde_res = []
    test_pts = [(0, 0), (0.2, 0.2), (-0.2, -0.3), (0.4, -0.4)]
    eps = 1e-6
    for xi, yi in test_pts:
        lap1 = ((p1(xi + eps, yi) - 2 * p1(xi, yi) + p1(xi - eps, yi)) / eps**2
                + (p1(xi, yi + eps) - 2 * p1(xi, yi) + p1(xi, yi - eps)) / eps**2)
        lap2 = ((p2(xi + eps, yi) - 2 * p2(xi, yi) + p2(xi - eps, yi)) / eps**2
                + (p2(xi, yi + eps) - 2 * p2(xi, yi) + p2(xi, yi - eps)) / eps**2)
        res1 = abs(-D1 * lap1 + A11 * p1(xi, yi) + A12 * p2(xi, yi))
        res2 = abs(A21 * p1(xi, yi) - D2 * lap2 + A22 * p2(xi, yi))
        pde_res.append(res1 + res2)

    # BC residuals
    bc_res = []
    h = 1e-6
    for yj in np.linspace(-0.4, 0.4, 9):
        dp1 = (p1(-0.5 + h, yj) - p1(-0.5, yj)) / h
        dp2 = (p2(-0.5 + h, yj) - p2(-0.5, yj)) / h
        bc_res.append(abs(-D1 * dp1 - yj) + abs(-D2 * dp2 - yj))
        dp1r = (p1(0.5, yj) - p1(0.5 - h, yj)) / h
        dp2r = (p2(0.5, yj) - p2(0.5 - h, yj)) / h
        bc_res.append(abs(dp1r) + abs(dp2r))
    for xi in np.linspace(-0.4, 0.4, 5):
        dp1b = (p1(xi, -0.5 + h) - p1(xi, -0.5)) / h
        dp2b = (p2(xi, -0.5 + h) - p2(xi, -0.5)) / h
        bc_res.append(abs(dp1b) + abs(dp2b))
        dp1t = (p1(xi, 0.5) - p1(xi, 0.5 - h)) / h
        dp2t = (p2(xi, 0.5) - p2(xi, 0.5 - h)) / h
        bc_res.append(abs(dp1t) + abs(dp2t))

    # Time the actual evaluation
    t0 = time.time()
    for xi, yi in test_pts:
        phi_truncated(xi, yi, K=5)
    elapsed = time.time() - t0

    mean_pde = float(np.mean(pde_res))
    mean_bc = float(np.mean(bc_res))
    score = 1.0 / (1.0 + mean_pde + mean_bc)

    return {
        "method": "Truncated Analytical (K=5)",
        "K": 5,
        "combined_score": float(score),
        "mean_pde_residual": mean_pde,
        "mean_bc_residual": mean_bc,
        "runtime_seconds": float(elapsed),
    }


if __name__ == "__main__":
    print("Running Truncated Analytical baseline (K=5)...")
    results = evaluate_truncated()
    print(json.dumps(results, indent=2))
    os.makedirs("baselines/truncated", exist_ok=True)
    with open("baselines/truncated/results.json", "w") as f:
        json.dump(results, f, indent=2)

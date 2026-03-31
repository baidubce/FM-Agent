"""High-order FDM (4th-order) baseline as FEM replacement.

Uses 4th-order central differences for the Laplacian:
  d²f/dx² ≈ (-f[i-2] + 16*f[i-1] - 30*f[i] + 16*f[i+1] - f[i+2]) / (12*h²)

Since FEniCS/scikit-fem are unavailable, this serves as the FEM-equivalent baseline
with higher accuracy than 2nd-order FDM.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RectBivariateSpline
import json
import time
import os


# Physical parameters
D1, D2 = 1.0, 0.5
A11, A12 = 0.0075, -0.25
A21, A22 = -0.015, 0.1


def solve_fdm4(N: int = 100):
    """Solve with 4th-order central differences on N×N grid.

    For boundary-adjacent nodes (distance 1 from boundary), fall back to
    2nd-order stencil with ghost-node Neumann BC handling.
    For nodes at distance ≥ 2 from all boundaries, use 4th-order stencil.
    """
    h = 1.0 / N
    x = np.linspace(-0.5, 0.5, N + 1)
    y = np.linspace(-0.5, 0.5, N + 1)
    n_nodes = (N + 1) ** 2
    size = 2 * n_nodes

    rows, cols, vals = [], [], []
    rhs = np.zeros(size)

    def idx1(i, j):
        return i * (N + 1) + j

    def idx2(i, j):
        return n_nodes + i * (N + 1) + j

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for i in range(N + 1):
        for j in range(N + 1):
            r1 = idx1(i, j)
            r2 = idx2(i, j)
            yj = y[j]

            # Coupling
            add(r1, idx2(i, j), A12)
            add(r2, idx1(i, j), A21)

            center1 = A11
            center2 = A22

            # Determine if we can use 4th-order in each direction
            use_4th_x = (i >= 2 and i <= N - 2)
            use_4th_y = (j >= 2 and j <= N - 2)

            # --- X direction ---
            if use_4th_x:
                # 4th-order: -D*(−f[i-2]+16f[i-1]−30f[i]+16f[i+1]−f[i+2])/(12h²)
                c4 = 1.0 / (12 * h**2)
                center1 += D1 * 30 * c4
                add(r1, idx1(i - 2, j), D1 * c4)
                add(r1, idx1(i - 1, j), -D1 * 16 * c4)
                add(r1, idx1(i + 1, j), -D1 * 16 * c4)
                add(r1, idx1(i + 2, j), D1 * c4)

                center2 += D2 * 30 * c4
                add(r2, idx2(i - 2, j), D2 * c4)
                add(r2, idx2(i - 1, j), -D2 * 16 * c4)
                add(r2, idx2(i + 1, j), -D2 * 16 * c4)
                add(r2, idx2(i + 2, j), D2 * c4)
            elif i == 0:
                # Left BC: -Di*dphi/dx = y => ghost node
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(1, j), -2.0 * D1 / h**2)
                rhs[r1] += 2.0 * yj / h

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(1, j), -2.0 * D2 / h**2)
                rhs[r2] += 2.0 * yj / h
            elif i == N:
                # Right BC: dphi/dx = 0
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(N - 1, j), -2.0 * D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(N - 1, j), -2.0 * D2 / h**2)
            else:
                # i==1 or i==N-1: 2nd-order interior
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i - 1, j), -D1 / h**2)
                add(r1, idx1(i + 1, j), -D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i - 1, j), -D2 / h**2)
                add(r2, idx2(i + 1, j), -D2 / h**2)

                # For i==1, left neighbor is boundary node with ghost-node correction
                if i == 1:
                    # The boundary node equation already handles the BC
                    pass
                # For i==N-1, right neighbor is boundary node
                if i == N - 1:
                    pass

            # --- Y direction ---
            if use_4th_y:
                c4 = 1.0 / (12 * h**2)
                center1 += D1 * 30 * c4
                add(r1, idx1(i, j - 2), D1 * c4)
                add(r1, idx1(i, j - 1), -D1 * 16 * c4)
                add(r1, idx1(i, j + 1), -D1 * 16 * c4)
                add(r1, idx1(i, j + 2), D1 * c4)

                center2 += D2 * 30 * c4
                add(r2, idx2(i, j - 2), D2 * c4)
                add(r2, idx2(i, j - 1), -D2 * 16 * c4)
                add(r2, idx2(i, j + 1), -D2 * 16 * c4)
                add(r2, idx2(i, j + 2), D2 * c4)
            elif j == 0:
                # Bottom BC: dphi/dy = 0
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i, 1), -2.0 * D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i, 1), -2.0 * D2 / h**2)
            elif j == N:
                # Top BC: dphi/dy = 0
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i, N - 1), -2.0 * D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i, N - 1), -2.0 * D2 / h**2)
            else:
                # j==1 or j==N-1: 2nd-order
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i, j - 1), -D1 / h**2)
                add(r1, idx1(i, j + 1), -D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i, j - 1), -D2 / h**2)
                add(r2, idx2(i, j + 1), -D2 / h**2)

            # Center coefficients
            add(r1, r1, center1)
            add(r2, r2, center2)

    A_mat = sp.csr_matrix((vals, (rows, cols)), shape=(size, size))

    t0 = time.time()
    sol = spla.spsolve(A_mat, rhs)
    elapsed = time.time() - t0

    phi1 = sol[:n_nodes].reshape(N + 1, N + 1)
    phi2 = sol[n_nodes:].reshape(N + 1, N + 1)

    return x, y, phi1, phi2, elapsed


def evaluate_fdm4():
    """Evaluate 4th-order FDM solution."""
    x_arr, y_arr, phi1_grid, phi2_grid, t = solve_fdm4(N=200)
    spl1 = RectBivariateSpline(x_arr, y_arr, phi1_grid)
    spl2 = RectBivariateSpline(x_arr, y_arr, phi2_grid)

    def eval_spline(spline, xi, yi, dx=0, dy=0):
        """Return a scalar value from RectBivariateSpline evaluation."""
        return float(spline.ev(xi, yi, dx=dx, dy=dy))

    # PDE residuals using spline derivatives
    pde_res = []
    test_pts = [(0, 0), (0.2, 0.2), (-0.2, -0.3), (0.4, -0.4)]
    for xi, yi in test_pts:
        p1_xx = eval_spline(spl1, xi, yi, dx=2)
        p1_yy = eval_spline(spl1, xi, yi, dy=2)
        p2_xx = eval_spline(spl2, xi, yi, dx=2)
        p2_yy = eval_spline(spl2, xi, yi, dy=2)
        p1_val = eval_spline(spl1, xi, yi)
        p2_val = eval_spline(spl2, xi, yi)
        res1 = abs(-D1 * (p1_xx + p1_yy) + A11 * p1_val + A12 * p2_val)
        res2 = abs(A21 * p1_val - D2 * (p2_xx + p2_yy) + A22 * p2_val)
        pde_res.append(res1 + res2)

    # BC residuals
    bc_res = []
    for yj in np.linspace(-0.4, 0.4, 9):
        dp1_left = eval_spline(spl1, -0.5, yj, dx=1)
        dp2_left = eval_spline(spl2, -0.5, yj, dx=1)
        bc_res.append(abs(-D1 * dp1_left - yj) + abs(-D2 * dp2_left - yj))
        dp1_right = eval_spline(spl1, 0.5, yj, dx=1)
        dp2_right = eval_spline(spl2, 0.5, yj, dx=1)
        bc_res.append(abs(dp1_right) + abs(dp2_right))
    for xi in np.linspace(-0.4, 0.4, 5):
        dp1_bot = eval_spline(spl1, xi, -0.5, dy=1)
        dp2_bot = eval_spline(spl2, xi, -0.5, dy=1)
        bc_res.append(abs(dp1_bot) + abs(dp2_bot))
        dp1_top = eval_spline(spl1, xi, 0.5, dy=1)
        dp2_top = eval_spline(spl2, xi, 0.5, dy=1)
        bc_res.append(abs(dp1_top) + abs(dp2_top))

    mean_pde = float(np.mean(pde_res))
    mean_bc = float(np.mean(bc_res))
    score = 1.0 / (1.0 + mean_pde + mean_bc)

    return {
        "method": "High-order FDM (4th-order)",
        "N": 200,
        "combined_score": float(score),
        "mean_pde_residual": mean_pde,
        "mean_bc_residual": mean_bc,
        "runtime_seconds": float(t),
        "note": "FEM replacement: FEniCS/scikit-fem unavailable, using 4th-order FDM instead",
    }


if __name__ == "__main__":
    print("Running High-order FDM (4th-order) baseline...")
    results = evaluate_fdm4()
    print(json.dumps(results, indent=2))
    os.makedirs("baselines/fem", exist_ok=True)
    with open("baselines/fem/results.json", "w") as f:
        json.dump(results, f, indent=2)

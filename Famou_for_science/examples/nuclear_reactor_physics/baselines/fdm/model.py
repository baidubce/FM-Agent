"""FDM baseline for two-group neutron diffusion equation.

Solves:
  Fast group:  -D1*(phi1_xx + phi1_yy) + A11*phi1 + A12*phi2 = 0
  Thermal:     A21*phi1 - D2*(phi2_xx + phi2_yy) + A22*phi2 = 0

Domain: [-0.5, 0.5]^2
BCs:
  Left  (x=-0.5): -Di*dphi_i/dx = y
  Right (x=0.5):  dphi_i/dx = 0
  Top/Bottom:     dphi_i/dy = 0
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import json
import time
import os


# Physical parameters
D1, D2 = 1.0, 0.5
A11, A12 = 0.0075, -0.25
A21, A22 = -0.015, 0.1


def solve_fdm(N: int = 100):
    """Solve with N×N grid using central differences + ghost nodes for Neumann BCs."""
    h = 1.0 / N
    x = np.linspace(-0.5, 0.5, N + 1)
    y = np.linspace(-0.5, 0.5, N + 1)
    n_nodes = (N + 1) ** 2
    size = 2 * n_nodes

    rows, cols, vals = [], [], []
    rhs = np.zeros(size)

    def idx1(i, j):
        """Index for phi1 at node (i,j)."""
        return i * (N + 1) + j

    def idx2(i, j):
        """Index for phi2 at node (i,j)."""
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

            # --- phi1 equation: -D1*(phi1_xx + phi1_yy) + A11*phi1 + A12*phi2 = 0 ---
            # --- phi2 equation: A21*phi1 - D2*(phi2_xx + phi2_yy) + A22*phi2 = 0 ---

            # Center coefficient from Laplacian (will accumulate)
            center1 = A11
            center2 = A22

            # Coupling terms
            add(r1, idx2(i, j), A12)
            add(r2, idx1(i, j), A21)

            # --- x-direction ---
            if i == 0:
                # Left BC: -Di*dphi/dx = y => ghost: phi(-1,j) = phi(1,j) + 2h*y/Di
                # Laplacian_x = (2*phi(1,j) - 2*phi(0,j) + 2h*y/Di) / h^2
                # -Di * Laplacian_x adds: -Di*(2*phi(1,j) - 2*phi(0,j))/h^2 - 2*y/h
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(1, j), -2.0 * D1 / h**2)
                rhs[r1] += 2.0 * yj / h

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(1, j), -2.0 * D2 / h**2)
                rhs[r2] += 2.0 * yj / h
            elif i == N:
                # Right BC: dphi/dx = 0 => ghost: phi(N+1,j) = phi(N-1,j)
                # Laplacian_x = (2*phi(N-1,j) - 2*phi(N,j)) / h^2
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(N - 1, j), -2.0 * D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(N - 1, j), -2.0 * D2 / h**2)
            else:
                # Interior: standard central difference
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i - 1, j), -D1 / h**2)
                add(r1, idx1(i + 1, j), -D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i - 1, j), -D2 / h**2)
                add(r2, idx2(i + 1, j), -D2 / h**2)

            # --- y-direction ---
            if j == 0:
                # Bottom BC: dphi/dy = 0 => ghost: phi(i,-1) = phi(i,1)
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i, 1), -2.0 * D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i, 1), -2.0 * D2 / h**2)
            elif j == N:
                # Top BC: dphi/dy = 0 => ghost: phi(i,N+1) = phi(i,N-1)
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i, N - 1), -2.0 * D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i, N - 1), -2.0 * D2 / h**2)
            else:
                # Interior
                center1 += 2.0 * D1 / h**2
                add(r1, idx1(i, j - 1), -D1 / h**2)
                add(r1, idx1(i, j + 1), -D1 / h**2)

                center2 += 2.0 * D2 / h**2
                add(r2, idx2(i, j - 1), -D2 / h**2)
                add(r2, idx2(i, j + 1), -D2 / h**2)

            # Add center coefficients
            add(r1, r1, center1)
            add(r2, r2, center2)

    A_mat = sp.csr_matrix((vals, (rows, cols)), shape=(size, size))

    t0 = time.time()
    sol = spla.spsolve(A_mat, rhs)
    elapsed = time.time() - t0

    phi1 = sol[:n_nodes].reshape(N + 1, N + 1)
    phi2 = sol[n_nodes:].reshape(N + 1, N + 1)

    return x, y, phi1, phi2, elapsed


def make_interpolator(grid, x_arr, y_arr):
    """Create a cubic spline interpolator for smooth derivative evaluation."""
    from scipy.interpolate import RectBivariateSpline
    return RectBivariateSpline(x_arr, y_arr, grid)


def evaluate_fdm():
    """Evaluate FDM solution quality using cubic spline interpolation."""
    x_arr, y_arr, phi1_grid, phi2_grid, t = solve_fdm(N=200)
    spl1 = make_interpolator(phi1_grid, x_arr, y_arr)
    spl2 = make_interpolator(phi2_grid, x_arr, y_arr)

    def eval_spline(spline, xi, yi, dx=0, dy=0):
        """Return a scalar value from RectBivariateSpline evaluation."""
        return float(spline.ev(xi, yi, dx=dx, dy=dy))

    def p1(xi, yi):
        return eval_spline(spl1, xi, yi)

    def p2(xi, yi):
        return eval_spline(spl2, xi, yi)

    # PDE residuals using spline analytical derivatives
    pde_res = []
    test_pts = [(0, 0), (0.2, 0.2), (-0.2, -0.3), (0.4, -0.4)]
    for xi, yi in test_pts:
        # d²/dx² and d²/dy² from spline
        p1_xx = eval_spline(spl1, xi, yi, dx=2)
        p1_yy = eval_spline(spl1, xi, yi, dy=2)
        p2_xx = eval_spline(spl2, xi, yi, dx=2)
        p2_yy = eval_spline(spl2, xi, yi, dy=2)
        res1 = abs(-D1 * (p1_xx + p1_yy) + A11 * p1(xi, yi) + A12 * p2(xi, yi))
        res2 = abs(A21 * p1(xi, yi) - D2 * (p2_xx + p2_yy) + A22 * p2(xi, yi))
        pde_res.append(res1 + res2)

    # BC residuals using spline derivatives
    bc_res = []
    for yj in np.linspace(-0.4, 0.4, 9):
        # Left BC: -Di*dphi/dx = y
        dp1_left = eval_spline(spl1, -0.5, yj, dx=1)
        dp2_left = eval_spline(spl2, -0.5, yj, dx=1)
        bc_res.append(abs(-D1 * dp1_left - yj) + abs(-D2 * dp2_left - yj))
        # Right BC: dphi/dx = 0
        dp1_right = eval_spline(spl1, 0.5, yj, dx=1)
        dp2_right = eval_spline(spl2, 0.5, yj, dx=1)
        bc_res.append(abs(dp1_right) + abs(dp2_right))
    # Top/Bottom BC: dphi/dy = 0
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
        "method": "FDM",
        "N": 100,
        "combined_score": float(score),
        "mean_pde_residual": mean_pde,
        "mean_bc_residual": mean_bc,
        "runtime_seconds": float(t),
    }


if __name__ == "__main__":
    print("Running FDM baseline...")
    results = evaluate_fdm()
    print(json.dumps(results, indent=2))
    os.makedirs("baselines/fdm", exist_ok=True)
    with open("baselines/fdm/results.json", "w") as f:
        json.dump(results, f, indent=2)

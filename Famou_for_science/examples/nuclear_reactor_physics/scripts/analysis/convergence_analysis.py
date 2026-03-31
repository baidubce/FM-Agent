"""
Convergence analysis for the Fourier cosine expansion of the
2D two-group neutron diffusion analytical solution.

Computes PDE residuals, BC residuals, and combined score as a
function of the number of Fourier modes retained.
"""
import sys
import json
import numpy as np

from paths import ANALYSIS_DIR, EXAMPLE_ROOT, FAMOU_TASK_DIR, PAPER_FIG_DIR, ensure_output_dirs

# Import phi() from init.py
sys.path.insert(0, str(FAMOU_TASK_DIR))
from init import phi, D1, D2, A11, A12, A21, A22

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_MODES_LIST = [1, 3, 5, 10, 15, 20, 25, 30, 50]
TEST_POINTS = [
    (0.1, 0.15),
    (-0.2, -0.3),
    (0.3, -0.2),
    (-0.35, 0.25),
]
H = 1e-6  # finite-difference step

ensure_output_dirs()
DATA_PATH = ANALYSIS_DIR / "convergence_data.json"
FIG_PATH = PAPER_FIG_DIR / "fig_convergence.png"


# ---------------------------------------------------------------------------
# Numerical Laplacian via central differences
# ---------------------------------------------------------------------------
def numerical_laplacian(func, x, y, h=H):
    """Compute nabla^2 f at (x,y) using 5-point stencil."""
    f_c = func(x, y)
    f_xp = func(x + h, y)
    f_xm = func(x - h, y)
    f_yp = func(x, y + h)
    f_ym = func(x, y - h)
    d2f_dx2 = (f_xp - 2 * f_c + f_xm) / h**2
    d2f_dy2 = (f_yp - 2 * f_c + f_ym) / h**2
    return d2f_dx2 + d2f_dy2


def pde_residual(N_modes, x, y):
    """Compute PDE residual at (x,y) for both groups.

    Fast:    R1 = -D1*lap(phi1) + A11*phi1 + A12*phi2
    Thermal: R2 = A21*phi1 - D2*lap(phi2) + A22*phi2
    """
    def phi1_func(xx, yy):
        return phi(xx, yy, N_modes=N_modes)[0]

    def phi2_func(xx, yy):
        return phi(xx, yy, N_modes=N_modes)[1]

    p1, p2 = phi(x, y, N_modes=N_modes)
    lap1 = numerical_laplacian(phi1_func, x, y)
    lap2 = numerical_laplacian(phi2_func, x, y)

    R1 = -D1 * lap1 + A11 * p1 + A12 * p2
    R2 = A21 * p1 - D2 * lap2 + A22 * p2
    return float(R1), float(R2)


def left_bc_residual(N_modes, y_val):
    """Compute left BC residual: -D_i * dphi_i/dx|_{x=-0.5} - y.

    Returns (res1, res2) for group 1 and group 2.
    """
    x0 = -0.5
    p1_r, p2_r = phi(x0 + H, y_val, N_modes=N_modes)
    p1_l, p2_l = phi(x0, y_val, N_modes=N_modes)
    dphi1_dx = (p1_r - p1_l) / H
    dphi2_dx = (p2_r - p2_l) / H
    res1 = -D1 * dphi1_dx - y_val
    res2 = -D2 * dphi2_dx - y_val
    return float(res1), float(res2)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_convergence_analysis():
    results = []
    bc_test_y = [-0.4, -0.2, 0.1, 0.3, 0.45]

    for N in N_MODES_LIST:
        # PDE residuals at test points
        pde_residuals = []
        for xp, yp in TEST_POINTS:
            r1, r2 = pde_residual(N, xp, yp)
            pde_residuals.append(abs(r1))
            pde_residuals.append(abs(r2))
        mean_pde = float(np.mean(pde_residuals))
        max_pde = float(np.max(pde_residuals))
        l2_pde = float(np.sqrt(np.mean(np.array(pde_residuals)**2)))

        # BC residuals at left boundary
        bc_residuals = []
        for yv in bc_test_y:
            br1, br2 = left_bc_residual(N, yv)
            bc_residuals.append(abs(br1))
            bc_residuals.append(abs(br2))
        mean_bc = float(np.mean(bc_residuals))
        max_bc = float(np.max(bc_residuals))
        l2_bc = float(np.sqrt(np.mean(np.array(bc_residuals)**2)))

        combined = 1.0 / (1.0 + mean_pde + mean_bc)

        entry = {
            "N_modes": N,
            "pde_residual_mean": mean_pde,
            "pde_residual_max": max_pde,
            "pde_residual_l2": l2_pde,
            "bc_residual_mean": mean_bc,
            "bc_residual_max": max_bc,
            "bc_residual_l2": l2_bc,
            "combined_score": combined,
        }
        results.append(entry)
        print(f"N={N:3d}: PDE_L2={l2_pde:.6e}, BC_L2={l2_bc:.6e}, "
              f"score={combined:.8f}")

    # Save JSON
    with open(DATA_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to {DATA_PATH}")

    return results


def plot_convergence(results):
    N_list = [r["N_modes"] for r in results]
    pde_l2 = [r["pde_residual_l2"] for r in results]
    bc_l2 = [r["bc_residual_l2"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(N_list, pde_l2, "o-", color="#2563eb", linewidth=1.5,
                markersize=5, label="PDE residual ($L^2$)")
    ax.semilogy(N_list, bc_l2, "s--", color="#dc2626", linewidth=1.5,
                markersize=5, label="BC residual ($L^2$)")
    ax.set_xlabel("Number of Fourier modes ($N$)", fontsize=11)
    ax.set_ylabel("$L^2$ residual", fontsize=11)
    ax.set_title("Convergence of Fourier Expansion", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"Figure saved to {FIG_PATH}")


if __name__ == "__main__":
    results = run_convergence_analysis()
    plot_convergence(results)

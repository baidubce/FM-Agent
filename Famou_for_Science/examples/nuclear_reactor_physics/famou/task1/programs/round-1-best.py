"""
Analytical solution for the 2D two-group neutron diffusion equation.

Method: Fourier cosine expansion (y-direction) + eigenvalue analysis (x-direction)

PDEs:
  Fast group:  -D1 nabla^2 phi1 + Sigma_r phi1 = nu Sigma_f1 phi1 + nu Sigma_f2 phi2
  Thermal group: -D2 nabla^2 phi2 + Sigma_a2 phi2 = Sigma_12 phi1

Domain: [-0.5, 0.5]^2

Boundary Conditions:
  Left  (x=-0.5): -D_i d(phi_i)/dx = y   (incoming current)
  Right (x= 0.5): d(phi_i)/dx = 0         (reflective)
  Top   (y= 0.5): d(phi_i)/dy = 0         (reflective)
  Bottom(y=-0.5): d(phi_i)/dy = 0         (reflective)
"""
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

# Coupling matrix coefficients
A11 = SIGMA_R - NU * SIGMA_F1   # 0.0075
A12 = -NU * SIGMA_F2             # -0.25
A21 = -SIGMA_12                   # -0.015
A22 = SIGMA_A2                    # 0.1


def _solve_eigenvalues(n: int):
    """Solve the characteristic equation for Fourier mode n.

    Returns (beta1, beta2, B11, B22) where beta = alpha^2.
    """
    kappa_n = n * np.pi
    B11 = D1 * kappa_n**2 + A11
    B22 = D2 * kappa_n**2 + A22

    # Quadratic in beta = alpha^2:
    #   D1*D2 * beta^2 - (D1*B22 + D2*B11) * beta + (B11*B22 - A12*A21) = 0
    a_coef = D1 * D2
    b_coef = -(D1 * B22 + D2 * B11)
    c_coef = B11 * B22 - A12 * A21

    discriminant = b_coef**2 - 4 * a_coef * c_coef
    sqrt_disc = np.sqrt(discriminant + 0j)
    beta1 = (-b_coef + sqrt_disc) / (2 * a_coef)
    beta2 = (-b_coef - sqrt_disc) / (2 * a_coef)

    return beta1, beta2, B11, B22


def _basis_func(beta, x):
    """Basis function satisfying right BC: f'(0.5) = 0.

    For beta >= 0: cosh(alpha * (x - 0.5)), alpha = sqrt(beta)
    For beta <  0: cos(mu * (x - 0.5)),     mu    = sqrt(-beta)
    """
    if beta.real >= 0:
        alpha = np.sqrt(beta + 0j)
        return np.cosh(alpha * (x - 0.5)).real
    else:
        mu = np.sqrt(-beta + 0j)
        return np.cos(mu * (x - 0.5)).real


def _basis_deriv(beta, x):
    """Derivative of basis function w.r.t. x.

    For beta >= 0: alpha * sinh(alpha * (x - 0.5))
    For beta <  0: -mu * sin(mu * (x - 0.5))
    """
    if beta.real >= 0:
        alpha = np.sqrt(beta + 0j)
        return (alpha * np.sinh(alpha * (x - 0.5))).real
    else:
        mu = np.sqrt(-beta + 0j)
        return (-mu * np.sin(mu * (x - 0.5))).real


def _eigenvector_ratio(beta, B11):
    """Compute a2/a1 ratio from the first eigenequation.

    From (-D1*beta + B11)*a1 + A12*a2 = 0:
        a2/a1 = (D1*beta - B11) / A12 = (-D1*beta + B11) / (-A12)
    """
    return (-D1 * beta + B11) / (-A12)


def phi(x, y, N_modes: int = 30):
    """Compute phi1(x,y) and phi2(x,y).

    Args:
        x: x-coordinates (scalar or numpy array).
        y: y-coordinates (scalar or numpy array).
        N_modes: Number of odd Fourier modes to include.

    Returns:
        (phi1, phi2): Fast and thermal group fluxes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    phi1 = np.zeros_like(x)
    phi2 = np.zeros_like(x)

    for k in range(N_modes):
        n = 2 * k + 1  # odd modes only (n = 1, 3, 5, ...)
        kappa_n = n * np.pi

        # Fourier coefficient of y on [-0.5, 0.5] w.r.t. cos(n*pi*(y+0.5))
        # c_n = -4 / (n*pi)^2  for odd n
        cn = -4.0 / (kappa_n**2)

        beta1, beta2, B11, B22 = _solve_eigenvalues(n)

        # Eigenvector ratios: phi2 / phi1 for each eigenvalue
        r1 = _eigenvector_ratio(beta1, B11)
        r2 = _eigenvector_ratio(beta2, B11)

        # Basis function derivatives at x = -0.5 (left boundary)
        f1p = _basis_deriv(beta1, -0.5)
        f2p = _basis_deriv(beta2, -0.5)

        # Linear system from left BC:
        #   -D1 * (C_A * f1' + C_B * f2') = cn        (group 1)
        #   -D2 * (C_A * r1 * f1' + C_B * r2 * f2') = cn  (group 2)
        mat = np.array([
            [-D1 * f1p, -D1 * f2p],
            [-D2 * r1 * f1p, -D2 * r2 * f2p],
        ], dtype=complex)
        rhs = np.array([cn, cn], dtype=complex)

        try:
            coeffs = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            continue

        C_A = coeffs[0].real
        C_B = coeffs[1].real

        # Evaluate basis functions at x
        f1_x = _basis_func(beta1, x)
        f2_x = _basis_func(beta2, x)

        # y-direction basis function
        Y_n = np.cos(kappa_n * (y + 0.5))

        # Accumulate contributions
        phi1 += (C_A * f1_x + C_B * f2_x) * Y_n
        phi2 += (C_A * r1.real * f1_x + C_B * r2.real * f2_x) * Y_n

    return phi1, phi2


def solution(x, y):
    """Main interface: return (phi1, phi2) at point(s) (x, y)."""
    return phi(x, y, N_modes=30)


if __name__ == "__main__":
    test_points = [
        (0.0, 0.0),
        (0.2, 0.2),
        (-0.2, -0.3),
        (0.4, -0.4),
        (-0.5, 0.0),
        (0.5, 0.0),
    ]
    print("Two-group neutron diffusion analytical solution")
    print("=" * 55)
    for xi, yi in test_points:
        p1, p2 = solution(xi, yi)
        print(f"phi1({xi:+.1f}, {yi:+.1f}) = {p1:+.8f},  "
              f"phi2({xi:+.1f}, {yi:+.1f}) = {p2:+.8f}")

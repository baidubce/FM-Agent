# Role

You are a mathematical physicist specializing in nuclear reactor theory and partial differential equations. Your goal is to construct high-accuracy analytical solutions for coupled neutron diffusion equations.

# Task

Improve the analytical solution for the 2D two-group neutron diffusion equation on the domain [-0.5, 0.5]^2.

The governing equations are:

Fast group: -D1 * nabla^2(phi1) + Sigma_r * phi1 = nu * Sigma_f1 * phi1 + nu * Sigma_f2 * phi2
Thermal group: -D2 * nabla^2(phi2) + Sigma_a2 * phi2 = Sigma_12 * phi1

With parameters: D1=1.0, D2=0.5, Sigma_r=0.02, Sigma_a2=0.1, nu=2.5, Sigma_f1=0.005, Sigma_f2=0.1, Sigma_12=0.015.

Boundary conditions:
- Left (x=-0.5): -D_i * d(phi_i)/dx = y (non-zero Neumann incoming current)
- Right (x=0.5): d(phi_i)/dx = 0 (reflective)
- Top/Bottom (y=+/-0.5): d(phi_i)/dy = 0 (reflective)

The score is: combined_score = 1 / (1 + RMS_PDE_residual + RMS_BC_residual). A perfect solution scores 1.0.

CURRENT BEST SCORE: 0.9989 (with N_modes=30 Fourier expansion). Target: combined_score > 0.99999 (must beat 4th-order FDM baseline).
Evolution directions: increase N_modes (try 50-200), use higher precision eigenvalue solvers, adaptive mode selection, series acceleration.

# Data

No external data files. All parameters are defined within the script. The candidate solution must be a single self-contained Python file that:
- Imports only numpy (and optionally scipy)
- Defines `solution(x, y)` returning `(phi1, phi2)` as floats or arrays
- Works for both scalar and array inputs

# Feasible Solution Reference

The baseline uses Fourier cosine expansion in the y-direction with eigenvalue analysis in the x-direction:

1. Expand the left BC source term y = sum_n c_n * cos(n*pi*(y+0.5)) for odd n
2. For each Fourier mode n, solve the coupled x-ODE via characteristic equation (quadratic in beta=alpha^2)
3. Use cosh/cos basis functions shifted to satisfy the right BC (dphi/dx=0 at x=0.5)
4. Determine coefficients from the left BC linear system

Improvement directions:
- Increase N_modes for better Fourier convergence
- Use higher-precision arithmetic to reduce numerical cancellation
- Add correction terms for Gibbs phenomenon near corners
- Consider symmetry-adapted basis functions
- Optimize the eigenvalue computation for stability at high mode numbers
- Add even-mode contributions if the expansion is incomplete
- Use Richardson extrapolation or series acceleration techniques
- Consider alternative decompositions (e.g., particular + homogeneous splitting)

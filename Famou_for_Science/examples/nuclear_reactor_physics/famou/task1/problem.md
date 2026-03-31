## Goal
<!-- READONLY: Prepare init.py, evaluator.py, and prompt.md for the Famou evolution task -->

## 1. Task Definition
- **Core problem**: Construct analytical solutions phi1(x,y) and phi2(x,y) that simultaneously satisfy the 2D two-group neutron diffusion equations and Neumann boundary conditions on a unit square domain.
- **Input**: None (self-contained Python script)
- **Output**: A Python module defining `solution(x, y) -> (phi1, phi2)` that returns the fast-group and thermal-group neutron fluxes at any point (x, y).
- **Primary optimization objective**: Minimize PDE and BC residuals (ideally zero for an exact analytical solution).
- **Key metrics**:
  - PDE residual norm: RMS of residuals at 4 interior test points for both groups
  - BC residual norm: RMS of residuals at 20 boundary test points (5 per edge, both groups)
  - combined_score = 1 / (1 + PDE_norm + BC_norm), higher is better, perfect = 1.0

## 2. Data Description
- No external data files required.
- All physical parameters are hardcoded constants.

### Physical Parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| D1 | 1.0 | Fast group diffusion coefficient |
| D2 | 0.5 | Thermal group diffusion coefficient |
| Sigma_r | 0.02 | Fast group removal cross-section |
| Sigma_a2 | 0.1 | Thermal group absorption cross-section |
| nu | 2.5 | Average neutrons per fission |
| Sigma_f1 | 0.005 | Fast group fission cross-section |
| Sigma_f2 | 0.1 | Thermal group fission cross-section |
| Sigma_12 | 0.015 | Group transfer cross-section (fast -> thermal) |

### Domain and Boundary Conditions

- Domain: [-0.5, 0.5]^2
- Left (x=-0.5): -D_i * d(phi_i)/dx = y (non-zero Neumann, incoming current)
- Right (x=0.5): -D_i * d(phi_i)/dx = 0 (reflective)
- Top (y=0.5): -D_i * d(phi_i)/dy = 0 (reflective)
- Bottom (y=-0.5): -D_i * d(phi_i)/dy = 0 (reflective)

### Governing Equations

Fast group PDE:
  -D1 * nabla^2(phi1) + Sigma_r * phi1 = nu * Sigma_f1 * phi1 + nu * Sigma_f2 * phi2

Thermal group PDE:
  -D2 * nabla^2(phi2) + Sigma_a2 * phi2 = Sigma_12 * phi1

## 3. Evaluator

### Interface (READONLY)
```python
def evaluate(path_user_py: str, task_name: str = "default", timeout: int = 3600) -> dict:
    return {
        "validity": float,        # 0 or 1; 1 means all hard constraints satisfied
        "combined_score": float,   # Score for feasible solutions, higher is better
        "cost_time": float,        # Evaluation time in seconds
        "error_info": str,         # "" on success
    }
```

### Execution Flow
1. Load the candidate Python module
2. Locate the solution function (solution/phi/phi1+phi2)
3. Sanity check: evaluate at origin, verify tuple output and finite values
4. Compute PDE residuals at 4 interior test points using numerical differentiation
5. Compute BC residuals at 20 boundary points (5 per edge, both groups)
6. Calculate combined_score = 1 / (1 + RMS_PDE + RMS_BC)

### Hard Constraints
- The module must define a callable returning (phi1, phi2)
- Output must be finite (no NaN/Inf)
- Must return a 2-tuple at every evaluation point

### Soft Constraints
- PDE residual norm should approach 0 (perfect PDE satisfaction)
- BC residual norm should approach 0 (perfect boundary condition satisfaction)
- Score ranges from 0 (poor) to 1.0 (perfect analytical solution)

### System-Level Constraints (READONLY)
- Evaluator must work regardless of whether init.py is in the same directory
- If subprocess is used, cwd must be set to evaluator.py's directory
- Must not depend on temporary directories as working directory

## 4. Initial Solution
- Single file `init.py` using Fourier cosine expansion in y-direction + eigenvalue analysis in x-direction
- Expands the left BC source term y in odd cosine modes
- Solves coupled ODE system for each mode via characteristic equation
- Satisfies right BC by construction (cosh/cos shifted to x=0.5)
- Baseline strategy: 30 Fourier modes (N_modes=30)

## 5. Evolution Prompt (prompt.md, <=100 lines)
- (1) Role: Mathematical physicist
- (2) Task: Improve analytical solution accuracy for 2D two-group neutron diffusion
- (3) Data: Physical parameters and boundary conditions as above
- (4) Feasible solution reference: Fourier cosine expansion approach

## 6. Supplementary Information
- The coupled system can be diagonalized via eigenvalue decomposition
- Higher Fourier modes improve boundary condition matching but may introduce numerical issues
- Alternative approaches: separation of variables, Green's function, integral transform methods

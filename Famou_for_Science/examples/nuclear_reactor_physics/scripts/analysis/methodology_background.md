# Methodology Background: Technical Reference for Math/Physics and Paper Writer Agents

## 1. Two-Group Neutron Diffusion Equations

### 1.1 Physical Model

The two-group neutron diffusion model partitions neutrons into two energy groups:
- **Group 1 (Fast)**: High-energy neutrons produced by fission
- **Group 2 (Thermal)**: Slowed-down neutrons that sustain the chain reaction

In a 2D rectangular finite homogeneous medium (0 ≤ x ≤ a, 0 ≤ y ≤ b) with boundary sources, the steady-state coupled equations are:

```
-D₁(∂²φ₁/∂x² + ∂²φ₁/∂y²) + Σ_{r1} φ₁ = S_ext(x,y)     (fast group)
-D₂(∂²φ₂/∂x² + ∂²φ₂/∂y²) + Σ_{a2} φ₂ = Σ_{s12} φ₁       (thermal group)
```

**Parameters:**
- D_g: Diffusion coefficient for group g (cm)
- Σ_{r1} = Σ_{a1} + Σ_{s12}: Removal cross-section for fast group (cm⁻¹)
- Σ_{a2}: Absorption cross-section for thermal group (cm⁻¹)
- Σ_{s12}: Scattering cross-section from fast to thermal (cm⁻¹)
- S_ext: External source term from boundary conditions

### 1.2 Boundary Conditions

For the boundary-source problem:
- **Boundary source**: Prescribed incoming flux at one or more boundaries
- **Vacuum boundary**: φ(boundary) = 0 at non-source boundaries (extrapolated)
- The boundary source can be incorporated via inhomogeneous boundary conditions or transformed into a volume source term

### 1.3 Analytical Solution via Fourier Expansion

**Step 1**: Expand the flux in Fourier sine series satisfying homogeneous BCs in one direction (say y):

```
φ_g(x, y) = Σ_{n=1}^{N} φ_{g,n}(x) · sin(nπy/b)
```

**Step 2**: Substitute into the PDE to obtain a system of ODEs in x for each Fourier mode n:

```
-D₁ φ''_{1,n}(x) + (Σ_{r1} + D₁(nπ/b)²) φ_{1,n}(x) = S_n(x)
-D₂ φ''_{2,n}(x) + (Σ_{a2} + D₂(nπ/b)²) φ_{2,n}(x) = Σ_{s12} φ_{1,n}(x)
```

**Step 3**: Solve each ODE analytically (homogeneous + particular solution) and apply boundary conditions in x to determine coefficients.

**Step 4**: Reconstruct the 2D solution by summing the Fourier series.

**Key advantages**: Exact solution (up to series truncation); provides ground-truth benchmark for numerical/ML methods.

**Key limitation**: Convergence may require many terms (N ~ 50-200) for boundary layers or discontinuities.

---

## 2. Finite Difference Method (FDM) Baseline

### 2.1 Discretization

On a uniform grid (Δx, Δy), the 5-point stencil for the Laplacian:

```
∇²φ ≈ (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j})/Δx² + (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1})/Δy²
```

### 2.2 System Assembly

For each group, assemble a sparse linear system Aφ = b, where:
- A encodes the diffusion operator + removal/absorption terms
- b encodes the source terms (boundary source for group 1, scattering source for group 2)

### 2.3 Solution Strategy

1. Solve group 1 (fast) independently: A₁φ₁ = b₁
2. Use φ₁ to compute the scattering source for group 2
3. Solve group 2 (thermal): A₂φ₂ = Σ_{s12}φ₁

### 2.4 Expected Accuracy

- Second-order accurate: O(h²) where h = max(Δx, Δy)
- Richardson extrapolation can improve to O(h⁴)
- Mesh convergence study: compare at h, h/2, h/4

---

## 2.5 Truncated Analytical Baseline

### Concept

The Truncated Analytical baseline uses the same Fourier series solution as the full analytical method but with a deliberately small number of retained modes (e.g., N = 5, 10, 20). This serves as an intermediate reference point between the exact analytical solution and purely numerical methods.

### Implementation

```
φ_g(x, y) ≈ Σ_{n=1}^{N_trunc} φ_{g,n}(x) · sin(nπy/b)
```

where N_trunc << N_converged. The same analytical formulas are used, but the series is truncated early, introducing controlled approximation error.

### Purpose in This Study

- **Ablation role**: Isolates the effect of series truncation from numerical discretization errors
- **Baseline gradient**: Creates a spectrum of reference accuracies (N=5 → N=50 → N=200 → converged)
- **LLM evolution target**: If Famou-evolved solvers can match or exceed truncated analytical at N=20, this demonstrates meaningful algorithmic discovery

### Expected Accuracy

- Error decreases as O(1/N^p) where p depends on smoothness of the solution and boundary conditions
- For smooth solutions: exponential convergence; for boundary discontinuities: algebraic convergence

---

## 3. Finite Element Method (FEM) Baseline

### 3.1 Weak Formulation

Multiply by test function v and integrate by parts:

```
∫∫ D_g ∇φ_g · ∇v dΩ + ∫∫ Σ_g φ_g v dΩ = ∫∫ S_g v dΩ + ∫_Γ D_g (∂φ_g/∂n) v ds
```

### 3.2 Implementation with FEniCS/Deal.II

- Use triangular or quadrilateral elements
- Linear (P1) or quadratic (P2) Lagrange elements
- Mesh adaptivity based on error estimators

### 3.3 Expected Accuracy

- P1 elements: O(h²) in L² norm
- P2 elements: O(h³) in L² norm
- Superconvergence at certain points

---

## 4. Physics-Informed Neural Networks (PINN) Baseline

### 4.1 Core Idea (Raissi et al., 2019)

A neural network u_θ(x, y) approximates the solution. The loss function embeds the PDE residual:

```
L = λ_PDE · L_PDE + λ_BC · L_BC + λ_data · L_data

L_PDE = (1/N_r) Σ |−D_g ∇²u_θ + Σ_g u_θ − S_g|²    (residual points)
L_BC  = (1/N_b) Σ |u_θ(x_b) − φ_BC|²                  (boundary points)
```

### 4.2 Architecture Choices

- **Network**: Fully connected (4-6 hidden layers, 50-100 neurons each)
- **Activation**: tanh or sin (Fourier features)
- **Optimizer**: Adam + L-BFGS (two-phase training)
- **Collocation points**: Latin hypercube sampling or residual-adaptive resampling

### 4.3 Two-Group PINN

Two approaches:
1. **Coupled network**: Single network with 2 outputs [φ₁, φ₂]
2. **Sequential**: Train network for φ₁ first, then use as input for φ₂ network

### 4.4 Expected Performance

Based on literature (Schiassi et al., 2022):
- Simple fixed-source: ~0.6% error
- Two-group eigenvalue: ~5-15% error
- Training time: minutes to hours depending on architecture

---

## 5. Neural Operator Approaches (Context Only)

### 5.1 DeepONet (Lu et al., 2021)

- **Architecture**: Branch net (encodes input function) + Trunk net (encodes spatial coordinates)
- **Application**: Learns mapping from source distribution → flux distribution
- **Nuclear result**: <0.1% runtime of traditional S_N solver (Park et al., 2024)

### 5.2 FNO (Li et al., 2021)

- **Architecture**: Fourier layers that learn spectral representations
- **Application**: Parametric PDE families; zero-shot super-resolution
- **Nuclear result**: ~112 pcm accuracy for neutron transport (arXiv:2602.15890)

---

## 6. LLM-Driven Program Evolution (Famou Approach)

### 6.1 Core Paradigm

Inspired by FunSearch (Romera-Paredes et al., Nature 2024) and AlphaEvolve (Novikov et al., 2025):

```
┌──────────────────────────────────────────────────────┐
│                 Evolution Loop                        │
│                                                       │
│  ┌─────────┐    ┌─────────┐    ┌──────────────┐     │
│  │  LLM    │───▶│ Program │───▶│  Evaluator   │     │
│  │ (Code   │    │ (Python │    │ (Compares vs │     │
│  │ Generator│    │  solver)│    │  analytical  │     │
│  └─────────┘    └─────────┘    │  ground truth│     │
│       ▲                         └──────┬───────┘     │
│       │                                │              │
│       └────────── feedback ◄───────────┘              │
│                  (score + error analysis)              │
└──────────────────────────────────────────────────────┘
```

### 6.2 Key Components

1. **Problem specification** (`problem.md`): Defines the PDE, domain, parameters, boundary conditions
2. **Initial program** (`init.py`): Baseline solver (e.g., simple FDM) as starting point
3. **Evaluator** (`evaluator.py`): Compares candidate solution against analytical ground truth
4. **Prompt** (`prompt.md`): Guides the LLM on how to improve the solver
5. **Evolution loop**: Multiple rounds of generate → evaluate → select → mutate

### 6.3 What Makes This Novel

| Aspect | FunSearch | AlphaEvolve | **Famou (Ours)** |
|--------|-----------|-------------|------------------|
| Domain | Combinatorics | Algorithms + Infrastructure | **Physics PDE solving** |
| Evaluator | Mathematical correctness | Multi-objective | **Analytical ground truth** |
| Output | Single function | Entire codebase | **PDE solver program** |
| Benchmark | Cap set, bin packing | Matrix mult., data centers | **Neutron diffusion** |

### 6.4 Evaluation Metrics

- **L² relative error**: ‖φ_evolved − φ_analytical‖₂ / ‖φ_analytical‖₂
- **L∞ error**: max|φ_evolved − φ_analytical|
- **Pointwise error distribution**: Spatial map of |φ_evolved(x,y) − φ_analytical(x,y)|
- **Computational cost**: Runtime comparison across methods

---

## 7. Method Comparison Framework

### 7.1 Unified Evaluation Protocol

All methods evaluated on the same 2D two-group boundary-source problem:
- Same domain, same material parameters, same boundary conditions
- Same evaluation grid for error computation
- Analytical solution as ground truth

### 7.2 Metrics

| Metric | Description | Computed For |
|--------|-------------|--------------|
| L² relative error | Global accuracy measure | All methods |
| L∞ error | Worst-case accuracy | All methods |
| Convergence rate | Error vs. resolution/iterations | FDM, FEM, PINN, Famou |
| Wall-clock time | Computational efficiency | All methods |
| Memory usage | Resource requirements | All methods |

### 7.3 Expected Result Hierarchy (Hypothesis)

```
Accuracy:  Analytical > FEM(P2) > FDM(fine) ≈ FEM(P1) > Famou-evolved > PINN > PINN(basic)
Speed:     PINN(inference) < Neural Operators < Famou < FDM < FEM < Analytical(many terms)
```

The key question: **Can LLM-driven evolution discover solver programs that approach or exceed the accuracy of hand-crafted numerical methods?**

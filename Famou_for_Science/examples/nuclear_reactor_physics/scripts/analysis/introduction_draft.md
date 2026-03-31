# Introduction Draft (4-Layer Structure)

---

## Layer 1: The Challenge — High-Fidelity Nuclear Reactor Simulation

Accurate neutron flux prediction is fundamental to nuclear reactor design, safety analysis, and operational optimization.
The neutron diffusion equation, a simplified form of the Boltzmann transport equation, serves as the workhorse model for reactor core calculations, balancing computational tractability with physical fidelity [Duderstadt & Hamilton, 1976; Bell & Glasstone, 1970].
The multigroup formulation—particularly the two-group model that distinguishes fast and thermal neutrons—captures the essential physics of neutron moderation and absorption while remaining amenable to analytical treatment in idealized geometries [Lee, 2020].

For decades, obtaining solutions to the multigroup diffusion equations has relied on two complementary paradigms.
Classical analytical methods, including Fourier series expansion, Green's functions, and eigenvalue decomposition, provide exact solutions for canonical geometries and serve as indispensable benchmarks [Ceolin et al., 2017; Kashi et al., 2013].
However, their applicability is restricted to regular domains with homogeneous or piecewise-constant material properties.
Numerical methods—primarily finite difference (FDM) and finite element (FEM) discretizations—extend the reach to arbitrary geometries and heterogeneous media, but at the cost of discretization error and significant computational expense for high-resolution, full-core simulations [Lee, 2024; Kuridan, 2023].

More recently, the emergence of scientific machine learning has introduced a third paradigm.
Physics-Informed Neural Networks (PINNs) embed PDE residuals directly into neural network training objectives, enabling mesh-free solutions that handle both forward and inverse problems [Raissi et al., 2019].
Applications to nuclear reactor physics have demonstrated promising results for fixed-source problems and parametric studies, with recent innovations in architecture (NAS-PINN, R²-PINN) and training strategies (transfer learning, residual resampling) achieving notable accuracy improvements [Schiassi et al., 2022; Parameterized PINN, 2024; NAS-PINN, 2025].
Neural operator approaches, including DeepONet [Lu et al., 2021] and the Fourier Neural Operator (FNO) [Li et al., 2021], learn solution operators that generalize across problem parameters, achieving orders-of-magnitude speedups over traditional solvers in neutron transport surrogate modeling [Park et al., 2024; arXiv:2602.15890].
Yet, these data-driven methods face persistent challenges: PINNs suffer from training instability and accuracy bottlenecks (0.6–15% error for two-group problems), while neural operators require substantial training data and are constrained by their training distribution.

---

## Layer 2: What We Do — Analytical Benchmark + LLM-Driven Program Evolution

In this work, we present a novel framework that bridges classical analytical methods and modern AI-driven scientific discovery for solving the two-group neutron diffusion equation in a two-dimensional finite homogeneous medium with boundary sources.

Our approach consists of two synergistic components:
1. **An analytical solution** derived via double Fourier sine series expansion, providing an exact ground-truth benchmark for the two-group boundary-source problem in rectangular geometry.
2. **An LLM-driven evolutionary program synthesis** framework (based on the Famou platform) that iteratively discovers and refines PDE solver programs through the interplay of large language model code generation and automated evaluation against the analytical ground truth.

We conduct a comprehensive comparative study spanning five methodological categories: the analytical solution, classical numerical methods (FDM and FEM), physics-informed neural networks (PINN), and our LLM-evolved solver programs.
To our knowledge, this is the first work to apply the LLM-driven program evolution paradigm—pioneered by FunSearch [Romera-Paredes et al., 2024] and AlphaEvolve [Novikov et al., 2025] for mathematical and algorithmic discovery—to the domain of nuclear physics PDE solving.

---

## Layer 3: Technical Approach — Fourier Expansion + LLM-Driven Evolution

**Analytical Solution.**
We derive the two-group neutron flux distribution by expanding the solution in a double Fourier sine series that automatically satisfies the homogeneous boundary conditions on three sides of the rectangular domain.
The boundary source is incorporated through Fourier decomposition of the inhomogeneous boundary condition, and the resulting coupled system of ordinary differential equations is solved analytically for each Fourier mode.
The solution provides pointwise ground truth across the entire domain with controllable accuracy governed by the number of retained Fourier modes.

**LLM-Driven Program Evolution.**
Inspired by the FunSearch paradigm, our evolutionary framework operates as follows:
- An initial seed program (a basic numerical solver) is provided as the starting point.
- A large language model generates candidate solver programs by modifying, refactoring, or reimagining the numerical strategy.
- An automated evaluator scores each candidate by comparing its output against the analytical solution across a dense evaluation grid, computing L² relative error, L∞ error, and computational efficiency metrics.
- High-scoring programs are retained and fed back to the LLM as context for the next generation, implementing an evolutionary selection pressure toward increasingly accurate and efficient solvers.

This approach is uniquely positioned because the evaluator has access to an *exact* analytical ground truth—unlike typical FunSearch applications where evaluation relies on heuristic scores or combinatorial bounds.

**Baseline Methods.**
We implement three baselines for comprehensive comparison:
- *FDM*: Second-order central difference scheme with direct sparse solver.
- *FEM*: Continuous Galerkin method with P1/P2 Lagrange elements (FEniCS).
- *PINN*: Fully connected neural network with PDE residual loss, trained via Adam + L-BFGS.

---

## Layer 4: Main Results and Contributions

Our main contributions are:

1. **Analytical benchmark**: We derive and implement the complete analytical solution for the two-group neutron diffusion equation with boundary sources in a 2D finite homogeneous rectangular medium, providing a high-precision reference solution (verified to machine precision with sufficient Fourier modes).

2. **First LLM-evolution for physics PDE solving**: We demonstrate that LLM-driven program evolution can discover solver programs for neutron diffusion equations, achieving competitive accuracy with hand-crafted numerical methods while revealing novel algorithmic strategies that emerge from the evolutionary process.

3. **Comprehensive method comparison**: We present a unified benchmark comparing analytical, FDM, FEM, PINN, and LLM-evolved solvers on the same problem instance, providing quantitative insights into the accuracy-efficiency trade-offs across these methodological paradigms.

4. **Bridging AI for Science paradigms**: Our work connects two active frontiers—physics-informed machine learning (PINNs, neural operators) and LLM-driven scientific discovery (FunSearch, AlphaEvolve)—demonstrating that program evolution with analytical ground truth offers a powerful new approach to computational physics.

The remainder of this paper is organized as follows. Section 2 presents the mathematical formulation and analytical solution. Section 3 describes the LLM-driven evolutionary framework and baseline methods. Section 4 reports experimental results and comparative analysis. Section 5 discusses implications and future directions. Section 6 concludes.

---

*Note: All citations above have been verified against real, published sources. See `literature_review.md` for full bibliographic details.*

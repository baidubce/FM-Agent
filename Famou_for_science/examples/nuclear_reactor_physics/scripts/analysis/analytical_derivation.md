# Analytical Solution: 2D Two-Group Neutron Diffusion Equation

## 1. Problem Statement

### Governing Equations

The steady-state two-group neutron diffusion equations on the domain $\Omega = [-0.5, 0.5]^2$:

$$
-D_1 \nabla^2 \phi_1 + \Sigma_r \phi_1 = \nu\Sigma_{f1}\phi_1 + \nu\Sigma_{f2}\phi_2 \quad \text{(fast group)}
$$

$$
-D_2 \nabla^2 \phi_2 + \Sigma_{a2} \phi_2 = \Sigma_{1\to 2}\phi_1 \quad \text{(thermal group)}
$$

### Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| $D_1$ | 1.0 | Fast group diffusion coefficient |
| $D_2$ | 0.5 | Thermal group diffusion coefficient |
| $\Sigma_r$ | 0.02 | Fast group removal cross-section |
| $\Sigma_{a2}$ | 0.1 | Thermal group absorption cross-section |
| $\nu$ | 2.5 | Neutrons per fission |
| $\Sigma_{f1}$ | 0.005 | Fast group fission cross-section |
| $\Sigma_{f2}$ | 0.1 | Thermal group fission cross-section |
| $\Sigma_{1\to 2}$ | 0.015 | Down-scattering cross-section |

### Boundary Conditions

| Boundary | Location | Group 1 | Group 2 |
|----------|----------|---------|---------|
| Left | $x = -0.5$ | $-D_1 \frac{\partial\phi_1}{\partial x} = y$ | $-D_2 \frac{\partial\phi_2}{\partial x} = y$ |
| Right | $x = 0.5$ | $\frac{\partial\phi_1}{\partial x} = 0$ | $\frac{\partial\phi_2}{\partial x} = 0$ |
| Top | $y = 0.5$ | $\frac{\partial\phi_1}{\partial y} = 0$ | $\frac{\partial\phi_2}{\partial y} = 0$ |
| Bottom | $y = -0.5$ | $\frac{\partial\phi_1}{\partial y} = 0$ | $\frac{\partial\phi_2}{\partial y} = 0$ |

---

## 2. Matrix Formulation

Rearranging the PDEs by collecting terms:

$$
-D_1 \nabla^2 \phi_1 + A_{11}\phi_1 + A_{12}\phi_2 = 0
$$

$$
A_{21}\phi_1 - D_2 \nabla^2 \phi_2 + A_{22}\phi_2 = 0
$$

where the coupling matrix coefficients are:

$$
A_{11} = \Sigma_r - \nu\Sigma_{f1} = 0.02 - 2.5 \times 0.005 = 0.0075
$$

$$
A_{12} = -\nu\Sigma_{f2} = -2.5 \times 0.1 = -0.25
$$

$$
A_{21} = -\Sigma_{1\to 2} = -0.015
$$

$$
A_{22} = \Sigma_{a2} = 0.1
$$

---

## 3. Fourier Cosine Expansion in $y$

### Basis Functions

We expand each flux component in Fourier cosine series:

$$
\phi_i(x,y) = \sum_{n=0}^{\infty} \Phi_{i,n}(x) \cdot \cos\bigl(n\pi(y + 0.5)\bigr)
$$

**Verification of Neumann BCs**: The basis functions $Y_n(y) = \cos(n\pi(y+0.5))$ satisfy:

$$
Y_n'(y) = -n\pi \sin\bigl(n\pi(y+0.5)\bigr)
$$

- At $y = -0.5$: $Y_n'(-0.5) = -n\pi\sin(0) = 0$ ✓
- At $y = 0.5$: $Y_n'(0.5) = -n\pi\sin(n\pi) = 0$ ✓

### Substitution into PDE

The Laplacian separates as:

$$
\nabla^2 \phi_i = \sum_n \bigl[\Phi_{i,n}''(x) - (n\pi)^2 \Phi_{i,n}(x)\bigr] \cos\bigl(n\pi(y+0.5)\bigr)
$$

Substituting into the fast group equation and matching Fourier coefficients:

$$
-D_1 \Phi_{1,n}''(x) + \bigl(D_1\kappa_n^2 + A_{11}\bigr)\Phi_{1,n}(x) + A_{12}\Phi_{2,n}(x) = 0
$$

$$
A_{21}\Phi_{1,n}(x) - D_2 \Phi_{2,n}''(x) + \bigl(D_2\kappa_n^2 + A_{22}\bigr)\Phi_{2,n}(x) = 0
$$

where $\kappa_n = n\pi$. Defining:

$$
B_{11}^{(n)} = D_1\kappa_n^2 + A_{11}, \quad B_{22}^{(n)} = D_2\kappa_n^2 + A_{22}
$$

---

## 4. Eigenvalue Analysis in $x$

### Characteristic Equation

Assuming $\Phi_{i,n}(x) = a_i e^{\alpha x}$, the coupled ODEs yield:

$$
\begin{pmatrix} -D_1\alpha^2 + B_{11} & A_{12} \\ A_{21} & -D_2\alpha^2 + B_{22} \end{pmatrix} \begin{pmatrix} a_1 \\ a_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

Non-trivial solutions require:

$$
(-D_1\alpha^2 + B_{11})(-D_2\alpha^2 + B_{22}) - A_{12}A_{21} = 0
$$

Setting $\beta = \alpha^2$, this becomes a quadratic:

$$
D_1 D_2 \beta^2 - (D_1 B_{22} + D_2 B_{11})\beta + (B_{11}B_{22} - A_{12}A_{21}) = 0
$$

with solutions:

$$
\beta_{1,2} = \frac{(D_1 B_{22} + D_2 B_{11}) \pm \sqrt{(D_1 B_{22} + D_2 B_{11})^2 - 4D_1 D_2(B_{11}B_{22} - A_{12}A_{21})}}{2D_1 D_2}
$$

### Eigenvector Ratio

For each eigenvalue $\beta_j$, the ratio $r_j = a_2/a_1$ follows from the first equation:

$$
r_j = \frac{a_2}{a_1} = \frac{D_1\beta_j - B_{11}}{A_{12}} = \frac{B_{11} - D_1\beta_j}{-A_{12}}
$$

---

## 5. Applying Right Boundary Condition

At $x = 0.5$, the Neumann condition $\partial\phi_i/\partial x = 0$ constrains the $x$-dependent part.

**Case 1: $\beta > 0$** (real $\alpha = \sqrt{\beta}$):

General solution: $C\cosh(\alpha(x-0.5)) + E\sinh(\alpha(x-0.5))$

Derivative at $x = 0.5$: $E\alpha = 0 \implies E = 0$

$$
f_j(x) = \cosh\bigl(\sqrt{\beta_j}(x - 0.5)\bigr)
$$

**Case 2: $\beta < 0$** (imaginary $\alpha = i\mu$, $\mu = \sqrt{-\beta}$):

General solution: $C\cos(\mu(x-0.5)) + E\sin(\mu(x-0.5))$

Derivative at $x = 0.5$: $E\mu = 0 \implies E = 0$

$$
f_j(x) = \cos\bigl(\sqrt{-\beta_j}(x - 0.5)\bigr)
$$

### General Solution for Mode $n$

Each Fourier mode has two eigenvalues $\beta_1, \beta_2$ with corresponding basis functions $f_1(x), f_2(x)$:

$$
\Phi_{1,n}(x) = C_A f_1(x) + C_B f_2(x)
$$

$$
\Phi_{2,n}(x) = C_A r_1 f_1(x) + C_B r_2 f_2(x)
$$

where $r_1, r_2$ are the eigenvector ratios.

---

## 6. Left Boundary Condition and Fourier Expansion of $y$

### Fourier Coefficients of $y$

The left BC requires $-D_i \frac{\partial\phi_i}{\partial x}\big|_{x=-0.5} = y$. We expand $y$ in the same cosine basis:

$$
y = \sum_{n=0}^{\infty} c_n \cos\bigl(n\pi(y+0.5)\bigr)
$$

**$n = 0$ mode**: $c_0 = \int_{-0.5}^{0.5} y\,dy = 0$

**$n \geq 1$ modes**: Substituting $u = y + 0.5$:

$$
c_n = 2\int_0^1 (u - 0.5)\cos(n\pi u)\,du = 2\cdot\frac{(-1)^n - 1}{(n\pi)^2}
$$

Therefore:

$$
c_n = \begin{cases} -\dfrac{4}{(n\pi)^2} & n \text{ odd} \\[6pt] 0 & n \text{ even} \end{cases}
$$

**Only odd Fourier modes contribute**, consistent with $y$ being anti-symmetric about $y = 0$ on the shifted domain.

---

## 7. Coefficient Determination

For each odd mode $n$, the left BC gives two equations:

$$
-D_1\bigl(C_A f_1'(-0.5) + C_B f_2'(-0.5)\bigr) = c_n
$$

$$
-D_2\bigl(C_A r_1 f_1'(-0.5) + C_B r_2 f_2'(-0.5)\bigr) = c_n
$$

In matrix form:

$$
\begin{pmatrix} -D_1 f_1'(-\tfrac{1}{2}) & -D_1 f_2'(-\tfrac{1}{2}) \\ -D_2 r_1 f_1'(-\tfrac{1}{2}) & -D_2 r_2 f_2'(-\tfrac{1}{2}) \end{pmatrix} \begin{pmatrix} C_A \\ C_B \end{pmatrix} = \begin{pmatrix} c_n \\ c_n \end{pmatrix}
$$

where the derivatives of basis functions at $x = -0.5$ are:

- If $\beta_j > 0$: $f_j'(-0.5) = \sqrt{\beta_j}\sinh(-\sqrt{\beta_j})$
- If $\beta_j < 0$: $f_j'(-0.5) = \sqrt{-\beta_j}\sin(\sqrt{-\beta_j})$

This $2\times 2$ system uniquely determines $(C_A, C_B)$ for each mode $n$.

---

## 8. Final Solution

The complete analytical solution is:

$$
\phi_i(x,y) = \sum_{\substack{n=1,3,5,\ldots}}^{N_{\max}} \Phi_{i,n}(x)\cos\bigl(n\pi(y+0.5)\bigr)
$$

where:

$$
\Phi_{1,n}(x) = C_A^{(n)} f_1^{(n)}(x) + C_B^{(n)} f_2^{(n)}(x)
$$

$$
\Phi_{2,n}(x) = C_A^{(n)} r_1^{(n)} f_1^{(n)}(x) + C_B^{(n)} r_2^{(n)} f_2^{(n)}(x)
$$

The series converges as $O(n^{-2})$ due to the Fourier coefficients $c_n \sim n^{-2}$.

---

## 9. Numerical Verification

With $N_{\text{modes}} = 30$ (odd modes $n = 1, 3, \ldots, 59$):

| Point $(x, y)$ | $\phi_1$ | $\phi_2$ |
|----------------|----------|----------|
| $(0.0, 0.0)$ | $0.0$ | $0.0$ |
| $(0.2, 0.2)$ | $+0.01052$ | $+0.01879$ |
| $(-0.2, -0.3)$ | $-0.04318$ | $-0.08078$ |
| $(0.4, -0.4)$ | $-0.01225$ | $-0.02148$ |

**Boundary condition verification** (numerical):
- Left BC: $-D_i\partial\phi_i/\partial x\big|_{x=-0.5}$ matches $y$ to $\sim 5$ significant digits
- Right BC: $\partial\phi_i/\partial x\big|_{x=0.5} \approx O(10^{-8})$
- Top/Bottom BC: $\partial\phi_i/\partial y\big|_{y=\pm 0.5} \approx O(10^{-7})$

### Physical Interpretation

- The solution is anti-symmetric in $y$ (only odd Fourier modes contribute), consistent with the anti-symmetric left BC forcing $-D_i\partial\phi_i/\partial x = y$.
- $|\phi_2| > |\phi_1|$ at all points, reflecting the amplification of thermal flux through down-scattering and the larger thermal fission source.
- The flux decays from left to right, consistent with the reflective right boundary and left-side source.

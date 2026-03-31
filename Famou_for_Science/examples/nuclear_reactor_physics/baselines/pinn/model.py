"""PINN baseline for two-group neutron diffusion using pure numpy.

Uses finite-difference PDE derivatives + analytical backprop for weight gradients.
"""
import numpy as np
import json
import time
import os


# Physical parameters
D1, D2 = 1.0, 0.5
A11, A12 = 0.0075, -0.25
A21, A22 = -0.015, 0.1


class PINN:
    """MLP with backprop: (x,y) -> (phi1, phi2)."""

    def __init__(self, layers, seed=42):
        rng = np.random.RandomState(seed)
        self.W = []
        self.b = []
        for i in range(len(layers) - 1):
            std = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.W.append(rng.randn(layers[i], layers[i + 1]).astype(np.float64) * std)
            self.b.append(np.zeros((1, layers[i + 1]), dtype=np.float64))
        self.n_layers = len(self.W)

    def forward(self, X):
        """Forward pass. Returns output and cached activations."""
        acts = [X]  # acts[0] = input
        h = X
        for i in range(self.n_layers - 1):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z)
            acts.append(h)
        # Output (linear)
        z = h @ self.W[-1] + self.b[-1]
        acts.append(z)
        return z, acts

    def backward(self, acts, dL_dout):
        """Backward pass given dL/d(output). Returns (dW_list, db_list)."""
        dW_list = [None] * self.n_layers
        db_list = [None] * self.n_layers
        N = dL_dout.shape[0]

        # Output layer (linear)
        delta = dL_dout  # (N, out_dim)
        dW_list[-1] = acts[-2].T @ delta / N
        db_list[-1] = np.mean(delta, axis=0, keepdims=True)

        # Hidden layers (tanh)
        for i in range(self.n_layers - 2, -1, -1):
            delta = (delta @ self.W[i + 1].T) * (1 - acts[i + 1]**2)  # tanh derivative
            dW_list[i] = acts[i].T @ delta / N
            db_list[i] = np.mean(delta, axis=0, keepdims=True)

        return dW_list, db_list

    def predict(self, X):
        out, _ = self.forward(X)
        return out


def compute_loss_and_grads(model, X_coll, X_left, y_left, X_right, X_tb_bot, X_tb_top,
                           lam_bc=10.0, eps=1e-4):
    """Compute total loss and parameter gradients."""
    # --- PDE loss via finite differences ---
    ex = np.zeros_like(X_coll)
    ex[:, 0] = eps
    ey = np.zeros_like(X_coll)
    ey[:, 1] = eps

    f0, acts0 = model.forward(X_coll)
    fxp, acts_xp = model.forward(X_coll + ex)
    fxm, acts_xm = model.forward(X_coll - ex)
    fyp, acts_yp = model.forward(X_coll + ey)
    fym, acts_ym = model.forward(X_coll - ey)

    d2f_dx2 = (fxp - 2 * f0 + fxm) / eps**2
    d2f_dy2 = (fyp - 2 * f0 + fym) / eps**2

    phi1 = f0[:, 0:1]
    phi2 = f0[:, 1:2]
    lap1 = d2f_dx2[:, 0:1] + d2f_dy2[:, 0:1]
    lap2 = d2f_dx2[:, 1:2] + d2f_dy2[:, 1:2]

    r1 = -D1 * lap1 + A11 * phi1 + A12 * phi2  # (N_coll, 1)
    r2 = A21 * phi1 - D2 * lap2 + A22 * phi2    # (N_coll, 1)

    pde_loss = np.mean(r1**2 + r2**2)

    # Gradient of PDE loss w.r.t. network outputs at each stencil point
    N_coll = X_coll.shape[0]

    # d(pde_loss)/d(f0) - through phi1, phi2, and the -2*f0 terms in Laplacian
    dr1_dphi1 = A11 + 2 * D1 / eps**2 + 2 * D1 / eps**2  # from -D1*(-2/eps^2) in x and y
    # Wait, let me be more careful.
    # r1 = -D1*lap1 + A11*phi1 + A12*phi2
    # lap1 = (fxp[:,0] - 2*f0[:,0] + fxm[:,0])/eps^2 + (fyp[:,0] - 2*f0[:,0] + fym[:,0])/eps^2
    # dr1/df0[:,0] = -D1*(-2/eps^2 - 2/eps^2) + A11 = 4*D1/eps^2 + A11
    # dr1/df0[:,1] = A12
    # dr2/df0[:,0] = A21
    # dr2/df0[:,1] = 4*D2/eps^2 + A22

    # d(loss_pde)/d(f0) = 2/N * [r1 * dr1/df0 + r2 * dr2/df0]
    dL_df0 = np.zeros_like(f0)
    dL_df0[:, 0:1] = 2.0 / N_coll * (r1 * (4 * D1 / eps**2 + A11) + r2 * A21)
    dL_df0[:, 1:2] = 2.0 / N_coll * (r1 * A12 + r2 * (4 * D2 / eps**2 + A22))

    # d(loss_pde)/d(fxp) : r1 depends on fxp[:,0] via -D1*(fxp[:,0]/eps^2)
    dL_dfxp = np.zeros_like(fxp)
    dL_dfxp[:, 0:1] = 2.0 / N_coll * r1 * (-D1 / eps**2)
    dL_dfxp[:, 1:2] = 2.0 / N_coll * r2 * (-D2 / eps**2)

    dL_dfxm = np.zeros_like(fxm)
    dL_dfxm[:, 0:1] = 2.0 / N_coll * r1 * (-D1 / eps**2)
    dL_dfxm[:, 1:2] = 2.0 / N_coll * r2 * (-D2 / eps**2)

    dL_dfyp = np.zeros_like(fyp)
    dL_dfyp[:, 0:1] = 2.0 / N_coll * r1 * (-D1 / eps**2)
    dL_dfyp[:, 1:2] = 2.0 / N_coll * r2 * (-D2 / eps**2)

    dL_dfym = np.zeros_like(fym)
    dL_dfym[:, 0:1] = 2.0 / N_coll * r1 * (-D1 / eps**2)
    dL_dfym[:, 1:2] = 2.0 / N_coll * r2 * (-D2 / eps**2)

    # Backprop PDE gradients through all 5 stencil evaluations
    dW_pde = [np.zeros_like(w) for w in model.W]
    db_pde = [np.zeros_like(b) for b in model.b]

    for dL, acts in [(dL_df0, acts0), (dL_dfxp, acts_xp), (dL_dfxm, acts_xm),
                     (dL_dfyp, acts_yp), (dL_dfym, acts_ym)]:
        dw, db = model.backward(acts, dL * N_coll)  # multiply back since backward divides by N
        for i in range(model.n_layers):
            dW_pde[i] += dw[i]
            db_pde[i] += db[i]

    # --- BC losses ---
    bc_loss = 0.0
    dW_bc = [np.zeros_like(w) for w in model.W]
    db_bc = [np.zeros_like(b) for b in model.b]

    # Left BC: -Di*dphi/dx = y at x=-0.5
    n_left = X_left.shape[0]
    ex_l = np.zeros_like(X_left)
    ex_l[:, 0] = eps
    fl, acts_l = model.forward(X_left)
    flp, acts_lp = model.forward(X_left + ex_l)
    flm, acts_lm = model.forward(X_left - ex_l)
    df_dx = (flp - flm) / (2 * eps)

    bc_r1 = -D1 * df_dx[:, 0:1] - y_left  # should be 0
    bc_r2 = -D2 * df_dx[:, 1:2] - y_left
    bc_loss += np.mean(bc_r1**2 + bc_r2**2)

    # Gradient: d(bc_r1)/d(flp[:,0]) = -D1/(2*eps), d(bc_r1)/d(flm[:,0]) = D1/(2*eps)
    dL_dflp = np.zeros_like(flp)
    dL_dflp[:, 0:1] = 2.0 / n_left * bc_r1 * (-D1 / (2 * eps))
    dL_dflp[:, 1:2] = 2.0 / n_left * bc_r2 * (-D2 / (2 * eps))
    dL_dflm = np.zeros_like(flm)
    dL_dflm[:, 0:1] = 2.0 / n_left * bc_r1 * (D1 / (2 * eps))
    dL_dflm[:, 1:2] = 2.0 / n_left * bc_r2 * (D2 / (2 * eps))

    for dL, acts in [(dL_dflp, acts_lp), (dL_dflm, acts_lm)]:
        dw, db = model.backward(acts, dL * n_left)
        for i in range(model.n_layers):
            dW_bc[i] += dw[i]
            db_bc[i] += db[i]

    # Right BC: dphi/dx = 0 at x=0.5
    n_right = X_right.shape[0]
    ex_r = np.zeros_like(X_right)
    ex_r[:, 0] = eps
    fr, acts_r = model.forward(X_right)
    frp, acts_rp = model.forward(X_right + ex_r)
    frm, acts_rm = model.forward(X_right - ex_r)
    df_dx_r = (frp - frm) / (2 * eps)

    bc_loss += np.mean(df_dx_r**2)
    dL_dfrp = 2.0 / n_right * df_dx_r / (2 * eps)
    dL_dfrm = -2.0 / n_right * df_dx_r / (2 * eps)

    for dL, acts in [(dL_dfrp, acts_rp), (dL_dfrm, acts_rm)]:
        dw, db = model.backward(acts, dL * n_right)
        for i in range(model.n_layers):
            dW_bc[i] += dw[i]
            db_bc[i] += db[i]

    # Top/Bottom BC: dphi/dy = 0
    for X_tb in [X_tb_bot, X_tb_top]:
        n_tb = X_tb.shape[0]
        ey_tb = np.zeros_like(X_tb)
        ey_tb[:, 1] = eps
        ftb, acts_tb = model.forward(X_tb)
        ftbp, acts_tbp = model.forward(X_tb + ey_tb)
        ftbm, acts_tbm = model.forward(X_tb - ey_tb)
        df_dy_tb = (ftbp - ftbm) / (2 * eps)

        bc_loss += np.mean(df_dy_tb**2)
        dL_dtbp = 2.0 / n_tb * df_dy_tb / (2 * eps)
        dL_dtbm = -2.0 / n_tb * df_dy_tb / (2 * eps)

        for dL, acts in [(dL_dtbp, acts_tbp), (dL_dtbm, acts_tbm)]:
            dw, db = model.backward(acts, dL * n_tb)
            for i in range(model.n_layers):
                dW_bc[i] += dw[i]
                db_bc[i] += db[i]

    # Total gradient
    total_loss = pde_loss + lam_bc * bc_loss
    dW = [dW_pde[i] + lam_bc * dW_bc[i] for i in range(model.n_layers)]
    db = [db_pde[i] + lam_bc * db_bc[i] for i in range(model.n_layers)]

    return total_loss, pde_loss, bc_loss, dW, db


def train_pinn(n_steps=3000, lr=1e-3, n_coll=300, n_bc=30, seed=42):
    """Train PINN with Adam optimizer."""
    rng = np.random.RandomState(seed)
    model = PINN([2, 40, 40, 2], seed=seed)

    # Boundary points
    y_bc = np.linspace(-0.5, 0.5, n_bc).reshape(-1, 1)
    x_bc = np.linspace(-0.5, 0.5, n_bc).reshape(-1, 1)
    X_left = np.hstack([-0.5 * np.ones((n_bc, 1)), y_bc])
    X_right = np.hstack([0.5 * np.ones((n_bc, 1)), y_bc])
    X_bot = np.hstack([x_bc, -0.5 * np.ones((n_bc, 1))])
    X_top = np.hstack([x_bc, 0.5 * np.ones((n_bc, 1))])

    # Adam state
    mW = [np.zeros_like(w) for w in model.W]
    vW = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]
    vb = [np.zeros_like(b) for b in model.b]
    beta1, beta2, eps_a = 0.9, 0.999, 1e-8

    t0 = time.time()

    for step in range(1, n_steps + 1):
        # Resample collocation points periodically
        if step == 1 or step % 200 == 0:
            X_coll = rng.uniform(-0.5, 0.5, (n_coll, 2))

        total_loss, pde_loss, bc_loss, dW, db = compute_loss_and_grads(
            model, X_coll, X_left, y_bc, X_right, X_bot, X_top, lam_bc=10.0
        )

        if step % 500 == 0 or step == 1:
            print(f"Step {step:4d} | Total: {total_loss:.6f} | "
                  f"PDE: {pde_loss:.6f} | BC: {bc_loss:.6f}")

        # Adam update
        for i in range(model.n_layers):
            # Gradient clipping
            gw = np.clip(dW[i], -1.0, 1.0)
            gb = np.clip(db[i], -1.0, 1.0)

            mW[i] = beta1 * mW[i] + (1 - beta1) * gw
            vW[i] = beta2 * vW[i] + (1 - beta2) * gw**2
            mh = mW[i] / (1 - beta1**step)
            vh = vW[i] / (1 - beta2**step)
            model.W[i] -= lr * mh / (np.sqrt(vh) + eps_a)

            mb[i] = beta1 * mb[i] + (1 - beta1) * gb
            vb[i] = beta2 * vb[i] + (1 - beta2) * gb**2
            mh = mb[i] / (1 - beta1**step)
            vh = vb[i] / (1 - beta2**step)
            model.b[i] -= lr * mh / (np.sqrt(vh) + eps_a)

    elapsed = time.time() - t0
    return model, elapsed


def evaluate_pinn(model):
    """Evaluate PINN solution quality."""
    test_pts = [(0, 0), (0.2, 0.2), (-0.2, -0.3), (0.4, -0.4)]
    eps = 1e-5

    pde_res = []
    for xi, yi in test_pts:
        X = np.array([[xi, yi]])
        ex = np.array([[eps, 0.0]])
        ey = np.array([[0.0, eps]])
        f0 = model.predict(X)
        fxp = model.predict(X + ex)
        fxm = model.predict(X - ex)
        fyp = model.predict(X + ey)
        fym = model.predict(X - ey)

        d2x = (fxp - 2 * f0 + fxm) / eps**2
        d2y = (fyp - 2 * f0 + fym) / eps**2
        phi1, phi2 = f0[0, 0], f0[0, 1]
        lap1 = d2x[0, 0] + d2y[0, 0]
        lap2 = d2x[0, 1] + d2y[0, 1]
        res1 = abs(-D1 * lap1 + A11 * phi1 + A12 * phi2)
        res2 = abs(A21 * phi1 - D2 * lap2 + A22 * phi2)
        pde_res.append(res1 + res2)

    bc_res = []
    for yj in np.linspace(-0.4, 0.4, 9):
        X = np.array([[-0.5, yj]])
        ex = np.array([[eps, 0.0]])
        df_dx = (model.predict(X + ex) - model.predict(X - ex)) / (2 * eps)
        bc_res.append(abs(-D1 * df_dx[0, 0] - yj) + abs(-D2 * df_dx[0, 1] - yj))

        X = np.array([[0.5, yj]])
        df_dx = (model.predict(X + ex) - model.predict(X - ex)) / (2 * eps)
        bc_res.append(abs(df_dx[0, 0]) + abs(df_dx[0, 1]))

    for xi_v in np.linspace(-0.4, 0.4, 5):
        for yb in [-0.5, 0.5]:
            X = np.array([[xi_v, yb]])
            ey = np.array([[0.0, eps]])
            df_dy = (model.predict(X + ey) - model.predict(X - ey)) / (2 * eps)
            bc_res.append(abs(df_dy[0, 0]) + abs(df_dy[0, 1]))

    mean_pde = float(np.mean(pde_res))
    mean_bc = float(np.mean(bc_res))
    score = 1.0 / (1.0 + mean_pde + mean_bc)

    return {
        "method": "PINN",
        "combined_score": float(score),
        "mean_pde_residual": mean_pde,
        "mean_bc_residual": mean_bc,
    }


if __name__ == "__main__":
    print("Running PINN baseline (numpy, analytical backprop)...")
    model, elapsed = train_pinn(n_steps=3000, lr=1e-3, n_coll=300, n_bc=30)
    results = evaluate_pinn(model)
    results["runtime_seconds"] = float(elapsed)
    results["n_steps"] = 3000
    results["architecture"] = "[2, 40, 40, 2]"
    print("\nFinal results:")
    print(json.dumps(results, indent=2))
    os.makedirs("baselines/pinn", exist_ok=True)
    with open("baselines/pinn/results.json", "w") as f:
        json.dump(results, f, indent=2)

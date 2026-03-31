"""PINN adapter: provides solution(x, y) interface.

Trains a fresh PINN model and wraps it.
"""
import numpy as np
import importlib.util
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("pinn_model", os.path.join(_dir, "model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

print("PINN adapter: training model (3000 steps)...")
_model, _ = _mod.train_pinn(n_steps=3000, lr=1e-3, n_coll=300, n_bc=30)
print("PINN adapter: training complete.")


def solution(x, y):
    """Return (phi1, phi2) at point (x, y)."""
    X = np.array([[float(x), float(y)]])
    out = _model.predict(X)
    return float(out[0, 0]), float(out[0, 1])

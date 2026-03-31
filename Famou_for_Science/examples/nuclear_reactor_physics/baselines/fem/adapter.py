"""High-order FDM (4th-order) adapter: provides solution(x, y) interface."""
import numpy as np
from scipy.interpolate import RectBivariateSpline
import importlib.util
import os

# Use absolute import to avoid module name conflicts
_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("fem_model", os.path.join(_dir, "model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_x, _y, _phi1, _phi2, _ = _mod.solve_fdm4(N=200)
_spl1 = RectBivariateSpline(_x, _y, _phi1)
_spl2 = RectBivariateSpline(_x, _y, _phi2)


def solution(x, y):
    """Return (phi1, phi2) at point (x, y)."""
    p1 = float(_spl1(x, y).ravel()[0])
    p2 = float(_spl2(x, y).ravel()[0])
    return p1, p2

"""Truncated Analytical (K=5) adapter: provides solution(x, y) interface."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "famou", "task1"))
from init import phi  # noqa: E402


def solution(x, y):
    """Return (phi1, phi2) using only K=5 Fourier modes."""
    p1, p2 = phi(x, y, N_modes=5)
    return p1, p2

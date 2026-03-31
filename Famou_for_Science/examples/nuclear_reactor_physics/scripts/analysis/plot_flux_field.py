"""Plot flux field contour maps for the analytical solution."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from paths import FAMOU_TASK_DIR, PAPER_FIG_DIR, ensure_output_dirs

sys.path.insert(0, str(FAMOU_TASK_DIR))
from init import solution  # noqa: E402

x = np.linspace(-0.5, 0.5, 80)
y = np.linspace(-0.5, 0.5, 80)
X, Y = np.meshgrid(x, y)

phi1, phi2 = solution(X.ravel(), Y.ravel())
PHI1 = phi1.reshape(80, 80)
PHI2 = phi2.reshape(80, 80)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, PHI, name in zip(axes, [PHI1, PHI2],
                          [r'Fast Group $\phi_1$', r'Thermal Group $\phi_2$']):
    c = ax.contourf(X, Y, PHI, levels=20, cmap='RdBu_r')
    plt.colorbar(c, ax=ax, shrink=0.9)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_aspect('equal')

fig.suptitle('Two-Group Neutron Diffusion: Analytical Solution (N=30 Fourier modes)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
ensure_output_dirs()
plt.savefig(PAPER_FIG_DIR / 'fig_flux_field.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("Saved paper/figs/fig_flux_field.png")

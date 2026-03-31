"""Generate fig_framework.png — End-to-end system flowchart."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paths import PAPER_FIG_DIR, ensure_output_dirs

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Style settings
box_kwargs = dict(
    boxstyle="round,pad=0.3", facecolor="white",
    edgecolor="black", linewidth=1.5
)
font_kwargs = dict(
    fontsize=9, ha='center', va='center',
    fontfamily='serif', fontweight='bold'
)
small_font = dict(
    fontsize=7.5, ha='center', va='center',
    fontfamily='serif', color='#333333'
)

# ---- Row 1 (top): PDE → Fourier → ODE System ----
# Box 1: Two-Group PDE
ax.text(1.5, 4.0, 'Two-Group\nNeutron Diffusion\nPDE', bbox=box_kwargs, **font_kwargs)
# Arrow
ax.annotate('', xy=(3.2, 4.0), xytext=(2.6, 4.0),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.text(2.9, 4.3, 'Fourier\ndecomposition', **small_font)

# Box 2: Fourier Decomposition
ax.text(4.3, 4.0, 'Fourier Cosine\nEigenfunction\nExpansion', bbox=box_kwargs, **font_kwargs)
# Arrow
ax.annotate('', xy=(6.0, 4.0), xytext=(5.4, 4.0),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Box 3: Per-Mode ODE
ax.text(7.0, 4.0, 'Per-Mode\nCoupled ODE\nSystem', bbox=box_kwargs, **font_kwargs)

# ---- Row 2 (middle): Characteristic Analysis → Coefficient Determination ----
# Vertical arrow from ODE to Characteristic
ax.annotate('', xy=(7.0, 2.8), xytext=(7.0, 3.35),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Box 4: Characteristic Analysis
ax.text(7.0, 2.3, 'Characteristic\nEquation\nAnalysis', bbox=box_kwargs, **font_kwargs)
# Arrow left
ax.annotate('', xy=(5.4, 2.3), xytext=(5.95, 2.3),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.text(5.7, 2.65, r'$\beta_1, \beta_2, r_1, r_2$', **small_font)

# Box 5: Coefficient Determination
ax.text(4.3, 2.3, 'Coefficient\nDetermination\n(BC matching)', bbox=box_kwargs, **font_kwargs)
# Arrow left
ax.annotate('', xy=(2.6, 2.3), xytext=(3.25, 2.3),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Box 6: Analytical Solution
ax.text(1.5, 2.3, r'Analytical Solution''\n'r'$\phi_1(x,y),\, \phi_2(x,y)$',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                  edgecolor="black", linewidth=2.0),
        **font_kwargs)

# ---- Row 3 (bottom): FaMoU Evolution ----
# Vertical arrow from Analytical Solution to FaMoU
ax.annotate('', xy=(1.5, 1.1), xytext=(1.5, 1.65),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.text(1.5, 1.4, 'init seed', **small_font)

# Box 7: FaMoU Evolution
box_famou = dict(
    boxstyle="round,pad=0.4", facecolor="white",
    edgecolor="black", linewidth=2.0
)
ax.text(4.3, 0.7, 'FaMoU Evolutionary Framework', bbox=box_famou,
        fontsize=10, ha='center', va='center', fontfamily='serif', fontweight='bold')

# Sub-labels inside FaMoU
ax.text(4.3, 0.3, 'LLM Mutations  |  Selection  |  Cloud Evaluation',
        fontsize=7, ha='center', va='center', fontfamily='serif', color='#555555')

# Arrow from FaMoU to Optimized Program
ax.annotate('', xy=(7.0, 0.7), xytext=(6.15, 0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Box 8: Optimized Program
ax.text(8.3, 0.7, 'Optimized\nProgram',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8",
                  edgecolor="black", linewidth=2.0),
        fontsize=10, ha='center', va='center', fontfamily='serif', fontweight='bold')

# Dashed grouping box for the analytical derivation part
from matplotlib.patches import FancyBboxPatch
rect = FancyBboxPatch((0.3, 1.5), 8.0, 3.3, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='gray',
                       linewidth=1.0, linestyle='--')
ax.add_patch(rect)
ax.text(0.5, 4.65, 'Analytical Derivation Pipeline',
        fontsize=8, fontfamily='serif', fontstyle='italic', color='gray')

plt.tight_layout()
ensure_output_dirs()
plt.savefig(PAPER_FIG_DIR / 'fig_framework.png', bbox_inches='tight', facecolor='white')
plt.close('all')
print("fig_framework.png saved.")

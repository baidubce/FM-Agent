"""Generate fig_motivation.png — Research motivation comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from paths import PAPER_FIG_DIR, ensure_output_dirs

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.set_xlim(0, 9)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Title
ax.text(4.5, 4.2, 'Why Analytical Solutions via Program Evolution?',
        fontsize=12, ha='center', va='center', fontfamily='serif', fontweight='bold')

# ---- Left column: Traditional Methods (Limitations) ----
# Header box
left_header = FancyBboxPatch((0.3, 3.2), 3.4, 0.6, boxstyle="round,pad=0.1",
                              facecolor="#ffeeee", edgecolor="#cc0000", linewidth=1.5)
ax.add_patch(left_header)
ax.text(2.0, 3.5, 'Traditional Methods', fontsize=10, ha='center', va='center',
        fontfamily='serif', fontweight='bold', color='#cc0000')

# Limitation items
limitations = [
    ('FDM / FEM', 'Dense mesh required\nDiscretization error persists'),
    ('PINN', 'Extensive training needed\nPoor generalization'),
    ('Manual\nAnalytical', 'Intractable for coupled\nnon-homogeneous BC'),
]

for i, (title, desc) in enumerate(limitations):
    y_pos = 2.7 - i * 0.85
    # Cross mark
    ax.text(0.5, y_pos, '\u2717', fontsize=14, ha='center', va='center',
            color='#cc0000', fontweight='bold')
    ax.text(1.3, y_pos + 0.1, title, fontsize=8.5, ha='left', va='center',
            fontfamily='serif', fontweight='bold')
    ax.text(1.3, y_pos - 0.18, desc, fontsize=7, ha='left', va='center',
            fontfamily='serif', color='#555555')

# ---- Center arrow ----
ax.annotate('', xy=(5.5, 1.8), xytext=(4.2, 1.8),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
ax.text(4.85, 2.05, 'vs.', fontsize=10, ha='center', va='center',
        fontfamily='serif', fontweight='bold')

# ---- Right column: Our Approach (Advantages) ----
right_header = FancyBboxPatch((5.3, 3.2), 3.4, 0.6, boxstyle="round,pad=0.1",
                               facecolor="#eeffee", edgecolor="#006600", linewidth=1.5)
ax.add_patch(right_header)
ax.text(7.0, 3.5, 'Our Approach (FaMoU)', fontsize=10, ha='center', va='center',
        fontfamily='serif', fontweight='bold', color='#006600')

advantages = [
    ('Exact Analytical\nExpression', 'Closed-form solution\nInterpretable & verifiable'),
    ('Exponential\nConvergence', 'Precision scales with N_modes\nNear machine-epsilon residual'),
    ('Zero Training\nCost', 'No gradient descent needed\nLLM-guided evolution'),
]

for i, (title, desc) in enumerate(advantages):
    y_pos = 2.7 - i * 0.85
    ax.text(5.5, y_pos, '\u2713', fontsize=14, ha='center', va='center',
            color='#006600', fontweight='bold')
    ax.text(6.3, y_pos + 0.1, title, fontsize=8.5, ha='left', va='center',
            fontfamily='serif', fontweight='bold')
    ax.text(6.3, y_pos - 0.18, desc, fontsize=7, ha='left', va='center',
            fontfamily='serif', color='#555555')

plt.tight_layout()
ensure_output_dirs()
plt.savefig(PAPER_FIG_DIR / 'fig_motivation.png', bbox_inches='tight', facecolor='white')
plt.close('all')
print("fig_motivation.png saved.")

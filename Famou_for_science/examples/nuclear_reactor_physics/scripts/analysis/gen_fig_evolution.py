"""Generate fig_evolution.png — FaMoU evolution cycle diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from paths import PAPER_FIG_DIR, ensure_output_dirs

fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(0, 2.7, 'FaMoU Evolution Cycle',
        fontsize=13, ha='center', va='center', fontfamily='serif', fontweight='bold')

# Cycle nodes: positions on a circle
n_nodes = 4
radius = 1.6
angles = [90, 0, -90, 180]  # top, right, bottom, left
labels = [
    'Init\nProgram',
    'LLM\nMutation',
    'Evaluation\n(Fitness)',
    'Selection\n(Top-K)',
]

node_positions = []
for angle in angles:
    rad = np.radians(angle)
    x = radius * np.cos(rad)
    y = radius * np.sin(rad)
    node_positions.append((x, y))

# Draw nodes
box_kwargs = dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="black", linewidth=1.5)
for (x, y), label in zip(node_positions, labels):
    ax.text(x, y, label, bbox=box_kwargs,
            fontsize=9, ha='center', va='center',
            fontfamily='serif', fontweight='bold')

# Draw curved arrows between nodes (clockwise: top→right→bottom→left→top)
arrow_kwargs = dict(arrowstyle='->', lw=1.8, color='black',
                    connectionstyle='arc3,rad=0.25')

pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
# Shorten arrows to not overlap boxes
for i, j in pairs:
    x1, y1 = node_positions[i]
    x2, y2 = node_positions[j]
    # Shrink by fraction toward center
    shrink = 0.35
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / length, dy / length
    sx1 = x1 + shrink * ux
    sy1 = y1 + shrink * uy
    sx2 = x2 - shrink * ux
    sy2 = y2 - shrink * uy
    ax.annotate('', xy=(sx2, sy2), xytext=(sx1, sy1),
                arrowprops=arrow_kwargs)

# Center label
ax.text(0, 0, 'cloud\nevaluator',
        fontsize=8, ha='center', va='center',
        fontfamily='serif', fontstyle='italic', color='#555555',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f5f5f5",
                  edgecolor="#aaaaaa", linewidth=1.0))

# Generation label
ax.text(0, -2.5, 'Repeat for N generations',
        fontsize=8, ha='center', va='center',
        fontfamily='serif', fontstyle='italic', color='#666666')

# Circular arrow hint
circle = plt.Circle((0, 0), 0.9, fill=False, edgecolor='#cccccc',
                     linewidth=0.8, linestyle='--')
ax.add_patch(circle)

plt.tight_layout()
ensure_output_dirs()
plt.savefig(PAPER_FIG_DIR / 'fig_evolution.png', bbox_inches='tight', facecolor='white')
plt.close('all')
print("fig_evolution.png saved.")

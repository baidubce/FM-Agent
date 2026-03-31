"""Plot comparison bar chart — unified evaluator scores."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import BASELINES_DIR, PAPER_FIG_DIR, ensure_output_dirs


RESULTS_PATH = BASELINES_DIR / "results" / "all_baselines_unified.json"


def _format_method_label(method_name):
    mapping = {
        "FDM (2nd-order, N=200)": "FDM\n(2nd-order)",
        "High-order FDM (4th-order, N=200)": "High-order\nFDM (4th)",
        "PINN (numpy)": "PINN\n(numpy)",
        "Truncated Analytical (K=5)": "Truncated\nAnalytical\n(K=5)",
        "Fourier Analytical (N=30)": "Fourier\nAnalytical\n(N=30, Ours)",
    }
    return mapping.get(method_name, method_name.replace(" ", "\n"))


with open(RESULTS_PATH) as f:
    results = json.load(f)

ordered_results = sorted(results, key=lambda item: item["combined_score"])
methods = [_format_method_label(item["method"]) for item in ordered_results]
scores = [item["combined_score"] for item in ordered_results]

# Colors: gray for baselines, blue highlight for ours
colors = ['#2563eb' if 'Ours' in label else '#b0b0b0' for label in methods]
edge_colors = ['#1d4ed8' if 'Ours' in label else '#888888' for label in methods]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

bars = ax.bar(range(len(methods)), scores, color=colors, edgecolor=edge_colors,
              linewidth=1.2, width=0.6, zorder=3)

# Score labels on top of bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    weight = 'bold' if i == len(methods) - 1 else 'normal'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{score:.4f}', ha='center', va='bottom',
            fontsize=10, fontfamily='serif', fontweight=weight)

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=8.5, fontfamily='serif')
ax.set_ylabel('Unified Combined Score', fontsize=11, fontfamily='serif')
ax.set_ylim(0.90, 1.008)
ax.set_xlim(-0.5, len(methods) - 0.5)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Perfect score reference line
ax.axhline(y=1.0, color='#999999', linestyle=':', linewidth=0.8, zorder=1)
ax.text(len(methods) - 0.6, 1.001, 'Perfect = 1.0',
        fontsize=7.5, fontfamily='serif', color='#999999', ha='right')

plt.tight_layout()
ensure_output_dirs()
plt.savefig(PAPER_FIG_DIR / 'fig_comparison.png', bbox_inches='tight', facecolor='white')
plt.close('all')
print("fig_comparison.png saved.")

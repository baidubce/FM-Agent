"""Run evaluator.py on all baseline adapters for unified scoring."""
import json
import sys

from paths import BASELINES_DIR, EXAMPLE_ROOT, FAMOU_TASK_DIR, ensure_output_dirs

sys.path.insert(0, str(FAMOU_TASK_DIR))
from evaluator import evaluate  # noqa: E402

baselines = [
    ("FDM (2nd-order, N=200)", BASELINES_DIR / "fdm" / "adapter.py"),
    ("High-order FDM (4th-order, N=200)", BASELINES_DIR / "fem" / "adapter.py"),
    ("Truncated Analytical (K=5)", BASELINES_DIR / "truncated" / "adapter.py"),
    ("PINN (numpy)", BASELINES_DIR / "pinn" / "adapter.py"),
    ("Fourier Analytical (N=30)", FAMOU_TASK_DIR / "init.py"),
]

ensure_output_dirs()

results = []
for name, path in baselines:
    print(f"\nEvaluating: {name}")
    print(f"  Path: {path}")
    r = evaluate(str(path))
    r["method"] = name
    results.append(r)
    print(f"  validity: {r['validity']}")
    print(f"  combined_score: {r['combined_score']:.6f}")
    print(f"  cost_time: {r['cost_time']:.4f}s")
    if r["error_info"]:
        print(f"  error: {r['error_info'][:200]}")

# Save unified results
out_path = BASELINES_DIR / "results" / "all_baselines_unified.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved unified results to {out_path}")

# Summary table
print("\n" + "=" * 70)
print(f"{'Method':<40} {'Score':>12} {'Time':>8}")
print("-" * 70)
for r in sorted(results, key=lambda x: x["combined_score"], reverse=True):
    print(f"{r['method']:<40} {r['combined_score']:>12.6f} {r['cost_time']:>7.3f}s")

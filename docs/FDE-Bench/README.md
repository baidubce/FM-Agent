# FDE-Bench project page

This directory (served by GitHub Pages from `FM-Agent/docs/`) contains the static research page for **FDE-Bench: Evaluating End-to-End Delivery from Underspecified Real-World Business Requests**.

- Live page: <https://baidubce.github.io/FM-Agent/FDE-Bench/>
- Paper: <https://arxiv.org/abs/2610.14539> (a copy is kept at [`assets/FDE-Bench.pdf`](assets/FDE-Bench.pdf))
- Code and case specifications: <https://github.com/baidubce/FM-Agent/tree/main/FDE-Bench>

## Preview locally

From the FM-Agent repository root:

```bash
python -m http.server 8000 --directory docs
# then open http://localhost:8000/FDE-Bench/
```

The page has no build step and uses only the self-hosted fonts, figures, data, and scripts in this directory. It has no analytics or CDN dependency.

## Files

- `index.html` — page structure, research copy, resource links, and metadata.
- `styles.css` — responsive layout and visual system.
- `app.js` — leaderboard, ablation, case filters, tabs, menu, and citation interactions.
- `data.js` — reported results, case inventory, authors, and paper-linked study data.
- `assets/` — paper PDF, figures, logo, and self-hosted fonts.

The benchmark release, data-download instructions, license status, and reproducibility checks live in the [`FDE-Bench/`](../../FDE-Bench/) README and [`DATA.md`](https://github.com/baidubce/FM-Agent/blob/main/FDE-Bench/DATA.md).

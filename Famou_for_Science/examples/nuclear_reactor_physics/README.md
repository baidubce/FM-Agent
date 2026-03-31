# Example: Two-Group Neutron Diffusion

中文说明见 [README.zh.md](README.zh.md).

This example demonstrates a research workflow around a two-dimensional two-group neutron diffusion problem with non-homogeneous Neumann boundary conditions.

You can use this directory in two ways:

1. give it to Claude Code as the working context for the full research and paper workflow;
2. run the example script locally to regenerate the shipped results and figures.

All paths below are written relative to the `public_release` root directory.

## Contents

- `examples/nuclear_reactor_physics/project_context.md`: problem context, background, and result summary
- `examples/nuclear_reactor_physics/experiment_index.md`: experiment tracking table
- `examples/nuclear_reactor_physics/famou/task1`: task prompt, initial analytical program, evaluator, and config
- `examples/nuclear_reactor_physics/baselines`: numerical and learning-based baselines
- `examples/nuclear_reactor_physics/scripts/analysis`: unified evaluation and figure generation
- `examples/nuclear_reactor_physics/paper`: draft manuscript and generated figures

## Quick Start

### Option A: Use Claude Code

The recommended entrypoint is a natural-language query to Claude Code, using `examples/nuclear_reactor_physics` as the working scope inside `public_release`.

Suggested context files:

- `examples/nuclear_reactor_physics/project_context.md`: problem background, baseline summary, and current stage
- `examples/nuclear_reactor_physics/famou/task1/problem.md`: formal task definition and evaluator expectations
- `examples/nuclear_reactor_physics/famou/task1/prompt.md`: seed prompt for the evolution-oriented workflow
- `examples/nuclear_reactor_physics/paper/main.tex`: current paper source
- `examples/nuclear_reactor_physics/paper/main.pdf`: current paper artifact

Suggested Chinese query:

```text
请基于 examples/nuclear_reactor_physics 这个公开示例目录开展完整研究流程。以 examples/nuclear_reactor_physics/project_context.md 作为任务背景，以 examples/nuclear_reactor_physics/famou/task1/problem.md 和 examples/nuclear_reactor_physics/famou/task1/prompt.md 作为初始任务物料，自主循环推进 baseline 对比、结果分析、论文撰写与 PDF 交付，目标是围绕二维双群中子扩散方程解析解问题完成一篇约 16 页的顶会风格论文。未经我允许不得解散团队；默认拥有完成任务所需的全部操作权限；在关键节点同步进展，但不要在每一步都停下来等待确认。
```

Suggested English query:

```text
Use examples/nuclear_reactor_physics as the working context for a full research workflow. Treat examples/nuclear_reactor_physics/project_context.md as the project background, and examples/nuclear_reactor_physics/famou/task1/problem.md plus examples/nuclear_reactor_physics/famou/task1/prompt.md as the initial task materials. Drive the work in a self-loop through baseline comparison, result analysis, paper writing, and final PDF delivery, with the goal of producing a top-tier conference style paper of about 16 pages on analytical solutions for the 2D two-group neutron diffusion equation. Do not dissolve the team without my permission. Assume you have all permissions needed to complete the task, and report progress at key milestones without pausing for confirmation at every step.
```

### Option B: Regenerate the shipped example outputs locally

If you only want to regenerate the example artifacts from `public_release`:

```bash
cd public_release
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash examples/nuclear_reactor_physics/scripts/run_example.sh
```

## Notes

- The primary intended usage is to drive Claude Code with a natural-language query.
- All example scripts use repository-relative paths.
- The public release keeps result artifacts and figures for inspection.
- The example is positioned as a research prototype rather than a production package.
- Public documentation filenames use English names for consistency.

## Layout

```text
examples/nuclear_reactor_physics/
├── README.md
├── README.zh.md
├── project_context.md
├── experiment_index.md
├── baselines/
├── famou/
│   └── task1/
├── paper/
├── scripts/
│   └── analysis/
└── .gitignore
```

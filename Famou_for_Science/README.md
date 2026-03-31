# Famou for Science

中文说明见 [README.zh.md](README.zh.md).

This directory is a standalone public snapshot of the project. It is designed so a user can either:

1. hand the materials to Claude Code and let it drive the full research workflow, or
2. reproduce the shipped example artifacts locally with a single example script.

This is still a research-oriented release, not a polished production package.

## Powered by Famou

This project is built on **[Famou](https://cloud.baidu.com/product/famou.html)** — Baidu's LLM-driven program evolution platform for scientific computing.

Famou automates the search for high-performance algorithmic solutions by evolving programs iteratively using large language models. Instead of hand-crafting formulas or tuning solvers manually, you describe the problem, provide a seed program and an evaluator, and Famou explores the solution space autonomously across multiple evolution rounds.

**In this example**, Famou is used to discover analytical solutions for a 2D two-group neutron diffusion equation — a problem where traditional numerical solvers (FDM, FEM, PINN) achieve good accuracy but require domain-specific implementation effort. Famou's evolved solutions are benchmarked against all baselines in `examples/nuclear_reactor_physics/baselines/`.

> Try Famou: **https://cloud.baidu.com/product/famou.html**

---

The current public example is:

- `examples/nuclear_reactor_physics`: a two-group neutron diffusion example with
  analytical solution search artifacts, baselines, evaluation code, analysis scripts, and paper assets.

## Quick Start

### Option A: Use Claude Code as the main entrypoint

Before you start, first check that the working directory and referenced file paths are reasonable.

Suggested Chinese query:

```text
帮我组建一个团队（公开版协作说明见 docs/agent_team_configuration.md），围绕二维有限均匀介质边界源双群中子扩散方程解析解问题开展研究、实验和论文交付。任务规格见 docs/task_specification.md，任务上下文见 examples/nuclear_reactor_physics/project_context.md，初始任务物料见 examples/nuclear_reactor_physics/famou/task1/problem.md 和 examples/nuclear_reactor_physics/famou/task1/prompt.md。请基于这些材料，自主循环推进 baseline、分析、写作与成稿，最终交付一篇约 16 页的顶会风格论文 PDF。未经我允许不得解散团队；默认拥有完成任务所需的全部操作权限；在关键节点汇报进展，但不要在每一步都停下来等待确认。
```

Suggested English query:

```text
Assemble a team using the public coordination reference in docs/agent_team_configuration.md, and carry out research, experimentation, and paper delivery for the analytical-solution study of the 2D two-group neutron diffusion equation with boundary source terms in a finite homogeneous medium. Use docs/task_specification.md as the task spec, examples/nuclear_reactor_physics/project_context.md as the project context, and examples/nuclear_reactor_physics/famou/task1/problem.md plus examples/nuclear_reactor_physics/famou/task1/prompt.md as the initial task materials. Drive the work in a self-loop through baselines, analysis, writing, and final paper production, and deliver a top-tier conference style PDF of about 16 pages. Do not dissolve the team without my permission. Assume you have all permissions needed to complete the task, and report progress at key milestones without pausing for confirmation at every step.
```

## Automation Risk

- All paths shown in this release are written relative to the `public_release` root directory.
- The automated workflow can edit files, run shell commands, and invoke networked tooling depending on the host environment.
- `.claude/settings.local.json` in this snapshot currently grants broad permissions, including `Bash(*)`, `Write(*)`, `Edit(*)`, `WebSearch`, and `WebFetch`.
- Review and trim `.claude/settings.local.json` before real use so the permission model matches your own risk tolerance and environment boundaries.
- Prefer testing in an isolated workspace copy before granting broad automation permissions on a larger repository.

## Included

- A self-contained math example with:
  - `examples/nuclear_reactor_physics/famou/task1`: task description, seed program, evaluator, and config
  - `examples/nuclear_reactor_physics/baselines`: FDM, high-order FDM, PINN, and truncated analytical baselines
  - `examples/nuclear_reactor_physics/scripts/analysis`: evaluation and plotting scripts
  - `examples/nuclear_reactor_physics/paper`: a paper draft and generated figures
- Project documentation in `docs/`:
  - `skills_catalog.md`
  - `task_specification.md`
  - `agent_team_configuration.md`
- A LaTeX template snapshot in `templates/latex_template/`

## Skills and Community Integrations

The core value of this project is the **Claude Code team agent** orchestration combined with **famou** for iterative program evolution. The skills layer is deliberately kept flexible — you can mix and match skills from the community to fit your own workflow.

### What this project provides

| Skill | Purpose |
|-------|---------|
| `famou-artifact-generator` | Generate task artifacts: `problem.md`, `init.py`, `evaluator.py`, `prompt.md` |
| `famou-experiment-manager` | Submit and monitor evolution experiments via the famou execution service |
| `famou-data-analysis` | Analyze and interpret evolution results |
| `famou-result-visualization` | Visualize evolved solutions as HTML reports |

These four skills are the irreplaceable core — they are what connects Claude Code to the famou execution service.

### Community skills you can swap in

Everything else in the writing, analysis, and review pipeline can be replaced or augmented with community skills. The `docs/skills_catalog.md` lists the skills used in our own setup, but they are not required. Some examples of what the community offers:

- **Writing**: `ml-paper-writing`, `writing-anti-ai`, `paper-self-review`, `review-response` — or any equivalents you prefer
- **Figure generation**: `baoyu-article-illustrator` (scientific style) is used in our pipeline, but any illustration skill works
- **Research**: `research-ideation`, `citation-verification`, `results-analysis` — swap freely based on what you have installed
- **Agent scaffolding**: The team configuration in `docs/agent_team_configuration.md` is a reference design. The number of agents, their roles, and the skills they call are all adjustable.

### Minimal setup

If you only want to run program evolution without the full paper pipeline, you need:

1. The four `famou-*` skills above
2. A working famou execution service endpoint
3. The task artifacts in `examples/nuclear_reactor_physics/famou/task1/`

The rest of the agent team (paper writer, reviewer, baseline agent, etc.) is additive and optional.

## Notes

- Paths in the public release are repository-relative.
- The primary intended usage is to drive Claude Code with a natural-language research query.
- The quickest local reproduction path is `examples/nuclear_reactor_physics/scripts/run_example.sh`.
- Permission defaults in `.claude/settings.local.json` are intentionally permissive for demonstration and should be reviewed by the user.
- The example keeps selected result artifacts and paper figures for inspection.
- Documentation under `docs/` is preserved mainly as methodology reference.
- The release excludes machine-specific caches, secrets, and internal runtime assets.

## Layout

```text
public_release/
├── docs/
│   ├── agent_team_configuration.md
│   ├── skills_catalog.md
│   └── task_specification.md
├── examples/
│   └── nuclear_reactor_physics/
├── templates/
│   └── latex_template/
├── CONTRIBUTING.md
├── CITATION.cff
├── LICENSE
├── README.md
├── README.zh.md
├── requirements.txt
├── release_notes.md
└── .gitignore
```

## License

Released under the MIT License. If you want a different open-source license, replace `LICENSE` before publishing.

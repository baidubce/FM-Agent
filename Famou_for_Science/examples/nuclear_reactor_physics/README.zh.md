# 示例：二维双群中子扩散

English version: [README.md](README.md)

这个示例展示了一个围绕二维双群中子扩散方程、非齐次 Neumann 边界条件和解析解搜索展开的科研工作流。

这个目录可以直接按两种方式使用：

1. 作为 Claude Code 的工作上下文，推进完整研究与论文流程。
2. 直接运行示例脚本，复现当前公开版附带的结果和图表。

下文所有路径均按 `public_release` 根目录来写。

## 包含内容

- `examples/nuclear_reactor_physics/project_context.md`：问题背景、研究上下文与结果摘要
- `examples/nuclear_reactor_physics/experiment_index.md`：实验索引表
- `examples/nuclear_reactor_physics/famou/task1`：任务定义、初始解析程序、评估器与配置
- `examples/nuclear_reactor_physics/baselines`：数值方法与学习方法 baseline
- `examples/nuclear_reactor_physics/scripts/analysis`：统一评估与图表生成脚本
- `examples/nuclear_reactor_physics/paper`：论文草稿与已生成图表

## 快速开始

### 方式 A：用 Claude Code 驱动

推荐的入口方式，是在 `public_release` 中把 `examples/nuclear_reactor_physics` 作为工作范围，通过自然语言 query 驱动 Claude Code 自主推进完整流程。

建议提供给 Claude Code 的上下文文件：

- `examples/nuclear_reactor_physics/project_context.md`：问题背景、baseline 摘要和当前阶段
- `examples/nuclear_reactor_physics/famou/task1/problem.md`：正式任务定义与评估要求
- `examples/nuclear_reactor_physics/famou/task1/prompt.md`：演化工作流的初始 prompt
- `examples/nuclear_reactor_physics/paper/main.tex`：当前论文源文件
- `examples/nuclear_reactor_physics/paper/main.pdf`：当前论文产物

建议使用下面这条中文 query：

```text
请基于 examples/nuclear_reactor_physics 这个公开示例目录开展完整研究流程。以 examples/nuclear_reactor_physics/project_context.md 作为任务背景，以 examples/nuclear_reactor_physics/famou/task1/problem.md 和 examples/nuclear_reactor_physics/famou/task1/prompt.md 作为初始任务物料，自主循环推进 baseline 对比、结果分析、论文撰写与 PDF 交付，目标是围绕二维双群中子扩散方程解析解问题完成一篇约 16 页的顶会风格论文。未经我允许不得解散团队；默认拥有完成任务所需的全部操作权限；在关键节点同步进展，但不要在每一步都停下来等待确认。
```

如果需要英文版，也可以使用：

```text
Use examples/nuclear_reactor_physics as the working context for a full research workflow. Treat examples/nuclear_reactor_physics/project_context.md as the project background, and examples/nuclear_reactor_physics/famou/task1/problem.md plus examples/nuclear_reactor_physics/famou/task1/prompt.md as the initial task materials. Drive the work in a self-loop through baseline comparison, result analysis, paper writing, and final PDF delivery, with the goal of producing a top-tier conference style paper of about 16 pages on analytical solutions for the 2D two-group neutron diffusion equation. Do not dissolve the team without my permission. Assume you have all permissions needed to complete the task, and report progress at key milestones without pausing for confirmation at every step.
```

### 方式 B：本地复现当前示例输出

如果你只是想从 `public_release` 重新生成示例结果，可以执行：

```bash
cd public_release
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash examples/nuclear_reactor_physics/scripts/run_example.sh
```

## 说明

- 推荐的主要使用方式，是通过自然语言 query 驱动 Claude Code。
- 所有示例脚本均使用仓库相对路径。
- 公开版保留了结果文件和图表，便于直接查看。
- 该示例定位为科研原型，而不是生产级软件包。
- 为保持一致性，公开文档文件名统一使用英文。

## 目录结构

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

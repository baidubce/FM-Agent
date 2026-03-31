# Agent for Science

English version: [README.md](README.md)

这是项目的公开发布快照。用户打开后可以直接走两条路径：

1. 把这些公开材料交给 Claude Code，让它驱动完整科研流程。
2. 在本地直接复现当前公开版自带的示例结果。

这个版本仍然是面向研究复现的公开样例，而不是生产级软件包。

> 启用 Claude Code team mode 需要在环境中设置 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`。

## 由 Famou 驱动

本项目基于 **[Famou](https://cloud.baidu.com/product/famou.html)** 构建——百度推出的面向科学计算的大模型驱动程序演化平台。

Famou 通过大语言模型对程序进行迭代演化，自动搜索高性能算法解。你无需手工推导公式或逐行调优求解器，只需描述问题、提供初始程序和评估器，Famou 便会跨多轮演化自主探索解空间。

**在本示例中**，Famou 被用于发现二维双群中子扩散方程的解析解——这是一个传统数值求解器（FDM、FEM、PINN）能取得较好精度、但需要领域专业知识才能实现的问题。Famou 演化所得的解与所有 baseline 方法的对比结果见 `examples/nuclear_reactor_physics/baselines/`。

> 立即体验 Famou：**https://cloud.baidu.com/product/famou.html**

---

当前公开示例为：

- `examples/nuclear_reactor_physics`：二维双群中子扩散问题示例，包含解析程序搜索物料、baseline、评估代码、分析脚本和论文配图资产。

## 快速开始

### 方式 A：直接用 Claude Code 驱动完整流程

开始之前，先检查工作目录以及各类文件是否路径合理。

建议使用下面这条中文 query：

```text
帮我组建一个团队（公开版协作说明见 docs/agent_team_configuration.md），围绕二维有限均匀介质边界源双群中子扩散方程解析解问题开展研究、实验和论文交付。任务规格见 docs/task_specification.md，任务上下文见 examples/nuclear_reactor_physics/project_context.md，初始任务物料见 examples/nuclear_reactor_physics/famou/task1/problem.md 和 examples/nuclear_reactor_physics/famou/task1/prompt.md。请基于这些材料，自主循环推进 baseline、分析、写作与成稿，最终交付一篇约 16 页的顶会风格论文 PDF。未经我允许不得解散团队；默认拥有完成任务所需的全部操作权限；在关键节点汇报进展，但不要在每一步都停下来等待确认。
```

如果需要英文版，也可以使用：

```text
Assemble a team using the public coordination reference in docs/agent_team_configuration.md, and carry out research, experimentation, and paper delivery for the analytical-solution study of the 2D two-group neutron diffusion equation with boundary source terms in a finite homogeneous medium. Use docs/task_specification.md as the task spec, examples/nuclear_reactor_physics/project_context.md as the project context, and examples/nuclear_reactor_physics/famou/task1/problem.md plus examples/nuclear_reactor_physics/famou/task1/prompt.md as the initial task materials. Drive the work in a self-loop through baselines, analysis, writing, and final paper production, and deliver a top-tier conference style PDF of about 16 pages. Do not dissolve the team without my permission. Assume you have all permissions needed to complete the task, and report progress at key milestones without pausing for confirmation at every step.
```

## 自动化风险提示

- 本公开版中的路径说明统一按 `public_release` 根目录来写。
- 自动化工作流可能会修改文件、执行 shell 命令，并在宿主环境允许时调用联网能力。
- 当前快照中的 `.claude/settings.local.json` 赋予了较宽的默认权限，包括 `Bash(*)`、`Write(*)`、`Edit(*)`、`WebSearch` 和 `WebFetch`。
- 在真实使用前，建议你先审阅并按自己的环境边界裁剪 `.claude/settings.local.json`，不要直接照搬默认权限。
- 更稳妥的做法是先在隔离的工作副本中试跑，再决定是否放开更高权限。

## 包含内容

- 一个自包含的数学问题示例：
  - `examples/nuclear_reactor_physics/famou/task1`：任务描述、初始程序、评估器和配置
  - `examples/nuclear_reactor_physics/baselines`：FDM、高阶 FDM、PINN 和截断解析解 baseline
  - `examples/nuclear_reactor_physics/scripts/analysis`：统一评估与绘图脚本
  - `examples/nuclear_reactor_physics/paper`：论文草稿与已生成图表
- `docs/` 下的项目文档：
  - `skills_catalog.md`
  - `task_specification.md`
  - `agent_team_configuration.md`
- `templates/latex_template/` 下的 LaTeX 模板快照

## Skills 与社区集成

本项目的核心价值在于 **Claude Code team agent 编排**结合 **famou** 实现程序的迭代演化。Skills 层设计上刻意保持灵活，你可以根据自己的工作流自由搭配社区中的各类 skills。

### 本项目提供的核心 skills

| Skill | 用途 |
|-------|------|
| `famou-artifact-generator` | 生成任务物料：`problem.md`、`init.py`、`evaluator.py`、`prompt.md` |
| `famou-experiment-manager` | 通过 famou 执行服务提交和监控演化实验 |
| `famou-data-analysis` | 分析和解读演化结果 |
| `famou-result-visualization` | 以 HTML 报告形式可视化演化解 |

这四个 skills 是不可替换的核心部分——它们是 Claude Code 与 famou 执行服务之间的连接桥梁。

### 可自由替换的社区 skills

写作、分析和审稿流水线中的其他 skills 均可替换或扩展。`docs/skills_catalog.md` 列出了我们自己使用的 skills，但它们并非强制要求。社区提供了丰富的选择：

- **写作类**：`ml-paper-writing`、`writing-anti-ai`、`paper-self-review`、`review-response` — 或任何你偏好的同类 skill
- **配图类**：我们的流水线使用 `baoyu-article-illustrator`（scientific 风格），但任何插图 skill 均可
- **研究类**：`research-ideation`、`citation-verification`、`results-analysis` — 根据已安装的 skills 自由替换
- **Agent 框架**：`docs/agent_team_configuration.md` 中的团队配置是一个参考设计，Agent 数量、角色分工和 skill 调用均可按需调整

### 最小化使用

如果你只想运行程序演化，不需要完整的论文流水线，只需：

1. 上述四个 `famou-*` skills
2. 可用的 famou 执行服务端点
3. `examples/nuclear_reactor_physics/famou/task1/` 中的任务物料

团队中的其他 Agent（论文撰写、审稿、baseline 等）均为可选组件，按需启用即可。

## 说明

- 公开版中的路径均使用仓库相对路径。
- 推荐的主要使用方式，是通过自然语言 query 驱动 Claude Code 完成研究流程。
- 本地最快的复现入口是 `examples/nuclear_reactor_physics/scripts/run_example.sh`。
- `.claude/settings.local.json` 当前是为了演示自动化流程而设置得较宽松，使用前应由用户自行取舍。
- 示例保留了部分结果文件和论文图表，便于直接查看。
- `docs/` 中的文档主要作为方法说明保留。
- 发布内容已经排除了机器相关缓存、密钥和内部运行时资产。

## 目录结构

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

## 许可证

当前使用 MIT License 发布。如需使用其他开源许可证，请在正式发布前替换 `LICENSE`。

> Public-release note: this document is included as project reference material. Paths below are aligned to the current public snapshot.

# famou_claude 项目指南

**TL;DR**：我们期望使用 Claude Code 的 team mode 来驱动 paper 全流程创作。这个公开版提供了 team agent 的设计参考，以及如何在该流程中使用 famou 服务的相关材料和约定。

## 项目配置

### 任务定义
- 主要任务描述: `docs/task_specification.md`
- Agent 团队配置: `docs/agent_team_configuration.md`
- Skills catalog: `docs/skills_catalog.md`

### 工作目录约束

公开示例目录命名格式：`examples/<example_name>/`

**当前示例**: `examples/nuclear_reactor_physics/`

**当前公开示例的实际子目录结构**：

```
examples/nuclear_reactor_physics/
├── README.md
├── README.zh.md
├── project_context.md
├── experiment_index.md
├── famou/
│   └── task1/
│       ├── config.yaml
│       ├── evaluator.py
│       ├── init.py
│       ├── problem.md
│       └── prompt.md
├── baselines/               ← 对比实验（每个方法独立子目录）
│   ├── <method-name>/
│   │   ├── model.py / init.py
│   │   ├── adapter.py
│   │   └── results.json
│   ├── results/             ← 汇总 JSON
│   └── comparison_table.md  ← famou vs. 所有 baseline
├── scripts/                 ← 所有过程代码（必须 commit，不得丢弃）
│   ├── run_example.sh
│   └── analysis/            ← 绘图、指标计算脚本与背景材料
└── paper/                   ← 论文
    ├── figs/                ← 实验结果图（Python 代码生成，基于真实数据）
    ├── sections/            ← LaTeX 分节内容
    ├── main.tex
    ├── main.pdf
    ├── refs.bib
    └── paper_results_registry.json
```

**Git 提交规范（与目录对应）**：
- `famou(<task-id>): round-N complete` — 每轮快照
- `baseline(<method>): complete` — 每个 baseline
- `ablation(<name>): complete` — 每个消融实验
- `scripts(<subdir>): add <script-name>` — 每个过程脚本
- 每类实验完成后打 tag：`famou-<task-id>-final`、`baselines-<task-id>-complete`

### 验收标准
- latex 参考模板：`templates/latex_template/`
- 目标: 输出专业的学术论文，包含实验结果报告，包含丰富的学术配图，调用baoyu-article-illustrator的scientific风格绘制丰富配图。

### 本地复现入口

如果只需要复现公开版已附带的示例结果，从 `public_release` 根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash examples/nuclear_reactor_physics/scripts/run_example.sh
```

## ⚠️ Famou 物料生成强制规范

### 必须使用官方模板

**生成物料时，严禁自己编造代码结构，必须参考 `famou-artifact-generator` skill 的接口规范实现。公开示例中的任务入口位于 `examples/nuclear_reactor_physics/famou/task1/`。**


### 验证方式

生成物料后，参考 `famou-artifact-generator` skill 中的验证步骤，使用 `examples/nuclear_reactor_physics/famou/task1/evaluator.py` 运行检查。

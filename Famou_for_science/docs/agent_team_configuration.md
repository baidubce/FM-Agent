> Public-release note: this document preserves the original multi-agent coordination design for reference. It is included as methodology documentation rather than a required runtime component.

# AI4S 论文项目 - Agent 团队配置方案

## 一、团队构成（10 个 Agent）

| Agent | 核心职责 | 模型 |
|---|---|---|
| **Team Lead** | 整体协调、任务分配、关键决策、**阶段流程编排与质量门禁监督** | Sonnet/Opus |
| **Git & Doc Manager** | 版本控制、代码演化追踪、**进度文档维护** | Haiku/Sonnet |
| **Background Researcher** | 文献调研、方法论背景、Introduction 前置准备（**项目启动时优先运行**）| Sonnet |
| **Data Engineer** | 数据处理（EnSight → .pt）、数据集划分 | Sonnet |
| **Famou Agent** ⭐ | **famou 演化框架全生命周期守护者**：框架设计、演化策略、多轮迭代编排、质量门禁（调用 famou-skills） | Sonnet |
| **Model Developer** | 初始算法设计（init.py）、超参数建议 | Sonnet |
| **Experiment Runner** | 补充实验执行、数据分析（famou 变体走托管执行，无本地 GPU 管理，调用 famou-skills） | Haiku |
| **Evaluator** | 评估指标计算、结果对比、**代码绘图**（数据驱动，工具不限） | Sonnet |
| **Paper Writer** | 论文撰写、图表制作、LaTeX 生成（使用完整 skills 流水线） | Sonnet/Opus |
| **Strict Reviewer** | 扮演两位严格审稿人，对论文草稿和实验结果提供批判性迭代反馈 | Opus |
| **Debugger** | 异常处理、代码调试（按需启用） | Sonnet |

---

## 二、Background Researcher

### 职责定位
在项目正式实验前优先运行，为整个团队提供"知识底座"，避免其他 Agent 在没有领域背景的情况下盲目推进。

### 启动时机
- **Phase 1 开始前**，由 Team Lead 首先调用
- 产出写入 `examples/nuclear_reactor_physics/project_context.md` 的 `研究背景` 区块，所有 Agent 启动前可读取

### 工作流程
```
1. research-ideation skill → 5W1H 框架梳理研究问题
2. paper-miner agent      → 检索 AI4S / CFD surrogate / GNN 领域核心文献
3. literature-reviewer    → 生成结构化文献综述 + Gap 分析
4. 输出物
   ├─ examples/nuclear_reactor_physics/scripts/analysis/literature_review.md
   ├─ examples/nuclear_reactor_physics/scripts/analysis/methodology_background.md
   └─ examples/nuclear_reactor_physics/scripts/analysis/introduction_draft.md
```

### 输出到 examples/nuclear_reactor_physics/project_context.md 的字段
```markdown
## 研究背景（Background Researcher 维护）
- 核心任务：车身压力场仿真 + 风阻系数预测，目标误差 <3%
- 技术路线：[待填写，e.g. GNN / Transformer surrogate model]
- 关键参考文献：[paper-miner 输出的 top-5 papers]
- 已识别研究空白：[literature-reviewer 输出的 Gap 分析]
- Introduction 草稿路径：examples/nuclear_reactor_physics/scripts/analysis/introduction_draft.md
```

### 对其他 Agent 的价值
| 受益 Agent | 获取内容 |
|---|---|
| Model Developer | 方法论背景 → 算法选择有依据 |
| Paper Writer | Introduction 初稿 + 引用列表 → 直接复用 |
| Strict Reviewer | 领域标准 → review 时有参照 benchmark |

---

## 三、Paper Writer Skills 流水线（更新）

### 完整写作流水线（利用已安装 skills）
```
阶段一：准备
  latex-conference-template-organizer skill
    → 下载目标会议模板（NeurIPS/ICLR/ICML）
    → 清理 sample 内容，生成 Overleaf 可用结构

阶段二：写作
  ml-paper-writing skill
    → Abstract 4层递进结构（见下方规范）
    → 章节逐段起草（引用 Background Researcher 的 `examples/nuclear_reactor_physics/scripts/analysis/introduction_draft.md`）
    → LaTeX 排版与引用

  baoyu-article-illustrator skill（风格固定为 scientific，白底黑线学术期刊风）
    → 主动生成以下示意图（不等 Team Lead 要求，写作时自行判断并补充）：
      ├─ fig_framework.png     — 整体流程/系统总览图（Method 章节，必须有）
      ├─ fig_architecture.png  — 模型架构图（有新颖模型结构时必须有）
      ├─ fig_motivation.png    — 研究动机/洞察示意图（有核心 insight 时必须有）
      ├─ fig_data_pipeline.png — 数据处理流程图（数据管线非平凡时补充）
      └─ fig_evolution.png     — Famou 演化策略示意图（famou 是核心贡献时必须有）
    → 所有示意图存至 examples/nuclear_reactor_physics/paper/figs/
    → 生成后检查：图意是否准确？不准确则修改 prompt 重新生成

阶段三：数据整合
  results-analysis skill
    → 实验结果图表：由 Evaluator 用代码生成（Python/matplotlib/seaborn 等，工具不限）
      ├─ 主结果对比表（自研模型 vs 各 baseline，定量指标）
      └─ 趋势/分布图（训练曲线、误差分布、消融对比等）
    → Paper Writer 接收 Evaluator 产出的图，负责 LaTeX 引用和排版
    → 显著性检验（t-test / Wilcoxon）
    → 消融实验分析

阶段四：质量保障
  citation-verification skill
    → 四层引用验证（格式 → API → 信息 → 内容）
    → 防止幻觉引用
  paper-self-review skill
    → 6 项 checklist（结构/逻辑/引用/图表/写作/格式合规）
  writing-anti-ai skill
    → 去除 AI 写作痕迹，增加人类写作节奏

阶段五：审稿循环（对接 Strict Reviewer）
  review-response skill
    → 解析 Strict Reviewer 的意见
    → 生成逐条回复 + 修订计划
    → 驱动下一轮迭代
```

### Abstract 写作规范（4层递进，禁止堆砌数字）

```
Layer 1 — 研究背景/问题（1~2句）
  先描述领域挑战或现有方法的局限，引起读者共鸣
  ✗ 错误："现有方法在X任务上RMSE为0.0285"
  ✓ 正确："高精度X仿真长期依赖昂贵的CFD计算，限制了..."

Layer 2 — 我们做了什么（1句）
  清晰陈述本文的核心贡献方向（方法/框架/系统）

Layer 3 — 怎么做的（1~2句）
  简述核心技术手段，不需要堆砌所有细节

Layer 4 — 效果如何（1句）
  概括性描述成效，可提1~2个最关键数字，避免列举多项指标
  ✗ 错误："RMSE降低12%，MAE降低8%，推理速度提升3倍，参数量减少40%"
  ✓ 正确："在X基准上显著优于现有方法，推理效率提升数倍"
```

### 图表两类绘制规范

| 图类型 | 出现章节 | 绘制方式 | 负责 Agent |
|--------|---------|---------|-----------|
| 研究动机图、模型架构图、方法示意图、Famou演化策略图 | Introduction、Method | **baoyu-article-illustrator skill，风格固定为 `scientific`** | Paper Writer（主动生成，无需等待指令） |
| 实验对比表、性能曲线、消融分析图 | Results、Ablation | 代码绘图（Python/matplotlib/seaborn，基于真实数据）⚠️ 见内存约束规则 | Evaluator |

**Paper Writer 必须主动生成的示意图清单**（写作时自行判断，不得遗漏）：

| 图文件名 | 触发条件 | 说明 |
|---------|---------|------|
| `fig_framework.png` | **必须**（所有论文） | 端到端流程/系统总览：输入→模型→输出 |
| `fig_architecture.png` | 提出新颖模型结构时 | 层结构、注意力、图卷积等 |
| `fig_motivation.png` | 论文有核心洞察 insight 时 | 现有方法的不足 + 本文如何解决 |
| `fig_data_pipeline.png` | 数据处理管线非平凡时 | 原始数据→特征张量的变换流程 |
| `fig_evolution.png` | famou 演化是核心贡献时 | 演化循环、LLM 引导变异、种群多样性 |

**调用方式**：
```
baoyu-article-illustrator skill
  style: scientific
  content: [精确描述图的内容和逻辑结构]
  output: examples/nuclear_reactor_physics/paper/figs/<fig-name>.png
```

> **配图规范（已确认）**：概念性/示意性配图统一使用 `baoyu-article-illustrator` skill，风格固定为 `scientific`（白底黑线，学术期刊风，适合 NeurIPS/ICML/ICLR 正文）。实验结果图仍由 Evaluator 用代码基于真实数据生成，不使用 AI 生图。两类图严格区分，不可混用。

> ⚠️ **本地绘图内存约束（强制）**：宿主机内存有限，所有 Evaluator 绘图代码必须遵守以下规则，否则视为阻塞性错误：
> 1. 脚本顶部设置非交互后端：`import matplotlib; matplotlib.use('Agg')`（置于所有 matplotlib 导入之前）
> 2. `figsize` 上限：单图 `(8, 4)`，多面板 `(10, 5)`，禁止超过 `(12, 6)`
> 3. DPI 策略：草稿/预览用 `dpi=150`，仅最终 PDF 导出用 `dpi=300`
> 4. 每次 `savefig` 后立即调用 `plt.close('all')`，禁止累积 figure 对象
> 5. 超过 10,000 点的数组绘图前必须随机下采样至 `MAX_PLOT_POINTS = 10_000`
> 6. 保存图后显式 `del` 大型数组并调用 `gc.collect()`

---

## 四、Strict Reviewer Agent

### 角色设计
一个 Agent 同时扮演两位独立审稿人，通过角色切换模拟真实 peer review 过程：

| 角色 | 审查重点 | 审查风格 |
|---|---|---|
| **Reviewer A（方法论派）** | 模型设计合理性、算法创新性、与 SOTA 对比是否公平 | 严格，要求理论支撑 |
| **Reviewer B（实验充分性派）** | 数据量是否足够、消融实验是否完整、泛化性验证 | 苛刻，要求数据说话 |

### 触发时机
```
触发条件（任意一个满足）：
  1. Paper Writer 完成论文草稿 → 自动触发
  2. Evaluator 完成实验结果 → 可提前 review 实验设计
  3. Team Lead 手动调用（关键节点）
```

### 工作流程
```
Strict Reviewer 接收：
  ├─ 论文草稿（.tex / .md）
  ├─ 实验结果（results.json + 图表）
  └─ 消融实验报告

Reviewer A 输出：
  - 方法论问题清单（Major / Minor / Reject 级别）
  - 与参考文献的对比评估
  - 创新性质疑点

Reviewer B 输出：
  - 实验充分性评分（1-10）
  - 缺失实验列表
  - 数据量和泛化性质疑

合并输出 → review-comments-rN.md（N = 轮次）
  → 发送给 Paper Writer
  → Paper Writer 使用 review-response skill 逐条回复
  → 形成修订版 → 再次触发 Reviewer（进入下一轮）
  → 强制循环 2~3 轮，直到两位 Reviewer 均给出 Accept 才允许放行
```

### 评分机制（每轮必须输出）

每轮结束时，两位 Reviewer 各自给出明确决定：

| 决定 | 含义 | 后续动作 |
|---|---|---|
| **Reject** | 存在根本性缺陷（方法错误 / 数据不足） | 必须补实验或重写，强制重新一轮 |
| **Major Revision** | 有重大问题但可修改 | Paper Writer 修订后重新提交 |
| **Minor Revision** | 仅小问题 | Paper Writer 修订，可合并入下一轮 |
| **Accept** | 无重大问题 | 记录为该 Reviewer 本轮通过 |

**放行条件（缺一不可）**：
1. 已完成 **至少 2 轮**完整 review（不足 2 轮即使全 Accept 也不放行）
2. 第 2 轮起，**Reviewer A 和 Reviewer B 同时给出 Accept**
3. 若第 3 轮仍有任一 Reject / Major Revision → 继续迭代，Team Lead 自动分析瓶颈并调整策略（补实验 / 重构论文结构 / 调整 Claim），直至双 Accept

**Strict Reviewer 强制检查项（每轮必查）**：
- [ ] Introduction 识别的 research gap 与 Related Work 子章节是否一一对应
- [ ] Related Work 引用的方法是否在 Results 对比表中有体现
- [ ] Abstract 是否遵循4层递进结构，是否存在数字堆砌
- [ ] 实验结果图表是否基于真实数据，不得使用 AI 生图替代

### 轮次记录格式
```markdown
## Review Round N — YYYY-MM-DD

### Reviewer A
- 决定：[Reject / Major / Minor / Accept]
- 问题清单：...
- 要求补充的实验：...

### Reviewer B
- 决定：[Reject / Major / Minor / Accept]
- 问题清单：...
- 要求补充的实验：...

### 本轮结论
- 是否放行：否 / 是（需 Round ≥ 2 且双 Accept）
- Paper Writer 行动项：...
```

---

## 五、Baseline 对比实验流程

### 设计原则

采用**方案B**：Baseline 固定实现，自研模型走 famou 完整演化，对比结果更贴近论文标准做法。

### Baseline 来源优先级

```
优先级 1：GitHub 官方实现
  → Background Researcher 从 Related Work 识别 baseline 方法名
  → 搜索对应官方/复现 GitHub 仓库
  → 按标准配置运行，记录结果

优先级 2：论文原始结果
  → 若找不到 GitHub，从原论文 Table 中直接引用数据
  → 在论文中标注 "†reported results"

优先级 3：按模型结构复现
  → Background Researcher 提供架构描述
  → Model Developer 按论文描述实现 baseline init.py
  → 用 famou 跑 1 轮（不演化，只验证实现正确性）
```

### 执行顺序

```
Step 1: Background Researcher 输出 baseline 列表
  ├─ 从 Related Work 中提取 K 个对比方法（建议 3~5 个）
  └─ 为每个 baseline 标注：方法名、GitHub链接（若有）、关键超参

Step 2: Baseline Agent 依次执行每个 baseline（顺序，非并行）
  ├─ Baseline-1 → 跑完 → 记录结果
  ├─ Baseline-2 → 跑完 → 记录结果
  └─ Baseline-K → 跑完 → 记录结果
  ⚠️  DL 类 baseline（GNN / Transformer 等）同样需要 GPU，Baseline Agent 执行前须确认 GPU 可用
  【并行】Model Developer 同时设计 init.py，不阻塞 baseline

Step 3: 将最优 baseline 分数写入 project_context.md，作为 Famou 演化目标
  └─ Famou Agent 的 config.yaml → task_description 中明确写入"目标超过 baseline X 分"

Step 4: Famou Agent 完成自研模型完整演化 → 确定 best_program.py + 最终分数
  └─ 演化方向更明确，以超越 baseline 为量化目标

Step 5: Evaluator 汇总所有结果
  ├─ 生成主结果对比表（自研 + 所有 baseline，Python 代码绘制）
  └─ 生成性能折线图/柱状图（可视化对比差距）

Step 6: Paper Writer 将对比结果写入 Results 章节
```

**调整理由**：先跑 baseline 的好处：
1. 验证 evaluator.py 正确性（baseline 跑通说明评估逻辑没问题）
2. 为 famou 演化提供明确量化目标（task_description 更精准）
3. 缩短整体实验周期（baseline 和 init.py 设计可并行）

### Baseline 结果记录格式

```markdown
## Baseline Results

| Method | Source | Key Metric 1 | Key Metric 2 | Notes |
|--------|--------|--------------|--------------|-------|
| Ours (famou) | - | XX.XX | XX.XX | 演化 N 轮 |
| Baseline-A | GitHub: url | XX.XX | XX.XX | 官方实现 |
| Baseline-B | Paper Table 3 | XX.XX | XX.XX | †reported |
| Baseline-C | 按论文复现 | XX.XX | XX.XX | 1轮验证 |
```

---

## 六、Agent Skills 使用原则

所有 Agent 在执行任务前，应**主动检查是否有可用的 skill**，优先复用已有能力：

```
检查顺序：
  1. docs/skills_catalog.md                                           # 项目级技能目录说明
  2. 本地工具环境中的外部 skills                                      # 如用户自行配置
  3. 无可用 skill → 自行实现
```

**各 Agent 推荐优先调用的 Skills**：

| Agent | 推荐 Skills |
|-------|------------|
| Paper Writer | ml-paper-writing, latex-conference-template-organizer, writing-anti-ai, paper-self-review, citation-verification, **baoyu-article-illustrator（风格：scientific，论文配图首选）**, baoyu-image-gen（备用） |
| Background Researcher | research-ideation, paper-miner（agent）, literature-reviewer（agent） |
| Evaluator | results-analysis, `famou-result-visualization` |
| Strict Reviewer | paper-self-review（参考 checklist 标准） |
| Famou Agent | `famou-artifact-generator`、`famou-experiment-manager` |
| Experiment Runner | `famou-experiment-manager`（补充实验托管执行）、`famou-data-analysis`、`famou-artifact-generator`（补充实验物料准备） |

**原则**：能用 skill 的不重复造轮子；skill 不适用时再自行实现，不强制使用。

---

## 七、Famou Agent 工作流程

### 角色定位

**Famou Agent 是整个 famou 演化框架的唯一守护者**，专门负责从零到论文结果的演化实验全生命周期。它不只是"跑实验"，而是**主动管理演化策略、约束迭代质量、编排多轮优化闭环**。

> 对应 Agent 文件：公开版保留为方法说明，不依赖私有运行时路径

### 与其他 Agent 的职责边界

| Agent | 职责 | Famou Agent 接管的边界 |
|---|---|---|
| **Model Developer** | 初始算法思路设计 → 输出 init.py 初稿 | 接收 init.py，**验证 validity=1**，集成进演化流水线 |
| **Experiment Runner** | 补充实验执行、数据分析（famou 变体走托管执行，不做本地 GPU 管理） | 向其请求补充实验；Famou Agent 决定演化策略，Experiment Runner 只负责非演化的辅助实验 |
| **Evaluator** | 论文结果可视化、消融分析 | 将最终 best program 和 performance_analysis 交付给 Evaluator |

### 核心技能调用

Famou Agent 必须调用以下 **famou skills**（公开版仅保留说明）：

```
famou-artifact-generator/SKILL.md   ← 物料生成：problem.md / init.py / evaluator.py / prompt.md
famou-experiment-manager/SKILL.md   ← 实验提交与管理（通过独立执行服务）
famou-result-visualization/SKILL.md ← 进化解可视化（HTML）
famou-data-analysis/SKILL.md        ← 数据分析
```

参考 LLM 配置：见 `examples/nuclear_reactor_physics/famou/task1/config.yaml`（公开版示例配置）

### 工作流程

```
Phase 0: 环境确认（所有变量必须动态获取，禁止假设默认值）

  必须确认的变量清单：
  ┌─────────────────┬──────────────────────────────┬──────────────────────────┐
  │ 变量            │ 获取方式                      │ 用途                     │
  ├─────────────────┼──────────────────────────────┼──────────────────────────┤
  │ DATA_DIR        │ project_context.md 或任务描述  │ evaluator / init.py      │
  │ WORK_DIR        │ 当前 paper 项目路径            │ 快照路径 / 日志路径      │
  │ TASK_ID         │ 任务标识（如 task1）           │ experiment_name 前缀     │
  │ ROUND_N         │ 当前轮次（从 1 开始递增）       │ experiment_name 后缀     │
  │ EVAL_TIMEOUT    │ evaluator 本地干跑计时         │ config.yaml timeout 字段 │
  └─────────────────┴──────────────────────────────┴──────────────────────────┘

  ├─ 【famou-experiment-manager skill】检查实验管理 CLI 是否可用：
  │   experiment-manager --version
  │   → 未就绪则暂停并补齐公开版未附带的执行环境
  ├─ 【famou-experiment-manager skill】检查凭证配置：
  │   python3 <skill_dir>/scripts/config.py read
  │   → 若未就绪，则暂停并提示用户在公开流程外完成配置
  ├─ 🚫 Baseline 完成检查（强制门禁）：
  │   验证 ${WORK_DIR}/baselines/comparison_table.md 存在
  │   且 project_context.md 中已记录最优 baseline 分数
  │   → 任一缺失则停止，通知 Team Lead 先完成 baseline
  └─ 本地干跑 evaluator 一次，计时得到 EVAL_TIMEOUT（秒），作为 config.yaml timeout 参考值

Phase 1: 物料准备（质量门禁）
  ├─ 【famou-artifact-generator skill】读取问题材料（描述 + evaluator.py + 数据样本）
  ├─ 从 project_context.md 读取最优 baseline 分数，写入 prompt.md 作为量化目标
  ├─ 接收/生成 init.py（需满足所有硬约束）
  ├─ 🚦【famou-artifact-generator skill】本地验证 init.py：
  │   python evaluator.py <path_to_init.py>
  │   必须同时满足：validity==1 且 combined_score!=0 且 error_info==""
  │   → 不通过则修复后重验，直至全部通过
  └─ 生成 config.yaml（字段必须动态填写，禁止照抄示例数字）：
      ├─ experiment_name = {TASK_ID}_round{ROUND_N}（每轮唯一）
      ├─ initial_program = "init.py"
      ├─ evaluator = "evaluator.py"
      ├─ system_message = "prompt.md"
      └─ evolve_config.timeout = EVAL_TIMEOUT

Phase 2: 多轮演化循环（默认最多 10 轮）
  ├─ A：【famou-experiment-manager skill】提交实验
  │       experiment-manager experiment create \
  │         --config <config.yaml绝对路径> \
  │         --experiment-name {TASK_ID}_round{ROUND_N} \
  │         --json
  │       → 记录返回的 experiment_id
  ├─ B：【famou-experiment-manager skill】轮询状态（每 10s 一次）
  │       experiment-manager experiment status <experiment_id> --json
  │       → 验证失败：分析错误，修复 evaluator/init.py，删除失败实验，回到 Phase 1
  │       → 验证通过并进入演化：继续等待
  │       → 演化完成（COMPLETE）：进入 C
  ├─ C：【famou-experiment-manager skill】拉取结果
  │       experiment-manager experiment results <experiment_id> \
  │         --output round-{ROUND_N}-evolution_log.json --json
  │       → 解析最佳分数、最佳程序、演化曲线
  ├─ D：终止判断（任一满足则终止循环）
  │       ├─ 最佳分数超过 baseline 目标且连续 2 轮无提升
  │       ├─ 已达最大轮数（10 轮）
  │       └─ 连续 3 轮最佳分数退步
  └─ E：下一轮准备（未终止时执行）
       ├─ 从结果中提取 best_program → 保存为 programs/round-{N}-best.py（永不删除）
       ├─ cp programs/round-{N}-best.py init.py（继承最优解，必须执行）
       ├─ ROUND_N += 1，更新 experiment_name
       └─ 依据演化信号微调 config.yaml（参考 Config 动态调优规则）

Phase 3: 实验完成报告
  ├─ 【famou-result-visualization skill】对最终 best_program.py 生成可视化 HTML
  └─ 输出汇总：总轮次 + 最终最优分数 + 各轮进展表 + 最佳程序分析 + 建议
```

### Config 动态调优规则

| 观察到的演化信号 | 调整动作 |
|---|---|
| 大量 validity=0，error_count 高 | 增强 prompt.md 中的代码规范约束 |
| 多轮分数停滞不前 | 增加 evolve_config.max_iterations 或 num_islands |
| 分数波动大，不稳定 | 降低 evolve_config.temperature（0.7 → 0.5） |
| 早期快速收敛 | 增大 evolve_config.num_islands，增加种群多样性 |
| 托管执行验证失败（状态轮询返回 error） | 分析执行服务日志，修复 evaluator/init.py 后重新提交 |

### 不可违背的原则

1. **baseline 先于 famou**：演化开始前必须确认 baselines/comparison_table.md 存在且最优 baseline 分数写入 project_context.md
2. **experiment_name 每轮唯一**：格式为 `{TASK_ID}_round{ROUND_N}`，轮次递增；相同 name 会与历史实验混淆
3. **evaluator.py 以问题定义为准，不得为提分而修改**
4. **每轮必须继承最优解**：从托管执行结果拉取 best_program 并 cp 为 init.py，是演化连续性的关键
5. **至少跑满 2 轮**，单轮结果不足以判断收敛
6. **validity=1 是硬门禁**：本地验证通过后再提交托管执行，节省实验时间
7. **托管执行环境必须就绪**：提交前用 config.py read 确认状态可用；缺失则先完成外部配置

### 每轮输出格式

```markdown
## Famou Round N 总结

- 状态: [运行中 / 完成 / 异常]
- 最佳分数: XX.XX（seed_0 基准: YY.YY，提升: +ZZ.ZZ%）
- 最佳程序: famou_data/<实验目录>/programs/<id>.py
- 关键洞察: [这个程序为何得分更高]
- 下轮 config 调整: [调整了什么 + 原因]
- 建议: [继续 / 停止 / 上报 Team Lead]
```

---

## 八、Git & Doc Manager 核心功能

### GitHub 留存策略

**核心原则**：每个实验结果必须和产生它的代码版本精确绑定，论文中每个数字都可追溯到对应 commit，确保不错位、可复现。

### GitHub 仓库目录结构

```
repo/
├── experiments/                        # 所有实验（每个独立目录，永不删除）
│   ├── exp-001-baseline/
│   │   ├── init.py                     # 初始程序
│   │   ├── best_program.py             # 最终最优程序
│   │   ├── config.yaml                 # 完整配置（含随机种子）
│   │   ├── results.json                # 所有评估指标（真实数据，不估计）
│   │   └── README.md                   # 实验说明 + 复现命令
│   ├── exp-002-famou-round1/
│   │   ├── programs/                   # 关键迭代程序快照
│   │   │   ├── iter-08-breakthrough.py # 拐点程序（必须保存）
│   │   │   └── iter-30-final.py
│   │   ├── evolution_log.json          # 完整演化历史（每轮分数）
│   │   └── results.json
│   └── exp-003-famou-round2/           # 每一轮演化独立目录
│
├── baselines/                          # 所有对比方法（独立目录）
│   ├── meanpool-mlp/
│   │   ├── model.py
│   │   ├── config.yaml
│   │   └── results.json
│   ├── meanstd-gbt/
│   └── pointnet-mlp/
│
├── paper/
│   └── paper_results_registry.json     # ⭐ 论文数字 → 实验ID 防错位映射
│
├── figs/                               # 论文配图（scientific 风格）
├── project_context.md                  # 全局上下文（所有 Agent 启动前必读）
├── experiment_index.md                 # 所有实验索引表（一览）
└── .gitignore                          # 排除大文件
```

### ⭐ 防错位核心机制：`paper_results_registry.json`

论文中**每一个数字**都必须在此文件登记，记录来源 experiment_id 和 git commit：

```json
{
  "table_1_main_results": {
    "our_method_ep30": {
      "value": "1.31%",
      "ci": "0.54%-2.11%",
      "experiment_id": "exp-003-famou-round2",
      "commit": "a3f9c12",
      "program": "experiments/exp-003-famou-round2/best_program.py",
      "reproduce_cmd": "python run.py --config experiments/exp-003-famou-round2/config.yaml"
    },
    "our_method_test25": {
      "value": "0.87%",
      "ci": "0.56%-1.23%",
      "experiment_id": "exp-003-famou-round2",
      "commit": "a3f9c12"
    },
    "meanpool_mlp_ep30": {
      "value": "15.6%",
      "experiment_id": "exp-001-baseline",
      "commit": "b2e8d45"
    }
  },
  "table_2_ablation": { }
}
```

**Paper Writer 在写论文数字前必须查此文件；Evaluator 产出新结果后必须立即更新此文件。**

### Famou 演化中间程序留存规则

Famou Agent 每轮演化结束后，Git & Doc Manager **必须**执行：

```bash
# 1. 保存当轮最优程序快照（不得覆盖，永久留存）
cp famou_data/<exp>/programs/best_program.py \
   experiments/exp-00N-famou-roundN/programs/iter-{N}-best.py

# 2. 导出演化日志
cp famou_data/<exp>/evolution_log.json \
   experiments/exp-00N-famou-roundN/evolution_log.json

# 3. 记录结果并 commit
git add experiments/exp-00N-famou-roundN/
git commit -m "exp(famou-round-N): best={score}% at iter={K}
- Features: {feature_summary}
- Updated paper_results_registry.json"

# 4. 打 tag（每轮必须）
git tag -a exp-round-N-best -m "Round N best: {score}%"
git push origin --tags
```

### 文档维护职责

1. **project_context.md**：全局上下文，所有 Agent 启动前必读
   - 当前阶段、已完成实验、待办任务
   - 关键决策记录、技术路线
   - 数据路径、当前最优结果汇总

2. **experiment_index.md**：所有实验索引表（一览无余）
   ```markdown
   | Exp ID | 描述 | 分数(EP30) | 分数(Test25) | Commit | 状态 |
   |--------|------|-----------|-------------|--------|------|
   | exp-001 | MeanPool+MLP Baseline | 15.6% | - | b2e8d45 | ✅完成 |
   | exp-002 | Famou Round 1 | 1.43% | - | c3f9a11 | ✅完成 |
   | exp-003 | Famou Round 2 | 1.31% | 0.87% | a3f9c12 | ✅完成 |
   ```

3. **自动同步机制**：
   - 每次 commit 后自动更新 experiment_index.md
   - 新实验结果产出后立即更新 paper_results_registry.json
   - 阶段切换时更新 project_context.md，向 Team Lead 广播

### .gitignore 配置

```
# 大文件（绝对不进 git）
*.pt
*.pth
*.h5
*.ckpt
*.pkl
*.npy
*.npz
*.csv
*.xlsx

# 运行时产物
__pycache__/
.ipynb_checkpoints/
logs/
outputs/
famou_data/*/logs/

# 但保留关键程序快照和结果
!experiments/**/best_program.py
!experiments/**/iter-*.py
!experiments/**/results.json
!experiments/**/evolution_log.json
```

### Commit 规范

```bash
# 新实验开始
git commit -m "exp(exp-00N): init baseline {method_name}
- Config: lr=0.001, epochs=100"

# 实验完成（同时更新 experiment_index.md + registry）
git commit -m "exp(exp-00N): complete, EP30={score}%
- Best program: iter-{K}
- Updated experiment_index.md, paper_results_registry.json"

# 里程碑
git tag -a v1.0-phase1-complete -m "Phase 1 complete, best EP30=1.31%"
git push origin --tags
```

---

## 九、三阶段协作流程

### 阶段 1：论文复现（少量人工干预）
```
Team Lead
  ├─ [优先] Background Researcher
  │   ├─ 文献调研 + 方法论背景
  │   ├─ 产出 literature_review.md / introduction_draft.md
  │   └─ 写入 project_context.md[研究背景]
  │
  ├─ Git & Doc Manager → 创建 phase-1 分支 + 初始化 project_context.md
  ├─ Data Engineer → 处理数据
  │   ├─ 【famou-data-analysis skill】理解数据格式、字段、分布
  │   └─ Git & Doc Manager → commit + 更新 PROGRESS.md
  │
  ├─ [并行启动]
  │   ├─ Background Researcher → 输出 baseline 候选列表（3~5 个方法）
  │   │   └─ Experiment Runner → 依次执行各 baseline → 记录结果 → 写入 project_context.md
  │   └─ Model Developer → 设计初始算法，输出 init.py 初稿（与 baseline 并行，不互相阻塞）
  │
  ├─ **Famou Agent** → 接收 init.py + baseline 基准分数（baseline 完成后才启动）
  │       ├─ Phase 0: 【famou-experiment-manager skill】确认实验管理 CLI 与凭证配置
  │       │           确认 DATA_DIR / WORK_DIR / TASK_ID / ROUND_N / EVAL_TIMEOUT
  │       │           🚫 强制门禁：baselines/comparison_table.md 存在 + project_context.md 有最优 baseline 分数
  │       ├─ Phase 1: 【famou-artifact-generator skill】本地验证 init.py
  │       │           (validity==1 + combined_score!=0 + error_info=="")
  │       │           生成 config.yaml / prompt.md（含 baseline 目标分数）
  │       ├─ Phase 2: 多轮演化循环
  │       │   ├─ 每轮：【famou-experiment-manager skill】托管执行提交 → 轮询状态 → 拉取结果
  │       │   ├─ 提取 best_program → 快照 → 继承最优解 → 调优 config
  │       │   └─ Git & Doc Manager → 每轮 commit round-N 快照文件
  │       └─ Phase 3: 【famou-result-visualization skill】生成可视化 HTML
  │                   输出最优程序 + 性能分析报告
  │                   Git & Doc Manager → commit + 更新 PROGRESS.md
  │
  ├─ Evaluator → 接收 Famou Agent 的最优程序，计算评估指标、对比、代码绘图
  │   └─ Git & Doc Manager → commit + 更新 PROGRESS.md
  │
  ├─ Paper Writer → 起草论文（使用完整 skills 流水线）
  │   ├─ 写作过程中主动调用 baoyu-article-illustrator skill 生成示意图（不等指令）
  │   │   ├─ fig_framework.png（必须）
  │   │   ├─ fig_architecture.png / fig_motivation.png / fig_evolution.png（按需）
  │   │   └─ 所有示意图存至 paper/figs/
  │   │
  │   ├─ 🚦【Team Lead 图表门禁】提交 Strict Reviewer 前，Team Lead 核查：
  │   │   ├─ fig_framework.png 是否存在？→ 不存在则 Block，退回 Paper Writer 补图
  │   │   ├─ 有新颖结构时 fig_architecture.png 是否存在？
  │   │   ├─ 有核心 insight 时 fig_motivation.png 是否存在？
  │   │   └─ 全部通过 → 放行进入 Strict Reviewer
  │   │
  │   ├─ [Round 1] Strict Reviewer → review 草稿，输出 Reject/Major/Minor/Accept
  │   ├─ Paper Writer → review-response skill + 修订
  │   ├─ [Round 2] Strict Reviewer → 再次 review（强制执行，不可跳过）
  │   ├─ Paper Writer → 修订
  │   └─ [Round N，直至双 Accept] Strict Reviewer → 持续迭代
  │       ├─ 若瓶颈在实验 → Experiment Runner 补充实验（famou 变体走托管执行）→ Evaluator 更新结果
  │       ├─ 若瓶颈在论文结构 → Paper Writer 重构
  │       ├─ 若瓶颈在示意图缺失/不清晰 → Paper Writer 用 baoyu-article-illustrator 补图/重绘
  │       └─ 若瓶颈在 Claim 过大 → Team Lead 自动收窄 Claim 范围
  └─ 双 Accept 后 → Git & Doc Manager 打 tag + 更新 project_context.md
```
**人工干预点**：算法选择、外部执行环境配置、异常处理

### 阶段 2：新任务自动化（零人工干预）
```
Team Lead（完全自主）
  ├─ Background Researcher → 增量更新文献（新任务相关）
  ├─ Git & Doc Manager → 创建 worktree + 初始化新 project_context.md
  ├─ Data Engineer → 【famou-data-analysis skill】自动分析数据
  ├─ Model Developer → 自动设计 init.py
  ├─ Famou Agent → 托管执行提交演化（famou-experiment-manager skill）
  ├─ Experiment Runner → 补充实验支持（baseline + ablation）
  ├─ Evaluator + Paper Writer → 生成报告（skills 流水线）
  │   └─ Strict Reviewer → review 循环
  └─ Git & Doc Manager → 持续更新 PROGRESS.md
```
**关键**：使用 `EnterWorktree` 实现环境隔离

### 阶段 3：规模化验证（并行任务）
```
Team Lead
  ├─ 并行 3 个 Model Developer（不同算法方向）
  ├─ 并行 3 个 Famou Agent（不同 TASK_ID，各自独立托管执行提交）
  ├─ Evaluator + Paper Writer → 最终论文（完整 skills 流水线）
  └─ Strict Reviewer → 最终把关（强制 2~N 轮 review，双 Accept 才放行，全程无人工介入）
```
**关键**：使用 `dispatching-parallel-agents` + `TaskList`

---

## 十、关键设计要点

### 1. 环境隔离（阶段 2）
```bash
git worktree add ../phase-2-worktree phase-2-automation
# 实验完成后
git merge phase-2-automation
git worktree remove ../phase-2-worktree
```

### 2. 任务分配
- **阶段 1**：Team Lead 手动分配（`TaskCreate`）
- **阶段 2**：Team Lead 自动分配
- **阶段 3**：Agent 自主认领（`TaskList`）

### 3. 通信机制
- 使用 `SendMessage` 点对点通信（避免 `broadcast`）
- 任务完成后向 Team Lead 报告
- **所有 Agent 启动前必读 `project_context.md` 同步进度**
- **任务完成后通知 Git & Doc Manager 更新文档**

### 4. 文档同步流程
```
Agent 完成任务 → SendMessage 给 Git & Doc Manager
  → Git & Doc Manager 更新 PROGRESS.md
  → 重要变更时更新 project_context.md
  → 通知 Team Lead
```

### 5. 实验提交监控（Team Lead 责任）

演化实验通过独立的托管执行服务提交，Team Lead 须确认以下事项后再放行：

**提交前检查清单：**
- [ ] 本地 `python evaluator.py <init.py>` 验证：validity==1，combined_score!=0，error_info==""
- [ ] `experiment-manager --version` 正常，`config.py read` 返回就绪状态
- [ ] config.yaml 的 experiment_name 格式为 `{TASK_ID}_round{ROUND_N}`，与历史实验不重复

**异常处理：**
```
托管执行验证失败（experiment-manager experiment status 返回 error）→
  ├─ experiment-manager experiment logs <id> 查看错误详情
  ├─ 修复 evaluator.py 或 init.py
  ├─ experiment-manager experiment delete <id> 删除失败实验
  └─ 重新提交
```

### 6. 图表质量监督（Team Lead 责任）

Paper Writer 提交 Strict Reviewer 前，Team Lead 执行图表门禁检查：

**必查项：**
- [ ] `paper/figs/fig_framework.png` 存在（所有论文强制）
- [ ] `paper/figs/fig_architecture.png` 存在（有新颖模型结构时）
- [ ] `paper/figs/fig_motivation.png` 存在（有核心 insight 时）
- [ ] `paper/figs/fig_evolution.png` 存在（famou 是核心贡献时）
- [ ] 所有示意图由 `baoyu-article-illustrator skill`（`style: scientific`）生成，非截图/网络图
- [ ] 所有实验结果图由 Evaluator 用 Python 代码基于真实数据生成，不使用 AI 生图

**决策规则：**

| 情况 | 动作 |
|------|------|
| fig_framework.png 缺失 | **Block**：退回 Paper Writer，调用 baoyu-article-illustrator 补图 |
| 实验结果图使用了 AI 生图 | **Reject**：必须换成 Python 代码生成的真实数据图 |
| Strict Reviewer 反映论文难以理解 | 检查方法/动机示意图是否缺失，退回补图 |
| 示意图内容与方法描述不一致 | 退回 Paper Writer，修改 prompt 重新生成 |

### 7. 文件管理策略
**只维护核心代码，排除大文件**

```bash
# .gitignore 配置（Git & Doc Manager 自动维护）
*.pth              # 模型权重
*.pt               # 数据集
*.h5               # 模型文件
*.ckpt             # checkpoint
*.pkl              # pickle 文件
*.npy              # numpy 数组
*.npz
*.csv              # 大数据文件
*.xlsx
*.png              # 实验图表（可选择性提交关键图表）
*.jpg
__pycache__/
.ipynb_checkpoints/
logs/
outputs/
```

**纳入版本控制**：
- 代码文件（`.py`, `.sh`, `.yaml`）
- 配置文件（`config.yaml`, `requirements.txt`）
- 文档文件（`.md`, `project_context.md`, `progress.md`）
- 关键结果（`results.json`, 小型图表）

---

## 十一、验收标准

| 阶段 | 核心指标 |
|---|---|
| **阶段 1** | 复现准确度 ≥95%，人工干预 ≤5 次 |
| **阶段 2** | 零人工干预，风阻误差 <3%，趋势准确率 >95% |
| **阶段 3** | 并行任务 ≥3 个，产出可投稿论文 |

---

## 十二、成本优化

### 精简版（4 个 Agent）
1. Team Lead
2. **Git & Doc Manager**（独立保留，负责版本控制 + 文档同步）
3. Data & Model Agent（合并）
4. Analysis & Report Agent（合并）

### 模型选择
- 简单任务（数据清洗、git）：`haiku`
- 复杂任务（训练、撰写）：`sonnet`
- 关键决策（Team Lead）：`opus`

---

## 十三、风险应对

| 风险 | 应对 |
|---|---|
| 阶段 2 自动化失败 | 降级半自动化，分析原因 |
| 结果不达标 | 人工优化，保留流程记录 |
| 成本超预期 | 优化提示词，使用 Haiku |
| **托管执行验证失败** | 查看执行服务日志，修复 evaluator/init.py，删除失败实验后重新提交 |

---

**版本**：v5.6 | **日期**：2026-03-14 | **更新**：Paper Writer 示意图规范完善（强制图清单 + baoyu-article-illustrator 调用规范），Team Lead 新增图表门禁监督机制（关键设计要点第6条），三阶段协作流程同步图表监督节点

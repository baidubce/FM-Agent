---
name: famou-result-visualization
description: 为 FaMou 进化算法生成可行解的 Python 代码解生成可视化结果页面。当用户提到"FaMou 可视化"、"把这个解可视化"、"可行解结果展示"、"evolution 结果"、"evolve 可视化"，或者提供了 Python 代码形式的问题解（路径规划、排课、背包、TSP、调度、机器学习等）需要直观展示效果时，必须使用此技能。即使用户只说"帮我可视化这个解"、"画出来看看"、"展示一下结果"，只要上下文涉及进化算法、优化问题的解，也应立即触发此技能。
---

# FaMou 进化算法解——结果可视化 Skill

**核心目标**：拿到 Python 代码形式的优化问题解，直接理解其语义，生成一个直观展示**解的效果**的交互式 HTML 页面。

不是展示进化过程，而是展示**解本身的结果**——路径规划画路径、排课画课表、背包问题画装载情况、图着色画上色的图……

---

## 第一步：收集输入

**必须有**：
1. **问题描述** — 是什么优化问题，输入规模，约束条件
2. **Python 代码形式的解** — FaMou 进化后的最终解（函数/数据结构/策略代码均可）

**可选补充**：
- 评估分数 / 适应度
- 初始解（用于对比）
- 问题的原始数据（如节点坐标、任务列表等）

若缺少必须项，直接询问用户补充。

---

## 第二步：理解解 → 规划可视化方案

**直接阅读和理解**用户提供的 Python 代码和问题描述，无需调用任何外部 API。

### 2.1 理解解的语义

阅读代码，提取：
- **问题类型**：路径规划 / 排课 / 背包 / 图论 / 调度 / 装箱 / 其他
- **解的核心数据结构**：是节点序列？时间表映射？选择集合？分配方案？
- **关键数值**：坐标、时间槽、容量、权重、颜色等可视化所需的具体数据

### 2.2 确定可视化类型

根据问题类型选择最合适的 `viz_type`：

| 问题类型 | viz_type | 视觉呈现 |
|---|---|---|
| TSP / VRP / 路径规划 | `path_map` | SVG 坐标系 + 节点连线路径 |
| 排课 / 时间表 | `schedule_grid` | 表格热力图，色块填充 |
| 背包 / 装箱 | `packing_rect` | SVG 矩形堆叠容器 |
| 图着色 / 社区检测 | `graph_color` | 节点着色图 |
| 作业调度 / 项目排期 | `gantt` | 横向甘特图 |
| 前后对比 / 多指标 | `bar_compare` | 对比柱状图 |
| 机器学习 / 神经网络 / 超参优化 | `ml_viz` | 网络结构图 / 训练曲线 / 超参热力图 |
| 其他 / 复杂策略 | `custom` | 关键指标仪表盘 + 文字说明 |

### 2.3 提取绘图数据

从代码中直接读取并整理渲染所需的具体数值，例如：
- `path_map`：节点坐标列表、访问顺序、总距离
- `schedule_grid`：资源列表、时间槽、各分配的 (资源, 时间槽, 名称)
- `packing_rect`：容器尺寸、各物品的 (x, y, w, h, 标签, 价值)
- `gantt`：任务列表，每项含 (名称, 开始, 结束, 资源)

---

## 第三步：生成 HTML 可视化文件

直接编写并输出 HTML 文件到 `famou_viz_result.html`（或用户指定路径）。

### 页面整体结构

```
┌──────────────────────────────────────────────────────────┐
│  [问题类型标签]  问题摘要                  关键指标卡片行  │
├────────────────────────────────────┬─────────────────────┤
│                                    │                     │
│   主可视化区域（视觉中心，≥50%）    │   解的亮点列表       │
│   路径图 / 课表 / 装箱图 / 甘特图   │                     │
│                                    │   评分 / 提升展示    │
├────────────────────────────────────┴─────────────────────┤
│   (可选) 进化前后对比 / 补充说明                          │
└──────────────────────────────────────────────────────────┘
```

### 设计规范

- **暗色科技风**：背景 `#030810`，卡片 `#080f1e`，边框 `#112240`
- **accent 色系**：主色 `#00c8ff`（蓝）搭配 `#00ff88`（绿）作为高亮
- **字体**：正文 `Noto Sans SC`，数字/代码 `IBM Plex Mono`（Google Fonts CDN）
- **入场动画**：各区块依次 `fadeUp`（`animation-delay` 递增）
- **主可视化动画**：路径逐段绘制，柱子从底部生长，节点弹入
- **交互**：hover 节点/格子/柱子时显示 tooltip
- **Tooltip**：固定定位，跟随鼠标，显示详细数值

### 文件模板

```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FaMou 解可视化 — {问题名称}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.5/babel.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* 内联全部样式，不依赖外部 CSS */
    /* 包含：CSS 变量、动画 keyframes、card/tag/tooltip/skeleton 等基础类 */
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    const { useState, useEffect, useRef } = React;

    // ── 从解中提取的数据（硬编码，来自对代码的分析）──
    const SOLUTION_DATA = { /* ... */ };

    // ── 可视化组件 ──
    // 根据 viz_type 实现对应组件

    // ── 页面骨架 ──
    function App() { /* 指标卡片 + 主可视化 + 亮点列表 */ }

    ReactDOM.render(<App />, document.getElementById('root'));
  </script>
</body>
</html>
```

---

## 注意事项

- **直接从代码中读数据**：将 Python 解里的关键数值（坐标、序列、映射等）直接硬编码进 HTML 的 `SOLUTION_DATA` 常量，不执行 Python 代码，只是读取其中的数据字面量
- **可视化忠于解的效果**：展示"这个解长什么样"，不是进化历史或算法流程
- **主可视化区域要大**：是页面视觉中心，占页面 50% 以上
- **数据规模适配**：节点/任务超过 100 个时，考虑抽样或聚合展示避免拥挤
- **降级处理**：无法识别具体问题类型时，fallback 到 `custom` 仪表盘展示关键数值
- **自包含**：所有依赖均通过 CDN 引入，文件本身可离线（CDN 可替换为本地）打开
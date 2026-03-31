# 项目上下文：二维双群中子扩散方程解析解论文

## 当前阶段
公开示例版本

## 任务描述
构造解析解 φ₁(x,y), φ₂(x,y) 满足二维双群中子扩散方程 + Neumann BC

## 物理参数
D1=1.0, D2=0.5, Σr=0.02, Σa2=0.1, ν=2.5, Σf1=0.005, Σf2=0.1, Σ1→2=0.015
域：[-0.5,0.5]²

## 边界条件
- 左 (x=-0.5): -D∂φ/∂x = y （非零 Neumann）
- 右 (x=0.5): -D∂φ/∂x = 0
- 上/下 (y=±0.5): -D∂φ/∂y = 0

## 测试点（PDE残差验证）
(0,0), (0.2,0.2), (-0.2,-0.3), (0.4,-0.4)

## 目录说明
- `famou/task1/`: 任务定义、初始程序、评估器
- `baselines/`: 对比方法及结果
- `scripts/analysis/`: 统一评估与出图脚本
- `paper/`: 论文草稿与配图

## 研究背景

### 问题定位
二维有限均匀介质边界源双群中子扩散方程是核反应堆物理中的经典问题。本项目构造解析解作为 ground truth，并对比多种求解方法（FDM、FEM、PINN、截断解析解、LLM驱动程序演化）。

### 文献调研总结（23篇已验证文献）

**领域1：核物理PDE求解方法（10篇）**
- 经典教材：Duderstadt & Hamilton (1976), Bell & Glasstone (1970), Lee (2020/2024), Kuridan (2023)
- Fourier方法：Ceolin FBPM系列 (2017-2021), Kashi et al. (2013), OSTI (1974)
- 近期解析：Momani et al. (2024) 球形反应堆双群HPM方法

**领域2：PINN在核物理中的应用（6篇）**
- 基础：Raissi et al. (2019) JCP — PINN原始论文
- 核物理应用：Schiassi (2022, 双群误差0.6-15%), TL-PINN (2023, 100x加速), Parameterized PINN (2024, 1000x加速), NAS-PINN (2025), R²-PINN (2025)

**领域3：神经算子（4篇）**
- 基础：DeepONet (Lu et al., 2021 NMI), FNO (Li et al., 2021 ICLR)
- 核应用：Park et al. (2024) 中子通量数字孪生, arXiv:2602.15890 (2026) DeepONet vs FNO中子输运

**领域4：LLM驱动程序演化（3篇）**
- FunSearch (Romera-Paredes et al., 2024 Nature) — 首次LLM科学发现
- AlphaEvolve (Novikov et al., 2025) — 进化整个代码库
- CodeEvolve (2025) — 开源替代方案

### 研究创新点
- **Research Gap**: LLM驱动程序演化此前未应用于物理PDE求解
- **本文贡献**: 首次将FunSearch/AlphaEvolve范式应用于中子扩散方程求解

### Baseline候选方法
| 方法 | 类别 | 预期精度 |
|------|------|---------|
| FDM | 数值方法 | O(h²) |
| FEM | 数值方法 | O(h^p) |
| PINN | AI/ML | ~0.6-15% |
| Truncated Analytical | 截断解析 | 取决于截断阶数N |
| Famou (LLM Evolution) | AI/LLM | 待验证 |

### 详细文件
- 文献综述：`scripts/analysis/literature_review.md`
- 技术背景：`scripts/analysis/methodology_background.md`
- Introduction初稿：`scripts/analysis/introduction_draft.md`

## 最优 Baseline 分数
- **High-order FDM (4th-order, N=200)**: combined_score=0.99999, PDE残差=9.92e-06, BC残差=8.77e-07, 耗时=6.597s
- **FDM (2nd-order, N=200)**: combined_score=0.99998, PDE残差=1.98e-05, BC残差=7.65e-07, 耗时=1.069s
- **Truncated Analytical (K=5)**: combined_score=0.99946, PDE残差=4.04e-05, BC残差=4.99e-04, 耗时=0.0003s
- **PINN (numpy)**: combined_score=0.94727, PDE残差=4.58e-02, BC残差=9.83e-03, 耗时=5.557s
- **当前最优**: High-order FDM 4th-order (0.99999)

### 收敛分析（截断解析解 N_modes 扫描）
| N_modes | combined_score | PDE残差L2 | BC残差L2 |
|---------|---------------|-----------|----------|
| 1 | 0.96979 | 5.19e-05 | 3.33e-02 |
| 5 | 0.99800 | 4.59e-05 | 3.07e-03 |
| 10 | 0.99963 | 3.88e-05 | 5.58e-04 |
| 20 | 0.99992 | 4.14e-05 | 9.66e-05 |
| 30 | 0.99996 | 4.14e-05 | 3.12e-05 |
| 50 | 0.99997 | 4.14e-05 | 6.97e-06 |

## 已完成实验
- [x] FDM baseline (N=200, 2nd-order) — combined_score=0.99998
- [x] High-order FDM (N=200, 4th-order) — combined_score=0.99999 (FEM替代)
- [x] Truncated Analytical (K=5) — combined_score=0.99946
- [x] PINN (numpy, MLP [2,40,40,2], 3000步) — combined_score=0.94727
- [x] 收敛分析（N_modes=1~50）
- [ ] Program evolution demo — public release does not include cloud submission workflow

## 关键决策记录
[待更新]

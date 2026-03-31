> Public-release note: this is the sanitized task specification used as project reference material.

# 任务数据与描述

---

## 任务：二维有限均匀介质边界源双群中子扩散方程解析解

### 任务 ID

Analytical Solution Helmholtz

---

## 一、任务目标

**核心目标**：构造两个函数 $\varphi_1(x,y)$ 和 $\varphi_2(x,y)$，使得它们同时满足：
1. 双群中子扩散方程（PDE）
2. 四条边界上的诺伊曼边界条件（BC）

---

## 二、物理方程

### 2.1 控制方程（PDE）

**快群方程**：
$$
-D_1 \nabla^2 \varphi_1 + \Sigma_r \varphi_1 = \nu \Sigma_{f1} \varphi_1 + \nu \Sigma_{f2} \varphi_2
$$

**热群方程**：
$$
-D_2 \nabla^2 \varphi_2 + \Sigma_{a2} \varphi_2 = \Sigma_{1\to2} \varphi_1
$$

**符号说明**：

| 符号 | 含义 |
|------|------|
| $\varphi_1$ | 快群中子通量（待求） |
| $\varphi_2$ | 热群中子通量（待求） |
| $D_1, D_2$ | 扩散系数 |
| $\Sigma_r$ | 快群移出截面 |
| $\Sigma_{a2}$ | 热群吸收截面 |
| $\nu$ | 每次裂变平均中子数 |
| $\Sigma_{f1}$ | 快群裂变截面 |
| $\Sigma_{f2}$ | 热群裂变截面 |
| $\Sigma_{1\to2}$ | 群间转移截面 |

**拉普拉斯算子**：
$$
\nabla^2 \varphi = \frac{\partial^2 \varphi}{\partial x^2} + \frac{\partial^2 \varphi}{\partial y^2}
$$

---

### 2.2 边界条件（BC）

**几何区域**：方形区域，边长为 1，中心在原点
- $x \in [-0.5, 0.5]$，$y \in [-0.5, 0.5]$

**左边界**（$x = -0.5$，非零诺伊曼条件）：
$$
\begin{cases}
-D_1 \frac{\partial \varphi_1}{\partial x} = y \\[4pt]
-D_2 \frac{\partial \varphi_2}{\partial x} = y
\end{cases}
$$

**右边界**（$x = 0.5$，零诺伊曼）：
$$
\begin{cases}
-D_1 \frac{\partial \varphi_1}{\partial x} = 0 \\[4pt]
-D_2 \frac{\partial \varphi_2}{\partial x} = 0
\end{cases}
$$

**上边界**（$y = 0.5$，零诺伊曼）：
$$
\begin{cases}
-D_1 \frac{\partial \varphi_1}{\partial y} = 0 \\[4pt]
-D_2 \frac{\partial \varphi_2}{\partial y} = 0
\end{cases}
$$

**下边界**（$y = -0.5$，零诺伊曼）：
$$
\begin{cases}
-D_1 \frac{\partial \varphi_1}{\partial y} = 0 \\[4pt]
-D_2 \frac{\partial \varphi_2}{\partial y} = 0
\end{cases}
$$

---

## 三、验收标准（如何计算残差）

### 3.1 残差计算方法

**步骤 1**：定义物理常数数值（用于代入计算）

| 符号 | 数值 |
|------|------|
| $D_1$ | 1.0 |
| $D_2$ | 0.5 |
| $\Sigma_r$ | 0.02 |
| $\Sigma_{a2}$ | 0.1 |
| $\nu$ | 2.5 |
| $\Sigma_{f1}$ | 0.005 |
| $\Sigma_{f2}$ | 0.1 |
| $\Sigma_{1\to2}$ | 0.015 |

---

**步骤 2**：计算 PDE 残差（内部点）

在以下测试点代入方程：
- $(0, 0)$
- $(0.2, 0.2)$
- $(-0.2, -0.3)$
- $(0.4, -0.4)$

**快群 PDE 残差**：
$$
\text{Res}_1 = -D_1 (\varphi_{1,xx} + \varphi_{1,yy}) + \Sigma_r \varphi_1 - (\nu \Sigma_{f1} \varphi_1 + \nu \Sigma_{f2} \varphi_2)
$$

**热群 PDE 残差**：
$$
\text{Res}_2 = -D_2 (\varphi_{2,xx} + \varphi_{2,yy}) + \Sigma_{a2} \varphi_2 - \Sigma_{1\to2} \varphi_1
$$

**期望**：$\text{Res}_1 = 0$ 且 $\text{Res}_2 = 0$（或接近 0）

---

**步骤 3**：计算边界条件残差

| 边界 | 位置 | 计算方式 | 期望值 |
|------|------|----------|--------|
| 左边界 | $x = -0.5$ | $-D \cdot \frac{\partial \varphi}{\partial x} - y$ | $= 0$ |
| 右边界 | $x = 0.5$ | $-D \cdot \frac{\partial \varphi}{\partial x}$ | $= 0$ |
| 上边界 | $y = 0.5$ | $-D \cdot \frac{\partial \varphi}{\partial y}$ | $= 0$ |
| 下边界 | $y = -0.5$ | $-D \cdot \frac{\partial \varphi}{\partial y}$ | $= 0$ |

---

### 3.2 判断标准

**合格的解析解**：
- 在所有内部点上，PDE 残差接近 0
- 在所有边界上，边界条件残差接近 0

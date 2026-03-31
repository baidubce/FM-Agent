# Baseline 对比表

## 统一评估（famou evaluator.py, h=1e-6, L2 norm）

| Method | unified_score | eval_time(s) | 备注 |
|--------|--------------|-------------|------|
| Fourier Analytical (N=30) | 0.9989 | 0.072 | ✅ 完整解析解（ground truth） |
| Truncated Analytical (K=5) | 0.9936 | 0.024 | ✅ 截断5个Fourier模 |
| PINN (numpy) | 0.9587 | 7.099 | ✅ MLP [2,40,40,2] 3000步 |
| High-order FDM (4th-order, N=200) | 0.9186 | 9.507 | ✅ FEM替代 |
| FDM (2nd-order, N=200) | 0.9186 | 1.847 | ✅ 标准有限差分 |
| **Ours (famou best)** | - | - | 演化中 |

> 注：FDM方法统一评分较低是因为spline插值在h=1e-6尺度下计算二阶导数精度受限，不代表FDM解本身精度差。

## 自评估（各方法独立评估器）

| Method | combined_score | PDE残差L2 | BC残差L2 | 运行时间(s) | 备注 |
|--------|---------------|-----------|----------|------------|------|
| High-order FDM (4th-order, N=200) | 0.99999 | 9.92e-06 | 8.77e-07 | 6.597 | FEM替代 |
| FDM (2nd-order, N=200) | 0.99998 | 1.98e-05 | 7.65e-07 | 1.069 | 标准有限差分 |
| Truncated Analytical (K=5) | 0.99946 | 4.04e-05 | 4.99e-04 | 0.0003 | 截断Fourier |
| PINN (numpy) | 0.94727 | 4.58e-02 | 9.83e-03 | 5.557 | MLP 3000步 |

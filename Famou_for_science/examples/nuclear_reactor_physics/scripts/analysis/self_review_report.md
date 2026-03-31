## Paper Self-Review Report — 2026-03-14

Reviewer: Math/Physics Agent (analytical solution specialist)

---

### ✅ 通过项

1. **Bibliography completeness**: All 23 `\cite{}` keys (across introduction.tex, related_work.tex) have matching entries in `refs.bib`. No orphan citations. Coverage spans all four research domains: classical nuclear physics (6), PINNs (6), neural operators (4), LLM program evolution (3).

2. **Related Work coverage**: All experimental baselines (FDM, FEM/High-order FDM, PINN, Truncated Analytical) are discussed in Related Work. The comparison table includes methods that are properly introduced and cited.

3. **Mathematical derivation correctness**: The Fourier expansion in method.tex (§4.2–§4.5) is rigorously derived and numerically verified. All boundary conditions satisfied to high precision. Eigenvalue equation, eigenvector ratios, and coefficient system are self-consistent.

4. **Ablation study data integrity**: All 9 data points in Table 2 (tab:ablation) match `convergence_data.json` exactly. Analysis of O(N⁻²) BC convergence and PDE floor is physically justified.

5. **Existing figures (4/6)**:
   - `fig_framework.png` ✅ (86 KB)
   - `fig_motivation.png` ✅ (85 KB)
   - `fig_evolution.png` ✅ (52 KB)
   - `fig_convergence.png` ✅ (66 KB)

---

### ⚠️ 需改进项

1. **Variable notation inconsistency** (problem.tex vs method.tex):
   - `problem.tex` uses `\varphi_1, \varphi_2` (via `\vph` command)
   - `method.tex` uses `\phi_1, \phi_2`
   - **建议**: Unify to `\phi_i` throughout, as method.tex is the more extensive section. Update problem.tex to match.

2. **Right boundary condition sign convention**:
   - `problem.tex` (eq:bc_right): $-D_i \partial\varphi_i/\partial x|_{x=0.5} = 0$
   - `method.tex` (eq:bc-right): $\partial\phi_i/\partial x|_{x=0.5} = 0$
   - Both are mathematically equivalent, but the notational inconsistency may confuse reviewers.
   - **建议**: Standardize to the form without the $-D_i$ factor (zero Neumann = zero gradient), or use current form in both.

3. **Duplicate parameter table**: Table 1 (`tab:params`) is defined in both `problem.tex` (line 39) and `method.tex` (line 24). This will cause a LaTeX duplicate label warning.
   - **建议**: Remove the table from `method.tex` §4.1 and cross-reference `Table~\ref{tab:params}` from `problem.tex`.

4. **Duplicate equation labels**: The governing equations are defined twice:
   - `problem.tex`: `eq:fast_group`, `eq:thermal_group`
   - `method.tex`: `eq:fast-group`, `eq:thermal-group` (different label format)
   - While they won't collide (different label strings), having two versions of the same equations wastes space.
   - **建议**: In method.tex §4.1, reference the equations from problem.tex instead of re-stating them.

5. **Introduction says "Fourier sine series"** (line 27, 32) but the actual solution uses **Fourier cosine series** (cos basis functions). This is factually incorrect.
   - **建议**: Change "Fourier sine series" → "Fourier cosine series" in introduction.tex (2 occurrences).

6. **Score formula inconsistency**:
   - `problem.tex` (eq:combined_score): $\text{Score} = \|\text{Res}_{\text{PDE}}\|_{L^2} + \|\text{Res}_{\text{BC}}\|_{L^2}$ (lower is better, perfect = 0)
   - `method.tex` (eq:combined-score): $S = 1/(1 + \bar{R}_{\text{PDE}} + \bar{R}_{\text{BC}})$ (higher is better, perfect = 1)
   - `comparison_table.md` and `convergence_data.json` use the $S = 1/(1+...)$ convention.
   - **建议**: Unify to the $S = 1/(1+...)$ convention used in the actual evaluator. Update problem.tex eq:combined_score.

7. **FaMoU acronym expansion inconsistency**:
   - `method.tex`: "**Fa**st **Mo**de m**U**tation" → FaMoU
   - This expansion doesn't match the casing well ("mUtation" for "U" is a stretch)
   - **建议**: Clarify or simplify the acronym. If FaMoU is an established project name, just introduce it without forced expansion.

8. **Conclusion contains unverified claims**:
   - "PDE residuals approaching machine precision ($\sim 10^{-14}$)" — actual PDE L2 residual is ~4.14e-5 (limited by numerical differentiation). The claim of $10^{-14}$ would only hold if the analytical solution were evaluated symbolically, not through the finite-difference evaluator.
   - **建议**: Change to "PDE residuals below $O(10^{-5})$ as measured by the numerical evaluator, with the analytical solution satisfying the PDE exactly in the continuous limit."

9. **Conclusion has placeholder text**: `\textcolor{red}{[X]}` appears 3 times for baseline comparison magnitudes.
   - **建议**: Fill in from `comparison_table.md` data once FaMoU results are available.

10. **Abstract claim "PDE residuals near machine precision"**: Same issue as conclusion — overstates the measured accuracy.
    - **建议**: Revise to match actual evaluator-measured residuals.

---

### ❌ 阻塞项（必须修复）

1. **Experiments section is 85% TODO**: Only the Ablation Study (§5.3) is complete. Missing:
   - §5.1 Experimental Setup (domain, metrics, hardware)
   - §5.2 Baselines (FDM, FEM, PINN, Truncated descriptions)
   - §5.3 Main Results (comparison table — data available in `comparison_table.md` but not yet in LaTeX)
   - §5.5 Convergence Analysis (FaMoU evolution rounds — awaiting Task#10)
   - §5.6 Qualitative Analysis (flux field visualization — awaiting fig_flux_field.png)
   - **依赖**: FaMoU 演化结果（Task#10 in_progress）

2. **Missing figures (2/6)**:
   - `fig_comparison.png` ❌ — baseline comparison bar chart (Task#22 in_progress)
   - `fig_flux_field.png` ❌ — flux distribution contour map (Task#20 in_progress)
   - These are referenced/needed in experiments but not yet generated.

3. **FaMoU evolution results unavailable**: Task#10 still in_progress. Without these results:
   - Cannot fill "Ours (FaMoU)" row in main comparison table
   - Cannot write §5.5 (score vs. evolution round)
   - Cannot complete conclusion with actual improvement magnitudes

---

### 页数估算

| Section | Est. Pages | Status |
|---------|-----------|--------|
| Title + Abstract | 0.5 | ✅ Complete |
| Introduction | 1.5 | ✅ Complete |
| Related Work | 2.5 | ✅ Complete |
| Problem Formulation | 2.0 | ✅ Complete (has overlap with Method) |
| Method | 4.5 | ✅ Complete (6 subsections + figure + table) |
| Experiments | 1.5 | ⚠️ Only ablation done (~1.5 of ~4 target pages) |
| Conclusion | 0.75 | ⚠️ Has placeholders |
| References | 1.0 | ✅ 23 entries |
| **Total** | **~14.25** | |

**距离 16 页目标还差 ~1.75 页。** 建议扩展：
- **Experiments §5.1–§5.3**: +2.0 pages（Setup + Baselines + Main Results table + analysis）
- **Experiments §5.5–§5.6**: +1.0 page（evolution convergence plot + flux field figure + discussion）
- **Consider reducing** Problem Formulation overlap with Method（currently ~1.5 pages of redundant content）— could save ~1 page and reinvest in Experiments

### 优先修复建议（按影响力排序）

1. 🔴 等待 FaMoU 结果（Task#10）→ 解锁 experiments 章节
2. 🔴 修复 Score 公式不一致（problem.tex vs method.tex）
3. 🟡 修复 "Fourier sine" → "Fourier cosine"（introduction.tex）
4. 🟡 统一变量符号 \varphi → \phi
5. 🟡 删除 method.tex 中重复的参数表
6. 🟡 修正 conclusion 中 "machine precision" 过度声明
7. 🟢 修正 FaMoU 缩写展开

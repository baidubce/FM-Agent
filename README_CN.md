<div align="right">
  <b>简体中文</b> | <a href="README.md">English</a>
</div>

<h1 align="center">伐谋 智能体</h1>

<div align="center">

🚩 <a href="https://cloud.baidu.com/product/famou.html" style="vertical-align:middle;"> **官方网址**</a> |
📄 **[技术报告](https://github.com/baidubce/FM-Agent/blob/main/docs/FMAgent_TechReport.pdf)** |
📌  <a href="https://arxiv.org/pdf/2510.26144" style="vertical-align:middle;"> **Arxiv 链接**</a> |
🧪 <a href="https://baidubce.github.io/FM-Agent/FDE-Bench/" style="vertical-align:middle;"> **FDE-Bench**</a> |
<a href="https://cloud.baidu.com/" style="vertical-align:middle;"><img src="docs/images/ACG.png" alt="ModelBuilder" width="16" height="16" style="vertical-align:middle;"/> **百度智能云**</a>
</div>


<p align="center">
  <img src="docs/images/main.png" width="700" height="700"/>
</p>
 
## 📰 最新动态
- **[2026-10]** 🔥 发布 **[FDE-Bench](https://baidubce.github.io/FM-Agent/FDE-Bench/)**：面向「需求不完整的真实业务请求」的端到端交付评测集，包含 49 个真实客户项目（29 个组合优化 + 20 个机器学习）和 266 个标注澄清点。[项目主页](https://baidubce.github.io/FM-Agent/FDE-Bench/) · [arXiv](https://arxiv.org/abs/2610.14539) · [代码](FDE-Bench/)
- **[2026-03]** 发布 **[Famou for Science](Famou_for_Science/)**：基于伐谋的自主科研工作流，附核反应堆物理示例。
- **[2025-10]** **[FM Agent](https://arxiv.org/abs/2510.26144)** 技术报告发布于 arXiv。

## 🗂️ 系列工作
| 工作 | 类型 | 简介 | 链接 |
| --- | --- | --- | --- |
| **FDE-Bench** | 评测集 | 评估智能体能否通过与模拟客户对话澄清缺失需求，并交付满足真实客户验收标准的方案。 | [项目主页](https://baidubce.github.io/FM-Agent/FDE-Bench/) · [arXiv](https://arxiv.org/abs/2610.14539) · [代码](FDE-Bench/) |
| **Famou for Science** | 应用 | 用伐谋端到端推进科研、实验与论文写作；示例为二维双群中子扩散方程解析解搜索。 | [目录](Famou_for_Science/) |
| **FM Agent** | 框架 | 结合大模型推理与大规模进化搜索的通用多智能体框架（详见下文）。 | [arXiv](https://arxiv.org/abs/2510.26144) · [技术报告](docs/FMAgent_TechReport.pdf) |

## FM Agent
FM Agent 是一种新颖的、通用的多智能体框架，它通过协同结合大语言模型驱动的推理和大规模进化搜索来解决复杂的现实世界难题。我们的系统展现出广泛的适用性，已在包括运筹学、机器学习、GPU 内核优化和经典数学问题在内的各种领域进行了评估。


## 技术优势
### ❄️ 冷启动初始化
这个阶段整合了多种生成智能体，旨在产生一个广泛且高质量的初始解空间。此外，借助一个可选的 expert-in-the-loop 设计，该框架确保了进化搜索从一个具有实用基础的起点开始，这在某些现实世界的复杂案例中能显著加速收敛。

### 🧬 自适应多样性驱动采样
我们新颖的采样策略协调多个并行进化岛，通过动态资源分配自适应地平衡探索与利用。该机制在算法谱系中保持了富有成效的多样性，同时系统地引导种群趋向全局最优。

### 🎯 领域特定评估
自定义评估器综合了多个关键标准——包括功能正确性、操作有效性和LLM 监督的质量评估——以生成细致入微、多维度的反馈。这种全面的评分机制提供了丰富且累积的信号，能够精确指导迭代优化过程。

### 🚀 分布式异步基础设施
我们的可扩展编排框架基于 Ray 构建，支持在分布式计算资源上进行细粒度、大规模的并发评估。这种架构确保了高效的资源利用，同时促进了对复杂、高维解空间的快速而系统的探索。
  
## 性能指标
FM Agent 在无人为干预或调优的情况下，自主达到了最先进的结果：在 ALE-Bench 上达到 **1976.3** (+5.2%)，在 MLE-bench 上达到 **43.56**% (+4.0pp)，在 KernelBench 上实现了高达 **20×** 的加速，并在多个经典数学问题上建立了新的最先进结果。

### MLE-bench
💥💥💥FM Agent is currently ranked first on the [MLE-bench Leaderboard](https://github.com/openai/mle-bench?tab=readme-ov-file).
<p align="center">
  <img src="docs/images/mlebench_result.png" width="500" height="500"/>
</p>


### ALE-Bench
<p align="center">
  <img src="docs/images/alebench_result.png" width="500" height="500"/>
</p>


### KernelBench
<p align="center">
  <img src="docs/images/kernelbench_result.png" width="500" height="500"/>
</p>

## 引用

如果您在研究中使用了FM Agent，请引用：

```bibtex
@misc{li2025fmagent,
      title={The FM Agent}, 
      author={Annan Li and Chufan Wu and Zengle Ge and Yee Hin Chong and Zhinan Hou and Lizhe Cao and Cheng Ju and Jianmin Wu and Huaiming Li and Haobo Zhang and Shenghao Feng and Mo Zhao and Fengzhi Qiu and Rui Yang and Mengmeng Zhang and Wenyi Zhu and Yingying Sun and Quan Sun and Shunhao Yan and Danyu Liu and Dawei Yin and Dou Shen},
      year={2025},
      eprint={2510.26144},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.26144}, 
}
```

如果您使用了 FDE-Bench，请引用：

```bibtex
@misc{li2026fdebench,
      title={FDE-Bench: Evaluating End-to-End Delivery from Underspecified Real-World Business Requests},
      author={Huaiming Li and Can Huang and Chufan Wu and Danyu Liu and Xiaomin Yuan and Qinze Li and Yui Lo and Wenhui Bai and Qianlong Wang and Zengle Ge and Xiyu Yan and Yuheng Cheng and Jinhui Ren and Guokai Chen and Jiaqun Liu and Haoran Li and Qianhui Liu and Annan Li and Jianmin Wu and Dawei Yin and Dou Shen},
      year={2026},
      eprint={2610.14539},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.14539},
}
```

## 许可证

本项目遵循 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

FDE-Bench 的 harness 代码同样采用 [MIT](FDE-Bench/LICENSE) 许可证，但其评测数据（`FDE-Bench/case/` 与数据压缩包）采用 [CC BY-NC-ND 4.0](FDE-Bench/LICENSE-DATA) 许可，详见 [`FDE-Bench/LICENSE_STATUS.md`](FDE-Bench/LICENSE_STATUS.md)；`docs/FDE-Bench/` 下的论文与图片版权归作者所有，内置字体沿用各自的 OFL 许可。

## 联系我们

- GitHub Issues: [提交问题](https://github.com/baidubce/FM-Agent/issues)

---


<div align="right">
  <a href="README_CN.md">简体中文</a> | <b>English</b>
</div>

<h1 align="center">Famou Agent</h1>

<div align="center">

🚩 <a href="https://cloud.baidu.com/product/famou.html" style="vertical-align:middle;"> **Official Website**</a> |
📄 **[Tech Report](https://github.com/baidubce/FM-Agent/blob/main/docs/FMAgent_TechReport.pdf)** |
📌  <a href="https://arxiv.org/pdf/2510.26144" style="vertical-align:middle;"> **Arxiv Link**</a> |
🧪 <a href="https://baidubce.github.io/FM-Agent/FDE-Bench/" style="vertical-align:middle;"> **FDE-Bench**</a> |
<a href="https://cloud.baidu.com/" style="vertical-align:middle;"><img src="docs/images/ACG.png" alt="ModelBuilder" width="16" height="16" style="vertical-align:middle;"/> **Baidu AI Cloud**</a>
</div>


<p align="center">
  <img src="docs/images/main.png" width="700" height="700"/>
</p>
 
## 📰 News
- **[2026-10]** 🔥 We release **[FDE-Bench](https://baidubce.github.io/FM-Agent/FDE-Bench/)**, a benchmark for end-to-end delivery from underspecified real-world business requests: 49 real customer projects (29 combinatorial optimization + 20 machine learning) with 266 annotated clarification targets. [Project Page](https://baidubce.github.io/FM-Agent/FDE-Bench/) · [arXiv](https://arxiv.org/abs/2610.14539) · [Code](FDE-Bench/)
- **[2026-03]** We release **[Famou for Science](Famou_for_Science/)**, an autonomous research workflow powered by Famou, with a nuclear reactor physics example.
- **[2025-10]** The **[FM Agent](https://arxiv.org/abs/2510.26144)** technical report is available on arXiv.

## 🗂️ Our Works
| Work | Type | Description | Links |
| --- | --- | --- | --- |
| **FDE-Bench** | Benchmark | Evaluates whether agents can clarify missing requirements with a simulated customer and deliver a solution that meets real customer acceptance criteria. | [Project Page](https://baidubce.github.io/FM-Agent/FDE-Bench/) · [arXiv](https://arxiv.org/abs/2610.14539) · [Code](FDE-Bench/) |
| **Famou for Science** | Application | Drives research, experiments, and paper writing end to end with Famou; the example searches for analytical solutions of the 2D two-group neutron diffusion equation. | [Directory](Famou_for_Science/) |
| **FM Agent** | Framework | General-purpose multi-agent framework combining LLM reasoning with large-scale evolutionary search (details below). | [arXiv](https://arxiv.org/abs/2510.26144) · [Tech Report](docs/FMAgent_TechReport.pdf) |

## FM Agent
FM Agent is a novel, general-purpose multi-agent framework that addresses complex real-world challenges by synergistically combining LLM-based reasoning and large-scale evolutionary search. Demonstrating broad applicability, our system has been evaluated across diverse domains, including operations research, machine learning, GPU kernel optimization, and classical mathematical problems.


## Technical Advantages
### ❄️ Cold-Start Initialization
This phase integrates  diverse generation agents to produce a broad yet high-quality initial solution space. Moreover, with an optional expert-in-the-loop design, the framework ensures evolutionary search begins from a pragmatically grounded foundation, significantly accelerating convergence, especially in some real-world complex cases.

### 🧬 Adaptive Diversity-Driven Sampling
Our novel sampling strategy orchestrates multiple parallel evolutionary islands, adaptively balancing exploration and exploitation through dynamic resource allocation. This mechanism maintains productive diversity across algorithmic lineages while systematically steering the population toward global optima.

### 🎯 Domain-Specific Evaluation
Custom evaluators synthesize multiple critical criteria—including functional correctness, operational effectiveness, and LLM-supervised quality assessment—to generate nuanced, multi-faceted feedback. This comprehensive scoring mechanism provides rich, cumulative signals that precisely guide the iterative refinement process.

### 🚀 Distributed Asynchronous Infrastructure
Built on Ray, our scalable orchestration framework enables fine-grained, large-scale concurrent evaluation across distributed computing resources. This architecture ensures efficient resource utilization while facilitating rapid and systematic exploration of complex, high-dimensional solution spaces.
  
## Performance Metrics
FM Agent reaches state-of-the-art results autonomously, without human interpretation or tuning — **1976.3** on ALE-Bench (+5.2%), **43.56**% on MLE-bench (+4.0pp), up to **20×** speedups on KernelBench, and establishes new state-of-the-art(SOTA) results on several classical mathematical problems.

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

## Citation

If you use FM Agent in your research, please cite:

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

If you use FDE-Bench, please cite:

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

FDE-Bench's harness code is also licensed under [MIT](FDE-Bench/LICENSE), but its benchmark data (`FDE-Bench/case/` and the data archives) are licensed under [CC BY-NC-ND 4.0](FDE-Bench/LICENSE-DATA). See [`FDE-Bench/LICENSE_STATUS.md`](FDE-Bench/LICENSE_STATUS.md) for details. The paper and figures under `docs/FDE-Bench/` remain with their authors, and the bundled fonts keep their own OFL licenses.

## Contact Us

- GitHub Issues: [Submit Issue](https://github.com/baidubce/FM-Agent/issues)

---

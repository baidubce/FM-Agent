<div align="right">
  <a href="README_CN.md">简体中文</a> | <b>English</b>
</div>

<h1 align="center">FM-Agent</h1>

<div align="center">

<a href="https://console.bce.baidu.com/qianfan/modelcenter/model/buildIn/list" style="vertical-align:middle;"><img src="docs/images/ACG.png" alt="ModelBuilder" width="16" height="16" style="vertical-align:middle;"/> **ModelBuilder**</a>

📄 **[Tech Report](https://github.com/baidubce/Qianfan-VL/blob/main/docs/qianfan_vl_report_comp.pdf)**

</div>


---

## Introduction
FM Agent is a novel, general-purpose multi-agent framework that addresses complex real-world challenges by synergistically combining LLM-based reasoning and large-scale evolutionary search. The core of FM Agent integrates several key innovations: 1) a cold-start initialization phase incorporating expert guidance, 2) a novel evolutionary sampling strategy for iterative optimization, 3) domain-specific evaluators that combine correctness, effectiveness, and LLM-supervised feedback, and 4) a distributed, asynchronous execution infrastructure built on Ray. Demonstrating broad applicability, our system has been evaluated across diverse domains, including operations research, machine learning, GPU kernel optimization, and classical mathematical problems.



## Performance Metrics
FM Agent reaches state-of-the-art results autonomously, without human interpretation or tuning — 1976.3 on ALE-Bench (+5.2%), 43.56% on MLE-Bench (+4.0pp), up to 20× speedups on KernelBench, and establishes new state-of-the-art(SOTA) results on several classical mathematical problems.

### MLE-Bench


### ALE-Bench


### KernelBench




## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact Us

- GitHub Issues: [Submit Issue](https://github.com/baidubce/FM-Agent/issues)

---

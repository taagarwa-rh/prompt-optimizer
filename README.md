# Prompt Optimizer

Improve your prompts with any LLM using Automatic Prompt Optimization (APO).

## Overview

![](./docs/_static/apo.png)
<center><small>From <i>"A Systematic Survey of Automatic Prompt Optimization Techniques"</i></small></center>

Automatic prompt optimization (APO) is a reinforcement learning technique to improve prompt performance.
At each iteration, new prompts are generated and scored against your AI system using your validation set.
Promising prompts are kept and used to seed the next generation of prompts.
The goal is to find the prompt that maximizes the AI system's performance on the evaluation metric you define.

## Installation

**uv (recommended)**

```sh
uv add git+https://github.com/taagarwa-rh/prompt-optimizer.git
```

## Usage

See [the docs](./docs/) for information on package usage.

## Roadmap

- [ ] Add prepackaged pipelines and metrics

## Citations

```
@inproceedings{Ramnath_2025,
   title={A Systematic Survey of Automatic Prompt Optimization Techniques},
   url={http://dx.doi.org/10.18653/v1/2025.emnlp-main.1681},
   DOI={10.18653/v1/2025.emnlp-main.1681},
   booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
   publisher={Association for Computational Linguistics},
   author={Ramnath, Kiran and Zhou, Kang and Guan, Sheng and Mishra, Soumya Smruti and Qi, Xuan and Shen, Zhengyuan and Wang, Shuai and Woo, Sangmin and Jeoung, Sullam and Wang, Yawei and Wang, Haozhu and Ding, Han and Lu, Yuzhe and Xu, Zhichao and Zhou, Yun and Srinivasan, Balasubramaniam and Yan, Qiaojing and Chen, Yueyan and Ding, Haibo and Xu, Panpan and Cheong, Lin Lee},
   year={2025},
   pages={33066–33098} }
```

```
@misc{zhou2023largelanguagemodelshumanlevel,
    title={Large Language Models Are Human-Level Prompt Engineers},
    author={Yongchao Zhou and Andrei Ioan Muresanu and Ziwen Han and Keiran Paster and Silviu Pitis and Harris Chan and Jimmy Ba},
    year={2023},
    eprint={2211.01910},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2211.01910},
}
```

```
@misc{pryzant2023automaticpromptoptimizationgradient,
    title={Automatic Prompt Optimization with "Gradient Descent" and Beam Search}, 
    author={Reid Pryzant and Dan Iter and Jerry Li and Yin Tat Lee and Chenguang Zhu and Michael Zeng},
    year={2023},
    eprint={2305.03495},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2305.03495}, 
}
```
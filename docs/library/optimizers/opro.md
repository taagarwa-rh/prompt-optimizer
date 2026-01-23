---
title: OPRO
---

# Optimization by PROmpting (APE)

## About

![](./_static/opro.png)

Optimization by PROmpting (OPRO) starts with a seed prompt.
At each iteration, a collection of past prompt candidates and scores and a random sample of input and output pairs from the validation set are formatted into a metaprompt.
The metaprompt is sent to the language model multiple times, each time asking it to provide a new candidate prompt that improves the instructions for the task.
These new candidates are scored and the iterations continue until the maximum depth is reached or the score threshold is exceeded.

## Citation

```
@misc{yang2024largelanguagemodelsoptimizers,
    title={Large Language Models as Optimizers},
    author={Chengrun Yang and Xuezhi Wang and Yifeng Lu and Hanxiao Liu and Quoc V. Le and Denny Zhou and Xinyun Chen},
    year={2024},
    eprint={2309.03409},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2309.03409},
}
```

## Source

::: src.prompt_optimizer.optimizers.opro
    options:
      heading_level: 2
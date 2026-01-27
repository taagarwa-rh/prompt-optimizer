# Prompt Optimizer

Improve your prompts with any LLM using Automatic Prompt Optimization (APO).

## Overview

![](./static/apo.png)
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

### Getting Started

Before you can use an optimizer, you must define your **validation set** and **evaluator**.

The **validation set** is a set of examples for your task, and should be a list of dictionaries.
For example, a simple QA validation set might look like:

```python
validation_set = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury"},
]
```

The **evaluator** is your scoring function for generated prompts.
It should take a prompt and your validation set and produce a set of predictions, one for each example in the validation set.
That score should represent how much you value the predictions from a prompt.
For example, a simple QA evaluator might look like:

```python
from datetime import datetime

from langchain_openai import ChatOpenAI
from prompt_optimizer import BasePrompt


def evaluator(prompt: BasePrompt, validation_set: list[dict]) -> float:
    """Prompt evaluator function."""
    predictions = []
    num_correct = 0
    llm = ChatOpenAI(model="gpt-5")
    for row in validation_set:
        # Get the prediction from your system
        question = row["question"]
        messages = [{"role": "system", "content": prompt.content}, {"role": "user", "content": question}]
        response = llm.invoke(messages)
        prediction = response.content.strip()
        predictions.append(prediction)

        # Reward exact matches
        actual = row["answer"]
        if actual == prediction:
            num_correct += 1
        else:
            num_correct += 0
    
    # Compute the score
    score = num_correct / len(validation_set)

    # Optionally, save the predictions and other info in metadata
    prompt.metadata["predictions"] = predictions
    prompt.metadata["run_date"] = datetime.now()

    return score
```

### Using Optimizers

Once you have your **validation set** and **evaluator** defined, you can set up an optimization pipeline.

Select one of the available optimizers to learn more about its usage:

- [OPRO](./library/optimizers/opro.md) (Recommended)
- [ProTeGi](./library/optimizers/protegi.md)
- [APE](./library/optimizers/ape.md)

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

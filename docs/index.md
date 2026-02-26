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
from prompt_optimizer import Prompt


def evaluator(prompt: Prompt, validation_set: list[dict]) -> list[str]:
    """Prompt evaluator function."""
    # Run the prompt through the AI system
    predictions = []
    num_correct = 0
    agent = ChatOpenAI(model="gpt-5", temperature=0.1)
    for row in validation_set:
        question = row["input"]
        messages = [{"role": "system", "content": prompt.content}, {"role": "user", "content": question}]
        response = agent.invoke(messages)
        prediction = response.content.strip()
        predictions.append(prediction)
    
        # Reward exact matches and collect errors
        actual = row["target"]
        if actual == prediction:
            num_correct += 1
        else:
            num_correct += 0
            # Save prediction error - Required for some optimizers
            error = PredictionError(input=question, prediction=prediction, actual=actual, feedback=None)
            prompt.errors.append(error)
    
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

- [PromptAgent](./optimizers/promptagent.md) (Recommended)
- [OPRO](./optimizers/opro.md)
- [ProTeGi](./optimizers/protegi.md)
- [APE](./optimizers/ape.md)

For example, **PromptAgent** usage looks like:

```py
from lagnchain_openai import ChatOpenAI
from prompt_optimizer.optimizers import PromptAgentOptimizer

# A langchain ChatModel for generating new prompts
client = ChatOpenAI(model="gpt-5", temperature=0.7)

# Initialize the optimizer
baseline_prompt = "Answer the user's questions to the best of your ability."
optimizer = PromptAgentOptimizer(
    client=client,
    seed_prompts=[baseline_prompt],
    validation_set=validation_set,
    max_depth=3,
    evaluator=evaluator,
)

# Run the optimization
optimized_prompt = optimizer.run()

# Print the optimized prompt
print(optimized_prompt.content)
# "Provide a simple answer to the user's question. Use as few words as possible."
```

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

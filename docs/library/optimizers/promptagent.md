---
title: PromptAgent
---

# PromptAgent

## About

![](./static/promptagent.png)

PromptAgent optimizes a prompt by scoring, generating feedback, and generating new prompts based on the feedback.
PromptAgent starts from a seed prompt with known errors on the training data.
At each step, a scored prompt and a sample of its errors are passed to a language model to produce a feedback "action".
Then the prompt, errors, trajectory, and feedback action are passed to a language model to produce new prompts.
These new prompts are scored and the best prompt from each branch (`search_mode="beam"`) or all prompts (`search_mode="greedy"`) are retained for the next step

## Usage

The `PromptAgentOptimizer` requires a description of the failures after each step.
You must provide this feedback in your evaluator by capturing errors and saving them in the prompt object's `errors` attribute.

> [!IMPORTANT] Important Note
> Your evaluator function \*\*MUST\*\* save any errors to the prompt object's `errors` attribute. Otherwise the optimization will fail.

```python
from lagnchain_openai import ChatOpenAI
from prompt_optimizer import BasePrompt, PredictionError
from prompt_optimizer.optimizers import PromptAgentOptimizer

# Simple QA validation set
validation_set = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury"},
    {"question": "What is the longest river in the world?", "answer": "Nile"},
    {"question": "What is the smallest river in the world?", "answer": "Reprua River"},
]

# A langchain ChatModel for generating new prompts
client = ChatOpenAI(model="gpt-5", temperature=0.7)

# Evaluator function
def evaluator(prompt: BasePrompt, validation_set: list[dict]) -> list[str]:
    """Prompt evaluator function."""
    # Run the prompt through the AI system
    predictions = []
    num_correct = 0
    agent = get_agent()
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
            # Save prediction error - Required for PromptAgentOptimizer
            error = PredictionError(input=question, prediction=prediction, actual=actual, feedback=None)
            prompt.errors.append(error)
    
    # Compute the score
    score = num_correct / len(validation_set)
    
    return score

# Initialize the optimizer
optimizer = PromptAgentOptimizer(
    client=client,
    seed_prompts=[baseline_prompt],
    validation_set=validation_set,
    max_depth=3,
    evaluator=evaluator,
)

# Run the optimization
optimized_prompt = optimizer.run()
```

## Citation

```
@misc{wang2023promptagentstrategicplanninglanguage,
    title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
    author={Xinyuan Wang and Chenxi Li and Zhen Wang and Fan Bai and Haotian Luo and Jiayou Zhang and Nebojsa Jojic and Eric P. Xing and Zhiting Hu},
    year={2023},
    eprint={2310.16427},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2310.16427},
}
```

## Source

::: src.prompt_optimizer.optimizers.protegi
    options:
      heading_level: 2

---
title: ProTeGi
---

# Prompt Optimization with Textual Gradients (ProTeGi)

## About

![](./static/protegi.png)

Prompt Optimization with Textual Gradients (ProTeGi) starts from an initial prompt and iteratively updates using feedback from the language model, called "gradients".
The initial prompt is scored using the validation set and the failures are collected.
The failures are described to the language model, and the language model is asked to provide multiple feedbacks (gradients) to describe how the prompt should be improved.
The language model then uses each of these gradients to generate a number of new prompts, and these new prompts are scored.
Then the language model generates variations
The top scoring prompt is retained and becomes the initial prompt for the next iteration. 
The iterations continue until the maximum iteration depth is reached or the score threshold is exceeded.

A variation of this using "greedy" search instead keeps all prompts at each iteration, leading to larger trees of prompts.

## Usage

The `ProtegiOptimizer` requires a description of the failures after each step.
You must provide this feedback in your evaluator by assigning an `"error_string"` in the prompt's metadata.

> [!IMPORTANT] Important Note
> Your evaluator function \*\*MUST\*\* create the "error_string" and save it in metadata. Otherwise the optimization will fail.

```python
from lagnchain_openai import ChatOpenAI
from prompt_optimizer import ProtegiOptimizer

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
    errors = []
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
            error = f"Question: '{question}'\nPrediction: '{prediction}'\nActual Answer: {actual}"
            errors.append(error)
    
    # Compute the score
    score = num_correct / len(validation_set)
    
    # Save the error string (required for ProtegiOptimizer)
    prompt.metadata["error_string"] = "\n\n".join(errors)
    
    return score

# Initialize the optimizer
optimizer = ProtegiOptimizer(
    client=client,
    seed_prompts=[baseline_prompt],
    validation_set=validation_set,
    evaluator=evaluator,
)

# Run the optimization
optimized_prompt = optimizer.run()
```

## Citation

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

## Source

::: src.prompt_optimizer.optimizers.protegi
    options:
      heading_level: 2

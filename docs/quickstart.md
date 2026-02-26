# Quickstart

## Installation

**uv (recommended)**

```sh
uv add git+https://github.com/taagarwa-rh/prompt-optimizer.git
```

## Validation Set and Evaluator

Before you can use an optimizer, you must define your **validation set** and **evaluator**.

The **validation set** is a set of examples for your task, and should be a list of dictionaries.
We'll create a validation set for a basic question-answering (QA) system.

```python
validation_set = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury"},
]
```

The **evaluator** is your scoring function for generated prompts.
It should take a prompt and your validation set and produce a set of predictions, one for each example in the validation set.
Then it should score the prompt with a numeric score.
You may want to reward a prompt for getting the right answer, or answering in as few tokens as possible, or other objectives.

We'll create a simple QA evaluator that invokes GPT-5 directly, then checks for an exact match between the expected answer and the response.

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

## Using Optimizers

Once you have your **validation set** and **evaluator** defined, you can set up an optimization pipeline.

We'll use **PromptAgent** to optimize our prompt:

```py
from langchain_openai import ChatOpenAI
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

## Next Steps

Check out the available optimizers to learn more about their capabilities and usage:

- [PromptAgent](./optimizers/promptagent.md) (Recommended)
- [OPRO](./optimizers/opro.md)
- [ProTeGi](./optimizers/protegi.md)
- [APE](./optimizers/ape.md)

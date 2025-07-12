# Prompt Optimizer

Optimize your prompts using Reinforcement Learning.

## Setup

See the [example](./examples/simple_optimize.py) for an sample pipeline setup.

There are three key components you need to define in order to execute the optimizer:

1. [Dataset](#1-dataset) - your test cases
2. [Pipeline](#2-pipeline) - a function that executes your AI pipeline
3. [Reward Function](#3-reward-function) - a function that rewards outputs from your pipeline and collects errors

### 1. Dataset
A pandas DataFrame of test cases. Can include any columns, but generally it will include a column for input and a column for output (e.g. question and answer).

```python
import pandas as pd

data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"}
]
dataset = pd.DataFrame(data)
```

### 2. Pipeline
A function that takes a prompt and a single row from your dataset as kwargs and produces an output (e.g., a string response). We are using an OpenAI client that pings Ollama locally to make our completions, but you can use any logic here to produce your output (e.g. RAG, structured output, classifier, etc.).

```python
from openai import Client

model_name = "qwen2.5:32b" # Or your preferred model
client = Client(base_url="http://localhost:11434/v1", api_key="NONE")

def pipeline(prompt: str, question: str, **kwargs) -> str:
    messages = [
        {"role": "system", "content": prompt}, 
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(messages=messages, model=model_name)
    result = response.choices[0].message.content.strip()
    return result
```

### 3. Reward Function
A function that takes a list of prediction (pipeline outputs) and other columns from your dataset as kwargs and returns a dictionary, with a reward and an error_string explaining what the predictions got wrong.

```python
def reward_func(predictions: list[str], question: list[str], answer: list[str], **kwargs) -> dict:
    reward = 0.0
    errors = []
    for i, prediction in enumerate(predictions):
        actual = answer[i]
        ques = question[i]
        if actual == prediction:
            reward += 1
        else:
            rewards += 0
            errors.append(f"Failed to answer the question: {ques}\nPredicted: {prediction}\nActual: {actual}")
    return {
        "reward": reward,
        "error_string": "\n\n".join(errors) if errors else None
    }
```

## Usage
With your pipeline, reward function, and dataset defined, you can build a prompt optimizer. We are passing an OpenAI client pointing to Ollama locally to use for generating new prompts. The `GradientOptimizer` and `StructuredGradientOptimizer` are based on the paper [Automatic Prompt Optimization with “Gradient Descent” and Beam Search](https://arxiv.org/pdf/2305.03495).

```python
from prompt_optimizer.optimizers import GradientOptimizer, StructuredGradientOptimizer

model_name = "qwen2.5:32b" # Or your preferred model
client = Client(base_url="http://localhost:11434/v1", api_key="NONE")

optimizer = GradientOptimizer(
    pipeline=pipeline, 
    reward, 
    dataset=dataset, 
    model_name=model_name, 
    client=client
)

# Or for more stable outputs
optimizer = StructuredGradientOptimizer(
    pipeline=pipeline, 
    reward_func=reward_func, 
    dataset=dataset, 
    model_name=model_name, 
    client=client
)
```

Once you have an optimizer defined, you can use the `optimize` method to generate new prompts based on your baseline prompt.

Here is the prompt before optimization:
```python
baseline_prompt = "Please answer every question the user asks to the best of your ability."
response = pipeline(prompt=baseline_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
```
```
'To Kill a Mockingbird' was written by Harper Lee. It was published in 1960 and became widely known for its portrayal of racial injustice in the American South.
```

Now we optimize it and view the best prompt:
```python
prompts = optimizer.optimize(baseline_prompt=baseline_prompt)
print(prompts[0].prompt)
```
```
Please provide only the most direct answer to the user's question without any additional information or explanation.
```

Now we use the best prompt in our pipeline:
```python
new_prompt = prompts[0].prompt
response = pipeline(prompt=new_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
```
```
Harper Lee
```


## Roadmap

- [ ] Add sample pipelines and metrics
- [ ] Improve stability of default prompts
- [ ] AgentPrompt Optimizer from [arXiv](https://arxiv.org/pdf/2310.16427)
- [ ] APE Optimizer from [arXiv](https://arxiv.org/abs/2211.01910)
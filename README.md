# Prompt Optimizer

Optimize your prompts with gradient prompt tuning.

## Setup

See the [example](./examples/simple_optimize.py) for an sample pipeline setup.

There are three key components you need to define in order to execute the optimizer:

1. Dataset - your test cases
2. Pipeline - a function that executes your AI pipeline
3. Metric - a function that scores outputs from your pipeline

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
A function that takes a prompt and a row from your dataset as kwargs and produces an output (e.g., a string response). We are using an OpenAI client that pings Ollama locally to make our completions, but you can use any logic here to produce your output (e.g. RAG, structured output, classifier, etc.).

```python
from openai import Client

model_name = "qwen2.5:32b" # Or your preferred model
client = Client(base_url="http://localhost:11434/v1", api_key="NONE")

def pipeline(prompt: str, question: str, **kwargs) -> str:
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    response = client.chat.completions.create(messages=messages, model=model_name)
    result = response.choices[0].message.content.strip()
    return result
```

### 3. Metric
A function that takes a prediction (a pipeline output) and a corresponding row from your dataset as kwargs and produce a dictionary, with a score <= 1 and an error_string explaining what it got wrong.

```python
def metric(prediction: str, question: str, answer: str, **kwargs) -> dict:
    if answer == prediction:
        score = 1
        error_string = None
    else:
        score = 0
        error_string = f"Failed to answer the question: {question}\nPredicted: {prediction}\nActual: {answer}"
    return {
        "score": score,
        "error_string": error_string
    }
```

## Usage
With your pipeline, metric, and dataset defined, you can build a prompt optimizer. We are passing an OpenAI client pointing to Ollama locally to use as our prompt optimizer.

```python
model_name = "qwen2.5:32b" # Or your preferred model
client = Client(base_url="http://localhost:11434/v1", api_key="NONE")

optimizer = GradientOptimizer(pipeline=pipeline, metric=metric, dataset=dataset, model_name=model_name, client=client)
```

Once you have an optimizer defined, you can use the `optimize` method to generate new prompts based on your baseline prompt.

Here is the prompt before optimization:
```python
baseline_prompt = "Please answer every question the user asks to the best of your ability."
response = pipeline(prompt=baseline_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
# "'To Kill a Mockingbird' was written by Harper Lee. It was published in 1960 and became widely known for its portrayal of racial injustice in the American South."
```

Now we optimize it and view the best prompt:
```python
prompts = optimizer.optimize(baseline_prompt=baseline_prompt)
print(prompts[0].prompt)
# Please provide only the most direct answer to the user's question without any additional information or explanation.
```

Now we use the best prompt in our pipeline:
```python
new_prompt = prompts[0].prompt
response = pipeline(prompt=new_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
# "Harper Lee"
```


## Roadmap

- [ ] MLFlow integration for metric tracking and more
- [ ] In depth reporting on intermediate prompts and results
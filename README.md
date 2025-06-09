# Prompt Optimizer

Optimize your prompts with gradient prompt tuning

## Usage

```python
import pandas as pd
from openai import Client
from prompt_optimizer.optimizers import GradientOptimizer

model_name = "qwen2.5:7b"
client = Client(base_url="http://localhost:11434/v1", api_key="NONE")

def pipeline(prompt: str, question: str, **kwargs) -> str:
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    response = client.chat.completions.create(messages=messages, model=model_name)
    result = response.choices[0].message.content.strip()
    print(result)
    return result

def metric(dataset: pd.DataFrame, predictions: pd.Series) -> MetricResult:
    score = 0
    errors = []
    for i in range(len(dataset)):
        answer = dataset.iloc[i]["answer"]
        prediction = predictions[i]
        if answer == prediction:
            score += 1
        else:
            question = dataset.iloc[i]["question"]
            error = f"Failed to answer the question: {question}\nPredicted: {prediction}\nActual: {answer}"
            errors.append(error)
    error_string = "\n\n".join(errors)
    score = round(score / len(dataset), 2)
    return MetricResult(score=score, error_string=error_string)

data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"}
]
dataset = pd.DataFrame(data)

optimizer = GradientOptimizer(pipeline=pipeline, metric=metric, dataset=dataset, model_name=model_name, client=client)

baseline_prompt = "Please answer every question the user asks to the best of your ability."

results = optimizer.optimize(baseline_prompt=baseline_prompt)

print(results)
```
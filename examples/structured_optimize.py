import pandas as pd
from openai import Client
import logging

from prompt_optimizer.optimizers import StructuredGradientOptimizer

logging.basicConfig(level=logging.INFO)

# 1. Create an OpenAI client pointing to Ollama for optimizing the prompt
model_name = "qwen2.5:32b" # Or your preferred model
client = Client(base_url="http://localhost:11434/v1", api_key="NONE")

# 2. Create your dataset of test cases
data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"}
]
dataset = pd.DataFrame(data)

# 3. Define your pipeline. This should take prompt and a row from your dataset as kwargs and produce an output (e.g., a string response).
def pipeline(prompt: str, question: str, **kwargs) -> str:
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    response = client.chat.completions.create(messages=messages, model=model_name)
    result = response.choices[0].message.content.strip()
    return result

# 4. Define your metric function. This should take a prediction (a pipeline output) and a corresponding row from your dataset as kwargs and produce a dictionary, 
# with a score <= 1 and an error_string explaining what it got wrong. This metric checks for an exact match between the predicted and actual answer.
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

# 5. Define your Optimizer, requires a pipeline, metric, dataset, and a model name. 
# We are passing an OpenAI client as well so we can use Ollama locally.
optimizer = StructuredGradientOptimizer(pipeline=pipeline, metric=metric, dataset=dataset, model_name=model_name, client=client)

# 6. Demonstrate the baseline prompt
baseline_prompt = "Please answer every question the user asks to the best of your ability."
response = pipeline(prompt=baseline_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
# "'To Kill a Mockingbird' was written by Harper Lee. It was published in 1960 and became widely known for its portrayal of racial injustice in the American South."

# 7. Optimize your baseline prompt
prompts = optimizer.optimize(baseline_prompt=baseline_prompt, score_threshold=1.0)

# 8. View the newly created prompts (they are automatically sorted by score)
print("--- PROMPTS ---")
for i, prompt in enumerate(prompts):
    print(f"# Prompt {i+1}")
    print("Score:", prompt.score)
    print("Prompt:", prompt.prompt)
    
# # Prompt 1
# Score: 1.0
# Prompt: Answer each question with a precise response without additional context or explanation.

# 9. Demonstrate the effectiveness of the new prompt
new_prompt = prompts[0].prompt
response = pipeline(prompt=new_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
# "Harper Lee"
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
    response = client.chat.completions.create(messages=messages, model=model_name, temperature=0.1, seed=42)
    result = response.choices[0].message.content.strip()
    return result

# 4. Define your reward function. This should take the predictions (pipeline outputs) and other columns from your dataset as kwargs and produce a dictionary, 
# with a reward and an error_string explaining what it got wrong. This function rewards exact matches between the predicted and actual answer.
def reward_func(predictions: list[str], question: list[str], answer: list[str], **kwargs) -> dict:
    reward = 0.0
    errors = []
    for i, prediction in enumerate(predictions):
        actual = answer[i]
        ques = question[i]
        if actual == prediction:
            reward += 1
        else:
            reward += 0
            errors.append(f"Failed to answer the question: {ques}\nPredicted: {prediction}\nActual: {actual}")
    return {
        "reward": reward,
        "error_string": "\n\n".join(errors) if errors else None
    }

# 5. Define your Optimizer, requires a pipeline, reward_func, dataset, and a model name. 
# We are passing an OpenAI client as well so we can use Ollama locally.
optimizer = StructuredGradientOptimizer(pipeline=pipeline, reward_func=reward_func, dataset=dataset, model_name=model_name, client=client)

# 6. Demonstrate the baseline prompt
baseline_prompt = "Please answer every question the user asks to the best of your ability."
response = pipeline(prompt=baseline_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
# "'To Kill a Mockingbird' was written by Harper Lee. It was published in 1960 and became widely known for its portrayal of racial injustice in the American South."

# 7. Optimize your baseline prompt. Set a depth of 1 since this is a simple task and should converge quickly.
prompts = optimizer.optimize(baseline_prompt=baseline_prompt, depth=1)

# 8. View the newly created prompts (they are automatically sorted by reward)
print("--- PROMPTS ---")
for i, prompt in enumerate(prompts):
    print(f"# Prompt {i+1}")
    print("Reward:", prompt.reward)
    print("Prompt:", prompt.prompt)
    
# # Prompt 1
# Reward: 1.0
# Prompt: Answer each question with a precise response without additional context or explanation.

# 9. Demonstrate the effectiveness of the new prompt
new_prompt = prompts[0].prompt
response = pipeline(prompt=new_prompt, question="Who wrote 'To Kill a Mockingbird'?")
print(response)
# "Harper Lee"
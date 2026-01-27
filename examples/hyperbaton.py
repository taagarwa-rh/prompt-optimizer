import json
import os
from urllib.request import urlopen

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from prompt_optimizer.optimizers import OPROOptimizer
from prompt_optimizer.pipeline.base import BasePrompt

load_dotenv()

base_url = os.environ.get("OPENAI_BASE_URL")
api_key = os.environ.get("OPENAI_API_KEY")

def get_agent():
    """Get the LLM client for the AI system."""
    model = "qwen3:0.6b"
    client = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=0.0, max_completion_tokens=5)
    return client

def get_optimizer_client():
    """Get the LLM client for the optimizer."""
    model = "gpt-oss-20b"
    client = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=1.0, max_completion_tokens=2000)
    return client

def get_dataset():
    """Get validation and test data."""
    url = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/refs/heads/main/bbh/hyperbaton.json"
    json_bytes = urlopen(url).read().decode("utf-8", "replace")
    dataset = json.loads(json_bytes)["examples"]
    train_set = dataset[:50]
    test_set = dataset[50:100]

    # Show a sample question
    print(train_set[0])
    return train_set, test_set

def evaluator(prompt: BasePrompt, validation_set: list[dict]) -> list[str]:
    """Prompt evaluator function."""
    # Run the prompt through the AI system
    predictions = []
    num_correct = 0
    agent = get_agent()
    for row in validation_set:
        # Get the prediction
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

    # Compute the score
    score = num_correct / len(validation_set)
    
    # Save the predictions and score
    prompt.metadata["predictions"] = predictions
    
    return score

def main():
    """Run optimization."""
    # Get datasets
    train_set, test_set = get_dataset()
    
    # Demonstrate the baseline prompt
    baseline_prompt = BasePrompt(content="You are a grammar expert. Please answer each question with '(A)' or '(B)' only, and no other thoughts.")
    baseline_prompt.score = evaluator(prompt=baseline_prompt, validation_set=train_set)
    test_score = evaluator(prompt=baseline_prompt, validation_set=test_set)
    predictions = baseline_prompt.metadata["predictions"]
    for row, response in zip(test_set[:3], predictions[:3]):
        print(f"Question: {row["input"]}")
        print(f"Response: {response}")
        print(f"Actual Answer: {row['target']}")
        print()
    print(f"Baseline Prompt Train Score: {baseline_prompt.score}")
    print(f"Baseline Prompt Test Score: {test_score}")

    # Define your Optimizer
    client = get_optimizer_client()
    optimizer = OPROOptimizer(
        client=client,
        seed_prompts=[baseline_prompt],
        validation_set=train_set,
        max_depth=3,
        evaluator=evaluator,
        input_field="input",
        output_field="target",
        output_path="out.jsonl",
    )

    # Optimize the prompt and output the intermediate prompts to a jsonl file
    best_prompt = optimizer.run()

    # View the newly created prompt
    print(f"BEST PROMPT:\n{best_prompt.content}")
    print()

    # Demonstrate the effectiveness of the new prompt
    score = evaluator(prompt=best_prompt, validation_set=test_set)
    predictions = best_prompt.metadata["predictions"]
    for row, response in zip(test_set[:3], predictions[:3]):
        print(f"Question: {row["input"]}")
        print(f"Response: {response}")
        print(f"Actual Answer: {row['target']}")
        print()
    print(f"Best Prompt Train Score: {best_prompt.score}")
    print(f"Best Prompt Test Score: {score}")
    
if __name__ == "__main__":
    main()
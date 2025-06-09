"""Gradient prompt tuning based on the paper: https://arxiv.org/pdf/2305.03495"""

from typing import Callable, Any, Union
import mlflow
from openai import Client
from pydantic import BaseModel
import re
import pandas as pd
from pandas import DataFrame, Series

GRADIENT_PROMPT = """I'm trying to write a zero-shot classifier prompt for metadata extraction.
My current prompt is:
"{prompt}"
But this prompt gets the following examples wrong:
{error_string}
give {num_feedbacks} reasons why the prompt could
have gotten these examples wrong.
Wrap each reason with <START> and <END>"""


REWRITE_PROMPT = """I'm trying to write a zero-shot classifier prompt for metadata extraction.
My current prompt is:
"{prompt}"
But it gets the following examples wrong:
{error_string}
Based on these examples the problem with this prompt is that {gradient}
Based on the above information, I wrote {steps_per_gradient} different improved prompts.
Each prompt is wrapped with <START> and <END>.
The {steps_per_gradient} new prompts are:
"""

PipelineOutputType = Union[str, BaseModel]

class MetricResult(BaseModel):
    
    score: float | int | str
    error_string: str
    
class PromptResult(BaseModel):
    
    prompt: str
    score: float | int | str
    error_string: str
    
def example_pipeline(prompt: str, **kwargs) -> PipelineOutputType:
    pass

def example_metric(dataset: DataFrame, predictions: Series) -> MetricResult:
    pass

class GradientOptimizer:
    def __init__(
        self, 
        pipeline: Callable[[str, Any], PipelineOutputType],
        metric: Callable[[DataFrame, Series], MetricResult],
        dataset: DataFrame,
        model_name: str,
        client: Client = None, 
        num_feedbacks: int = 3, 
        steps_per_gradient: int = 3,  
        gradient_prompt: str = GRADIENT_PROMPT, 
        rewrite_prompt: str = REWRITE_PROMPT, 
    ):
        self.pipeline = pipeline
        self.metric = metric
        self.dataset = dataset.copy()
        self.model_name = model_name
        self.client = client if client is not None else Client()
        self.num_feedbacks = num_feedbacks
        self.steps_per_gradient = steps_per_gradient
        self.gradient_prompt = gradient_prompt
        self.rewrite_prompt = rewrite_prompt
        
    def _predict(self, prompt: str) -> Series:
        baseline_predictions = self.dataset.apply(lambda x: self.pipeline(prompt=prompt, **x.to_dict()), axis=1)
        return baseline_predictions
        
    def _evalute(self, predictions: Series) -> MetricResult:
        metric_result = self.metric(dataset=self.dataset, predictions=predictions)
        return metric_result
    
    def _extract_responses(self, content: str):
        """Extract the responses between <START> and <END>."""
        pattern = r"<START>(.*?)</?END>"
        matches = re.findall(pattern, content, flags=re.DOTALL)
        if len(matches) == 0:
            # TODO: Something here...
            return []
        return matches
    
    def _generate(self, prompt_template: str, template_kwargs: dict, **kwargs) -> str:
        prompt = prompt_template.format(**template_kwargs)
        print(prompt)
        response = self.client.completions.create(prompt=prompt, model=self.model_name, **kwargs)
        result = response.choices[0].text.strip()
        print(result)
        return result
    
    def _generate_new_prompts(self, prompt: str, error_string: str, **kwargs) -> list[str]:
        template_kwargs = {
            "prompt": prompt,
            "error_string": error_string,
            "num_feedbacks": self.num_feedbacks,
            "steps_per_gradient": self.steps_per_gradient,
        }
        gradient_response = self._generate(prompt_template=self.gradient_prompt, template_kwargs=template_kwargs, **kwargs)
        gradients = self._extract_responses(gradient_response)
        all_new_prompts = []
        for gradient in gradients:
            template_kwargs.update({"gradient": gradient})
            rewrite_response = self._generate(prompt_template=self.rewrite_prompt, template_kwargs=template_kwargs, **kwargs)
            new_prompts = self._extract_responses(rewrite_response)
            all_new_prompts.extend(new_prompts)
        return all_new_prompts
    
    
    def optimize(self, baseline_prompt: str, run_name: str = None, max_iters: int = 3, **kwargs) -> list[PromptResult]: # TODO: add minimum improvement threshold
        with mlflow.start_run(run_name=run_name):            
            # Get first results
            baseline_predictions = self._predict(prompt=baseline_prompt)
            baseline_metric_result = self._evalute(predictions=baseline_predictions)
            baseline_prompt_result = PromptResult(prompt=baseline_prompt, score=baseline_metric_result.score, error_string=baseline_metric_result.error_string)
            prompt_results = [baseline_prompt_result]
            
            # Start iteration loop
            prev_prompts: list[PromptResult] = [baseline_prompt_result]
            for i in range(max_iters):
                new_prompts = []
                for prompt in prev_prompts:
                    if prompt.score >= 1.0:
                        continue

                    for new_prompt in self._generate_new_prompts(prompt=prompt.prompt, error_string=prompt.error_string):
                        predictions = self._predict(prompt=new_prompt)
                        metric_result = self._evalute(predictions=predictions)
                        prompt_result = PromptResult(prompt=new_prompt, score=metric_result.score, error_string=metric_result.error_string)
                        prompt_results.append(prompt_result)
                        new_prompts.append(prompt_result)
                prev_prompts = new_prompts
            
        return prompt_results

if __name__ == "__main__":
    
    model_name = "qwen2.5:7b"
    client = Client(base_url="http://localhost:11434/v1", api_key="NONE")
    
    def pipeline(prompt: str, question: str, **kwargs) -> str:
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        response = client.chat.completions.create(messages=messages, model=model_name)
        result = response.choices[0].message.content.strip()
        print(result)
        return result
    
    def metric(dataset: DataFrame, predictions: Series) -> MetricResult:
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
        score = score / len(dataset)
        return MetricResult(score=score, error_string=error_string)
    
    data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
        {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"}
    ]
    dataset = DataFrame(data)
    
    optimizer = GradientOptimizer(pipeline=pipeline, metric=metric, dataset=dataset, model_name=model_name, client=client)
    
    baseline_prompt = "Please answer every question the user asks to the best of your ability."
    
    results = optimizer.optimize(baseline_prompt=baseline_prompt)
    
    print(results)
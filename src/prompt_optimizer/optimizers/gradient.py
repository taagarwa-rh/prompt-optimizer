"""Gradient prompt tuning based on the paper: https://arxiv.org/pdf/2305.03495"""

import logging
from typing import Callable, Any, Iterable, Generator
import re

from openai import Client
import pandas as pd

from .base import BaseOptimizer, MetricResult, PromptResult

logger = logging.getLogger(__name__)

GRADIENT_PROMPT = """I'm trying to write a zero-shot classifier prompt.
My current prompt is:
"{prompt}"
But this prompt gets the following examples wrong:
{error_string}
give {num_feedbacks} reasons why the prompt could
have gotten these examples wrong.
Wrap each reason with <START> and <END>"""


REWRITE_PROMPT = """I'm trying to write a zero-shot classifier prompt.
My current prompt is:
"{prompt}"
But it gets the following examples wrong:
{error_string}
Based on these examples the problem with this prompt is that {gradient}
Based on the above information, I wrote {steps_per_gradient} different improved prompts.
Each prompt is wrapped with <START> and <END>.
The {steps_per_gradient} new prompts are:
"""

PipelineOutputType = Any

class GradientOptimizer(BaseOptimizer):
    """Prompt optimizer with "Gradient Descent" using methods outlined in https://arxiv.org/pdf/2305.03495."""
    
    def __init__(
        self, 
        pipeline: Callable[[str], PipelineOutputType],
        metric: Callable[[PipelineOutputType], dict],
        dataset: pd.DataFrame,
        model_name: str,
        client: Client = None, 
        num_feedbacks: int = 3, 
        steps_per_gradient: int = 3,
    ):
        """Initialize GradientOptimizer.

        Args:
            pipeline (Callable[[str], PipelineOutputType]): 
                Function that takes a prompt string and a row from `dataset` as kwargs and returns an output (e.g. a string).
                Output can be of any type that can be passed to `metric`.
            metric (Callable[[PipelineOutputType], dict]):
                Function that takes a pipeline output and a row from `dataset` as kwargs and returns a dictionary with `score` and `error_string` keys.
                The `score` key should be a float <= 1.0 and `error_string` should be a string describing the issues with the output or None.
            dataset (pandas.DataFrame):
                Pandas dataframe containing all test cases. Each row will be passed to `pipeline` and `metric` as kwargs.
            model_name (str):
                Name of the LLM at the endpoint to use to generate gradients and new prompts.
            client (openai.Client, optional): 
                OpenAI client to use for generating gradients and new prompts. If not specified, a default one will be created for you.
            num_feedbacks (int, optional): 
                Number of feedbacks (gradients) to generate for each prompt. This won't be enforced and the LLM may generate more or less than this. Defaults to 3.
            steps_per_gradient (int, optional): 
                Number of steps (new prompts) to generate per gradient. This won't be enforced and the LLM may generate more or less than this. Defaults to 3.
        """
        self.pipeline = pipeline
        self.metric = metric
        self.dataset = dataset.copy()
        self.model_name = model_name
        self.client = client if client is not None else Client()
        self.num_feedbacks = num_feedbacks
        self.steps_per_gradient = steps_per_gradient
        
    def _predict(self, prompt: str) -> Iterable[PipelineOutputType]:
        """Generate a prediction for each datapoint using the pipeline.

        Args:
            prompt (str): Prompt to pass as a keyword argument `prompt` to the pipeline.

        Returns:
            Iterable[PipelineOutputType]: Outputs from the pipeline, one for each datapoint in the same order as the dataset.
        """
        predictions = self.dataset.apply(lambda x: self.pipeline(prompt=prompt, **x.to_dict()), axis=1)
        return predictions.to_list()
        
    def _evalute(self, predictions: Iterable[PipelineOutputType]) -> MetricResult:
        """Evaluate the predictions for each datapoint using the metric.

        Args:
            predictions (Iterable[PipelineOutputType]): Outputs from the pipeline, one for each datapoint in the same order as the dataset.

        Raises:
            ValueError: If the metric function does not return a dictionary with keys 'score' and 'error_string'.

        Returns:
            MetricResult: The aggregate evaluation results from the metric.
        """
        dataset_ = self.dataset.copy()
        dataset_["prediction"] = predictions
        metric_results: list[dict] = dataset_.apply(lambda x: self.metric(**x.to_dict()), axis=1).to_list()
        if any(not isinstance(r, dict) for r in metric_results) \
            or any("score" not in r for r in metric_results) \
            or any("error_string" not in r for r in metric_results):
            raise ValueError("Metric results must be dictionaries with keys 'score' and 'error_string'.")
        metric_result = MetricResult(
            score = sum([r["score"] for r in metric_results]) / len(metric_results),
            error_string = "\n\n".join([r["error_string"] for r in metric_results if r["error_string"] is not None])
        )
        return metric_result
    
    def _extract_responses(self, content: str) -> list[str]:
        """Extract the responses between <START> and <END>.

        Args:
            content (str): Output string from an LLM generation request.

        Returns:
            list[str]: List of all responses within <START> and </?END> or <START> and </?START>.
        """
        pattern = r"<START>(.*?)(?:<\/?END>|<\/?START>)"
        matches = re.findall(pattern, content, flags=re.DOTALL)
        return matches
    
    def _generate(self, prompt_template: str, template_kwargs: dict, **kwargs) -> list[str]:
        """Generate a completion for a given template and kwargs and parse the results.

        Args:
            prompt_template (str): Template for the prompt.
            template_kwargs (dict): Key word arguments to fill the template values.
            kwargs: Additional kwargs to pass to the OpenAI client.completions.create (e.g. temperature)

        Returns:
            list[str]: The parsed generation results.
        """
        prompt = prompt_template.format(**template_kwargs)
        raw_response = self.client.completions.create(prompt=prompt, model=self.model_name, **kwargs)
        raw_text = raw_response.choices[0].text.strip()
        responses = self._extract_responses(raw_text)
        return responses
    
    def _generate_new_prompts(self, prompt: str, error_string: str, **kwargs) -> Generator[str, None, None]:
        """Generate a number of new prompts based on the given prompt and error string.

        Args:
            prompt (str): Prompt to generate new prompts off of.
            error_string (str): Description of the errors with the `prompt`.

        Returns:
            Generator[str]: New prompts that attempt to correct the errors presented in `error_string`.
        """
        template_kwargs = {
            "prompt": prompt,
            "error_string": error_string,
            "num_feedbacks": self.num_feedbacks,
            "steps_per_gradient": self.steps_per_gradient,
        }
        gradients = self._generate(prompt_template=GRADIENT_PROMPT, template_kwargs=template_kwargs, **kwargs)
        gradients = gradients[:self.num_feedbacks]
        for gradient in gradients:
            template_kwargs.update({"gradient": gradient})
            new_prompts = self._generate(prompt_template=REWRITE_PROMPT, template_kwargs=template_kwargs, **kwargs)
            new_prompts = new_prompts[:self.steps_per_gradient]
            for new_prompt in new_prompts:
                yield new_prompt
    
    
    def optimize(self, baseline_prompt: str, max_iters: int = 3, score_threshold: float = None, prune: bool = False, **kwargs) -> list[PromptResult]:
        """Optimize a prompt.

        Args:
            baseline_prompt (str): Starting prompt to optimize.
            max_iters (int, optional): 
                Maxium number of iterations of generating new prompts from gradients. 
                Each iteration will produce (approximately) num_feedbacks * steps_per_gradient prompts per prompt generated in the previous iteration. 
                Defaults to 3.
            score_threshold (float, optional): 
                If the score for any prompt exceeds this score, stop the iterations immedately. If not specified, new prompts will continue to be generated
                for any prompt not scoring over 1.0.
            prune (bool, optional):
                If True, prune any new prompts that do not meet or exceed their base prompt's score. Defaults to False.

        Returns:
            list[PromptResult]: List of prompt results, including scores and error strings for the prompts.
        """
        # Get baseline results
        logger.info("Getting baseline results")
        baseline_predictions = self._predict(prompt=baseline_prompt)
        baseline_metric_result = self._evalute(predictions=baseline_predictions)
        baseline_prompt_result = PromptResult(prompt=baseline_prompt, score=baseline_metric_result.score, error_string=baseline_metric_result.error_string)
        prompt_results = [baseline_prompt_result]
        logger.info(f"Baseline prompt - Score: {baseline_prompt_result.score:.2f}")
        
        # Start iteration loop
        prev_prompts: list[PromptResult] = [baseline_prompt_result]
        logger.info("Starting iterations")
        for i in range(max_iters):
            new_prompts = []
            logger.info(f"Iteration {i} - {len(prev_prompts)} prompts will be iterated")
            for j, prompt in enumerate(prev_prompts):
                # Skip the prompt if it's already perfect
                if prompt.score >= 1.0:
                    continue
                
                # Otherwise generate some new prompts and evaluate them
                logger.info(f"Iteration {i} - Prompt {j} - Generating new prompts")
                k = 1
                for new_prompt in self._generate_new_prompts(prompt=prompt.prompt, error_string=prompt.error_string, **kwargs):
                    logger.info(f"Iteration {i} - Prompt {j} - New Prompt {k} - Evaluating new prompt")
                    predictions = self._predict(prompt=new_prompt)
                    metric_result = self._evalute(predictions=predictions)
                    prompt_result = PromptResult(prompt=new_prompt, score=metric_result.score, error_string=metric_result.error_string)
                    logger.info(f"Iteration {i} - Prompt {j} - New Prompt {k} - Score: {prompt_result.score:2f}")
                    # Check if we should prune this prompt
                    if prune:
                        if prompt_result.score >= prompt.score:
                            prompt_results.append(prompt_result)
                            new_prompts.append(prompt_result)
                    else:
                        prompt_results.append(prompt_result)
                        new_prompts.append(prompt_result)
                    # Stop the iterations if we've found a prompt that beats the score_threshold
                    if score_threshold is not None and any(prompt.score >= score_threshold for prompt in prompt_results):
                        return sorted(prompt_results, key=lambda x: x.score, reverse=True)
                    k += 1
            prev_prompts = new_prompts
            
        return sorted(prompt_results, key=lambda x: x.score, reverse=True)

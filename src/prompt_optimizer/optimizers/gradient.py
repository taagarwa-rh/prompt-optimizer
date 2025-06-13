"""ProTeGi with Successive Rejects based on the paper: https://arxiv.org/pdf/2305.03495."""

import logging
import re
from typing import Any, Callable, Generator, Iterable, Literal

import pandas as pd
from openai import Client

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
        """
        Initialize GradientOptimizer.

        Args:
            pipeline (Callable[[str], PipelineOutputType]):
                Function that takes a prompt string and a row from `dataset` as kwargs and returns an output (e.g. a string).
                Output can be of any type that can be passed to `metric`.
            metric (Callable[[PipelineOutputType], dict]):
                Function that takes a pipeline output and a row from `dataset` as kwargs and returns a dictionary with `score` and `error_string` keys.
                The `score` key should be a float or int and `error_string` should be a string describing the issues with the output or None.
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
        """
        Generate a prediction for each datapoint using the pipeline.

        Args:
            prompt (str): Prompt to pass as a keyword argument `prompt` to the pipeline.

        Returns:
            Iterable[PipelineOutputType]: Outputs from the pipeline, one for each datapoint in the same order as the dataset.

        """
        predictions = self.dataset.apply(lambda x: self.pipeline(prompt=prompt, **x.to_dict()), axis=1)
        return predictions.to_list()

    def _evalute(self, predictions: Iterable[PipelineOutputType]) -> MetricResult:
        """
        Evaluate the predictions for each datapoint using the metric.

        Args:
            predictions (Iterable[PipelineOutputType]): Outputs from the pipeline, one for each datapoint in the same order as the dataset.

        Raises:
            ValueError: If the metric function does not return a dictionary with keys 'score' and 'error_string'.

        Returns:
            MetricResult: The aggregate evaluation results from the metric.

        """
        raw_metric_result: dict = self.metric(predictions=predictions, **self.dataset.to_dict(orient="list"))
        try:
            metric_result = MetricResult.model_validate(raw_metric_result)
        except Exception as e:
            logger.error(e)
            raise ValueError("Metric results must be dictionaries with keys 'score' and 'error_string'.")
        return metric_result

    def _extract_responses(self, content: str) -> list[str]:
        """
        Extract the responses between <START> and <END>.

        Args:
            content (str): Output string from an LLM generation request.

        Returns:
            list[str]: List of all responses within <START> and </?END> or <START> and </?START>.

        """
        pattern = r"<START>(.*?)(?:<\/?END>|<\/?START>)"
        matches = re.findall(pattern, content, flags=re.DOTALL)
        return matches

    def _generate(self, prompt_template: str, template_kwargs: dict, **kwargs) -> list[str]:
        """
        Generate a completion for a given template and kwargs and parse the results.

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
        """
        Generate a number of new prompts based on the given prompt and error string.

        Args:
            prompt (str): Prompt to generate new prompts off of.
            error_string (str): Description of the errors with the `prompt`.
            kwargs: Additional kwargs to pass to the OpenAI client.completions.create (e.g. temperature)

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
        gradients = gradients[: self.num_feedbacks]
        for gradient in gradients:
            template_kwargs.update({"gradient": gradient})
            new_prompts = self._generate(prompt_template=REWRITE_PROMPT, template_kwargs=template_kwargs, **kwargs)
            new_prompts = new_prompts[: self.steps_per_gradient]
            for new_prompt in new_prompts:
                yield new_prompt

    def _filter_batch(self, batch: list[PromptResult], search_mode: Literal["greedy", "beam"]) -> list[PromptResult]:
        """Select prompts from the batch based on the search_mode."""
        # If it's greedy search, we keep all the prompts from the batch
        if search_mode == "greedy":
            return batch
        # If it's beam search, we keep only the best prompt from the batch
        elif search_mode == "beam":
            best_prompt = max(batch, key=lambda x: x.score)
            return [best_prompt]

    def optimize(
        self, baseline_prompt: str, depth: int = 3, score_threshold: float = None, search_mode: Literal["greedy", "beam"] = "beam", **kwargs
    ) -> list[PromptResult]:
        """
        Optimize a prompt.

        Performs a tree-based search of new prompts from gradients until `depth` is reached or `score_threshold` is met.

        Each iteration will produce at most num_feedbacks * steps_per_gradient new prompts per prompt generated in the previous iteration.
        For example, if num_feedbacks is 3 and steps_per_gradient is 3, then the first iteration will produce 9 new prompts.

        If search_mode is "beam", the only the best prompt from each iteration is kept and the rest discarded.
        For the above example, 1 prompt would be kept from the first iteration and the second iteration would produce at most
        9 new prompts (1 prompts * 3 num_feedbacks * 3 steps_per_gradient).
        Setting num_feedbacks to 1 and steps_per_gradient to 1 would result in a monte carlo simulation where each iteration
        produces one feedback and one new prompt.

        If search_mode is "greedy", all prompts from the previous iteration are kept.
        For the above example, 9 prompts would be kept from the first iteration and the second iteration would produce at most
        81 new prompts (9 prompts * 3 num_feedbacks * 3 steps_per_gradient).

        Args:
            baseline_prompt (str): Starting prompt to optimize.
            depth (int, optional):
                Maxium number of iterations of generating new prompts from gradients.
                Each iteration will produce (at most) num_feedbacks * steps_per_gradient prompts per prompt generated in the previous iteration.
                Defaults to 3.
            score_threshold (float, optional):
                If the score for any prompt exceeds this score, stop the iterations immedately. If not specified, new prompts will continue to be generated
                for any prompt not scoring over 1.0.
            search_mode (Literal["greedy", "beam"], optional):
                Search mode for discovering new prompts. "greedy" keeps all prompts generated in each iteration.
                "beam" selects the top prompt from each minibatch based on their scores.
            kwargs:
                Keyword arguments to pass to the openai.client.completions.create call, e.g. `temperature`.

        Returns:
            list[PromptResult]: List of prompt results, including scores and error strings for the prompts.

        """
        # Check inputs
        if search_mode not in ["greedy", "beam"]:
            raise ValueError("search_mode must be either 'greedy' or 'beam'")

        # Get baseline results
        logger.info("Getting baseline results")
        baseline_predictions = self._predict(prompt=baseline_prompt)
        baseline_metric_result = self._evalute(predictions=baseline_predictions)
        baseline_prompt_result = PromptResult(
            prompt=baseline_prompt, score=baseline_metric_result.score, error_string=baseline_metric_result.error_string
        )
        prompt_results = [baseline_prompt_result]
        logger.info(f"Baseline prompt - Score: {baseline_prompt_result.score:.4f}")

        # Start iteration loop
        prompts: list[PromptResult] = [baseline_prompt_result]
        logger.info("Starting iterations")
        for i in range(depth):
            new_prompts = []
            logger.info(f"Iteration {i + 1} - {len(prompts)} prompts will be iterated")
            for j, prompt in enumerate(prompts):
                # Otherwise generate a minibatch of new prompts and evaluate them
                logger.info(f"Iteration {i + 1} - Minibatch {j + 1} - Generating new prompts")
                k = 0
                minibatch_prompt_results = []
                for new_prompt in self._generate_new_prompts(prompt=prompt.prompt, error_string=prompt.error_string, **kwargs):
                    logger.info(f"Iteration {i + 1} - Minibatch {j + 1} - New Prompt {k + 1} - Evaluating new prompt")
                    predictions = self._predict(prompt=new_prompt)
                    metric_result = self._evalute(predictions=predictions)
                    prompt_result = PromptResult(prompt=new_prompt, score=metric_result.score, error_string=metric_result.error_string)
                    logger.info(f"Iteration {i + 1} - Minibatch {j + 1} - New Prompt {k + 1} - Score: {prompt_result.score:4f}")
                    minibatch_prompt_results.append(prompt_result)
                    # Stop the iterations if we've found a prompt that beats the score_threshold
                    if score_threshold is not None and prompt_result.score >= score_threshold:
                        prompt_results.extend(minibatch_prompt_results)
                        return sorted(prompt_results, key=lambda x: x.score, reverse=True)
                    k += 1

                # Log the top score from this batch
                minibatch_best_score = max(minibatch_prompt_results, key=lambda x: x.score)
                logger.info(f"Iteration {i + 1} - Minibatch {j + 1} - Best score: {minibatch_best_score.score:.4f}")

                # Save every prompt to the prompt results
                prompt_results.extend(minibatch_prompt_results)

                # Save prompts for next iteration based on search_mode
                filtered_prompt_results = self._filter_batch(minibatch_prompt_results, search_mode=search_mode)
                new_prompts.extend(filtered_prompt_results)

            # Log the top score from this iteration
            batch_best_prompt = max(new_prompts, key=lambda x: x.score)
            logger.info(f"Iteration {i + 1} - Best score: {batch_best_prompt.score:.4f}")
            logger.info(f"Iteration {i + 1} - Best prompt: {batch_best_prompt.score:.4f}")

            # Initalize the next iteration with the new prompts
            prompts = new_prompts

        return sorted(prompt_results, key=lambda x: x.score, reverse=True)

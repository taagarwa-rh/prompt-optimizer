import logging
import random
import re
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from rich.progress import track

from prompt_optimizer import Prompt
from prompt_optimizer.types import ClientType, ScoreType, ValidationSetType

from .base import BaseOptimizer

logger = logging.getLogger(__name__)

ERROR_STRING_TEMPLATE = """<{index}>
The model's input is:
{input}
The model's response is:
{prediction}
The correct response is:
{actual}"""

ERROR_FEEDBACK_TEMPLATE = """I'm writing prompts for a language model designed for a task.
My current prompt is:
{prompt}

But this prompt gets the following examples wrong:
{error_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt."""


STATE_TRANSIT_PROMPT_TEMPLATE = """I'm writing prompts for a language model designed for a task.
My current prompt is:
{prompt}

But this prompt gets the following examples wrong:
{error_string}

Based on these errors, the problems with this prompt and the reasons are:
{error_feedback}

There is a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write {steps_per_gradient} new prompts following these guidelines:

1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve
based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.
The new prompts are:"""


class PromptAgentOptimizer(BaseOptimizer):
    """
    PromptAgent Optimizer.

    Based on PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization.

    ```
    @misc{wang2023promptagentstrategicplanninglanguage,
        title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
        author={Xinyuan Wang and Chenxi Li and Zhen Wang and Fan Bai and Haotian Luo and Jiayou Zhang and Nebojsa Jojic and Eric P. Xing and Zhiting Hu},
        year={2023},
        eprint={2310.16427},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2310.16427},
    }
    ```
    """

    def __init__(
        self,
        *,
        client: ClientType,
        seed_prompts: list[Prompt],
        validation_set: ValidationSetType,
        max_depth: int,
        evaluator: Callable[[Prompt, ValidationSetType], ScoreType],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 5,
        expand_width: int = 3,
        num_samples: int = 2,
        search_mode: Literal["beam", "greedy"] = "beam",
        score_threshold: Optional[Union[float, int]] = None,
    ):
        """
        Initialize the PromptAgent Optimizer.

        Args:
            client (ClientType):
                Language model client to use for prompt generation and feedback.
            seed_prompts (list[Prompt]):
                List of prompts to seed generation.
            validation_set (ValidationSetType):
                Set of examples to evaluate the prompt on.
            max_depth (int):
                Maximum iteration depth for prompt generation.
            evaluator (Callable[[Prompt, ValidationSetType], ScoreType]):
                Function that takes a prompt and the validation data and returns a score.
            output_path (Union[str, Path], optional):
                Path to store run results. Should be a .jsonl file path.
                If None, no outputs will be written to disk. Defaults to None.
            batch_size (int, optional):
                Number of errors to sample for each action / new prompt generation. Defaults to 5.
            expand_width (int, optional):
                Number of feedback actions to generate per prompt. Defaults to 3.
            num_samples (int, optional):
                Number of new prompts to generate per feedback action. Defaults to 2.
            search_mode (Literal["beam", "greedy"], optional):
                Mode for filtering prompt candidates after each step. "greedy" keeps all prompts from the previous step.
                "beam" keeps only the highest scoring prompt from each branch of the previous step. Defaults to "beam".
            score_threshold (float, optional):
                Threshold for early convergence. If a prompt exceeds this score after any iteration, the optimization loop
                immediately ends. If set to None, the optimization loop will not terminate early. Defaults to None.

        """
        super().__init__(
            client=client,
            seed_prompts=seed_prompts,
            validation_set=validation_set,
            max_depth=max_depth,
            evaluator=evaluator,
            output_path=output_path,
        )
        self.batch_size = batch_size
        self.expand_width = expand_width
        self.num_samples = num_samples
        self.search_mode = search_mode
        self.score_threshold = score_threshold
        self.PARENT_KEY = "_parent"

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

    def _generate(self, metaprompt_template: str, template_kwargs: dict) -> str:
        """
        Generate a completion for a given template and kwargs and parse the results.

        Args:
            metaprompt_template (str): Template for the metaprompt.
            template_kwargs (dict): Key word arguments to fill the template values.
            kwargs: Additional kwargs to pass to the OpenAI client.completions.create (e.g. temperature)

        Returns:
            list[str]: The parsed generation results.

        """
        metaprompt = metaprompt_template.format(**template_kwargs)
        input = [{"role": "user", "content": metaprompt}]
        raw_response = self.client.invoke(input=input)
        response = raw_response.content.strip()
        return response

    def _map_trajectory(self, prompt: Optional[Prompt] = None):
        """Map the trajectory of the prompt."""
        if prompt is None:
            return ""
        parent = prompt.metadata.get(self.PARENT_KEY, None)
        return (self._map_trajectory(parent) + "\n\n" + f"Prompt: {prompt.content}\nScore: {prompt.score}").strip()

    def generate_prompt_candidates(self, *, prompts: list[Prompt], **kwargs) -> list[Prompt]:
        """Generate prompt candidates using gradients."""
        prompt_candidates = []
        for prompt in track(prompts, description="Generating prompt candidates", transient=True):
            if len(prompt.errors) == 0:
                continue

            # Map prompt trajectory
            trajectory_prompts = self._map_trajectory(prompt=prompt)

            for _ in range(self.expand_width):
                # Sample and collect errors into error string
                error_sample = random.choices(prompt.errors, k=self.batch_size)
                error_string = "\n\n".join(
                    [
                        ERROR_STRING_TEMPLATE.format(index=i + 1, input=error.input, prediction=error.prediction, actual=error.actual)
                        for i, error in enumerate(error_sample)
                    ]
                )

                # Generate actions
                template_kwargs = {
                    "prompt": prompt.content,
                    "error_string": error_string,
                    "steps_per_gradient": self.num_samples,
                    "trajectory_prompts": trajectory_prompts,
                }
                action = self._generate(metaprompt_template=ERROR_FEEDBACK_TEMPLATE, template_kwargs=template_kwargs)

                # Generate new prompts from the action
                template_kwargs.update({"error_feedback": action})
                raw_new_prompts = self._generate(metaprompt_template=STATE_TRANSIT_PROMPT_TEMPLATE, template_kwargs=template_kwargs)
                new_prompts = self._extract_responses(raw_new_prompts)
                new_prompts = new_prompts[: self.num_samples]
                metadata = {self.PARENT_KEY: prompt, "_action": action, "_resampled": False}
                new_prompt_candidates = [Prompt(content=new_prompt, metadata=metadata) for new_prompt in new_prompts]

                # Save prompts to prompt candidates
                prompt_candidates.extend(new_prompt_candidates)

        return prompt_candidates

    def _get_best_prompt(self, prompts: list[Prompt]):
        """Get the highest scoring prompt."""
        if any(prompt.score is None for prompt in prompts):
            raise ValueError("All prompts must be scored before calling this function.")
        return max(prompts, key=lambda x: x.score)

    def select_prompt_candidates(self, *, prompts: list[Prompt], validation_set: ValidationSetType) -> list[Prompt]:
        """Select prompt candidates according to the search mode."""
        self._score_prompts(prompts=prompts, validation_set=validation_set)
        # If this is the first iteration, keep all prompts
        if len(self._p) == 1:
            return prompts

        # Otherwise, keep based on search_mode
        elif self.search_mode == "greedy":
            # Keep all prompts
            return prompts

        elif self.search_mode == "beam":
            # Select the best prompt in each branch
            # Split prompts by parent
            parent_to_prompts = {prompt.metadata[self.PARENT_KEY]: [] for prompt in prompts}
            for prompt in prompts:
                parent_to_prompts[prompt.metadata[self.PARENT_KEY]].append(prompt)

            # Get the best prompt from each parent
            return [self._get_best_prompt(prompts=branch_prompts) for _, branch_prompts in parent_to_prompts.items()]

    def check_early_convergence(self, *, all_prompts: list[list[Prompt]]):
        """Check if the early convergence criteria is met."""
        if self.score_threshold is None:
            return False

        # Flatten all iterations
        prompts = sum(all_prompts, start=[])

        # Check if early convergence criteria is met
        highest_score = max(prompts, key=lambda x: x.score).score
        if highest_score >= self.score_threshold:
            return True
        return False

    def select_best_prompt(self, *, all_prompts: list[list[Prompt]]) -> Prompt:
        """Select the top scoring prompt."""
        # Flatten all iterations
        prompts = sum(all_prompts, start=[])

        # Select the single prompt with the highest score
        best_prompt = self._get_best_prompt(prompts=prompts)
        logger.info(f"Best score: {best_prompt.score:.3f}")
        return best_prompt

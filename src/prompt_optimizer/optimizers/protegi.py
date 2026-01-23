import logging
import re
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from rich.progress import track

from prompt_optimizer import BasePrompt
from prompt_optimizer.pipeline import BasePipeline
from prompt_optimizer.types import ClientType, ScoreType, ValidationSetType

logger = logging.getLogger(__name__)

GENERATE_GRADIENT_PROMPT_TEMPLATE = """I'm trying to write a zero-shot classifier prompt. My current prompt is:

"{prompt}"

But this prompt gets the following examples wrong:

{error_string}

give {num_feedbacks} reasons why the prompt could have gotten these examples wrong. Wrap each reason with <START> and <END>."""


INCORPORTATE_GRADIENT_PROMPT_TEMPLATE = """I'm trying to write a zero-shot classifier prompt. My current prompt is:

"{prompt}"

But it gets the following examples wrong:

{error_string}

Based on these examples the problem with this prompt is that {gradient}. Based on the above information, give {steps_per_gradient} different improved prompts. Each prompt must be wrapped with <START> and <END>."""

RESAMPLING_PROMPT_TEMPLATE = """Generate a variation of the following instruction while keeping the semantic meaning.

Instruction:

{prompt}"""


class ProtegiOptimizer(BasePipeline):
    """
    ProTeGi Optimizer.

    Based on ProTeGi with Successive Rejects.

    ```
    @misc{pryzant2023automaticpromptoptimizationgradient,
        title={Automatic Prompt Optimization with "Gradient Descent" and Beam Search},
        author={Reid Pryzant and Dan Iter and Jerry Li and Yin Tat Lee and Chenguang Zhu and Michael Zeng},
        year={2023},
        eprint={2305.03495},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2305.03495},
    }
    ```
    """

    def __init__(
        self,
        *,
        client: ClientType,
        seed_prompts: list[BasePrompt],
        validation_set: ValidationSetType,
        max_depth: int,
        evaluator: Callable[[BasePrompt, ValidationSetType], ScoreType],
        output_path: Optional[Union[str, Path]] = None,
        num_feedbacks: int = 3,
        steps_per_gradient: int = 3,
        num_resample: int = 3,
        search_mode: Literal["greedy", "beam"] = "beam",
        score_threshold: Optional[Union[float, int]] = None,
    ):
        """
        Initialize the ProTeGi optimizer.

        Args:
            client (ClientType):
                Language model client to use for prompt generation and feedback.
            seed_prompts (list[BasePrompt]):
                List of prompts to seed generation.
            validation_set (ValidationSetType):
                Set of examples to evaluate the prompt on.
            max_depth (int):
                Maximum iteration depth for prompt generation.
            evaluator (Callable[[BasePrompt, ValidationSetType], ScoreType]):
                Function that takes a prompt and the validation data and returns a score.
            output_path (Union[str, Path], optional):
                Path to store run results. Should be a .jsonl file path.
                If None, no outputs will be written to disk. Defaults to None.
            num_feedbacks (int, optional):
                Number of feedbacks to generate per prompt. Defaults to 3.
            steps_per_gradient (int, optional):
                Number of new prompts to generate per feedback. Defaults to 3.
            num_resample (int, optional):
                Number of Monte Carlo rewrites per new prompt generated from feedback. The paper recommends
                setting this equal to steps_per_gradient. Defaults to 3.
            search_mode (Literal["greedy", "beam"], optional):
                Mode for filtering prompt candidates after each step. "greedy" keeps all prompts from the previous step.
                "beam" keeps only the highest scoring prompt from the previous step. Defaults to "beam".
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
        self.num_feedbacks = num_feedbacks
        self.steps_per_gradient = steps_per_gradient
        self.num_resample = num_resample
        self.search_mode = search_mode
        self.score_threshold = score_threshold

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

    def generate_prompt_candidates(self, *, prompts: list[BasePrompt], **kwargs) -> list[BasePrompt]:
        """Generate prompt candidates using gradients."""
        prompt_candidates = []
        for prompt in track(prompts, description="Generating prompt candidates", transient=True):
            # Generate gradients
            template_kwargs = {
                "prompt": prompt.content,
                "error_string": prompt.metadata.get("error_string", ""),
                "num_feedbacks": self.num_feedbacks,
                "steps_per_gradient": self.steps_per_gradient,
            }
            raw_gradients = self._generate(metaprompt_template=GENERATE_GRADIENT_PROMPT_TEMPLATE, template_kwargs=template_kwargs)
            gradients = self._extract_responses(raw_gradients)
            gradients = gradients[: self.num_feedbacks]
            # Generate prompts for each gradient
            for gradient in gradients:
                template_kwargs.update({"gradient": gradient})
                raw_new_prompts = self._generate(metaprompt_template=INCORPORTATE_GRADIENT_PROMPT_TEMPLATE, template_kwargs=template_kwargs)
                new_prompts = self._extract_responses(raw_new_prompts)
                new_prompts = new_prompts[: self.steps_per_gradient]
                metadata = {"_origin_prompt": prompt.content, "_gradient": gradient, "_resampled": False}
                new_prompt_candidates = [BasePrompt(content=new_prompt, metadata=metadata) for new_prompt in new_prompts]

                # Resample new prompts
                for new_prompt in new_prompts:
                    varied_prompts = [
                        self._generate(metaprompt_template=RESAMPLING_PROMPT_TEMPLATE, template_kwargs={"prompt": new_prompt})
                        for _ in range(self.num_resample)
                    ]
                    metadata = {"_origin_prompt": new_prompt, "_gradient": None, "_resampled": True}
                    varied_prompts = [BasePrompt(content=new_prompt) for new_prompt in varied_prompts]
                    new_prompt_candidates.extend(varied_prompts)

                # Save prompts to prompt candidates
                prompt_candidates.extend(new_prompt_candidates)

        # Add back the initial prompts to the pool
        prompt_candidates = prompts + prompt_candidates

        return prompt_candidates

    def _get_best_prompt(self, prompts: list[BasePrompt]):
        """Get the highest scoring prompt."""
        if any(prompt.score is None for prompt in prompts):
            raise ValueError("All prompts must be scored before calling this function.")
        return max(prompts, key=lambda x: x.score)

    def select_prompt_candidates(self, *, prompts: list[BasePrompt], validation_set: ValidationSetType) -> list[BasePrompt]:
        """Select prompt candidates according to the search mode."""
        self._score_prompts(prompts=prompts, validation_set=validation_set)
        if self.search_mode == "greedy":
            return prompts
        elif self.search_mode == "beam":
            return [self._get_best_prompt(prompts=prompts)]

    def check_early_convergence(self, *, all_prompts: list[list[BasePrompt]]):
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

    def select_best_prompt(self, *, all_prompts: list[list[BasePrompt]]) -> BasePrompt:
        """Select the top scoring prompt."""
        # Flatten all iterations
        prompts = sum(all_prompts, start=[])

        # Select the single prompt with the highest score
        best_prompt = self._get_best_prompt(prompts=prompts)
        logger.info(f"Best score: {best_prompt.score:.3f}")
        return best_prompt

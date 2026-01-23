import logging
import random
from typing import Callable, Optional, Union

from rich.progress import track

from prompt_optimizer import BasePrompt
from prompt_optimizer.pipeline import BasePipeline
from prompt_optimizer.types import ClientType, ScoreType, ValidationSetType

logger = logging.getLogger(__name__)

FORWARD_GENERATION_TEMPLATE = """I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:

{input_output_pairs}
...

The instruction was """

RESAMPLING_PROMPT_TEMPLATE = """Generate a variation of the following instruction while keeping the semantic meaning.

Input: {instruction}
Output: """


class APEOptimizer(BasePipeline):
    """
    APE Optimizer.

    Based on Automatic Prompt Engineer from Zhou, et. al.

    ```
    @misc{zhou2023largelanguagemodelshumanlevel,
        title={Large Language Models Are Human-Level Prompt Engineers},
        author={Yongchao Zhou and Andrei Ioan Muresanu and Ziwen Han and Keiran Paster and Silviu Pitis and Harris Chan and Jimmy Ba},
        year={2023},
        eprint={2211.01910},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2211.01910},
    }
    ```

    """

    def __init__(
        self,
        *,
        client: ClientType,
        validation_set: ValidationSetType,
        max_depth: int,
        evaluator: Callable[[BasePrompt, ValidationSetType], ScoreType],
        input_field: str,
        output_field: str,
        num_initial_prompts: int = 10,
        num_exemplars: int = 5,
        k_percent: float = 0.5,
        score_threshold: Optional[Union[float, int]] = None,
    ):
        """
        Initialize the APE optimizer.

        Args:
            client (ClientType):
                Language model client to use for prompt generation and feedback.
            validation_set (ValidationSetType):
                Set of examples to evaluate the prompt on.
            max_depth (int):
                Maximum iteration depth for prompt generation.
            evaluator (Callable[[BasePrompt, ValidationSetType], ScoreType]):
                Function that takes a prompt and the validation data and returns a score.
            input_field (str):
                Field in the validation set that represents the input. Used in forward generation in
                the "Input:" field.
            output_field (str):
                Field in the validation set that represents the output. Used in forward generation in
                the "Output:" field.
            num_initial_prompts (int):
                Number of prompts to create in the initial forward generation. Defaults to 10.
            num_exemplars (int):
                Number of exemplars from the validation set to provide for forward generation.
                A random sample of input and output pairs of this size will be provided to the LLM
                during forward generation. Defaults to 5.
            k_percent (str, optional):
                Top k% of candidate prompts to retain between iterations. Defaults to 0.5.
            score_threshold (float, optional):
                Threshold for early convergence. If a prompt exceeds this score after any iteration, the optimization loop
                immediately ends. If set to None, the optimization loop will not terminate early. Defaults to None.
        """
        super().__init__(
            client=client,
            seed_prompts=[],  # There is no seeding for APE
            validation_set=validation_set,
            max_depth=max_depth,
            evaluator=evaluator,
        )
        self.num_initial_prompts = num_initial_prompts
        self.num_exemplars = num_exemplars
        self.input_field = input_field
        self.output_field = output_field
        self.k_percent = k_percent
        self.score_threshold = score_threshold

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

    def generate_prompt_candidates(self, *, prompts: list[BasePrompt], validation_set: ValidationSetType) -> list[BasePrompt]:
        """Generate prompt candidates using forward generation or resampling."""
        prompt_candidates = []

        # Do forward generation if there are no prompts
        if len(prompts) == 0:
            for i in track(range(self.num_initial_prompts), description="Generating new prompts", transient=True):
                sample = random.choices(validation_set, k=self.num_exemplars)
                input_output_pairs = "\n".join(
                    [f"**Input:** {exemplar[self.input_field]}   **Output:** {exemplar[self.output_field]}" for exemplar in sample]
                )
                template_kwargs = {"input_output_pairs": input_output_pairs}
                new_prompt = self._generate(metaprompt_template=FORWARD_GENERATION_TEMPLATE, template_kwargs=template_kwargs)
                prompt_candidates.append(BasePrompt(content=new_prompt))

        # Otherwise, resample the prompts
        else:
            for prompt in track(prompts, description="Generating new prompts", transient=True):
                template_kwargs = {"instruction": prompt.content}
                new_prompt = self._generate(metaprompt_template=RESAMPLING_PROMPT_TEMPLATE, template_kwargs=template_kwargs)
                prompt_candidates.append(BasePrompt(content=new_prompt))

        return prompt_candidates

    def _get_best_prompt(self, prompts: list[BasePrompt]):
        if any(prompt.score is None for prompt in prompts):
            raise ValueError("All prompts must be scored before calling this function.")
        return max(prompts, key=lambda x: x.score)

    def select_prompt_candidates(self, *, prompts: list[BasePrompt], validation_set: ValidationSetType) -> list[BasePrompt]:
        """Select the top scoring k% of prompts."""
        self._score_prompts(prompts=prompts, validation_set=validation_set)
        # Select the top scoring k% of prompts
        k = int(len(prompts) * self.k_percent)
        best_prompts = sorted(prompts, key=lambda x: x.score, reverse=True)[:k]
        return best_prompts

    def check_early_convergence(self, *, all_prompts: list[list[BasePrompt]]):
        """Check if any prompt exceeds the score threshold."""
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
        """Select the highest scoring prompt."""
        # Flatten all iterations
        prompts = sum(all_prompts, start=[])

        # Select the single prompt with the highest score
        best_prompt = self._get_best_prompt(prompts=prompts)
        logger.info(f"Best score: {best_prompt.score:.3f}")
        return best_prompt

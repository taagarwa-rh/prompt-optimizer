import logging
import random
from typing import Callable, Optional, Union

from rich.progress import track

from prompt_optimizer import BasePrompt
from prompt_optimizer.pipeline import BasePipeline
from prompt_optimizer.types import ClientType, ScoreType, ValidationSetType

logger = logging.getLogger(__name__)


METAPROMPT_TEMPLATE = """I have some texts along with their corresponding scores. The texts are arranged in ascending order based on their scores, where higher scores indicate better quality.

{instructions_and_scores}

The following exemplars show how to apply your text: you replace <INS> in each input with your text, then read the input and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same. 

{input_output_pairs}

Write your new text that is different from the old ones and has a score as high as possible. Write the text in square brackets."""


class OPROOptimizer(BasePipeline):
    """
    OPRO Optimizer.

    Based on Optimization by PROmpting from Yang, et. al. 2024

    ```
    @misc{yang2024largelanguagemodelsoptimizers,
        title={Large Language Models as Optimizers},
        author={Chengrun Yang and Xuezhi Wang and Yifeng Lu and Hanxiao Liu and Quoc V. Le and Denny Zhou and Xinyun Chen},
        year={2024},
        eprint={2309.03409},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2309.03409},
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
        input_field: str,
        output_field: str,
        num_candidates_per_step: int = 20,
        num_exemplars: int = 3,
        max_demonstration_prompts: int = 20,
        score_threshold: Optional[Union[float, int]] = None,
    ):
        """
        Initialize the APE optimizer.

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
            input_field (str):
                Field in the validation set that represents the input. Used in candidate generation in
                the "input:" field.
            output_field (str):
                Field in the validation set that represents the output. Used in candidate generation in
                the "output:" field.
            num_candidates_per_step (int, optional):
                Number of candidates to create at each step. Defaults to 20.
            num_exemplars (int, optional):
                Number of exemplars from the validation set to provide in the metaprompt.
                A random sample of input and output pairs of this size will be provided to the LLM
                during candidate generation. Defaults to 3.
            max_demonstration_prompts (int, optional):
                Maximum number of demostration prompts to provide in the metaprompt.
                Defaults to 20.
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
        )
        self.num_candidates_per_step = num_candidates_per_step
        self.num_exemplars = num_exemplars
        self.max_demonstration_prompts = max_demonstration_prompts
        self.input_field = input_field
        self.output_field = output_field
        self.score_threshold = score_threshold

    def _extract_response(self, content: str) -> str:
        """
        Extract the responses between square brackets.

        Args:
            content (str): Output string from an LLM generation request.

        Returns:
            str: Response captured between square brackets, or the whole response if there are no square brackets.

        """
        if "[" not in content or "]" not in content:
            return content

        # Get everything after the first bracket
        parsed_content = content.split("[", maxsplit=1)[1]

        # Get everything before the last bracket
        parsed_content = parsed_content[::-1].split("]", maxsplit=1)[1][::-1]

        return parsed_content

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
        """Generate prompt candidates."""
        # Sort prompts by score, highest to lowest
        sorted_prompts = sorted(prompts, key=lambda x: x.score, reverse=True)[: self.max_demonstration_prompts]

        # Format prompts into instructions_and_scores
        instructions_and_scores = "\n\n".join([f"text:\n{prompt.content}\nscore:\n{prompt.score}" for prompt in sorted_prompts])

        # Generate prompt candidates
        prompt_candidates = []
        for _ in track(range(self.num_candidates_per_step), description="Generating prompt candidates", transient=True):
            # Format a sample of questions into input_output_pairs
            sample = random.choices(validation_set, k=self.num_exemplars)
            input_output_pairs = "\n\n".join(
                [f"input:\n<INS>\n{row[self.input_field]}\noutput:\n{row[self.output_field]}" for row in sample]
            )

            # Generate prompt candidate
            template_kwargs = {"instructions_and_scores": instructions_and_scores, "input_output_pairs": input_output_pairs}
            response = self._generate(metaprompt_template=METAPROMPT_TEMPLATE, template_kwargs=template_kwargs)
            prompt_candidate = self._extract_response(response)

            prompt_candidates.append(BasePrompt(content=prompt_candidate))

        return prompt_candidates

    def select_prompt_candidates(self, *, prompts: list[BasePrompt], validation_set: ValidationSetType) -> list[BasePrompt]:
        """Select prompt candidates."""
        # Score the prompts
        self._score_prompts(prompts=prompts, validation_set=validation_set)
        # Return all prompts, no filtering is done
        return prompts

    def check_early_convergence(self, *, all_prompts: list[list[BasePrompt]]) -> bool:
        """Detect early convergence."""
        if self.score_threshold is None:
            return False

        # Flatten all iterations
        prompts = sum(all_prompts, start=[])

        # Check if early convergence criteria is met
        highest_score = max(prompts, key=lambda x: x.score).score
        if highest_score >= self.score_threshold:
            return True
        return False

    def _get_best_prompt(self, prompts: list[BasePrompt]):
        """Get the highest scoring prompt."""
        if any(prompt.score is None for prompt in prompts):
            raise ValueError("All prompts must be scored before calling this function.")
        return max(prompts, key=lambda x: x.score)

    def select_best_prompt(self, all_prompts: list[list[BasePrompt]]) -> BasePrompt:
        """Select the top scoring prompt."""
        # Flatten all iterations
        prompts = sum(all_prompts, start=[])

        # Select the single prompt with the highest score
        best_prompt = self._get_best_prompt(prompts=prompts)
        logger.info(f"Best score: {best_prompt.score:.3f}")
        return best_prompt

import logging
from abc import ABC
from pathlib import Path
from typing import Callable, Optional, Union

from rich.progress import track

from prompt_optimizer import Prompt
from prompt_optimizer.types import ClientType, ScoreType, ValidationSetType

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Base pipeline for most APO algorithms.

    Based on Algorithm 1 in https://arxiv.org/abs/2502.16923.
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
    ):
        """
        Initialize the inference evaluation pipeline.

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

        """
        self.client = client
        self.seed_prompts = seed_prompts
        self.validation_set = validation_set
        self.max_depth = max_depth
        self.evaluator = evaluator
        self.output_path = output_path
        self._p: list[list[Prompt]] = []
        self._g: list[list[Prompt]] = []

    ## Class Utilities ##

    def get_all_prompts(self, include_candidates: bool = False) -> list[list[Prompt]]:
        """
        Get all the prompts from the latest training run.

        The default behavior returns a list of lists, where each internal list contains the
        retained candidates after one iteration step.
        Setting include_candidates to True will also include all generated candidate prompts.

        Args:
            include_candidates (bool, optional):
                Whether to include all the candidate prompts in the output.
                If True, candidate prompts from each iteration will be included.
                Defaults to False.

        Returns:
            list[list[Prompt]]:
                List of lists where each list contains the prompts from each iteration.
                E.g. list[0] contains prompts from the first iteration, list[1] the second, etc.
                If include_candidates is False, each inner list contains only the retained prompts at each iteration.
                If include_candidates is True, each inner list contains all candidate prompts at each iteration,
                including those that were discarded.

        """
        # Decide whether to include candidates
        if include_candidates:
            all_prompts = self._g
        else:
            all_prompts = self._p

        return all_prompts

    def save_prompts(self, output_path: Optional[Union[str, Path]]):
        """Save prompts in jsonl format."""
        # Exit if no output path is set
        if self.output_path is None:
            return

        # Get and deduplicate prompts
        prompts = sum(self._p, start=[])
        prompts = list(set(prompts))

        # Save the prompts to the file
        lines = [prompt.model_dump_json() for prompt in prompts]
        with open(output_path, "w") as f:
            for line in lines:
                f.write(line)
                f.write("\n")

    ## Scoring ##
    def _evaluate(self, prompt: Prompt, validation_set: ValidationSetType) -> ScoreType:
        """
        Evaluate the prompt using the evaluator function.

        Args:
            prompt (Prompt): Prompt to pass as a keyword argument `prompt` to the evaluator.
            validation_set (ValidationSetType): Set of examples to evaluate the prompt on.

        Returns:
            ScoreType: The aggregate evaluation results.

        """
        score = self.evaluator(prompt=prompt, validation_set=validation_set)
        return score

    def _score_prompts(self, prompts: list[Prompt], validation_set: ValidationSetType) -> list[Prompt]:
        """
        Score all the prompts using the evaluator.

        Args:
            prompts (list[Prompt]): The prompts to score.
            validation_set (ValidationSetType): Set of examples to evaluate the prompt on.

        Returns:
            list[Prompts]: The scored prompts with the 'score' field set.
        """
        # BUG: Scores being assigned to existing objects instead of returning new ones
        for prompt in track(prompts, description="Evaluating Prompts", transient=True):
            if prompt.score is None:
                try:
                    score = self._evaluate(prompt=prompt, validation_set=validation_set)
                    prompt.score = score
                except Exception as e:
                    logger.error(f"Error scoring prompt {prompt.content}: {e}")
                    prompt.score = 0.0
        return prompts

    ## Optimization Loop ##

    def generate_prompt_candidates(self, *, prompts: list[Prompt], validation_set: ValidationSetType) -> list[Prompt]:
        """Generate prompt candidates."""
        raise NotImplementedError("This method must be implemented by the child class")

    def select_prompt_candidates(self, *, prompts: list[Prompt], validation_set: ValidationSetType) -> list[Prompt]:
        """Select prompt candidates."""
        raise NotImplementedError("This method must be implemented by the child class")

    def check_early_convergence(self, *, all_prompts: list[list[Prompt]]) -> bool:
        """Detect early convergence."""
        raise NotImplementedError("This method must be implemented by the child class")

    def select_best_prompt(self, all_prompts: list[list[Prompt]]) -> Prompt:
        """Select the best prompt."""
        raise NotImplementedError("This method must be implemented by the child class")

    def run(self) -> Prompt:
        """Run the optimization pipeline."""
        # Score seed_prompts
        self.seed_prompts = self._score_prompts(self.seed_prompts, self.validation_set)

        # Initialize objects
        self._p = [self.seed_prompts]
        self._g = [self.seed_prompts]

        # Iterate until max depth
        for t in track(range(1, self.max_depth + 1), description="Step", total=self.max_depth):
            # Generate prompt candidates
            g_t = self.generate_prompt_candidates(prompts=self._p[t - 1], validation_set=self.validation_set)
            self._g.append(g_t)
            # Select prompt candidates
            p_t = self.select_prompt_candidates(prompts=self._g[t], validation_set=self.validation_set)
            self._p.append(p_t)
            # Check for early convergence
            if self.check_early_convergence(all_prompts=self._p):
                break

        # Save prompts if requested
        self.save_prompts(output_path=self.output_path)

        # Return best prompt
        return self.select_best_prompt(all_prompts=self._p)

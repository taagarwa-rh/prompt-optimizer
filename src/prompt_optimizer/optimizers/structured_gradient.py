"""Gradient prompt tuning with structured outputs based on the paper: https://arxiv.org/pdf/2305.03495."""

import logging
from typing import Generator

from pydantic import BaseModel

from .gradient import GradientOptimizer

logger = logging.getLogger(__name__)

GRADIENT_PROMPT = """I'm trying to write a zero-shot classifier prompt.
My current prompt is:
"{prompt}"
But this prompt gets the following examples wrong:
{error_string}
give {num_feedbacks} reasons why the prompt could
have gotten these examples wrong.
Respond using JSON only."""

REWRITE_PROMPT = """I'm trying to write a zero-shot classifier prompt.
My current prompt is:
"{prompt}"
But it gets the following examples wrong:
{error_string}
Based on these examples the problem with this prompt is that {gradient}
Based on the above information, write {steps_per_gradient} different improved prompts.
Respond using JSON only."""


class GradientResponse(BaseModel):
    """Gradient response model."""

    feedbacks: list[str]


class RewriteResponse(BaseModel):
    """Rewrite response model."""

    prompts: list[str]


class StructuredGradientOptimizer(GradientOptimizer):
    """A more stable version of the GradientOptimizer using structured generation."""

    def _generate(self, prompt_template: str, template_kwargs: dict, response_format: BaseModel, **kwargs) -> list[str]:
        """
        Generate a completion for a given template and kwargs and parse the results.

        Args:
            prompt_template (str): Template for the prompt.
            template_kwargs (dict): Key word arguments to fill the template values.
            response_format (BaseModel): Pydantic model for LLM response structure.
            kwargs: Additional kwargs to pass to the openai.client.completions.parse (e.g. temperature)

        Returns:
            list[str]: The parsed generation results.

        """
        prompt = prompt_template.format(**template_kwargs)
        messages = [{"role": "user", "content": prompt}]
        raw_response = self.client.beta.chat.completions.parse(
            messages=messages, model=self.model_name, response_format=response_format, **kwargs
        )
        response_model: BaseModel = raw_response.choices[0].message.parsed
        responses = response_model.feedbacks if hasattr(response_model, "feedbacks") else response_model.prompts
        return responses

    def _generate_new_prompts(self, prompt: str, error_string: str, num_feedbacks: int, steps_per_gradient: int, **kwargs) -> Generator[str, None, None]:
        """
        Generate a number of new prompts based on the given prompt and error string.

        Args:
            prompt (str): 
                Prompt to generate new prompts off of.
            error_string (str): 
                Description of the errors with the `prompt`.
            num_feedbacks (int, optional):
                Number of feedbacks (gradients) to generate for each prompt. This won't be enforced and the LLM may generate more or less than this.
            steps_per_gradient (int, optional):
                Number of steps (new prompts) to generate per gradient. This won't be enforced and the LLM may generate more or less than this.
            kwargs: Additional kwargs to pass to the OpenAI client.completions.create (e.g. temperature)

        Returns:
            Generator[str]: New prompts that attempt to correct the errors presented in `error_string`.

        """
        template_kwargs = {
            "prompt": prompt,
            "error_string": error_string,
            "num_feedbacks": num_feedbacks,
            "steps_per_gradient": steps_per_gradient,
        }
        gradients = self._generate(
            prompt_template=GRADIENT_PROMPT, template_kwargs=template_kwargs, response_format=GradientResponse, **kwargs
        )
        gradients = gradients[: num_feedbacks]
        for gradient in gradients:
            template_kwargs.update({"gradient": gradient})
            new_prompts = self._generate(
                prompt_template=REWRITE_PROMPT, template_kwargs=template_kwargs, response_format=RewriteResponse, **kwargs
            )
            new_prompts = new_prompts[: steps_per_gradient]
            for new_prompt in new_prompts:
                yield new_prompt

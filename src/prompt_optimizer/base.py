from typing import Any, Optional, Self

from pydantic import BaseModel, model_validator

from .types import PromptContentType, ScoreType


class PredictionError(BaseModel):
    """
    Save prediction errors for error-based optimizers.

    Args:
        input (Any):
            The input data. This could be a question, request, messages, or any other relevant
            information for feedback generation.
        prediction (Any):
            The prediction made by the model/agent/application. This could be a chat response,
            classification label, or any other relevant information.
        actual (Any, optional):
            The expected output. If the error is a result of the prediction not matching the expected output,
            this should be set. Otherwise, you should set `feedback` to explain why it failed. Defaults to None.
        feedback (str, optional):
            Any additional feedback to provide to the feedback generation. If the error is due to the prediction
            not meeting some metric not satisfied by providing the actual (e.g. generation length was too long)
            you should provide that feedback here. Defaults to None.

    """

    input: Any
    prediction: Any
    actual: Optional[Any] = None
    feedback: Optional[str] = None

    @model_validator(mode="after")
    def validate_presence(self) -> Self:
        """Ensure that either actual or feedback is set."""
        if not (self.actual or self.feedback):
            raise ValueError("Either actual or feedback must be set.")
        return self


class BasePrompt(BaseModel):
    """
    Base prompt object.

    Args:
        content (str):
            Content of the prompt.
        score (ScoreType, optional):
            Score of the prompt. This should be given by an `evaluator` function.
            Defaults to None.
        errors (list[PredictionError]):
            List of errors caught during evaluation. This is required for some
            optimizers. Defaults to None.
        metadata (dict[str, Any], optional):
            Optional metadata to associate with the prompt. Some optimizers
            also store variables here, each prefixed by `_`. Defaults to None.

    """

    content: PromptContentType
    score: ScoreType = None
    errors: list[PredictionError] = []
    metadata: dict[str, Any] = {}

    def __hash__(self):
        """Hash the prompt content."""
        return hash(self.content)

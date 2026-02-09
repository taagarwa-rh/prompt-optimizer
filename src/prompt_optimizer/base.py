from typing import Any, Optional

from pydantic import BaseModel

from .types import PromptContentType, ScoreType


class PredictionError(BaseModel):
    """Save prediction errors for error-based optimizers."""

    input: Any
    prediction: Any
    actual: Any
    feedback: Optional[str] = None


class BasePrompt(BaseModel):
    """Base prompt object."""

    content: PromptContentType
    score: ScoreType = None
    errors: list[PredictionError] = []
    metadata: dict[str, Any] = {}

    def __hash__(self):
        """Hash the prompt content."""
        return hash(self.content)

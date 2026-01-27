from typing import Any

from pydantic import BaseModel

from .types import PromptContentType, ScoreType


class BasePrompt(BaseModel):
    """Base prompt object."""

    content: PromptContentType
    score: ScoreType = None
    metadata: dict[str, Any] = {}

    def __hash__(self):
        """Hash the prompt content."""
        return hash(self.content)

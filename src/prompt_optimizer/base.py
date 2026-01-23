from typing import Any

from pydantic import BaseModel

from .types import PromptContentType, ScoreType


class BasePrompt(BaseModel):
    """Base prompt object."""

    content: PromptContentType
    score: ScoreType = None
    metadata: dict[str, Any] = {}

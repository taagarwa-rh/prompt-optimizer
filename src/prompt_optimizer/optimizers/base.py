from abc import ABC
from typing import Optional

from pydantic import BaseModel

from .types import PipelineOutputType, Prompt, Reward


class RewardFuncResult(BaseModel):
    """Reward function result model."""

    reward: Reward
    error_string: Optional[str] = None


class PromptResult(BaseModel):
    """Prompt result model."""

    id: int
    step: int
    parent_id: Optional[int] = None
    gradient: Optional[str] = None
    prompt: Prompt
    predictions: list[PipelineOutputType]
    reward: Reward
    error_string: Optional[str] = None


class BaseOptimizer(ABC):
    """Base class for optimizers."""

    def __init__(self):
        """Initialize."""
        pass

    def optimize(self, baseline_prompt: str, **kwargs):
        """Optimize a prompt."""
        pass

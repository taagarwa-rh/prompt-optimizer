from abc import ABC
from typing import Optional

from pydantic import BaseModel


class MetricResult(BaseModel):
    """Metric result model."""

    score: float | int
    error_string: Optional[str] = None


class PromptResult(BaseModel):
    """Prompt result model."""

    prompt: str
    score: float | int
    error_string: Optional[str] = None


class BaseOptimizer(ABC):
    """Base class for optimizers."""

    def __init__(self):
        """Initialize."""
        pass

    def optimize(self, baseline_prompt: str, **kwargs):
        """Optimize a prompt."""
        pass

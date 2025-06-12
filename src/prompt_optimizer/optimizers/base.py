from typing import Optional

from pydantic import BaseModel


class MetricResult(BaseModel):
    score: float | int
    error_string: Optional[str] = None


class PromptResult(BaseModel):
    prompt: str
    score: float | int
    error_string: Optional[str] = None


class BaseOptimizer:
    def __init__(self):
        pass

    def optimize(self, baseline_prompt: str, **kwargs):
        pass

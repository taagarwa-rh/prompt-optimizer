from pydantic import BaseModel
from typing import Optional

class MetricResult(BaseModel):
    
    score: float
    error_string: Optional[str] = None
    
class PromptResult(BaseModel):
    
    prompt: str
    score: float
    error_string: Optional[str] = None

class BaseOptimizer:
    
    def __init__(self):
        pass
    
    def optimize(self, baseline_prompt: str, **kwargs):
        pass
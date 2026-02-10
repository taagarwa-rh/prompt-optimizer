import pytest

from pydantic import ValidationError

from prompt_optimizer.base import BasePrompt, PredictionError


class TestBasePrompt:
    
    def test_init(self):
        """Test initialization schemes."""
        result = BasePrompt(content="Hello, world!", score=0.9)
        assert result.content == "Hello, world!"
        assert result.score == 0.9
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0
        
        result.metadata["test"] = "test"
        assert len(result.metadata) == 1
        assert result.metadata.get("test") == "test"
        assert result.metadata.get("other") is None
        
    
class TestPredictionError:
    
    def test_init(self):
        """Test initialization."""
        result = PredictionError(input="What is the best city?", prediction="Paris", actual="Rome")
        assert result.input == "What is the best city?"
        assert result.prediction == "Paris"
        assert result.actual == "Rome"
        
        result = PredictionError(input="What is the best city?", prediction="Paris", feedback="The prediction was too short.")
        assert result.feedback == "The prediction was too short."
    
    def test_validation(self):
        """Test model validation."""
        with pytest.raises(ValidationError):
            PredictionError(input="What is the best city?", prediction="Paris")
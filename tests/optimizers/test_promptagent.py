import pytest

from prompt_optimizer.optimizers.promptagent import PromptAgentOptimizer
from prompt_optimizer.base import BasePrompt

class TestPromptAgentOptimizer:
    
    @pytest.fixture
    def mock_cls(self, mock_client, mock_seed_prompts, mock_validation_set, exact_match_evaluator) -> PromptAgentOptimizer:
        cls = PromptAgentOptimizer(
            client=mock_client,
            seed_prompts=mock_seed_prompts,
            validation_set=mock_validation_set,
            max_depth=3,
            evaluator=exact_match_evaluator
        )
        
        return cls
    
    def test_initialize(self, mock_cls: PromptAgentOptimizer, mock_client, mock_seed_prompts, mock_validation_set, exact_match_evaluator):
        """Test class initialization."""
        assert isinstance(mock_cls, PromptAgentOptimizer)
        assert mock_cls.client == mock_client
        assert mock_cls.seed_prompts == mock_seed_prompts
        assert mock_cls.validation_set == mock_validation_set
        assert mock_cls.max_depth == 3
        assert mock_cls.evaluator == exact_match_evaluator
        
    
    def test_map_trajectory(self, mock_cls: PromptAgentOptimizer):
        """Test prompt trajectory mapping."""
        
        prompt1 = BasePrompt(content="What is the capital of France?", score=0.7)
        prompt2 = BasePrompt(content="What is the largest city in France?", score=0.8, metadata={mock_cls.PARENT_KEY: prompt1})
        prompt3 = BasePrompt(content="What is France's capital city?", score=0.9, metadata={mock_cls.PARENT_KEY: prompt2})
        
        # Base case - no parents, no trajectory
        result = mock_cls._map_trajectory(prompt=prompt1)
        assert isinstance(result, str)
        assert result.count("Prompt:") == 1
        assert prompt1.content in result
        assert result == "Prompt: What is the capital of France?\nScore: 0.7"
        
        # First generation case - one parent, one item trajectory
        result = mock_cls._map_trajectory(prompt=prompt2)
        assert isinstance(result, str)
        assert result.count("Prompt:") == 2
        assert prompt1.content in result
        assert prompt2.content in result
        assert result == "Prompt: What is the capital of France?\nScore: 0.7\n\nPrompt: What is the largest city in France?\nScore: 0.8"

        # Second generation case - Second parent, two items trajectory
        result = mock_cls._map_trajectory(prompt=prompt3)
        assert isinstance(result, str)
        assert result.count("Prompt:") == 3
        assert prompt1.content in result
        assert prompt2.content in result
        assert prompt3.content in result
        assert result == "Prompt: What is the capital of France?\nScore: 0.7\n\nPrompt: What is the largest city in France?\nScore: 0.8\n\nPrompt: What is France's capital city?\nScore: 0.9"
        
    
    def test_select_prompt_candidates(self, mock_cls: PromptAgentOptimizer, mock_validation_set):
        """Test prompt candidate selection"""
        
        # Create branches of prompts
        # In each branch, child1 is the best prompt, and should be selected when search_mode == "beam"
        parent1 = BasePrompt(content="What is the capital of France?")
        child1_1 = BasePrompt(content="What is the largest city in France?", score=0.9, metadata={mock_cls.PARENT_KEY: parent1})
        child2_1 = BasePrompt(content="What is the largest city in Spain?", score=0.8, metadata={mock_cls.PARENT_KEY: parent1})
        child3_1 = BasePrompt(content="What is the capital?", score=0.7, metadata={mock_cls.PARENT_KEY: parent1})
        
        parent2 = BasePrompt(content="What is the largest planet in the Solar System?")
        child1_2 = BasePrompt(content="What is the largest planet?", score=0.9, metadata={mock_cls.PARENT_KEY: parent2})
        child2_2 = BasePrompt(content="What is the largest gas giant?", score=0.8, metadata={mock_cls.PARENT_KEY: parent2})
        child3_2 = BasePrompt(content="What is Jupiter?", score=0.7, metadata={mock_cls.PARENT_KEY: parent2})
        
        # Select prompt candidates with search_mode == "beam"
        mock_cls.search_mode = "beam"
        prompts = [child1_1, child2_1, child3_1, child1_2, child2_2, child3_2]
        result = mock_cls.select_prompt_candidates(prompts=prompts, validation_set=mock_validation_set)
        
        # Unit test
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(prompt, BasePrompt) for prompt in result)
        
        # Assert child1 (the best prompt) was selected in both branches
        assert set(result) == {child1_1, child1_2}
        
        # Select prompt candidates with search_mode == "greedy"
        mock_cls.search_mode = "greedy"
        prompts = [child1_1, child2_1, child3_1, child1_2, child2_2, child3_2]
        result = mock_cls.select_prompt_candidates(prompts=prompts, validation_set=mock_validation_set)
        
        # Unit test
        assert isinstance(result, list)
        assert len(result) == 6
        assert all(isinstance(prompt, BasePrompt) for prompt in result)
        
        # Assert all children were kept
        assert set(result) == set(prompts)
        
        
import pytest

INPUT_KEY = "input"
OUTPUT_KEY = "ouput"

@pytest.fixture()
def mock_client():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    class MockClient(GenericFakeChatModel):
        pass
    return MockClient

@pytest.fixture()
def mock_seed_prompts():
    from prompt_optimizer.base import BasePrompt
    seed_prompts = [BasePrompt(content="test")]
    return seed_prompts


def dataset_map():
    return {
        "What is the capital of France?": "Paris",
        "What is the largest planet in the Solar System?": "Jupiter",
        "What is the smallest planet in the Solar System?": "Mercury",
    }

@pytest.fixture()
def mock_validation_set():
    validation_set = [{INPUT_KEY: key, OUTPUT_KEY: value} for key, value in dataset_map().items()]
    return validation_set

@pytest.fixture()
def exact_match_evaluator():
    from prompt_optimizer.base import PredictionError, BasePrompt
    
    def evaluator(prompt: BasePrompt, validation_set: list[dict]):
        # Run the prompt through the AI system
        predictions = []
        num_correct = 0
        for row in validation_set:
            # Get the prediction
            input = row[INPUT_KEY]
            prediction = dataset_map()[input]
            predictions.append(prediction)

            # Reward exact matches and collect errors
            actual = row[OUTPUT_KEY]
            if actual == prediction:
                num_correct += 1
            else:
                num_correct += 0
                prompt.errors.append(PredictionError(input=input, prediction=prediction, actual=actual, feedback=None))

        # Compute the score
        score = num_correct / len(validation_set)

        return score
        
    return evaluator
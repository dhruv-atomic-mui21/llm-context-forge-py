"""
Shared sample data for tests.
"""

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning is transforming the technology landscape. "
    "Natural language processing enables computers to understand text. "
    "Deep learning models require large amounts of training data. "
    "Transformers have revolutionized the field of NLP. "
    "Attention mechanisms allow models to focus on relevant parts. "
    "Pre-trained models can be fine-tuned for specific tasks. "
    "Context windows limit the amount of text models can process. "
    "Token counting is essential for managing API costs. "
    "Prompt engineering helps get better results from LLMs. "
) * 5

CODE_SNIPPET = """
def calculate_cost(tokens, rate):
    return (tokens / 1000) * rate

class CostEstimator:
    def __init__(self, model):
        self.model = model
        
    def estimate(self, text):
        return len(text) * 0.001
"""

MARKDOWN_DOC = """
# LLM Context Forge

A tool for managing context.

## Token Counting

Accurate token blocks.

### Supported Models

- GPT-4o
- Claude 3.5 Sonnet
"""

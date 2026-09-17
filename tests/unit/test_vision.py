"""
Unit tests for Multimodal Vision Token Counter.
"""

import io
import pytest
from PIL import Image

from llm_context_forge.vision import (
    VisionTokenCounter,
    OpenAIVisionStrategy,
    AnthropicVisionStrategy,
    GoogleVisionStrategy,
    GenericVisionStrategy,
    count_image_tokens,
)


class TestOpenAIVisionStrategy:
    def setup_method(self):
        self.strategy = OpenAIVisionStrategy()

    def test_low_detail_fixed_cost(self):
        assert self.strategy.calculate_tokens(100, 100, detail="low") == 85
        assert self.strategy.calculate_tokens(2048, 2048, detail="low") == 85

    def test_high_detail_512x512_single_tile(self):
        # 512x512: 1 tile -> 170 * 1 + 85 = 255
        assert self.strategy.calculate_tokens(512, 512, detail="high") == 255

    def test_high_detail_1024x1024_four_tiles(self):
        # 1024x1024 scaled down to 768x768 -> ceil(768/512)*ceil(768/512) = 2*2 = 4 tiles -> 4*170 + 85 = 765
        assert self.strategy.calculate_tokens(1024, 1024, detail="high") == 765

    def test_auto_detail_threshold(self):
        # <= 512x512 defaults to low detail
        assert self.strategy.calculate_tokens(512, 512, detail="auto") == 85
        # > 512x512 defaults to high detail
        assert self.strategy.calculate_tokens(1024, 1024, detail="auto") == 765

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            self.strategy.calculate_tokens(-10, 100)
        with pytest.raises(ValueError):
            self.strategy.calculate_tokens(100, 0)

    def test_invalid_detail(self):
        with pytest.raises(ValueError):
            self.strategy.calculate_tokens(100, 100, detail="ultra")


class TestAnthropicVisionStrategy:
    def setup_method(self):
        self.strategy = AnthropicVisionStrategy()

    def test_standard_scaling(self):
        # 750 pixels per token
        assert self.strategy.calculate_tokens(750, 750) == 750

    def test_downscale_above_1568(self):
        tokens_3000 = self.strategy.calculate_tokens(3000, 3000)
        tokens_1568 = self.strategy.calculate_tokens(1568, 1568)
        assert tokens_3000 == tokens_1568


class TestGoogleVisionStrategy:
    def setup_method(self):
        self.strategy = GoogleVisionStrategy()

    def test_small_image_base_patch(self):
        assert self.strategy.calculate_tokens(300, 300) == 258

    def test_tiled_image(self):
        # 1000x1000 -> 2x2 patches -> 4 * 258 = 1032
        assert self.strategy.calculate_tokens(1000, 1000) == 1032


class TestVisionTokenCounter:
    def setup_method(self):
        self.counter = VisionTokenCounter()

    def test_model_resolution(self):
        assert isinstance(self.counter.resolve_strategy("gpt-4o"), OpenAIVisionStrategy)
        assert isinstance(self.counter.resolve_strategy("claude-3-5-sonnet"), AnthropicVisionStrategy)
        assert isinstance(self.counter.resolve_strategy("gemini-1.5-pro"), GoogleVisionStrategy)
        assert isinstance(self.counter.resolve_strategy("custom-llm"), GenericVisionStrategy)

    def test_count_with_bytes_buffer(self):
        # Create a dummy in-memory 800x600 PNG image
        img = Image.new("RGB", (800, 600), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        tokens_file = self.counter.count_file(buf, model="gpt-4o", detail="high")
        tokens_direct = self.counter.count(800, 600, model="gpt-4o", detail="high")
        assert tokens_file == tokens_direct

    def test_helper_function(self):
        tokens = count_image_tokens(1024, 1024, model="gpt-4o", detail="high")
        assert tokens == 765

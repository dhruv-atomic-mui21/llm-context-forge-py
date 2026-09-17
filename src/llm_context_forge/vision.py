"""
Multimodal Vision Token Sizing Module.

Provides exact and deterministic token calculation strategies for image inputs
across major LLM providers (OpenAI, Anthropic, Google Gemini).
"""

from __future__ import annotations

import io
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class VisionStrategy(ABC):
    """Abstract base strategy for computing vision tokens for a model family."""

    @abstractmethod
    def calculate_tokens(self, width: int, height: int, detail: str = "auto") -> int:
        """
        Calculate token count for given image dimensions.

        Args:
            width: Image width in pixels (> 0).
            height: Image height in pixels (> 0).
            detail: Fidelity tier ("low", "high", or "auto").

        Returns:
            Calculated token integer.
        """
        pass


class OpenAIVisionStrategy(VisionStrategy):
    """
    OpenAI Vision token sizing algorithm (GPT-4o, GPT-4-turbo).

    - Detail 'low': Fixed 85 tokens regardless of dimensions.
    - Detail 'high':
        1. Scale to fit within a 2048 x 2048 box while preserving aspect ratio.
        2. Scale such that the shortest side is at most 768px.
        3. Count 512 x 512 pixel tiles: ceil(w / 512) * ceil(h / 512).
        4. Formula: 170 tokens per tile + 85 base tokens.
    - Detail 'auto': Evaluates dimensions; defaults to low if <= 512x512, otherwise high.
    """

    BASE_TOKENS = 85
    TILE_TOKENS = 170
    TILE_SIZE = 512
    MAX_BOX = 2048
    TARGET_SHORT_SIDE = 768

    def calculate_tokens(self, width: int, height: int, detail: str = "auto") -> int:
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be strictly positive, got {width}x{height}")

        detail_mode = detail.lower()
        if detail_mode == "low":
            return self.BASE_TOKENS

        if detail_mode == "auto":
            if width <= 512 and height <= 512:
                return self.BASE_TOKENS
            detail_mode = "high"

        if detail_mode != "high":
            raise ValueError(f"Invalid detail setting: {detail}. Expected 'low', 'high', or 'auto'.")

        # Step 1: Scale to fit inside 2048 x 2048
        w, h = float(width), float(height)
        if w > self.MAX_BOX or h > self.MAX_BOX:
            scale = self.MAX_BOX / max(w, h)
            w *= scale
            h *= scale

        # Step 2: Scale shortest side to 768px
        min_side = min(w, h)
        if min_side > self.TARGET_SHORT_SIDE:
            scale = self.TARGET_SHORT_SIDE / min_side
            w *= scale
            h *= scale

        # Step 3: Compute 512 x 512 tiles
        tiles_x = math.ceil(w / self.TILE_SIZE)
        tiles_y = math.ceil(h / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Total tokens
        return (total_tiles * self.TILE_TOKENS) + self.BASE_TOKENS


class AnthropicVisionStrategy(VisionStrategy):
    """
    Anthropic Vision token sizing algorithm (Claude 3, 3.5 Sonnet / Haiku / Opus).

    - If either dimension > 1568px, scale down preserving aspect ratio to fit inside 1568x1568.
    - Formula: ceil((width * height) / 750).
    """

    MAX_BOX = 1568
    TOKEN_DIVISOR = 750

    def calculate_tokens(self, width: int, height: int, detail: str = "auto") -> int:
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be strictly positive, got {width}x{height}")

        w, h = float(width), float(height)
        if w > self.MAX_BOX or h > self.MAX_BOX:
            scale = self.MAX_BOX / max(w, h)
            w *= scale
            h *= scale

        tokens = math.ceil((w * h) / self.TOKEN_DIVISOR)
        return max(tokens, 1)


class GoogleVisionStrategy(VisionStrategy):
    """
    Google Gemini Vision token sizing algorithm (Gemini 1.5 Pro, Flash).

    - Gemini tiles images into 768 x 768 patches.
    - Each patch costs 258 tokens.
    - Small images (<= 384x384) use single base patch (258 tokens).
    """

    PATCH_SIZE = 768
    PATCH_TOKENS = 258

    def calculate_tokens(self, width: int, height: int, detail: str = "auto") -> int:
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be strictly positive, got {width}x{height}")

        if width <= 384 and height <= 384:
            return self.PATCH_TOKENS

        tiles_x = math.ceil(width / self.PATCH_SIZE)
        tiles_y = math.ceil(height / self.PATCH_SIZE)
        return tiles_x * tiles_y * self.PATCH_TOKENS


class GenericVisionStrategy(VisionStrategy):
    """Fallback heuristic strategy for models without vendor-specific vision docs."""

    def calculate_tokens(self, width: int, height: int, detail: str = "auto") -> int:
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be strictly positive, got {width}x{height}")
        # Default ~800 pixels per token heuristic with base 85 minimum
        return max(85, math.ceil((width * height) / 800))


class VisionTokenCounter:
    """
    Unified manager for multimodal image token estimation.
    """

    def __init__(self, default_model: str = "gpt-4o"):
        self.default_model = default_model
        self._strategies: Dict[str, VisionStrategy] = {
            "openai": OpenAIVisionStrategy(),
            "anthropic": AnthropicVisionStrategy(),
            "google": GoogleVisionStrategy(),
            "generic": GenericVisionStrategy(),
        }

    def resolve_strategy(self, model: Optional[str] = None) -> VisionStrategy:
        """Resolve the appropriate VisionStrategy based on model identifier."""
        target = (model or self.default_model).lower()

        if any(prefix in target for prefix in ("gpt-4", "gpt-4o", "chatgpt", "o1", "o3")):
            return self._strategies["openai"]
        if "claude" in target:
            return self._strategies["anthropic"]
        if "gemini" in target:
            return self._strategies["google"]

        return self._strategies["generic"]

    def count(
        self,
        width: int,
        height: int,
        model: Optional[str] = None,
        detail: str = "auto"
    ) -> int:
        """
        Calculate token count for an image from its dimensions.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            model: Target model name (defaults to instance default).
            detail: Fidelity tier ("low", "high", "auto").

        Returns:
            Calculated token integer.
        """
        strategy = self.resolve_strategy(model)
        return strategy.calculate_tokens(width, height, detail=detail)

    def count_file(
        self,
        file_input: Union[str, Path, bytes, BinaryIO],
        model: Optional[str] = None,
        detail: str = "auto"
    ) -> int:
        """
        Inspect image file or byte buffer and compute tokens using header dimensions.

        Args:
            file_input: Path string, Path object, raw image bytes, or file-like buffer.
            model: Target model name.
            detail: Fidelity tier ("low", "high", "auto").

        Returns:
            Calculated token integer.

        Raises:
            RuntimeError: If Pillow is not installed.
            ValueError: If image cannot be read or parsed.
        """
        if not _PIL_AVAILABLE:
            raise RuntimeError(
                "Pillow is required for count_file(). Install via `pip install pillow`."
            )

        width, height = self._extract_dimensions(file_input)
        return self.count(width, height, model=model, detail=detail)

    @staticmethod
    def _extract_dimensions(file_input: Union[str, Path, bytes, BinaryIO]) -> Tuple[int, int]:
        """Extract (width, height) without loading entire image pixels into memory."""
        try:
            if isinstance(file_input, (str, Path)):
                with Image.open(file_input) as img:
                    return img.size
            elif isinstance(file_input, bytes):
                with Image.open(io.BytesIO(file_input)) as img:
                    return img.size
            elif hasattr(file_input, "read"):
                # Seekable buffer
                current_pos = file_input.tell() if hasattr(file_input, "tell") else 0
                with Image.open(file_input) as img:
                    size = img.size
                if hasattr(file_input, "seek"):
                    file_input.seek(current_pos)
                return size
            else:
                raise ValueError(f"Unsupported file_input type: {type(file_input)}")
        except Exception as e:
            raise ValueError(f"Failed to extract image dimensions: {e}") from e


def count_image_tokens(
    width: int,
    height: int,
    model: str = "gpt-4o",
    detail: str = "auto"
) -> int:
    """
    Convenience function to compute image tokens directly from dimensions.

    Example:
        >>> count_image_tokens(1024, 768, model="gpt-4o", detail="high")
        765
    """
    counter = VisionTokenCounter(default_model=model)
    return counter.count(width, height, model=model, detail=detail)

"""
Context Compression Engine

Reduces context size while preserving important information.
Strategies:

  - EXTRACTIVE:  Keep only the most important sentences (TF-IDF scoring).
  - TRUNCATE:    Keep the beginning, discard the rest.
  - MIDDLE_OUT:  Keep the start and end, remove the middle.
  - MAP_REDUCE:  Split into chunks → summarise each → merge summaries.
"""

import math
import re
from enum import Enum
from collections import Counter
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


class CompressionStrategy(Enum):
    """Available compression strategies."""
    EXTRACTIVE = "extractive"
    TRUNCATE = "truncate"
    MIDDLE_OUT = "middle_out"
    MAP_REDUCE = "map_reduce"


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    text: str
    original_tokens: int
    compressed_tokens: int
    strategy: str

    @property
    def ratio(self) -> float:
        """Compression ratio (0-1, lower = more compressed)."""
        if self.original_tokens == 0:
            return 1.0
            
        return self.compressed_tokens / self.original_tokens

    @property
    def savings_pct(self) -> float:
        """Percentage of tokens saved."""
        return (1.0 - self.ratio) * 100


class ContextCompressor:
    """
    Compresses text and conversation history to fit token budgets.

    Uses extractive summarisation (sentence scoring), truncation,
    middle-out removal, and map-reduce strategies.
    """

    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, default_model: str = "gpt-4o"):
        """
        Initialise compressor.

        Args:
            default_model: Model for token counting.
        """
        from llm_context_forge.tokenizer import TokenCounter
        self._counter = TokenCounter(default_model)
        self.default_model = default_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        text: str,
        target_tokens: int,
        strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        model: Optional[str] = None,
    ) -> CompressionResult:
        """
        Compress *text* to fit within *target_tokens*.

        Args:
            text:          Text to compress.
            target_tokens: Desired maximum token count.
            strategy:      Compression strategy.
            model:         Model for token counting.

        Returns:
            CompressionResult with the compressed text and stats.
        """
        model = model or self.default_model
        original_tokens = self._counter.count(text, model)

        if original_tokens <= target_tokens:
            return CompressionResult(
                text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                strategy=strategy.value,
            )

        if strategy == CompressionStrategy.EXTRACTIVE:
            result_text = self._compress_extractive(text, target_tokens, model)
        elif strategy == CompressionStrategy.TRUNCATE:
            result_text = self._compress_truncate(text, target_tokens, model)
        elif strategy == CompressionStrategy.MIDDLE_OUT:
            result_text = self.middle_out(text, target_tokens, model)
        elif strategy == CompressionStrategy.MAP_REDUCE:
            result_text = self._compress_map_reduce(text, target_tokens, model)
        else:
            result_text = self._compress_extractive(text, target_tokens, model)

        return CompressionResult(
            text=result_text,
            original_tokens=original_tokens,
            compressed_tokens=self._counter.count(result_text, model),
            strategy=strategy.value,
        )

    def compress_conversation(
        self,
        messages: List[Dict[str, str]],
        target_tokens: int,
        model: Optional[str] = None,
        preserve_recent: int = 4,
    ) -> List[Dict[str, str]]:
        """
        Compress a conversation by summarising older messages.

        Keeps the *preserve_recent* newest non-system messages intact
        and compresses older ones into a summary block.

        Args:
            messages:        ChatML message list.
            target_tokens:   Token budget.
            model:           Model name.
            preserve_recent: Number of recent messages to keep verbatim.

        Returns:
            Compressed message list.
        """
        model = model or self.default_model

        if not messages:
            return []

        # Separate system messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]

        # If already fits, return as-is
        total = sum(self._counter.count(m["content"], model) for m in messages)
        if total <= target_tokens:
            return messages

        # Split into old + recent
        if len(other_msgs) <= preserve_recent:
            recent = other_msgs
            old = []
        else:
            old = other_msgs[:-preserve_recent]
            recent = other_msgs[-preserve_recent:]

        # Calculate budget for summary
        recent_tokens = sum(
            self._counter.count(m["content"], model) for m in recent
        )
        system_tokens = sum(
            self._counter.count(m["content"], model) for m in system_msgs
        )
        summary_budget = max(50, target_tokens - recent_tokens - system_tokens)

        # Compress old messages into a summary
        result: List[Dict[str, str]] = list(system_msgs)

        if old:
            old_text = "\\n".join(
                f"[{m['role']}]: {m['content']}" for m in old
            )
            summary = self._compress_extractive(old_text, summary_budget, model)
            result.append({
                "role": "system",
                "content": f"[Conversation summary]: {summary}",
            })

        result.extend(recent)
        return result

    def extract_key_sentences(
        self,
        text: str,
        n: int = 5,
    ) -> List[str]:
        """
        Extract the *n* most important sentences using TF-IDF scoring.

        Args:
            text: Input text.
            n:    Number of sentences to extract.

        Returns:
            List of key sentences in original order.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= n:
            return sentences

        scores = self._score_sentences(sentences)
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[:n])

        return [sentences[i] for i in top_indices]

    def middle_out(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """
        Keep the start and end of *text*, removing the middle.

        Useful for long documents where the beginning (context setup)
        and end (recent/conclusion) matter most.

        Args:
            text:       Text to compress.
            max_tokens: Target token limit.
            model:      Model name.

        Returns:
            Compressed text with middle removed.
        """
        model = model or self.default_model

        if self._counter.count(text, model) <= max_tokens:
            return text

        marker = "\\n\\n[... middle content removed for brevity ...]\\n\\n"
        marker_tokens = self._counter.count(marker, model)
        available = max_tokens - marker_tokens

        # Split budget: 60% start, 40% end
        start_budget = int(available * 0.6)
        end_budget = available - start_budget

        start_text = self._counter.truncate_to_fit(text, start_budget, model, suffix="")
        end_text = self._get_tail(text, end_budget, model)

        return start_text + marker + end_text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compress_extractive(
        self,
        text: str,
        target_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Keep most important sentences until target is met."""
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:target_tokens * 4]  # fallback

        scores = self._score_sentences(sentences)
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: scores[i],
            reverse=True,
        )

        selected_indices: List[int] = []
        token_sum = 0

        for idx in ranked:
            s_tokens = self._counter.count(sentences[idx], model)
            if token_sum + s_tokens <= target_tokens:
                selected_indices.append(idx)
                token_sum += s_tokens

        # Preserve original order
        selected_indices.sort()
        return " ".join(sentences[i] for i in selected_indices)

    def _compress_truncate(
        self,
        text: str,
        target_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Simple prefix truncation."""
        return self._counter.truncate_to_fit(text, target_tokens, model)

    def _compress_map_reduce(
        self,
        text: str,
        target_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """
        Map-reduce style: split → extract key sentences per chunk → merge.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:target_tokens * 4]

        # Create chunks of ~10 sentences
        chunk_size = 10
        chunks = [
            sentences[i:i + chunk_size]
            for i in range(0, len(sentences), chunk_size)
        ]

        # Extract 2 key sentences per chunk
        key_per_chunk = max(1, target_tokens // (len(chunks) * 20))
        summaries: List[str] = []
        for chunk in chunks:
            key_sents = self._score_and_select(chunk, min(key_per_chunk, len(chunk)))
            summaries.extend(key_sents)

        result = " ".join(summaries)

        # Final trim if still too long
        if self._counter.count(result, model) > target_tokens:
            result = self._counter.truncate_to_fit(result, target_tokens, model)

        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        parts = self._SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score sentences using a simple TF-IDF-inspired heuristic.

        Factors:
          - Term frequency (unique words).
          - Position bonus (first/last sentences score higher).
          - Length penalty (very short sentences score lower).
        """
        # Build document frequency
        all_words: List[str] = []
        sentence_words: List[List[str]] = []
        for s in sentences:
            words = re.findall(r'\w+', s.lower())
            sentence_words.append(words)
            all_words.extend(words)

        word_freq = Counter(all_words)
        num_sentences = len(sentences)
        scores: List[float] = []

        for i, words in enumerate(sentence_words):
            if not words:
                scores.append(0.0)
                continue

            # TF component: sum of inverse frequency
            tf_score = sum(1.0 / (word_freq[w] + 1) for w in set(words))

            # Position bonus
            position_score = 0.0
            if i == 0:
                position_score = 2.0
            elif i == num_sentences - 1:
                position_score = 1.5
            elif i < num_sentences * 0.2:
                position_score = 1.0

            # Length factor (prefer medium-length sentences)
            length_factor = min(len(words) / 10.0, 1.5)

            scores.append(tf_score + position_score + length_factor)

        return scores

    def _score_and_select(
        self,
        sentences: List[str],
        n: int,
    ) -> List[str]:
        """Score sentences and return top *n* in original order."""
        if len(sentences) <= n:
            return sentences

        scores = self._score_sentences(sentences)
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[:n])
        return [sentences[i] for i in top_indices]

    def _get_tail(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Get the last *max_tokens* worth of text."""
        words = text.split()
        tail: List[str] = []
        token_sum = 0

        for word in reversed(words):
            w_tokens = self._counter.count(word, model)
            if token_sum + w_tokens > max_tokens:
                break
            tail.append(word)
            token_sum += w_tokens

        tail.reverse()
        return " ".join(tail)


if __name__ == "__main__":
    print("LLM Context Forge — Context Compressor")
    print("=" * 40)
    print("Usage:")
    print("  compressor = ContextCompressor()")
    print("  result = compressor.compress(text, target_tokens=500)")
    print(f"  print(result.ratio, result.savings_pct)")

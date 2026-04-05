"""
Context Window Manager

Provides priority-based context assembly and conversation management
for LLM interactions. Key features:

  - Tag content blocks with priorities; the manager greedily packs the
    highest-priority blocks first.
  - Conversation history auto-trimming that preserves recent messages.
  - Token usage analytics per block and overall.
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


class Priority(IntEnum):
    """Block priority — lower numeric value = higher importance."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    OPTIONAL = 4


@dataclass
class ContextBlock:
    """
    A single unit of context to be packed into a context window.

    Attributes:
        content:     The text content.
        priority:    Importance level.
        label:       Human-readable label for tracking.
        token_count: Number of tokens (populated by ContextWindow).
        metadata:    Arbitrary metadata (source file, timestamp, etc.).
        included:    Whether this block was included in the last assembly.
    """
    content: str
    priority: Priority
    label: str
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    included: bool = False


class ContextWindow:
    """
    Priority-based context window manager.

    Add content blocks with priorities, then call ``assemble()`` to
    greedily pack the most important content into *max_tokens*.
    """

    def __init__(self, default_model: str = "gpt-4o"):
        """
        Initialise context window.

        Args:
            default_model: Model for token counting.
        """
        from layoutlm_forge.tokenizer import TokenCounter
        self._counter = TokenCounter(default_model)
        self.default_model = default_model
        self._blocks: List[ContextBlock] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_block(
        self,
        content: str,
        priority: Priority = Priority.MEDIUM,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextBlock:
        """
        Add a content block to the window.

        Args:
            content:  Text content.
            priority: Importance level.
            label:    Human-readable label.
            metadata: Arbitrary metadata.

        Returns:
            The created ContextBlock.
        """
        token_count = self._counter.count(content)
        block = ContextBlock(
            content=content,
            priority=priority,
            label=label or f"block_{len(self._blocks)}",
            token_count=token_count,
            metadata=metadata or {},
        )
        self._blocks.append(block)
        return block

    def assemble(
        self,
        max_tokens: int = 4096,
        separator: str = "\\n\\n---\\n\\n",
    ) -> str:
        """
        Assemble the context window by greedily packing blocks
        in priority order.

        Args:
            max_tokens: Maximum token budget.
            separator:  String between blocks.

        Returns:
            Assembled context string.
        """
        sep_tokens = self._counter.count(separator)

        # Sort by priority (ascending = most important first)
        sorted_blocks = sorted(self._blocks, key=lambda b: b.priority)

        included_texts: List[str] = []
        used_tokens = 0

        for block in sorted_blocks:
            cost = block.token_count + (sep_tokens if included_texts else 0)
            if used_tokens + cost <= max_tokens:
                block.included = True
                included_texts.append(block.content)
                used_tokens += cost
            else:
                block.included = False

        return separator.join(included_texts)

    def usage(self) -> Dict[str, Any]:
        """
        Return token usage statistics.

        Returns:
            Dict with per-block and total usage info.
        """
        total = sum(b.token_count for b in self._blocks)
        included = sum(b.token_count for b in self._blocks if b.included)
        excluded = total - included

        return {
            "total_tokens": total,
            "included_tokens": included,
            "excluded_tokens": excluded,
            "num_blocks": len(self._blocks),
            "num_included": sum(1 for b in self._blocks if b.included),
            "num_excluded": sum(1 for b in self._blocks if not b.included),
            "blocks": [
                {
                    "label": b.label,
                    "priority": b.priority.name,
                    "tokens": b.token_count,
                    "included": b.included,
                }
                for b in self._blocks
            ],
        }

    def overflow_blocks(self) -> List[ContextBlock]:
        """Return blocks that did not fit in the last assembly."""
        return [b for b in self._blocks if not b.included]

    def included_blocks(self) -> List[ContextBlock]:
        """Return blocks that were included in the last assembly."""
        return [b for b in self._blocks if b.included]

    def clear(self) -> None:
        """Remove all blocks."""
        self._blocks.clear()

    def remove_block(self, label: str) -> bool:
        """Remove a block by label. Returns True if found."""
        for i, b in enumerate(self._blocks):
            if b.label == label:
                self._blocks.pop(i)
                return True
        return False

    def to_messages(
        self,
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> List[Dict[str, str]]:
        """
        Build a ChatML message list with the assembled context
        injected as the system prompt.

        Args:
            system_prompt: Base system prompt.
            max_tokens:    Token budget.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        context = self.assemble(max_tokens=max_tokens)
        full_system = f"{system_prompt}\\n\\n{context}" if context else system_prompt
        return [{"role": "system", "content": full_system}]


class ConversationManager:
    """
    Manages multi-turn conversation history with automatic trimming.

    Keeps the most recent messages when the token budget is exceeded,
    optionally preserving the system message.
    """

    def __init__(self, default_model: str = "gpt-4o"):
        """
        Initialise conversation manager.

        Args:
            default_model: Model for token counting.
        """
        from layoutlm_forge.tokenizer import TokenCounter
        self._counter = TokenCounter(default_model)
        self.default_model = default_model
        self._messages: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a message to the conversation.

        Args:
            role:     ``"system"``, ``"user"``, or ``"assistant"``.
            content:  Message content.
            metadata: Optional metadata (timestamp auto-added).
        """
        self._messages.append({
            "role": role,
            "content": content,
            "tokens": self._counter.count(content),
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        })

    def get_context(
        self,
        max_tokens: int = 4096,
        preserve_system: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Return messages trimmed to fit *max_tokens*.

        The system message (if any) is always preserved. Older user/assistant
        messages are dropped first.

        Args:
            max_tokens:      Token budget.
            preserve_system: Keep the system message.

        Returns:
            Trimmed list of ``{"role": ..., "content": ...}`` dicts.
        """
        system_msgs = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        budget = max_tokens
        result: List[Dict[str, str]] = []

        # Reserve space for system messages
        if preserve_system and system_msgs:
            for sm in system_msgs:
                budget -= sm["tokens"]
                result.append({"role": sm["role"], "content": sm["content"]})

        # Add messages from newest to oldest
        selected: List[Dict[str, str]] = []
        for msg in reversed(other_msgs):
            if budget - msg["tokens"] >= 0:
                budget -= msg["tokens"]
                selected.append({"role": msg["role"], "content": msg["content"]})
            else:
                break

        # Reverse to restore chronological order
        selected.reverse()
        result.extend(selected)

        return result

    def token_usage(self) -> Dict[str, Any]:
        """
        Return conversation token statistics.

        Returns:
            Dict with total tokens, per-role breakdown, and message count.
        """
        total = sum(m["tokens"] for m in self._messages)
        by_role: Dict[str, int] = {}
        for m in self._messages:
            by_role[m["role"]] = by_role.get(m["role"], 0) + m["tokens"]

        return {
            "total_tokens": total,
            "message_count": len(self._messages),
            "by_role": by_role,
        }

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Raw message list."""
        return list(self._messages)


if __name__ == "__main__":
    print("LayoutLM Forge — Context Window Manager")
    print("=" * 40)
    print("Usage:")
    print("  window = ContextWindow()")
    print('  window.add_block("System prompt...", Priority.CRITICAL)')
    print('  window.add_block("User docs...", Priority.HIGH)')
    print("  assembled = window.assemble(max_tokens=4096)")

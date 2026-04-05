"""
Tests for ContextWindow and ConversationManager
"""

import pytest
from layoutlm_forge.context import ContextWindow, ContextBlock, Priority, ConversationManager

class TestContextWindow:
    def setup_method(self):
        self.window = ContextWindow("gpt-4o")

    def test_add_block(self):
        block = self.window.add_block("Hello", Priority.HIGH, "greeting")
        assert block.label == "greeting"
        assert block.priority == Priority.HIGH
        assert block.token_count > 0

    def test_add_block_default_priority(self):
        block = self.window.add_block("Hello")
        assert block.priority == Priority.MEDIUM

    def test_assemble_empty(self):
        result = self.window.assemble(max_tokens=100)
        assert result == ""

    def test_assemble_single_block(self):
        self.window.add_block("Test content", Priority.HIGH, "test")
        result = self.window.assemble(max_tokens=1000)
        assert "Test content" in result

    def test_assemble_priority_order(self):
        self.window.add_block("Low priority", Priority.LOW, "low")
        self.window.add_block("Critical stuff", Priority.CRITICAL, "crit")
        result = self.window.assemble(max_tokens=1000)
        # Critical should come first in the assembled text
        assert result.index("Critical stuff") < result.index("Low priority")

    def test_assemble_respects_token_limit(self):
        self.window.add_block("word " * 100, Priority.HIGH, "big")
        self.window.add_block("small", Priority.LOW, "small")
        result = self.window.assemble(max_tokens=20)
        # Should include whatever fits
        assert len(result) > 0

    def test_overflow_blocks(self):
        self.window.add_block("word " * 100, Priority.LOW, "big")
        self.window.assemble(max_tokens=5)
        overflow = self.window.overflow_blocks()
        assert len(overflow) >= 1

    def test_included_blocks(self):
        self.window.add_block("Hi", Priority.CRITICAL, "small")
        self.window.assemble(max_tokens=1000)
        included = self.window.included_blocks()
        assert len(included) == 1

    def test_usage_stats(self):
        self.window.add_block("Hello", Priority.HIGH, "a")
        self.window.add_block("World", Priority.LOW, "b")
        self.window.assemble(max_tokens=1000)
        usage = self.window.usage()
        assert "total_tokens" in usage
        assert "included_tokens" in usage
        assert "num_blocks" in usage
        assert usage["num_blocks"] == 2

    def test_clear(self):
        self.window.add_block("Hello", Priority.HIGH, "a")
        self.window.clear()
        assert self.window.assemble(max_tokens=100) == ""

    def test_remove_block(self):
        self.window.add_block("Hello", Priority.HIGH, "removable")
        assert self.window.remove_block("removable") is True
        assert self.window.remove_block("nonexistent") is False

    def test_to_messages(self):
        self.window.add_block("Context info", Priority.HIGH, "info")
        msgs = self.window.to_messages("You are helpful.", max_tokens=1000)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "Context info" in msgs[0]["content"]


class TestPriority:
    def test_ordering(self):
        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH < Priority.MEDIUM
        assert Priority.MEDIUM < Priority.LOW
        assert Priority.LOW < Priority.OPTIONAL


class TestConversationManager:
    def setup_method(self):
        self.conv = ConversationManager("gpt-4o")

    def test_add_message(self):
        self.conv.add_message("user", "Hello")
        assert len(self.conv.messages) == 1
        assert self.conv.messages[0]["role"] == "user"

    def test_get_context_empty(self):
        result = self.conv.get_context(max_tokens=100)
        assert result == []

    def test_get_context_preserves_system(self):
        self.conv.add_message("system", "You are helpful.")
        self.conv.add_message("user", "Hi")
        result = self.conv.get_context(max_tokens=1000)
        roles = [m["role"] for m in result]
        assert "system" in roles

    def test_get_context_trims_old_messages(self):
        self.conv.add_message("system", "System prompt")
        for i in range(20):
            self.conv.add_message("user", f"Message {i} " * 50)
        result = self.conv.get_context(max_tokens=200)
        assert len(result) < 21

    def test_get_context_keeps_recent(self):
        self.conv.add_message("user", "Old message")
        self.conv.add_message("user", "Recent message")
        result = self.conv.get_context(max_tokens=1000)
        contents = [m["content"] for m in result]
        assert "Recent message" in contents

    def test_token_usage(self):
        self.conv.add_message("user", "Hello")
        self.conv.add_message("assistant", "Hi there!")
        usage = self.conv.token_usage()
        assert usage["total_tokens"] > 0
        assert usage["message_count"] == 2
        assert "user" in usage["by_role"]
        assert "assistant" in usage["by_role"]

    def test_clear(self):
        self.conv.add_message("user", "Hello")
        self.conv.clear()
        assert len(self.conv.messages) == 0

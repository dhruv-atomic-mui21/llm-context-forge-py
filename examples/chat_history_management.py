"""
Chat History Management Example

Shows how to automatically trim long conversation histories
while preserving recent messages and system prompts.
"""

from llm_context_forge.context import ConversationManager

def main():
    manager = ConversationManager("gpt-4o")

    # 1. Setup system prompt
    manager.add_message("system", "You are an AI assistant helping with Python code.")

    # 2. Simulate a long conversation
    for i in range(1, 11):
        manager.add_message("user", f"Here is block {i} of my code: " + "print('hello')\n" * i * 50)
        manager.add_message("assistant", f"Code block {i} looks good.")

    usage = manager.token_usage()
    print(f"Total messages: {usage['message_count']}")
    print(f"Total tokens in raw history: {usage['total_tokens']:,}")

    # 3. Retrieve context to fit inside a limited budget
    # Let's say we only have 500 tokens budget left for the history
    trimmed = manager.get_context(max_tokens=500, preserve_system=True)
    
    print("\n--- Trimmed Context ---")
    print(f"Trimmed messages: {len(trimmed)}")
    for msg in trimmed:
        content = msg['content']
        display = content if len(content) < 50 else content[:50] + "..."
        print(f"[{msg['role'].upper()}] {display}")

if __name__ == "__main__":
    main()

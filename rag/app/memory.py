# New file: rag/memory/conversation_memory.py
"""
Lightweight conversation memory wrapper using LangChain.
This ONLY handles conversation history - all your custom retrieval/embedding stays the same.
"""

from typing import List, Dict
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage


class ConversationMemory:
    """
    Lightweight wrapper around LangChain memory.
    """

    def __init__(self, k: int = 10, memory_key: str = "chat_history"):
        """
        Initialize conversation memory.

        Args:
            k: Number of recent messages to keep (default: 10 = 5 exchanges)
            memory_key: Key for memory in LangChain format
        """
        self.memory = ConversationBufferWindowMemory(
            k=k, return_messages=True, memory_key=memory_key
        )
        self.k = k

    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        elif role == "assistant":
            self.memory.chat_memory.add_ai_message(content)

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history in simple format.
        Returns: [{"role": "user/assistant", "content": "..."}, ...]
        """
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """
        Get history formatted for OpenAI/Gemini API calls.
        Returns: [{"role": "user/assistant", "content": "..."}, ...]
        """
        return self.get_history()

    def clear(self):
        """Clear conversation history."""
        self.memory.clear()

    def get_buffer_string(self) -> str:
        """Get history as a formatted string (for prompt injection)."""
        return self.memory.buffer

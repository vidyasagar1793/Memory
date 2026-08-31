# memory.py
import logging
from typing import List
from .core_models import Message, Role

# Setup logging for tracing our agent's inner workings
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShortTermMemory")

class ShortTermMemory:
    def __init__(self, max_tokens: int = 4000):
        """
        Initializes the working memory.
        :param max_tokens: The safety limit before we start pruning older messages.
        """
        self.history: List[Message] = []
        self.max_tokens = max_tokens
        
    def add(self, message: Message):
        """Adds a new message to memory and triggers the compaction check."""
        self.history.append(message)
        logger.info(f"Added {message.role.value} message. Total messages: {len(self.history)}")
        self._enforce_token_limit()

    def get_context(self) -> List[dict]:
        """Returns the cleanly formatted payload ready to be sent to an LLM."""
        return [msg.to_llm_format() for msg in self.history]

    def _estimate_tokens(self, text: str) -> int:
        """A rough heuristic: 1 word is roughly 1.3 tokens."""
        return int(len(text.split()) * 1.3)

    def _get_current_token_count(self) -> int:
        return sum(self._estimate_tokens(msg.content) for msg in self.history)

    def _enforce_token_limit(self):
        """
        Sliding Window logic: If we exceed max_tokens, pop the oldest messages,
        but strictly preserve the SYSTEM prompt at index 0.
        """
        while self._get_current_token_count() > self.max_tokens and len(self.history) > 1:
            # We want to preserve history[0] assuming it's the system prompt.
            # So we remove the second item (index 1).
            if len(self.history) > 1 and self.history[1].role != Role.SYSTEM:
                removed_msg = self.history.pop(1)
                logger.warning(f"Memory limit reached. Pruned older message from role: {removed_msg.role.value}")
            else:
                # Edge case: If for some reason we only have system prompts left, break.
                break

    def clear(self, keep_system_prompt: bool = True):
        """Wipes memory for a fresh task."""
        if keep_system_prompt and self.history and self.history[0].role == Role.SYSTEM:
            self.history = [self.history[0]]
        else:
            self.history = []
        logger.info("Memory cleared.")
